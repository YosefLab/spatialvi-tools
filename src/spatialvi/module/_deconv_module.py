"""Deconvolution module for spatial transcriptomics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.distributions import Normal, Dirichlet
from torch.distributions import kl_divergence as kl

from scvi.distributions import NegativeBinomial
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import FCLayers

from spatialvi.module._base import BaseSpatialModule

if TYPE_CHECKING:
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class DeconvolutionModule(BaseSpatialModule):
    """Module for spatial deconvolution.

    This module estimates cell type proportions from spatial
    transcriptomics data using reference single-cell data.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_cell_types
        Number of cell types to deconvolve.
    n_batch
        Number of batches.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate for neural networks.
    use_subcell_variation
        Whether to model within-cell-type variation.
    n_subcell_factors
        Number of subcell factors if using subcell variation.
    """

    def __init__(
        self,
        n_input: int,
        n_cell_types: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_subcell_variation: bool = False,
        n_subcell_factors: int = 5,
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.n_cell_types = n_cell_types
        self.use_subcell_variation = use_subcell_variation
        self.n_subcell_factors = n_subcell_factors

        # Proportion encoder
        self.proportion_encoder = FCLayers(
            n_in=n_input,
            n_out=n_cell_types,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.proportion_logits = nn.Linear(n_cell_types, n_cell_types)

        # Cell type-specific expression profiles (basis)
        self.cell_type_profiles = nn.Parameter(torch.randn(n_cell_types, n_input))

        # Subcell variation encoder
        if use_subcell_variation:
            self.subcell_encoder = FCLayers(
                n_in=n_input,
                n_out=n_subcell_factors * n_cell_types,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
            self.subcell_decoder = nn.ModuleList([
                nn.Linear(n_subcell_factors, n_input)
                for _ in range(n_cell_types)
            ])

        # Dispersion
        self.px_r = nn.Parameter(torch.randn(n_input))

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Get input for inference network."""
        return {
            "x": tensors.get("X"),
            "batch_index": tensors.get("batch"),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution],
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Get input for generative network."""
        return {
            "proportions": inference_outputs["proportions"],
            "subcell_factors": inference_outputs.get("subcell_factors"),
            "library": inference_outputs["library"],
            "batch_index": tensors.get("batch"),
        }

    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the inference network.

        Parameters
        ----------
        x
            Gene expression tensor.
        batch_index
            Batch indices.

        Returns
        -------
        Dictionary of inference outputs.
        """
        x_ = torch.log1p(x)

        # Encode proportions
        h = self.proportion_encoder(x_)
        proportion_logits = self.proportion_logits(h)
        proportions = torch.softmax(proportion_logits, dim=-1)

        # Alternative: Dirichlet posterior
        proportion_alpha = torch.exp(proportion_logits) + 1e-6
        qc = Dirichlet(proportion_alpha)

        # Library size
        library = x.sum(dim=-1, keepdim=True)

        result = {
            "proportions": proportions,
            "qc": qc,
            "library": library,
            "proportion_alpha": proportion_alpha,
        }

        # Subcell variation
        if self.use_subcell_variation:
            subcell_h = self.subcell_encoder(x_)
            subcell_factors = subcell_h.view(
                x.shape[0], self.n_cell_types, self.n_subcell_factors
            )
            result["subcell_factors"] = subcell_factors

        return result

    def generative(
        self,
        proportions: torch.Tensor,
        library: torch.Tensor,
        subcell_factors: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the generative network.

        Parameters
        ----------
        proportions
            Cell type proportions.
        library
            Library size.
        subcell_factors
            Subcell variation factors.
        batch_index
            Batch indices.

        Returns
        -------
        Dictionary of generative outputs.
        """
        # Get cell type profiles
        profiles = torch.softmax(self.cell_type_profiles, dim=-1)

        # Add subcell variation
        if self.use_subcell_variation and subcell_factors is not None:
            variations = []
            for ct in range(self.n_cell_types):
                var = self.subcell_decoder[ct](subcell_factors[:, ct])
                variations.append(var)
            variations = torch.stack(variations, dim=1)  # (batch, n_ct, n_genes)
            profiles = profiles.unsqueeze(0) + 0.1 * variations
            profiles = torch.softmax(profiles, dim=-1)
        else:
            profiles = profiles.unsqueeze(0).expand(proportions.shape[0], -1, -1)

        # Compute expected expression
        # proportions: (batch, n_ct)
        # profiles: (batch, n_ct, n_genes)
        px_scale = torch.einsum("bc,bcg->bg", proportions, profiles)
        px_rate = library * px_scale

        px_r = torch.exp(self.px_r)

        return {
            "px_rate": px_rate,
            "px_r": px_r,
            "px_scale": px_scale,
            "proportions": proportions,
            "cell_type_profiles": profiles,
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution],
        generative_outputs: dict[str, torch.Tensor | Distribution],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the loss.

        Parameters
        ----------
        tensors
            Dictionary of input tensors.
        inference_outputs
            Dictionary of inference outputs.
        generative_outputs
            Dictionary of generative outputs.
        kl_weight
            Weight for KL divergence term.

        Returns
        -------
        LossOutput.
        """
        x = tensors["X"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]

        # Reconstruction loss
        px = NegativeBinomial(mu=px_rate, theta=px_r)
        reconst_loss = -px.log_prob(x).sum(dim=-1)

        # KL divergence for proportions (Dirichlet prior)
        qc = inference_outputs["qc"]
        prior_alpha = torch.ones_like(inference_outputs["proportion_alpha"])
        prior = Dirichlet(prior_alpha)
        kl_proportions = kl(qc, prior)

        # Total loss
        loss = torch.mean(reconst_loss + kl_weight * kl_proportions)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_proportions,
        )

    @torch.inference_mode()
    def get_cell_type_expression(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get deconvolved cell type-specific expression.

        Parameters
        ----------
        tensors
            Dictionary of input tensors.

        Returns
        -------
        Cell type-specific expression of shape (n_spots, n_cell_types, n_genes).
        """
        inference_inputs = self._get_inference_input(tensors)
        inference_outputs = self.inference(**inference_inputs)

        generative_inputs = self._get_generative_input(tensors, inference_outputs)
        generative_outputs = self.generative(**generative_inputs)

        profiles = generative_outputs["cell_type_profiles"]
        proportions = generative_outputs["proportions"]
        library = inference_outputs["library"]

        # Cell type specific expression
        ct_expression = profiles * proportions.unsqueeze(-1) * library.unsqueeze(-1)

        return ct_expression
