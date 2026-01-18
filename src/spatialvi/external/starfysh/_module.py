"""Starfysh module for spatial deconvolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers
from torch import nn
from torch.distributions import Dirichlet

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class StarfyshModule(BaseModuleClass):
    """Neural network module for Starfysh deconvolution.

    Parameters
    ----------
    n_genes
        Number of genes.
    n_cell_types
        Number of cell types.
    n_factors
        Number of archetypal factors per cell type.
    n_hidden
        Number of hidden units.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    use_histology
        Whether to use histology features.
    histology_dim
        Dimension of histology features (if used).
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_factors: int = 5,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_histology: bool = False,
        histology_dim: int = 0,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_factors = n_factors
        self.use_histology = use_histology

        input_dim = n_genes
        if use_histology and histology_dim > 0:
            input_dim += histology_dim

        # Encoder
        self.encoder = FCLayers(
            n_in=input_dim,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )

        # Proportion predictor (Dirichlet parameters)
        self.proportion_head = nn.Sequential(
            nn.Linear(n_hidden, n_cell_types),
            nn.Softplus(),
        )

        # Cell type-specific factor loadings
        self.factor_loadings = nn.Parameter(torch.randn(n_cell_types, n_factors, n_genes))

        # Factor weights encoder
        self.factor_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_cell_types * n_factors),
        )

        # Library size encoder
        self.library_encoder = FCLayers(
            n_in=n_genes,
            n_out=1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Prior concentration for proportions
        self.register_buffer("prior_concentration", torch.ones(n_cell_types))

    def _get_inference_input(
        self,
        tensors: dict[str, Tensor],
        **kwargs,
    ) -> dict[str, Tensor | None]:
        """Get input for inference."""
        return {
            "x": tensors.get("X"),
            "histology": tensors.get("histology"),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, Tensor],
        inference_outputs: dict[str, Tensor | Distribution],
        **kwargs,
    ) -> dict[str, Tensor | None]:
        """Get input for generative."""
        return {
            "proportions": inference_outputs["proportions"],
            "factor_weights": inference_outputs["factor_weights"],
            "library": inference_outputs["library"],
        }

    def inference(
        self,
        x: Tensor,
        histology: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor | Distribution]:
        """Run inference.

        Parameters
        ----------
        x
            Gene expression tensor.
        histology
            Optional histology features.

        Returns
        -------
        Dictionary of inference outputs.
        """
        x_log = torch.log1p(x)

        # Concatenate histology if available
        if self.use_histology and histology is not None:
            encoder_input = torch.cat([x_log, histology], dim=-1)
        else:
            encoder_input = x_log

        # Encode
        h = self.encoder(encoder_input)

        # Predict proportion parameters (Dirichlet concentration)
        concentration = self.proportion_head(h) + 1.0  # Add 1 to ensure > 0

        # Sample proportions from Dirichlet
        q_proportions = Dirichlet(concentration)
        proportions = q_proportions.rsample()

        # Get factor weights
        factor_logits = self.factor_encoder(h)
        factor_logits = factor_logits.view(-1, self.n_cell_types, self.n_factors)
        factor_weights = torch.softmax(factor_logits, dim=-1)

        # Library size
        library = self.library_encoder(x_log)
        library = torch.exp(library)

        return {
            "proportions": proportions,
            "concentration": concentration,
            "q_proportions": q_proportions,
            "factor_weights": factor_weights,
            "library": library,
            "h": h,
        }

    def generative(
        self,
        proportions: Tensor,
        factor_weights: Tensor,
        library: Tensor,
        **kwargs,
    ) -> dict[str, Tensor | Distribution]:
        """Run generative model.

        Parameters
        ----------
        proportions
            Cell type proportions.
        factor_weights
            Factor weights per cell type.
        library
            Library size.

        Returns
        -------
        Dictionary of generative outputs.
        """
        # Get factor loadings (normalized)
        loadings = torch.softmax(self.factor_loadings, dim=-1)

        # Weighted factor expression per cell type
        ct_expr = torch.einsum(
            "bcf,cfg->bcg",
            factor_weights,
            loadings,
        )  # (batch, n_cell_types, n_genes)

        # Mix by proportions
        px_rate = torch.einsum("bc,bcg->bg", proportions, ct_expr)

        # Scale by library
        px_rate = px_rate * library

        return {
            "px_rate": px_rate,
            "px_scale": px_rate / library,
            "ct_expression": ct_expr,
        }

    def loss(
        self,
        tensors: dict[str, Tensor],
        inference_outputs: dict[str, Tensor | Distribution],
        generative_outputs: dict[str, Tensor | Distribution],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute loss.

        Parameters
        ----------
        tensors
            Input tensors.
        inference_outputs
            Inference outputs.
        generative_outputs
            Generative outputs.
        kl_weight
            Weight for KL divergence.

        Returns
        -------
        LossOutput.
        """
        x = tensors["X"]
        px_rate = generative_outputs["px_rate"]

        # Reconstruction loss (Poisson likelihood approximation)
        reconst_loss = torch.sum(
            px_rate - x * torch.log(px_rate + 1e-8),
            dim=-1,
        )

        # KL divergence for Dirichlet
        q_proportions = inference_outputs["q_proportions"]
        p_proportions = Dirichlet(self.prior_concentration.expand(x.shape[0], -1))
        kl_proportions = torch.distributions.kl_divergence(q_proportions, p_proportions)

        # Total loss
        loss = torch.mean(reconst_loss + kl_weight * kl_proportions)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_proportions,
        )

    @auto_move_data
    def forward(
        self,
        tensors: dict[str, Tensor],
        get_inference_input_kwargs: dict | None = None,
        get_generative_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        generative_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss: bool = True,
    ) -> tuple[dict, dict] | tuple[dict, dict, LossOutput]:
        """Forward pass."""
        get_inference_input_kwargs = get_inference_input_kwargs or {}
        get_generative_input_kwargs = get_generative_input_kwargs or {}
        inference_kwargs = inference_kwargs or {}
        generative_kwargs = generative_kwargs or {}
        loss_kwargs = loss_kwargs or {}

        inference_inputs = self._get_inference_input(tensors, **get_inference_input_kwargs)
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)

        generative_inputs = self._get_generative_input(tensors, inference_outputs, **get_generative_input_kwargs)
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)

        if compute_loss:
            losses = self.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
            return inference_outputs, generative_outputs, losses

        return inference_outputs, generative_outputs
