"""Niche-aware module for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.distributions import Normal, Categorical, Dirichlet
from torch.distributions import kl_divergence as kl

from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers

from spatialvi.module._base import BaseSpatialModule

if TYPE_CHECKING:
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class NicheModule(BaseSpatialModule):
    """Module for niche-aware spatial analysis.

    This module models cellular niches by learning representations
    that capture both cell-intrinsic and niche-specific factors.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_labels
        Number of cell type labels.
    n_batch
        Number of batches.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_niche_factors
        Number of niche factors.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate for neural networks.
    use_attention
        Whether to use attention for neighbor aggregation.
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_niche_factors: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_attention: bool = True,
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

        self.n_labels = n_labels
        self.n_niche_factors = n_niche_factors
        self.use_attention = use_attention

        # Cell-intrinsic encoder
        self.z_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent,
            n_cat_list=[n_batch] if n_batch > 0 else None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Niche composition encoder
        self.niche_encoder = FCLayers(
            n_in=n_labels,
            n_out=n_niche_factors,
            n_layers=2,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Attention mechanism for neighbors
        if use_attention:
            self.neighbor_attention = nn.MultiheadAttention(
                embed_dim=n_hidden,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.neighbor_proj = nn.Linear(n_input, n_hidden)

        # Decoder for expression
        self.decoder = FCLayers(
            n_in=n_latent + n_niche_factors,
            n_out=n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Cell type classifier
        self.classifier = nn.Linear(n_latent, n_labels)

        # Output layers
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_input, n_input),
            nn.Softmax(dim=-1),
        )
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
            "niche_composition": tensors.get("niche_composition"),
            "neighbor_indices": tensors.get("neighbor_indices"),
            "neighbor_expr": tensors.get("neighbor_expr"),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution],
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Get input for generative network."""
        return {
            "z": inference_outputs["z"],
            "niche_z": inference_outputs["niche_z"],
            "library": inference_outputs["library"],
            "batch_index": tensors.get("batch"),
        }

    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        niche_composition: torch.Tensor | None = None,
        neighbor_indices: torch.Tensor | None = None,
        neighbor_expr: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the inference network.

        Parameters
        ----------
        x
            Gene expression tensor.
        batch_index
            Batch indices.
        niche_composition
            Neighborhood cell type composition.
        neighbor_indices
            Indices of neighboring cells.
        neighbor_expr
            Expression of neighboring cells.

        Returns
        -------
        Dictionary of inference outputs.
        """
        x_ = torch.log1p(x)

        # Encode cell-intrinsic factors
        # Encoder returns (qz_m, qz_v, z)
        if batch_index is not None:
            qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        else:
            qz_m, qz_v, z = self.z_encoder(x_)

        # Create distribution for KL computation
        qz = Normal(qz_m, qz_v.sqrt())

        # Encode niche factors
        if niche_composition is not None:
            niche_z = self.niche_encoder(niche_composition)
        elif self.use_attention and neighbor_expr is not None:
            # Use attention over neighbors
            neighbor_h = self.neighbor_proj(torch.log1p(neighbor_expr))
            query = self.neighbor_proj(x_).unsqueeze(1)
            attn_out, _ = self.neighbor_attention(query, neighbor_h, neighbor_h)
            niche_z = attn_out.squeeze(1)[:, :self.n_niche_factors]
        else:
            niche_z = torch.zeros(x.shape[0], self.n_niche_factors, device=x.device)

        # Library size
        library = x.sum(dim=-1, keepdim=True)

        # Cell type prediction
        logits = self.classifier(z)

        return {
            "z": z,
            "qz": qz,
            "niche_z": niche_z,
            "library": library,
            "logits": logits,
        }

    def generative(
        self,
        z: torch.Tensor,
        niche_z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the generative network.

        Parameters
        ----------
        z
            Cell-intrinsic latent representation.
        niche_z
            Niche latent representation.
        library
            Library size.
        batch_index
            Batch indices.

        Returns
        -------
        Dictionary of generative outputs.
        """
        # Combine intrinsic and niche factors
        combined_z = torch.cat([z, niche_z], dim=-1)

        # Decode
        h = self.decoder(combined_z)
        px_scale = self.px_scale_decoder(h)
        px_rate = library * px_scale

        px_r = torch.exp(self.px_r)

        return {
            "px_rate": px_rate,
            "px_r": px_r,
            "px_scale": px_scale,
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

        # Reconstruction loss (negative binomial)
        from scvi.distributions import NegativeBinomial
        px = NegativeBinomial(mu=px_rate, theta=px_r)
        reconst_loss = -px.log_prob(x).sum(dim=-1)

        # KL divergence
        qz = inference_outputs["qz"]
        mean = torch.zeros_like(qz.loc)
        scale = torch.ones_like(qz.scale)
        kl_z = kl(qz, Normal(mean, scale)).sum(dim=-1)

        # Classification loss (if labels available)
        classification_loss = torch.tensor(0.0, device=x.device)
        if "labels" in tensors:
            labels = tensors["labels"].squeeze(-1).long()
            logits = inference_outputs["logits"]
            classification_loss = nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )

        # Total loss
        loss = torch.mean(reconst_loss + kl_weight * kl_z + classification_loss)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_z,
            extra_metrics={"classification_loss": classification_loss.mean()},
        )
