"""AMICI module for cell-cell interaction inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from spatialvi.nn import CrossAttention, NeighborAttention

if TYPE_CHECKING:
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class AMICIModule(BaseModuleClass):
    """AMICI module implementing cross-attention for cell-cell interactions.

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
    n_layers
        Number of hidden layers.
    n_attention_heads
        Number of attention heads.
    dropout_rate
        Dropout rate.
    gene_likelihood
        Distribution for gene expression.
    use_cell_type_attention
        Whether to use cell type-specific attention.
    interaction_layers
        Number of interaction modeling layers.
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int = 1,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        use_cell_type_attention: bool = True,
        interaction_layers: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_labels = n_labels
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        self.use_cell_type_attention = use_cell_type_attention

        # Cell-intrinsic encoder
        cat_list = [n_batch] if n_batch > 0 else None
        self.z_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Library encoder
        self.l_encoder = Encoder(
            n_input=n_input,
            n_output=1,
            n_cat_list=cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Cell type embedding
        if use_cell_type_attention and n_labels > 1:
            self.cell_type_embedding = nn.Embedding(n_labels, n_hidden)
        else:
            self.cell_type_embedding = None

        # Cross-attention layers for cell-cell interaction
        self.interaction_layers = nn.ModuleList(
            [
                CrossAttention(
                    embed_dim=n_hidden,
                    n_heads=n_attention_heads,
                    dropout=dropout_rate,
                )
                for _ in range(interaction_layers)
            ]
        )

        # Neighbor attention for aggregation
        self.neighbor_attention = NeighborAttention(
            input_dim=n_hidden,
            hidden_dim=n_hidden,
            n_heads=n_attention_heads,
            dropout=dropout_rate,
        )

        # Feature projection
        self.feature_proj = nn.Linear(n_input, n_hidden)

        # Interaction score predictor
        self.interaction_scorer = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

        # Decoder
        self.decoder = FCLayers(
            n_in=n_latent + n_hidden,  # latent + interaction context
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_input),
            nn.Softmax(dim=-1),
        )
        self.px_r = nn.Parameter(torch.randn(n_input))
        self.px_dropout = nn.Linear(n_hidden, n_input)

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Get input for inference network."""
        return {
            "x": tensors.get("X"),
            "batch_index": tensors.get("batch"),
            "labels": tensors.get("labels"),
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
            "interaction_context": inference_outputs["interaction_context"],
            "library": inference_outputs["library"],
            "batch_index": tensors.get("batch"),
        }

    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
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
        labels
            Cell type labels.
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
        # scvi Encoder returns tuple (q_m, q_v, z)
        if batch_index is not None:
            qz_m, qz_v, z = self.z_encoder(x_, batch_index)
            ql_m, ql_v, _ = self.l_encoder(x_, batch_index)
        else:
            qz_m, qz_v, z = self.z_encoder(x_)
            ql_m, ql_v, _ = self.l_encoder(x_)

        library = torch.exp(ql_m)

        # Project features
        h = self.feature_proj(x_)

        # Add cell type embeddings if available
        if self.cell_type_embedding is not None and labels is not None:
            ct_embed = self.cell_type_embedding(labels.long().squeeze(-1))
            h = h + ct_embed

        # Compute interaction context from neighbors
        interaction_context = torch.zeros_like(h)
        interaction_scores = torch.zeros(x.shape[0], 1, device=x.device)
        attention_weights = None

        if neighbor_expr is not None:
            batch_size, n_neighbors, n_genes = neighbor_expr.shape

            # Project neighbor features
            neighbor_h = self.feature_proj(torch.log1p(neighbor_expr.view(-1, n_genes)))
            neighbor_h = neighbor_h.view(batch_size, n_neighbors, -1)

            # Apply cross-attention layers
            query = h.unsqueeze(1)
            for layer in self.interaction_layers:
                query = layer(query, neighbor_h)
            interaction_context = query.squeeze(1)

            # Compute interaction scores
            h_expanded = h.unsqueeze(1).expand(-1, n_neighbors, -1)
            concat = torch.cat([h_expanded, neighbor_h], dim=-1)
            neighbor_scores = self.interaction_scorer(concat).squeeze(-1)
            interaction_scores = neighbor_scores.mean(dim=-1, keepdim=True)
            attention_weights = neighbor_scores

        return {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "library": library,
            "interaction_context": interaction_context,
            "interaction_scores": interaction_scores,
            "attention_weights": attention_weights,
        }

    def generative(
        self,
        z: torch.Tensor,
        interaction_context: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the generative network.

        Parameters
        ----------
        z
            Latent representation.
        interaction_context
            Interaction context from neighbors.
        library
            Library size.
        batch_index
            Batch indices.

        Returns
        -------
        Dictionary of generative outputs.
        """
        # Combine latent and interaction context
        decoder_input = torch.cat([z, interaction_context], dim=-1)

        # Decode
        h = self.decoder(decoder_input)
        px_scale = self.px_scale_decoder(h)
        px_rate = library * px_scale
        px_r = torch.exp(self.px_r)
        px_dropout = self.px_dropout(h)

        # Build distribution
        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r)
        else:
            px = torch.distributions.Poisson(rate=px_rate)

        return {
            "px": px,
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
        px = generative_outputs["px"]

        # Reconstruction loss
        reconst_loss = -px.log_prob(x).sum(dim=-1)

        # KL divergence - create Normal distributions from encoder outputs
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        # qz_v is already variance, need to convert to std for Normal
        qz_std = torch.sqrt(qz_v)
        qz_dist = Normal(qz_m, qz_std)
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_std)
        kl_z = kl(qz_dist, Normal(mean, scale)).sum(dim=-1)

        # Total loss
        loss = torch.mean(reconst_loss + kl_weight * kl_z)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_z,
        )

    @auto_move_data
    def forward(
        self,
        tensors: dict[str, torch.Tensor],
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
