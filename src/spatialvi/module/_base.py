"""Base module class for spatial models."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

if TYPE_CHECKING:
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class BaseSpatialModule(BaseModuleClass):
    """Base class for spatial transcriptomics modules.

    This class extends scvi-tools' BaseModuleClass with spatial-specific
    functionality including:
    - Spatial context encoding
    - Neighbor aggregation
    - Spatial regularization

    Parameters
    ----------
    n_input
        Number of input genes.
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
    use_spatial
        Whether to use spatial information.
    spatial_weight
        Weight for spatial regularization.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_spatial: bool = True,
        spatial_weight: float = 1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_spatial = use_spatial
        self.spatial_weight = spatial_weight

    @abstractmethod
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        spatial_coords: torch.Tensor | None = None,
        neighbor_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the inference (encoder) network.

        Parameters
        ----------
        x
            Gene expression tensor of shape (n_cells, n_genes).
        batch_index
            Batch indices of shape (n_cells,).
        spatial_coords
            Spatial coordinates of shape (n_cells, n_dims).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).

        Returns
        -------
        Dictionary of inference outputs including latent variables.
        """
        pass

    @abstractmethod
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the generative (decoder) network.

        Parameters
        ----------
        z
            Latent representation of shape (n_cells, n_latent).
        library
            Library size of shape (n_cells, 1).
        batch_index
            Batch indices of shape (n_cells,).

        Returns
        -------
        Dictionary of generative outputs including reconstructions.
        """
        pass

    @abstractmethod
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
        LossOutput containing reconstruction loss, KL divergence, etc.
        """
        pass

    def _compute_spatial_loss(
        self,
        z: torch.Tensor,
        neighbor_indices: torch.Tensor | None = None,
        neighbor_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute spatial regularization loss.

        Encourages neighboring cells to have similar latent representations.

        Parameters
        ----------
        z
            Latent representation of shape (n_cells, n_latent).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).
        neighbor_z
            Pre-computed neighbor latent representations.

        Returns
        -------
        Spatial regularization loss.
        """
        if not self.use_spatial or neighbor_indices is None:
            return torch.tensor(0.0, device=z.device)

        if neighbor_z is None:
            # Get neighbor representations
            # This assumes we have access to all z values
            # In practice, this is handled at the batch level
            return torch.tensor(0.0, device=z.device)

        # Compute pairwise distances in latent space
        # neighbor_z: (n_cells, n_neighbors, n_latent)
        z_expanded = z.unsqueeze(1)  # (n_cells, 1, n_latent)
        spatial_dist = torch.mean((z_expanded - neighbor_z) ** 2, dim=-1)  # (n_cells, n_neighbors)

        # Average over neighbors and cells
        spatial_loss = torch.mean(spatial_dist)

        return self.spatial_weight * spatial_loss

    def _aggregate_neighbors(
        self,
        x: torch.Tensor,
        neighbor_indices: torch.Tensor,
        aggregation: Literal["mean", "sum", "max"] = "mean",
    ) -> torch.Tensor:
        """Aggregate features from neighboring cells.

        Parameters
        ----------
        x
            Feature tensor of shape (n_cells, n_features).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).
        aggregation
            Aggregation method.

        Returns
        -------
        Aggregated neighbor features of shape (n_cells, n_features).
        """
        n_cells, n_features = x.shape
        n_neighbors = neighbor_indices.shape[1]

        # Gather neighbor features
        neighbor_x = x[neighbor_indices.long()]  # (n_cells, n_neighbors, n_features)

        if aggregation == "mean":
            return neighbor_x.mean(dim=1)
        elif aggregation == "sum":
            return neighbor_x.sum(dim=1)
        elif aggregation == "max":
            return neighbor_x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

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
        """Forward pass through the module.

        Parameters
        ----------
        tensors
            Dictionary of input tensors.
        get_inference_input_kwargs
            Keyword arguments for _get_inference_input.
        get_generative_input_kwargs
            Keyword arguments for _get_generative_input.
        inference_kwargs
            Keyword arguments for inference.
        generative_kwargs
            Keyword arguments for generative.
        loss_kwargs
            Keyword arguments for loss.
        compute_loss
            Whether to compute the loss.

        Returns
        -------
        Inference outputs, generative outputs, and optionally loss.
        """
        get_inference_input_kwargs = get_inference_input_kwargs or {}
        get_generative_input_kwargs = get_generative_input_kwargs or {}
        inference_kwargs = inference_kwargs or {}
        generative_kwargs = generative_kwargs or {}
        loss_kwargs = loss_kwargs or {}

        inference_inputs = self._get_inference_input(tensors, **get_inference_input_kwargs)
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)

        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)

        if compute_loss:
            losses = self.loss(
                tensors, inference_outputs, generative_outputs, **loss_kwargs
            )
            return inference_outputs, generative_outputs, losses

        return inference_outputs, generative_outputs
