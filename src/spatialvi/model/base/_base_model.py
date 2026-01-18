"""Base spatial model class."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from anndata import AnnData

from scvi.data import AnnDataManager
from scvi.model.base import BaseModelClass

from spatialvi._constants import REGISTRY_KEYS, SPATIAL_REGISTRY_KEYS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class BaseSpatialModel(BaseModelClass):
    """Base class for all spatial transcriptomics models.

    This class extends scvi-tools' BaseModelClass with spatial-specific
    functionality including:
    - Spatial coordinate handling
    - Neighbor graph construction
    - Spatial-aware data loading

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`setup_anndata`.
    """

    _module_cls = None  # To be set by subclasses

    def __init__(
        self,
        adata: AnnData,
        **kwargs,
    ):
        super().__init__(adata)
        self._spatial_key = None
        self._n_neighbors = None

    @property
    def spatial_key(self) -> str | None:
        """Key for spatial coordinates in obsm."""
        return self._spatial_key

    @property
    def n_neighbors(self) -> int | None:
        """Number of spatial neighbors."""
        return self._n_neighbors

    def _validate_spatial_data(self, adata: AnnData) -> None:
        """Validate that spatial data is properly registered.

        Parameters
        ----------
        adata
            AnnData object to validate.
        """
        if self._spatial_key is not None:
            if self._spatial_key not in adata.obsm:
                raise ValueError(
                    f"Spatial coordinates key '{self._spatial_key}' not found in adata.obsm"
                )

    @staticmethod
    def compute_spatial_neighbors(
        adata: AnnData,
        spatial_key: str = "spatial",
        n_neighbors: int = 20,
        index_key: str = "nn_index",
        dist_key: str = "nn_dist",
        metric: str = "euclidean",
    ) -> None:
        """Compute spatial nearest neighbors.

        Parameters
        ----------
        adata
            AnnData object.
        spatial_key
            Key in obsm for spatial coordinates.
        n_neighbors
            Number of neighbors to compute.
        index_key
            Key to store neighbor indices in obsm.
        dist_key
            Key to store neighbor distances in obsm.
        metric
            Distance metric to use.
        """
        from sklearn.neighbors import NearestNeighbors

        coords = adata.obsm[spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        # Remove self from neighbors
        adata.obsm[index_key] = indices[:, 1:].astype(np.int64)
        adata.obsm[dist_key] = distances[:, 1:].astype(np.float32)
        adata.uns[SPATIAL_REGISTRY_KEYS.NUM_NEIGHBORS_KEY] = n_neighbors

    @staticmethod
    def compute_niche_composition(
        adata: AnnData,
        labels_key: str,
        index_key: str = "nn_index",
        composition_key: str = "niche_composition",
    ) -> None:
        """Compute neighborhood cell type composition.

        Parameters
        ----------
        adata
            AnnData object.
        labels_key
            Key in obs for cell type labels.
        index_key
            Key in obsm for neighbor indices.
        composition_key
            Key to store composition in obsm.
        """
        labels = adata.obs[labels_key].values
        if hasattr(labels, "cat"):
            labels = labels.cat.codes.values

        n_labels = len(np.unique(labels))
        nn_indices = adata.obsm[index_key]
        n_obs, n_neighbors = nn_indices.shape

        # Compute composition
        composition = np.zeros((n_obs, n_labels), dtype=np.float32)
        for i in range(n_obs):
            neighbor_labels = labels[nn_indices[i]]
            for label in neighbor_labels:
                composition[i, label] += 1
        composition /= n_neighbors

        adata.obsm[composition_key] = composition

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> NDArray:
        """Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells to use.
        give_mean
            Whether to return the mean of the posterior.
        batch_size
            Batch size for data loader.

        Returns
        -------
        Latent representation array of shape (n_cells, n_latent).
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size or self.summary_stats.get("batch_size", 128),
        )

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if give_mean:
                if "qz_m" in outputs:
                    z = outputs["qz_m"]
                elif "qz" in outputs and hasattr(outputs["qz"], "loc"):
                    z = outputs["qz"].loc
                else:
                    z = outputs.get("z", outputs.get("latent"))
            else:
                z = outputs.get("z", outputs.get("latent"))

            latent.append(z.cpu().numpy())

        return np.concatenate(latent, axis=0)

    @abstractmethod
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: str | Sequence[str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | Literal["latent"] = 1.0,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
    ) -> NDArray:
        """Return normalized gene expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells to use.
        transform_batch
            Batch to condition on.
        gene_list
            Subset of genes to use.
        library_size
            Library size to use for normalization.
        n_samples
            Number of samples to draw from posterior.
        batch_size
            Batch size for data loader.
        return_mean
            Whether to return the mean of samples.

        Returns
        -------
        Normalized expression array.
        """
        pass