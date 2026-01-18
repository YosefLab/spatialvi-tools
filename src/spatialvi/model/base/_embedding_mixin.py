"""Embedding mixin for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingMixin:
    """Mixin for embedding generation and visualization.

    This mixin provides methods for:
    - Computing UMAP/t-SNE embeddings of latent representations
    - Generating visualizations
    - Batch-corrected embedding computation
    """

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        **kwargs,
    ) -> NDArray:
        """Get latent representation. To be implemented by subclass."""
        raise NotImplementedError("Subclass must implement get_latent_representation")

    def get_embedding(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        method: Literal["umap", "tsne", "pca"] = "umap",
        give_mean: bool = True,
        batch_size: int | None = None,
        n_components: int = 2,
        return_reducer: bool = False,
        **kwargs,
    ) -> NDArray | tuple[NDArray, object]:
        """Compute 2D/3D embedding of the latent representation.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of observations to use. If None, defaults to all observations.
        method
            Dimensionality reduction method. One of "umap", "tsne", or "pca".
        give_mean
            Whether to use the mean of the latent distribution.
        batch_size
            Minibatch size for computing latent representation.
        n_components
            Number of components for the embedding.
        return_reducer
            Whether to return the fitted reducer object.
        **kwargs
            Additional keyword arguments passed to the reducer.

        Returns
        -------
        Embedding array of shape (n_obs, n_components). If return_reducer is True,
        also returns the fitted reducer object.
        """
        latent = self.get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=give_mean,
            batch_size=batch_size,
        )

        if method == "umap":
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError(
                    "umap-learn is required for UMAP embedding. Install via `pip install umap-learn`"
                ) from None
            reducer = UMAP(n_components=n_components, **kwargs)
        elif method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=n_components, **kwargs)
        elif method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'umap', 'tsne', 'pca'")

        embedding = reducer.fit_transform(latent)

        if return_reducer:
            return embedding, reducer
        return embedding

    def add_embedding_to_adata(
        self,
        adata: AnnData | None = None,
        key_added: str = "X_spatialvi",
        method: Literal["umap", "tsne", "pca"] = "umap",
        give_mean: bool = True,
        batch_size: int | None = None,
        n_components: int = 2,
        **kwargs,
    ) -> None:
        """Add embedding to AnnData object.

        Parameters
        ----------
        adata
            AnnData object to modify. If None, uses training data.
        key_added
            Key to use for storing the embedding in obsm.
        method
            Dimensionality reduction method.
        give_mean
            Whether to use the mean of the latent distribution.
        batch_size
            Minibatch size for computing latent representation.
        n_components
            Number of components for the embedding.
        **kwargs
            Additional keyword arguments passed to the reducer.
        """
        adata = self._validate_anndata(adata)
        embedding = self.get_embedding(
            adata=adata,
            method=method,
            give_mean=give_mean,
            batch_size=batch_size,
            n_components=n_components,
            **kwargs,
        )
        adata.obsm[key_added] = embedding

    def add_latent_representation_to_adata(
        self,
        adata: AnnData | None = None,
        key_added: str = "X_latent",
        give_mean: bool = True,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """Add latent representation to AnnData object.

        Parameters
        ----------
        adata
            AnnData object to modify. If None, uses training data.
        key_added
            Key to use for storing the latent representation in obsm.
        give_mean
            Whether to use the mean of the latent distribution.
        batch_size
            Minibatch size for computing latent representation.
        **kwargs
            Additional keyword arguments.
        """
        adata = self._validate_anndata(adata)
        latent = self.get_latent_representation(
            adata=adata,
            give_mean=give_mean,
            batch_size=batch_size,
            **kwargs,
        )
        adata.obsm[key_added] = latent


class SpatialEmbeddingMixin(EmbeddingMixin):
    """Extended embedding mixin with spatial awareness.

    This mixin extends EmbeddingMixin with spatial-specific methods:
    - Spatial-constrained UMAP
    - Spatial-weighted embeddings
    - Joint spatial-transcriptomic embeddings
    """

    def get_spatial_embedding(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        spatial_weight: float = 0.5,
        give_mean: bool = True,
        batch_size: int | None = None,
        n_components: int = 2,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> NDArray:
        """Compute joint spatial-transcriptomic embedding.

        This method combines latent representation with spatial coordinates
        to create a joint embedding.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of observations to use. If None, defaults to all observations.
        spatial_weight
            Weight for spatial coordinates (0 to 1).
            0 = transcriptomic only, 1 = spatial only.
        give_mean
            Whether to use the mean of the latent distribution.
        batch_size
            Minibatch size for computing latent representation.
        n_components
            Number of components for the embedding.
        spatial_key
            Key in obsm for spatial coordinates.
        **kwargs
            Additional keyword arguments passed to the reducer.

        Returns
        -------
        Joint embedding array of shape (n_obs, n_components).
        """
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn is required for spatial embedding. Install via `pip install umap-learn`"
            ) from None

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)

        # Get latent representation
        latent = self.get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=give_mean,
            batch_size=batch_size,
        )

        # Get spatial coordinates
        coords = adata.obsm[spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values
        coords = coords[indices]

        # Normalize both representations
        from sklearn.preprocessing import StandardScaler

        latent_scaled = StandardScaler().fit_transform(latent)
        coords_scaled = StandardScaler().fit_transform(coords)

        # Combine with weighting
        combined = np.concatenate(
            [
                latent_scaled * (1 - spatial_weight),
                coords_scaled * spatial_weight,
            ],
            axis=1,
        )

        # Compute joint embedding
        reducer = UMAP(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(combined)

        return embedding

    def add_spatial_embedding_to_adata(
        self,
        adata: AnnData | None = None,
        key_added: str = "X_spatial_umap",
        spatial_weight: float = 0.5,
        give_mean: bool = True,
        batch_size: int | None = None,
        n_components: int = 2,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Add spatial-aware embedding to AnnData object.

        Parameters
        ----------
        adata
            AnnData object to modify. If None, uses training data.
        key_added
            Key to use for storing the embedding in obsm.
        spatial_weight
            Weight for spatial coordinates (0 to 1).
        give_mean
            Whether to use the mean of the latent distribution.
        batch_size
            Minibatch size for computing latent representation.
        n_components
            Number of components for the embedding.
        spatial_key
            Key in obsm for spatial coordinates.
        **kwargs
            Additional keyword arguments passed to the reducer.
        """
        adata = self._validate_anndata(adata)
        embedding = self.get_spatial_embedding(
            adata=adata,
            spatial_weight=spatial_weight,
            give_mean=give_mean,
            batch_size=batch_size,
            n_components=n_components,
            spatial_key=spatial_key,
            **kwargs,
        )
        adata.obsm[key_added] = embedding
