"""Custom data fields for spatial transcriptomics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from scvi.data.fields import ObsmField

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class SpatialCoordinatesField(ObsmField):
    """Field for spatial coordinates.

    Parameters
    ----------
    registry_key
        Key to register in the data registry.
    attr_key
        Key in adata.obsm for spatial coordinates.
    """

    def __init__(
        self,
        registry_key: str,
        attr_key: str = "spatial",
    ):
        super().__init__(
            registry_key=registry_key,
            attr_key=attr_key,
        )

    def validate_field(self, adata: "AnnData") -> None:
        """Validate spatial coordinates.

        Parameters
        ----------
        adata
            AnnData object to validate.
        """
        super().validate_field(adata)

        if self.attr_key in adata.obsm:
            coords = adata.obsm[self.attr_key]
            if hasattr(coords, "values"):
                coords = coords.values

            if coords.ndim != 2:
                raise ValueError(
                    f"Spatial coordinates must be 2D, got {coords.ndim}D"
                )

            if coords.shape[1] not in (2, 3):
                raise ValueError(
                    f"Spatial coordinates must have 2 or 3 dimensions, "
                    f"got {coords.shape[1]}"
                )


class NeighborIndexField(ObsmField):
    """Field for spatial neighbor indices.

    Parameters
    ----------
    registry_key
        Key to register in the data registry.
    attr_key
        Key in adata.obsm for neighbor indices.
    n_neighbors
        Expected number of neighbors.
    """

    def __init__(
        self,
        registry_key: str,
        attr_key: str = "nn_index",
        n_neighbors: int | None = None,
    ):
        super().__init__(
            registry_key=registry_key,
            attr_key=attr_key,
        )
        self.n_neighbors = n_neighbors

    def validate_field(self, adata: "AnnData") -> None:
        """Validate neighbor indices.

        Parameters
        ----------
        adata
            AnnData object to validate.
        """
        super().validate_field(adata)

        if self.attr_key in adata.obsm:
            indices = adata.obsm[self.attr_key]

            if indices.ndim != 2:
                raise ValueError(
                    f"Neighbor indices must be 2D, got {indices.ndim}D"
                )

            if indices.shape[0] != adata.n_obs:
                raise ValueError(
                    f"Neighbor indices must have {adata.n_obs} rows, "
                    f"got {indices.shape[0]}"
                )

            if self.n_neighbors is not None and indices.shape[1] != self.n_neighbors:
                logger.warning(
                    f"Expected {self.n_neighbors} neighbors, "
                    f"got {indices.shape[1]}"
                )

            # Check for valid indices
            if indices.max() >= adata.n_obs:
                raise ValueError(
                    f"Neighbor index {indices.max()} out of bounds "
                    f"for {adata.n_obs} observations"
                )


class NeighborDistanceField(ObsmField):
    """Field for spatial neighbor distances.

    Parameters
    ----------
    registry_key
        Key to register in the data registry.
    attr_key
        Key in adata.obsm for neighbor distances.
    """

    def __init__(
        self,
        registry_key: str,
        attr_key: str = "nn_dist",
    ):
        super().__init__(
            registry_key=registry_key,
            attr_key=attr_key,
        )

    def validate_field(self, adata: "AnnData") -> None:
        """Validate neighbor distances.

        Parameters
        ----------
        adata
            AnnData object to validate.
        """
        super().validate_field(adata)

        if self.attr_key in adata.obsm:
            distances = adata.obsm[self.attr_key]

            if distances.ndim != 2:
                raise ValueError(
                    f"Neighbor distances must be 2D, got {distances.ndim}D"
                )

            if (distances < 0).any():
                raise ValueError("Neighbor distances must be non-negative")


class NicheCompositionField(ObsmField):
    """Field for neighborhood cell type composition.

    Parameters
    ----------
    registry_key
        Key to register in the data registry.
    attr_key
        Key in adata.obsm for niche composition.
    n_cell_types
        Expected number of cell types.
    """

    def __init__(
        self,
        registry_key: str,
        attr_key: str = "niche_composition",
        n_cell_types: int | None = None,
    ):
        super().__init__(
            registry_key=registry_key,
            attr_key=attr_key,
        )
        self.n_cell_types = n_cell_types

    def validate_field(self, adata: "AnnData") -> None:
        """Validate niche composition.

        Parameters
        ----------
        adata
            AnnData object to validate.
        """
        super().validate_field(adata)

        if self.attr_key in adata.obsm:
            composition = adata.obsm[self.attr_key]

            if composition.ndim != 2:
                raise ValueError(
                    f"Niche composition must be 2D, got {composition.ndim}D"
                )

            if composition.shape[0] != adata.n_obs:
                raise ValueError(
                    f"Niche composition must have {adata.n_obs} rows, "
                    f"got {composition.shape[0]}"
                )

            # Check that composition sums to 1
            row_sums = composition.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-4):
                logger.warning(
                    "Niche composition rows do not sum to 1. "
                    "Consider normalizing."
                )

            if (composition < 0).any() or (composition > 1).any():
                raise ValueError(
                    "Niche composition values must be between 0 and 1"
                )
