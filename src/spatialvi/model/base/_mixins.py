"""Mixins for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SpatialMixin:
    """Mixin for spatial-aware analysis methods.

    This mixin provides methods for:
    - Spatial neighbor analysis
    - Distance-weighted computations
    - Coordinate-based embeddings
    """

    def get_spatial_neighbors(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Get spatial neighbor indices and distances.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Subset of indices to use.

        Returns
        -------
        Tuple of (neighbor_indices, neighbor_distances).
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)

        nn_index = adata.obsm.get("nn_index", None)
        nn_dist = adata.obsm.get("nn_dist", None)

        if nn_index is None or nn_dist is None:
            raise ValueError("Spatial neighbors not computed. Run `compute_spatial_neighbors` first.")

        return nn_index[indices], nn_dist[indices]

    def get_spatial_coordinates(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        spatial_key: str = "spatial",
    ) -> NDArray:
        """Get spatial coordinates.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Subset of indices to use.
        spatial_key
            Key in obsm for spatial coordinates.

        Returns
        -------
        Spatial coordinates array of shape (n_cells, n_dims).
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)

        coords = adata.obsm[spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        return coords[indices]


class NicheMixin:
    """Mixin for niche/microenvironment analysis.

    This mixin provides methods for:
    - Neighborhood composition analysis
    - Niche-aware differential expression
    - Cell-cell interaction inference
    """

    @torch.inference_mode()
    def predict_neighborhood(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Predict neighborhood cell type composition.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Subset of indices to use.
        batch_size
            Batch size for data loader.

        Returns
        -------
        Predicted composition array of shape (n_cells, n_labels).
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        compositions = []
        for tensors in scdl:
            inference_outputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_outputs)

            gen_inputs = self.module._get_generative_input(tensors, inference_outputs)
            gen_outputs = self.module.generative(**gen_inputs)

            if "niche_composition_pred" in gen_outputs:
                comp = gen_outputs["niche_composition_pred"]
                if hasattr(comp, "mean"):
                    comp = comp.mean
                compositions.append(comp.cpu().numpy())

        if not compositions:
            raise NotImplementedError("This model does not support neighborhood prediction.")

        return np.concatenate(compositions, axis=0)

    @torch.inference_mode()
    def predict_niche_activation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Predict niche activation patterns.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Subset of indices to use.
        batch_size
            Batch size for data loader.

        Returns
        -------
        Predicted niche activation array.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        activations = []
        for tensors in scdl:
            inference_outputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_outputs)

            gen_inputs = self.module._get_generative_input(tensors, inference_outputs)
            gen_outputs = self.module.generative(**gen_inputs)

            if "niche_activation_pred" in gen_outputs:
                act = gen_outputs["niche_activation_pred"]
                activations.append(act.cpu().numpy())

        if not activations:
            raise NotImplementedError("This model does not support niche activation prediction.")

        return np.concatenate(activations, axis=0)

    def differential_niche_expression(
        self,
        groupby: str,
        group1: str | Sequence[str],
        group2: str | Sequence[str] | None = None,
        adata: AnnData | None = None,
        niche_mode: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Perform niche-aware differential expression.

        Parameters
        ----------
        groupby
            Key in obs to group cells by.
        group1
            First group for comparison.
        group2
            Second group for comparison. If None, compare against rest.
        adata
            AnnData object.
        niche_mode
            Whether to use niche-aware mode.
        **kwargs
            Additional arguments for differential expression.

        Returns
        -------
        DataFrame with differential expression results.
        """
        # This is a placeholder - actual implementation depends on the model
        raise NotImplementedError("Niche-aware differential expression not implemented for this model.")


class DeconvolutionMixin:
    """Mixin for spatial deconvolution methods.

    This mixin provides methods for:
    - Cell type proportion estimation
    - Sub-cell-type variation analysis
    - Reference-based mapping
    """

    @torch.inference_mode()
    def get_proportions(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_dataframe: bool = True,
    ) -> NDArray | pd.DataFrame:
        """Get estimated cell type proportions per spot.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Subset of indices to use.
        batch_size
            Batch size for data loader.
        return_dataframe
            Whether to return as DataFrame with cell type names.

        Returns
        -------
        Cell type proportions array or DataFrame.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        proportions = []
        for tensors in scdl:
            inference_outputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_outputs)

            gen_inputs = self.module._get_generative_input(tensors, inference_outputs)
            gen_outputs = self.module.generative(**gen_inputs)

            if "proportions" in gen_outputs:
                prop = gen_outputs["proportions"]
                if hasattr(prop, "mean"):
                    prop = prop.mean
                proportions.append(prop.cpu().numpy())
            elif "qc" in inference_outputs:
                prop = inference_outputs["qc"]
                if hasattr(prop, "mean"):
                    prop = prop.mean
                proportions.append(prop.cpu().numpy())

        if not proportions:
            raise NotImplementedError("This model does not support proportion estimation.")

        result = np.concatenate(proportions, axis=0)

        if return_dataframe:
            # Get cell type names if available
            if hasattr(self, "_cell_type_names"):
                columns = self._cell_type_names
            else:
                columns = [f"CellType_{i}" for i in range(result.shape[1])]

            obs_names = adata.obs_names if indices is None else adata.obs_names[indices]
            return pd.DataFrame(result, index=obs_names, columns=columns)

        return result

    @torch.inference_mode()
    def get_scale_for_celltype(
        self,
        celltype: str | int,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Get gene expression scale for a specific cell type.

        Parameters
        ----------
        celltype
            Cell type name or index.
        adata
            AnnData object.
        indices
            Subset of indices to use.
        batch_size
            Batch size for data loader.

        Returns
        -------
        Gene expression scale array.
        """
        # Placeholder - implementation depends on model architecture
        raise NotImplementedError("Cell type-specific expression not implemented for this model.")
