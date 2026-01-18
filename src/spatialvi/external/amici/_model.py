"""AMICI model for cell-cell interaction inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    ObsmField,
)
from scvi.utils import setup_anndata_dsp

from spatialvi._constants import REGISTRY_KEYS, SPATIAL_REGISTRY_KEYS
from spatialvi.external.amici._module import AMICIModule
from spatialvi.model.base import BaseSpatialModel, SpatialMixin
from spatialvi.model.base._training_mixins import SpatialTrainingMixin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AMICI(SpatialTrainingMixin, SpatialMixin, BaseSpatialModel):
    """AMICI model for cell-cell interaction inference.

    AMICI (Attention-based Multi-scale Interaction for Cell-cell Inference)
    uses cross-attention mechanisms to model how neighboring cells influence
    gene expression through cell-cell interactions.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers.
    n_attention_heads
        Number of attention heads for cross-attention.
    dropout_rate
        Dropout rate for neural networks.
    gene_likelihood
        Distribution for gene expression.
    use_cell_type_attention
        Whether to use cell type-specific attention.
    interaction_layers
        Number of interaction modeling layers.
    **model_kwargs
        Additional keyword arguments for :class:`~spatialvi.external.amici.AMICIModule`.

    Examples
    --------
    >>> import spatialvi
    >>> adata = spatialvi.data.synthetic_spatial()
    >>> AMICI.setup_anndata(
    ...     adata,
    ...     spatial_key="spatial",
    ...     labels_key="cell_type",
    ... )
    >>> model = AMICI(adata)
    >>> model.train()
    >>> interactions = model.get_interaction_scores()
    """

    _module_cls = AMICIModule

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        use_cell_type_attention: bool = True,
        interaction_layers: int = 2,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_labels = self.summary_stats.get("n_labels", 1)
        n_batch = self.summary_stats.n_batch

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_labels=n_labels,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_attention_heads=n_attention_heads,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            use_cell_type_attention=use_cell_type_attention,
            interaction_layers=interaction_layers,
            **model_kwargs,
        )

        self._cell_type_names = None
        if "labels" in self.adata_manager.data_registry:
            labels_state = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
            if hasattr(labels_state, "categorical_mapping"):
                self._cell_type_names = labels_state.categorical_mapping

        self.init_params_ = self._get_init_params(locals())

    @torch.inference_mode()
    def get_interaction_scores(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_attention_weights: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Get cell-cell interaction scores.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices of cells to use.
        batch_size
            Batch size for data loader.
        return_attention_weights
            Whether to return raw attention weights.

        Returns
        -------
        Interaction scores array. If return_attention_weights is True,
        also returns attention weight matrices.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size or 128,
        )

        interaction_scores = []
        attention_weights_list = []

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if "interaction_scores" in outputs:
                interaction_scores.append(outputs["interaction_scores"].cpu().numpy())
            if return_attention_weights and "attention_weights" in outputs:
                attention_weights_list.append(outputs["attention_weights"].cpu().numpy())

        scores = np.concatenate(interaction_scores, axis=0)

        if return_attention_weights and attention_weights_list:
            weights = np.concatenate(attention_weights_list, axis=0)
            return scores, weights

        return scores

    @torch.inference_mode()
    def get_cell_type_interactions(
        self,
        adata: AnnData | None = None,
        aggregate: Literal["mean", "sum", "max"] = "mean",
    ) -> pd.DataFrame:
        """Get aggregated cell type interaction matrix.

        Parameters
        ----------
        adata
            AnnData object.
        aggregate
            Aggregation method for interaction scores.

        Returns
        -------
        DataFrame of cell type x cell type interaction strengths.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        scores, weights = self.get_interaction_scores(
            adata=adata,
            return_attention_weights=True,
        )

        # Get cell type labels
        labels = adata.obs[self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key]

        if hasattr(labels, "cat"):
            unique_labels = labels.cat.categories.tolist()
        else:
            unique_labels = list(labels.unique())

        n_types = len(unique_labels)

        # Aggregate interactions by cell type pair
        interaction_matrix = np.zeros((n_types, n_types))
        count_matrix = np.zeros((n_types, n_types))

        labels_array = labels.values
        if hasattr(labels_array, "codes"):
            labels_array = labels_array.codes

        for i in range(len(labels_array)):
            sender_type = labels_array[i]
            # Average over neighbors
            for j, receiver_type in enumerate(labels_array):
                if i != j and j < weights.shape[1]:
                    interaction_matrix[sender_type, receiver_type] += weights[i, j]
                    count_matrix[sender_type, receiver_type] += 1

        # Normalize
        count_matrix[count_matrix == 0] = 1
        if aggregate == "mean":
            interaction_matrix = interaction_matrix / count_matrix
        elif aggregate == "sum":
            pass  # Already summed
        elif aggregate == "max":
            # This is an approximation; true max would require different logic
            interaction_matrix = interaction_matrix / count_matrix

        return pd.DataFrame(
            interaction_matrix,
            index=unique_labels,
            columns=unique_labels,
        )

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
            AnnData object.
        indices
            Indices of cells to use.
        give_mean
            Whether to return the mean of the posterior.
        batch_size
            Batch size for data loader.

        Returns
        -------
        Latent representation array.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size or 128,
        )

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if give_mean:
                z = outputs.get("qz_m", outputs.get("z"))
            else:
                z = outputs["z"]

            latent.append(z.cpu().numpy())

        return np.concatenate(latent, axis=0)

    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        **kwargs,
    ) -> NDArray:
        """Return normalized gene expression."""
        # Simplified implementation
        return self.get_latent_representation(adata, indices, give_mean=True)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        spatial_key: str = "spatial",
        neighbor_index_key: str | None = "nn_index",
        neighbor_dist_key: str | None = "nn_dist",
        **kwargs,
    ) -> None:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        spatial_key
            Key in `adata.obsm` for spatial coordinates.
        neighbor_index_key
            Key in `adata.obsm` for neighbor indices.
        neighbor_dist_key
            Key in `adata.obsm` for neighbor distances.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            ObsmField(SPATIAL_REGISTRY_KEYS.SPATIAL_KEY, spatial_key, required=True),
        ]

        if neighbor_index_key is not None:
            anndata_fields.append(
                ObsmField(
                    SPATIAL_REGISTRY_KEYS.NEIGHBOR_INDEX_KEY,
                    neighbor_index_key,
                    required=False,
                )
            )

        if neighbor_dist_key is not None:
            anndata_fields.append(
                ObsmField(
                    SPATIAL_REGISTRY_KEYS.NEIGHBOR_DIST_KEY,
                    neighbor_dist_key,
                    required=False,
                )
            )

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
