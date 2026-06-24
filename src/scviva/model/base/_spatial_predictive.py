"""Generic spatial predictive mixin providing importance expression and neighbor abundance."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scvi import REGISTRY_KEYS, settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

logger = logging.getLogger(__name__)


class SpatialPredictiveMixin:
    """Shared predictive methods for spatial models with neighbor graphs.

    Applied to: SCVIVA, ResolVI.

    Provides:
    - ``get_neighbor_abundance``: cell-type composition of spatial neighborhoods.
      Two intentionally different contracts depending on the model:

      * ResolVI (Pyro): uses this mixin default — model-predicted abundance from
        posterior sampling (``PyroSampleMixin.sample_posterior``). Honors
        ``n_samples`` / ``return_mean`` / ``weights``.
      * SCVIVA (PyTorch): overrides this in its class body to return the
        *observed* precomputed niche composition (not a posterior quantity).

    - ``_get_label_names``: shared accessor for cell-type label names from the
      labels registry (used for DataFrame column labels). Note that a model whose
      neighbor-composition array is stored in a different column order (e.g. SCVIVA's
      observed ``niche_composition``) should keep its own stored columns rather than
      relabel via this accessor.

    Notes
    -----
    Importance-weighted expression (``get_normalized_expression_importance``) is
    **not** provided here: it requires the Pyro guide and likelihood reweighting,
    so it lives on ResolVI only. PyTorch models obtain importance weighting via
    ``get_normalized_expression(weights="importance")`` (``RNASeqMixin``).
    """

    def _get_label_names(self) -> np.ndarray:
        """Return cell-type label names from the labels registry.

        Single source of truth for cell-type column labels, shared across spatial
        models. Pulls the categorical mapping registered under
        :attr:`~scvi.REGISTRY_KEYS.LABELS_KEY`, so it does not depend on
        ``self._label_mapping`` (which ResolVI only sets in semisupervised mode).

        Returns
        -------
        Array of label names.
        """
        try:
            state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        except (KeyError, AttributeError) as e:
            raise AttributeError(
                "No labels were registered for this model, so cell-type names are "
                "unavailable. Register a `labels_key` in `setup_anndata`, or request "
                "numpy output (`return_numpy=True`)."
            ) from e
        return state_registry.categorical_mapping

    @torch.inference_mode()
    def get_neighbor_abundance(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        neighbor_key: str | None = None,
        n_samples: int = 1,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        summary_frequency: int = 2,
        weights: str | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        **kwargs,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the abundance of cell-types within spatial proximity of center cells.

        Uses Pyro posterior sampling (``PyroSampleMixin.sample_posterior``). Models without
        a Pyro module should override this method in their class body.

        Parameters
        ----------
        adata
            AnnData object. If ``None``, uses the model's registered adata.
        indices
            Cell indices. If ``None``, all cells are used.
        neighbor_key
            Obsm key containing the spatial neighbors of each cell.
        n_samples
            Number of posterior samples.
        n_samples_overall
            Number of posterior samples. Overrides ``n_samples``.
        batch_size
            Minibatch size for data loading.
        summary_frequency
            Compute summary after this many batches (reduces memory footprint).
        weights
            Spatial weights per neighbor. If ``None``, no spatial weighting is applied.
        return_mean
            Return the mean over samples.
        return_numpy
            Return :class:`~numpy.ndarray` instead of :class:`~pandas.DataFrame`.
        **kwargs
            Additional keyword arguments (ignored, for compatibility).

        Returns
        -------
        Cell-type abundance array or DataFrame of shape ``(n_cells, n_cell_types)``.
        """
        if adata:
            assert neighbor_key is not None, "Must provide `neighbor_key` if `adata` is provided."
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if neighbor_key is None:
            neighbor_key = self.adata_manager.registry["field_registries"]["index_neighbor"][
                "data_registry"
            ]["attr_key"]
            neighbor_obsm = adata.obsm[neighbor_key]
        else:
            neighbor_obsm = adata.obsm[neighbor_key]
        n_neighbors = neighbor_obsm.shape[-1]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is `False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        if batch_size is not None and batch_size % n_neighbors != 0:
            raise ValueError("Batch size must be divisible by the number of neighbors.")
        if batch_size is None:
            # Cap so the batch never exceeds n_obs (guide uses subsample_size=batch_size
            # on a plate of size n_obs — exceeding n_obs would raise a Pyro error).
            raw = n_neighbors * settings.batch_size
            capped = (min(raw, adata.n_obs) // n_neighbors) * n_neighbors
            batch_size = max(capped, n_neighbors)  # at least one cell's worth
        indices_ = neighbor_obsm[indices].reshape(-1)
        dl = self._make_data_loader(
            adata=adata, indices=indices_, shuffle=False, batch_size=batch_size
        )

        sampled_prediction = self.sample_posterior(
            input_dl=dl,
            model=self.module.model_corrected,
            return_sites=["probs_prediction"],
            summary_frequency=summary_frequency,
            num_samples=n_samples,
            return_samples=True,
        )
        flat_neighbor_abundance_ = sampled_prediction["posterior_samples"]["probs_prediction"]
        neighbor_abundance_ = flat_neighbor_abundance_.reshape(
            n_samples, len(indices), n_neighbors, -1
        )
        neighbor_abundance = np.average(neighbor_abundance_, axis=-2, weights=weights)

        if return_mean:
            neighbor_abundance = np.mean(neighbor_abundance, axis=0)

        if n_samples_overall is not None:
            neighbor_abundance = neighbor_abundance.reshape(-1, neighbor_abundance.shape[-1])
            n_samples_ = neighbor_abundance.shape[0]
            ind_ = np.random.choice(n_samples_, n_samples_overall, replace=True)
            neighbor_abundance = neighbor_abundance[ind_]

        if return_numpy is None or return_numpy is False:
            assert return_mean, "Only numpy output is supported when `return_mean` is False."
            n_labels = len(neighbor_abundance[-1])
            return pd.DataFrame(
                neighbor_abundance,
                columns=self._get_label_names()[:n_labels],
                index=adata.obs_names[indices],
            )
        else:
            return neighbor_abundance
