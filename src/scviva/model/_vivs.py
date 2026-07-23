from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, ObsmField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from scviva._constants import VIVS_REGISTRY_KEYS
from scviva.model.base import SpatialBaseModel
from scviva.module._vivs import VIVSModule

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class VIVS(VAEMixin, UnsupervisedTrainingMixin, SpatialBaseModel):
    """VIVS: calibrated identification of feature dependencies in multiomics :cite:p:`Boyeau24`.

    Identifies which genes in ``X`` are conditionally dependent on an external response
    ``Y`` (e.g. protein expression, niche composition) using a conditional randomization
    test (CRT), with a deep generative model of ``X`` as the knockoff sampler.

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`~scviva.model.VIVS.setup_anndata`.
    n_hidden
        Number of hidden units in the generative VAE (ignored if ``x_model`` is given).
    n_latent
        Latent dimensionality of the generative VAE (ignored if ``x_model`` is given).
    x_likelihood
        Gene-expression likelihood for the generative VAE: ``"nb"``, ``"zinb"``, or ``"poisson"``
        (ignored if ``x_model`` is given).
    xy_linear
        If ``True``, the importance-score net is a linear model instead of an MLP.
    xy_include_batch_in_input
        Whether to concatenate one-hot batch to the importance-score net's input.
    x_model
        An already-trained scviva-tools spatial model (e.g. :class:`~scviva.model.SCVIVA`,
        :class:`~scviva.model.DestVI`, :class:`~scviva.model.ResolVI`,
        :class:`~scviva.model.GIMVI`) or :class:`~scvi.model.SCVI`, registered on data
        compatible with ``adata``. When given, its trained module is reused (frozen) as
        the knockoff sampler, and VIVS's own generative training phase is skipped entirely.
    **module_kwargs
        Additional keyword arguments passed to :class:`~scviva.module.VIVSModule`.
    """

    _module_cls = VIVSModule

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        x_likelihood: Literal["nb", "zinb", "poisson"] = "nb",
        xy_linear: bool = False,
        xy_include_batch_in_input: bool = False,
        x_model: BaseModelClass | None = None,
        **module_kwargs,
    ):
        super().__init__(adata)
        summary_stats = self.summary_stats

        x_module = None
        if x_model is not None:
            if not x_model.is_trained:
                raise ValueError(
                    "`x_model` must already be trained (call `.train()` on it) before "
                    "being passed to VIVS."
                )
            x_module = x_model.module

        self.module = self._module_cls(
            n_input=summary_stats.n_vars,
            n_responses=summary_stats.n_Y,
            n_batch=summary_stats.n_batch,
            x_model_kwargs={
                "n_hidden": n_hidden,
                "n_latent": n_latent,
                "gene_likelihood": x_likelihood,
            },
            xy_linear=xy_linear,
            xy_include_batch_in_input=xy_include_batch_in_input,
            x_module=x_module,
            **module_kwargs,
        )
        self._model_summary_string = "VIVS model"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        y_obsm_key: str,
        layer: str | None = None,
        batch_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        y_obsm_key
            Key in ``adata.obsm`` for the response(s) ``Y`` whose conditional dependence on
            gene expression ``X`` is being tested (e.g. protein expression, niche composition).
        %(param_layer)s
        %(param_batch_key)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            ObsmField(VIVS_REGISTRY_KEYS.Y_KEY, y_obsm_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int | None = None,
        x_max_epochs: int | None = None,
        xy_max_epochs: int | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        **kwargs,
    ):
        """Train VIVS in two sequential phases.

        Phase 1 fits the generative VAE over ``X`` (skipped entirely if a pretrained
        ``x_model`` was supplied at construction). Phase 2 freezes it and fits the
        importance-score net for ``Y|X``. This order is required for CRT validity: the
        knockoff sampler must not be contaminated by information about ``Y``.
        """
        if not self.module.x_module_is_pretrained:
            self.module._phase = "x"
            super().train(
                max_epochs=x_max_epochs or max_epochs,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                early_stopping=early_stopping,
                **kwargs,
            )
            self.module.x_module.requires_grad_(False)
            self.module.x_module.eval()

        self.module._phase = "xy"
        super().train(
            max_epochs=xy_max_epochs or max_epochs,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            **kwargs,
        )
        self.is_trained_ = True

    @torch.inference_mode()
    def predict_t(
        self,
        adata: AnnData | None = None,
        indices=None,
        batch_size: int = 128,
    ) -> np.ndarray:
        """Raw per-cell importance-score-net predictions (no CRT knockoff perturbation)."""
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        results = []
        for tensors in dataloader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[VIVS_REGISTRY_KEYS.Y_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            xy_input = self.module.xy_input(x, batch_index)
            out = self.module.xy_module(xy_input, y)
            results.append(out["all_loss"].cpu().numpy())
        return np.concatenate(results, axis=0)
