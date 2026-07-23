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
from statsmodels.stats.multitest import multipletests

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

    @torch.inference_mode()
    def _encode_for_knockoffs(
        self, x: torch.Tensor, batch_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen generative VAE's encoder once per batch.

        The original JAX implementation samples ``z`` a single time per batch and only
        resamples the decoder's noise across MC knockoff draws (`vivs/_vivs.py:227-228,302`
        on VIVS `main`: ``z_rng`` is split once, ``randomize(...)`` is called once per batch,
        outside the MC loop). Re-drawing ``z`` fresh on every MC sample (as an earlier draft
        of this helper did) folds extra encoder-uncertainty variance into every null draw and
        is NOT what the reference algorithm does — call this once per batch, then call
        `_sample_knockoffs` many times with the same `z`/`library`.
        """
        inference_out = self.module.inference(x=x, batch_index=batch_index)
        return inference_out["z"], inference_out["library"]

    @torch.inference_mode()
    def _sample_knockoffs(
        self, z: torch.Tensor, library: torch.Tensor, batch_index: torch.Tensor
    ) -> torch.Tensor:
        """Sample one conditional replacement of X from the frozen generative VAE's decoder.

        `z`/`library` must come from `_encode_for_knockoffs`, called ONCE per batch outside
        the MC loop — only the decoder's `px.sample()` varies across MC knockoff draws,
        matching the reference algorithm exactly (see `_encode_for_knockoffs`'s docstring).
        """
        generative_out = self.module.generative(z=z, library=library, batch_index=batch_index)
        return generative_out["px"].sample()

    @staticmethod
    def _crt_pvalue(
        obs_t: np.ndarray, tilde_t: np.ndarray, n_mc_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-sided CRT p-value with BH correction, shared by every hypothesis-testing method.

        Parameters
        ----------
        obs_t
            Observed statistic, shape ``(..., n_responses)``.
        tilde_t
            Null statistics, shape ``(n_mc_samples, ..., n_responses)``.
        """
        pval = (1.0 + (obs_t >= tilde_t).sum(0)) / (1.0 + n_mc_samples)
        padj = np.stack(
            [multipletests(pval[..., r], method="fdr_bh")[1] for r in range(pval.shape[-1])],
            axis=-1,
        )
        return pval, padj

    @torch.inference_mode()
    def get_importance(
        self,
        adata: AnnData | None = None,
        indices=None,
        batch_size: int = 128,
        n_mc_samples: int = 500,
        use_vmap: Literal["auto", True, False] = "auto",
    ) -> dict:
        """Conditional-randomization-test importance of each gene for each response.

        Parameters
        ----------
        use_vmap
            Whether to vectorize the per-gene resampling loop with :func:`torch.vmap`.
            ``"auto"`` enables it when the number of genes is below 2000 (mirrors the
            original's own recommended gene-filtering ceiling). Disable if you hit an
            out-of-memory error.
        """
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        n_genes = self.summary_stats.n_vars
        n_responses = self.summary_stats.n_Y
        use_vmap = use_vmap if use_vmap != "auto" else n_genes < 2000

        obs_t_total = torch.zeros(n_responses)
        tilde_t_total = torch.zeros(n_mc_samples, n_genes, n_responses)
        n_obs = 0

        for tensors in dataloader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[VIVS_REGISTRY_KEYS.Y_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            batch_n = x.shape[0]
            n_obs += batch_n

            obs_all_loss = self.module.xy_module(self.module.xy_input(x, batch_index), y)[
                "all_loss"
            ]
            obs_t_total += obs_all_loss.sum(0)

            z, library = self._encode_for_knockoffs(x, batch_index)  # once per batch
            px_sample = self._sample_knockoffs(z, library, batch_index)  # (batch_n, n_genes)

            if use_vmap:
                try:
                    tilde_t_batch = self._get_importance_vmap(x, px_sample, batch_index, y)
                except RuntimeError as e:
                    raise RuntimeError(
                        "Out of memory while vmapping over genes. Try setting use_vmap=False."
                    ) from e
            else:
                tilde_t_batch = torch.stack(
                    [
                        self._compute_gene_statistic(x, px_sample[:, g], g, batch_index, y)
                        for g in range(n_genes)
                    ],
                    dim=0,
                )  # (n_genes, n_responses)
            # Only one MC sample per batch pass in this minimal loop-based version; repeat
            # `n_mc_samples` times by resampling. This mirrors `n_mc_per_pass=1` in the original.
            tilde_t_total[0] += tilde_t_batch
            for k in range(1, n_mc_samples):
                px_sample_k = self._sample_knockoffs(
                    z, library, batch_index
                )  # same z/library, fresh px draw
                if use_vmap:
                    try:
                        tilde_t_k = self._get_importance_vmap(x, px_sample_k, batch_index, y)
                    except RuntimeError as e:
                        raise RuntimeError(
                            "Out of memory while vmapping over genes. Try setting use_vmap=False."
                        ) from e
                else:
                    tilde_t_k = torch.stack(
                        [
                            self._compute_gene_statistic(x, px_sample_k[:, g], g, batch_index, y)
                            for g in range(n_genes)
                        ],
                        dim=0,
                    )
                tilde_t_total[k] += tilde_t_k

        obs_t = (obs_t_total / n_obs).numpy()
        null_t = (tilde_t_total / n_obs).numpy()
        pval, padj = self._crt_pvalue(obs_t, null_t, n_mc_samples)
        return {"obs_ts": obs_t, "null_ts": null_t, "pvalues": pval, "padj": padj}

    @torch.inference_mode()
    def get_cell_scores(
        self,
        gene_ids: list[int],
        response_ids: list[int] | None = None,
        adata: AnnData | None = None,
        indices=None,
        batch_size: int | None = None,
        n_mc_samples: int | None = None,
    ) -> dict:
        """Per-cell (unsummed) importance scores for a specific set of genes."""
        adata = self._validate_anndata(adata)
        batch_size = batch_size or 128
        n_mc_samples = n_mc_samples or 500
        response_ids = (
            response_ids if response_ids is not None else list(range(self.summary_stats.n_Y))
        )
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        tilde_t_mean_chunks, obs_t_chunks = [], []
        for tensors in dataloader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[VIVS_REGISTRY_KEYS.Y_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

            obs_all_loss = self.module.xy_module(self.module.xy_input(x, batch_index), y)[
                "all_loss"
            ][:, response_ids]
            obs_t = obs_all_loss.mean(-1, keepdim=True)  # (batch_n, 1)

            z, library = self._encode_for_knockoffs(x, batch_index)  # once per batch
            tilde_t_sum = torch.zeros(x.shape[0], len(response_ids))
            for _ in range(n_mc_samples):
                px_sample = self._sample_knockoffs(
                    z, library, batch_index
                )  # fresh px draw, same z
                x_perturbed = x.clone()
                x_perturbed[..., gene_ids] = px_sample[..., gene_ids]
                xy_input = self.module.xy_input(x_perturbed, batch_index)
                all_loss = self.module.xy_module(xy_input, y)["all_loss"][:, response_ids]
                tilde_t_sum += all_loss
            tilde_t_mean_chunks.append((tilde_t_sum / n_mc_samples).numpy())
            obs_t_chunks.append(obs_t.numpy())

        return {
            "tilde_t_mean": np.concatenate(tilde_t_mean_chunks, axis=0),
            "obs_t": np.concatenate(obs_t_chunks, axis=0),
        }

    def _get_importance_vmap(
        self,
        x: torch.Tensor,
        x_tilde: torch.Tensor,
        batch_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized version of `_compute_gene_statistic`, mapped over the gene axis.

        Mirrors `scvi.external.mrvi.MRVI`'s vmap usage pattern: `torch.vmap` over one
        axis, `randomness="different"` since downstream computation involves stochastic
        submodules (dropout in the importance-score net). Uses a non-mutating
        `torch.where` substitution rather than in-place indexed assignment
        (`x_perturbed[..., gene_id] = ...`) — verified empirically that the in-place
        form raises `RuntimeError: vmap: index_put_(...)` as soon as a dropout/BatchNorm
        submodule is called downstream inside the vmapped function; `torch.where` does not.
        """
        n_genes = x.shape[-1]

        def _statistic_for_gene(gene_id: torch.Tensor, x_tilde_gene: torch.Tensor) -> torch.Tensor:
            mask = torch.nn.functional.one_hot(gene_id, n_genes).bool()
            x_tilde_expanded = x_tilde_gene.unsqueeze(-1).expand(-1, n_genes)
            x_perturbed = torch.where(mask, x_tilde_expanded, x)
            xy_input = self.module.xy_input(x_perturbed, batch_index)
            return self.module.xy_module(xy_input, y)["all_loss"].sum(0)

        gene_ids = torch.arange(n_genes, device=x.device)
        return torch.vmap(_statistic_for_gene, in_dims=(0, 1), randomness="different")(
            gene_ids, x_tilde
        )

    def _compute_gene_statistic(
        self,
        x: torch.Tensor,
        x_tilde_gene: torch.Tensor,
        gene_id: int,
        batch_index: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Substitute one gene with its knockoff, recompute xy statistic (summed over cells)."""
        x_perturbed = x.clone()
        x_perturbed[..., gene_id] = x_tilde_gene
        xy_input = self.module.xy_input(x_perturbed, batch_index)
        return self.module.xy_module(xy_input, y)["all_loss"].sum(0)
