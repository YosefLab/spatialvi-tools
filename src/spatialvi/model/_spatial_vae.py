"""Spatial VAE model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.model._utils import _init_library_size
from scvi.utils import setup_anndata_dsp

from spatialvi._constants import REGISTRY_KEYS, SPATIAL_REGISTRY_KEYS
from spatialvi.model.base import BaseSpatialModel, SpatialMixin
from spatialvi.model.base._training_mixins import SpatialTrainingMixin
from spatialvi.module import SpatialVAEModule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SpatialVAE(SpatialTrainingMixin, SpatialMixin, BaseSpatialModel):
    """Spatial Variational Autoencoder for spatial transcriptomics.

    This model combines gene expression modeling with spatial context
    using a VAE framework. It can be used for:
    - Spatially-aware dimensionality reduction
    - Batch effect correction in spatial data
    - Imputation with spatial smoothing

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
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        Dispersion parameter for negative binomial. One of:
        - "gene": single dispersion per gene
        - "gene-batch": dispersion per gene and batch
        - "gene-cell": dispersion per gene and cell
    gene_likelihood
        Distribution for gene expression. One of:
        - "zinb": Zero-inflated negative binomial
        - "nb": Negative binomial
        - "poisson": Poisson
    latent_distribution
        Distribution for latent space. One of:
        - "normal": Normal distribution
        - "ln": Logistic normal
    use_spatial
        Whether to use spatial information in encoding.
    spatial_weight
        Weight for spatial regularization.
    **model_kwargs
        Additional keyword arguments for :class:`~spatialvi.module.SpatialVAEModule`.

    Examples
    --------
    >>> import spatialvi
    >>> adata = spatialvi.data.synthetic_spatial()
    >>> SpatialVAE.setup_anndata(adata, spatial_key="spatial")
    >>> model = SpatialVAE(adata)
    >>> model.train()
    >>> latent = model.get_latent_representation()
    """

    _module_cls = SpatialVAEModule

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_spatial: bool = True,
        spatial_weight: float = 1.0,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        library_log_means, library_log_vars = None, None
        if not self.summary_stats.get("use_size_factor", False):
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_spatial=use_spatial,
            spatial_weight=spatial_weight,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_cats_per_cov=n_cats_per_cov,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            **model_kwargs,
        )

        self._spatial_key = self.adata_manager.get_state_registry(
            SPATIAL_REGISTRY_KEYS.SPATIAL_KEY
        ).get("attr_key", "spatial") if SPATIAL_REGISTRY_KEYS.SPATIAL_KEY in self.adata_manager.data_registry else None

        self.init_params_ = self._get_init_params(locals())

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: str | Sequence[str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | Literal["latent"] = 1.0,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ) -> NDArray | dict[str, NDArray]:
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
        n_samples_overall
            Total number of samples across all cells.
        batch_size
            Batch size for data loader.
        return_mean
            Whether to return the mean of samples.
        return_numpy
            Whether to return numpy array.

        Returns
        -------
        Normalized expression array.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size or 128,
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [all_genes.get_loc(g) for g in gene_list]

        if n_samples_overall is not None:
            n_samples = 1
            indices = np.random.choice(indices, size=n_samples_overall, replace=True)

        exprs = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            generative_inputs = self.module._get_generative_input(
                tensors, outputs
            )

            for _ in range(n_samples):
                generative_outputs = self.module.generative(**generative_inputs)

                if library_size == "latent":
                    lib = outputs.get("library", torch.ones(tensors["X"].shape[0], 1))
                else:
                    lib = library_size

                px = generative_outputs["px"]
                if hasattr(px, "mu"):
                    rate = px.mu
                else:
                    rate = generative_outputs.get("px_rate", generative_outputs.get("rate"))

                if rate is not None:
                    expr = rate * lib
                    exprs.append(expr[:, gene_mask].cpu().numpy())

        exprs = np.concatenate(exprs, axis=0)

        if return_mean and n_samples > 1:
            exprs = exprs.reshape(-1, n_samples, exprs.shape[-1]).mean(axis=1)

        return exprs

    @torch.inference_mode()
    def get_spatial_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Get spatial-aware latent representation.

        This method returns a latent representation that incorporates
        spatial context from neighboring cells.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices of cells to use.
        batch_size
            Batch size for data loader.

        Returns
        -------
        Spatial latent representation array.
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

            if "z_spatial" in outputs:
                z = outputs["z_spatial"]
            elif "qz_m" in outputs:
                z = outputs["qz_m"]
            else:
                z = outputs.get("z", outputs.get("latent"))

            latent.append(z.cpu().numpy())

        return np.concatenate(latent, axis=0)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        spatial_key: str = "spatial",
        neighbor_index_key: str | None = None,
        neighbor_dist_key: str | None = None,
        **kwargs,
    ) -> None:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        spatial_key
            Key in `adata.obsm` for spatial coordinates.
        neighbor_index_key
            Key in `adata.obsm` for neighbor indices. If None, neighbors
            will not be used during training.
        neighbor_dist_key
            Key in `adata.obsm` for neighbor distances.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                required=False,
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ObsmField(
                SPATIAL_REGISTRY_KEYS.SPATIAL_KEY,
                spatial_key,
                required=True,
            ),
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

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)