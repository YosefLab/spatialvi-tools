"""ResolVI module implementation.

This module contains the neural network components for ResolVI,
a model for denoising cellular-resolved spatial transcriptomics data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pyro
    from pyro.distributions import (
        Categorical,
        Delta,
        Dirichlet,
        Exponential,
        Gamma,
        Independent,
        LogNormal,
        Multinomial,
        Normal,
        Poisson,
        constraints,
    )
    from pyro.nn import PyroModule

    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    PyroModule = torch.nn.Module

from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnTorchDataset
from scvi.module._classifier import Classifier
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder

if TYPE_CHECKING:
    pass

_RESOLVAE_PYRO_MODULE_NAME = "resolvae"


class RESOLVAEModel(PyroModule):
    """A PyroModule that serves as the model for the RESOLVAE class.

    This implements the generative model for ResolVI, which models
    signal diffusion and background contamination in spatial transcriptomics.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_obs
        Number of total input cells.
    n_neighbors
        Number of spatial neighbors to consider for diffusion.
    z_encoder
        Shared encoder between model (neighboring cells) and guide.
    expression_anntorchdata
        AnnTorchDataset containing expression data.
    n_batch
        Number of batches. If 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    mixture_k
        Number of components in the Mixture-of-Gaussian prior.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    n_cats_per_cov
        Number of categories for each extra categorical covariate.
    n_labels
        Number of cell-type labels in the dataset.
    dispersion
        One of:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    semisupervised
        Whether to use a semi-supervised model.
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    classifier_parameters
        Parameters for the cell-type classifier.
    prior_true_amount
        Prior for true_proportion.
    prior_diffusion_amount
        Prior for diffusion_proportion.
    sparsity_diffusion
        Prior for sparsity_diffusion.
    background_ratio
        Prior for background_proportion.
    prior_proportions_rate
        Rate parameter for the prior proportions.
    median_distance
        Kernel size in the RBF kernel for distances.
    encode_covariates
        Whether to concatenate covariates to expression in encoder.
    """

    def __init__(
        self,
        n_input: int,
        n_obs: int,
        n_neighbors: int,
        z_encoder: Encoder,
        expression_anntorchdata: AnnTorchDataset,
        n_batch: int = 0,
        n_hidden: int = 32,
        n_latent: int = 10,
        mixture_k: int = 100,
        n_layers: int = 2,
        n_cats_per_cov: Iterable[int] | None = None,
        n_labels: Iterable[int] | None = None,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        semisupervised: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        classifier_parameters: dict | None = None,
        prior_true_amount: float = 1.0,
        prior_diffusion_amount: float = 0.3,
        sparsity_diffusion: float = 3.0,
        background_ratio: float = 0.1,
        prior_proportions_rate: float = 10.0,
        median_distance: float = 1.0,
        encode_covariates: bool = False,
    ):
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro is required for RESOLVAEModel. Install with: pip install pyro-ppl")

        super().__init__(_RESOLVAE_PYRO_MODULE_NAME)
        self.z_encoder = z_encoder
        self.expression_anntorchdata = expression_anntorchdata
        self.register_buffer("gene_dummy", torch.ones([n_batch, n_input]))

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.mixture_k = mixture_k
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.semisupervised = semisupervised
        self.eps = torch.tensor(1e-6)
        self.encode_covariates = encode_covariates

        if self.dispersion == "gene":
            init_px_r = torch.full([n_input], 0.01)
        elif self.dispersion == "gene-batch":
            init_px_r = torch.full([n_input, n_batch], 0.01)
        else:
            raise ValueError(f"dispersion must be one of ['gene', 'gene-batch'], but input was {dispersion}.")
        self.register_buffer("px_r", init_px_r)

        self.register_buffer("median_distance", torch.tensor(median_distance))
        self.register_buffer("sparsity_diffusion", torch.tensor(sparsity_diffusion))

        if self.semisupervised:
            mixture_k = n_labels

        self.register_buffer("u_prior_logits", torch.ones([mixture_k]))
        if self.semisupervised:
            self.register_buffer("u_prior_means", torch.zeros([mixture_k, n_latent]))
            self.register_buffer("u_prior_scales", torch.zeros([mixture_k, n_latent]))
        else:
            self.register_buffer("u_prior_means", torch.randn([mixture_k, n_latent]))
            self.register_buffer("u_prior_scales", torch.zeros([mixture_k, n_latent]) - 1.0)

        self.register_buffer("diffusion_scale", torch.tensor([1]))
        self.register_buffer(
            "prior_proportions",
            torch.tensor(
                [
                    prior_true_amount,
                    prior_diffusion_amount,
                    10 * background_ratio * prior_true_amount + 1e-3,
                ]
            ),
        )
        self.register_buffer("prior_proportions_rate", torch.tensor([prior_proportions_rate]))

        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        if self.semisupervised:
            classifier_parameters = classifier_parameters or {}
            self.n_labels = n_labels
            cls_parameters = {
                "n_layers": 0,
                "n_hidden": 128,
                "dropout_rate": 0.0,
            }
            cls_parameters.update(classifier_parameters)
            self.classifier = Classifier(
                n_latent,
                n_labels=n_labels,
                use_batch_norm=False,
                use_layer_norm=True,
                **cls_parameters,
            )

    def _get_fn_args_from_batch(self, tensor_dict: dict[str, torch.Tensor]) -> Iterable | dict:
        """Extract function arguments from batch tensor dictionary."""
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        y = tensor_dict[REGISTRY_KEYS.LABELS_KEY].long().ravel()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensor_dict[cat_key] if cat_key in tensor_dict.keys() else None

        ind_x = tensor_dict[REGISTRY_KEYS.INDICES_KEY].long().ravel()
        distances_n = tensor_dict["distance_neighbor"]
        ind_neighbors = tensor_dict["index_neighbor"].long()

        x_n = self.expression_anntorchdata[ind_neighbors.cpu().numpy().flatten(), :]["X"]
        if isinstance(x_n, np.ndarray):
            x_n = torch.from_numpy(x_n)
        x_n = x_n.to(x.device)

        if x.layout is torch.sparse_csr or x.layout is torch.sparse_csc:
            x = x.to_dense()
        if x_n.layout is torch.sparse_csr or x_n.layout is torch.sparse_csc:
            x_n = x_n.to_dense()
        x_n = x_n.reshape(x.shape[0], -1)
        library = torch.log(torch.sum(x, dim=1, keepdim=True))

        return (), {
            "x": x,
            "ind_x": ind_x,
            "library": library,
            "y": y,
            "batch_index": batch_index,
            "cat_covs": cat_covs,
            "x_n": x_n,
            "distances_n": distances_n,
        }

    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        """Forward pass - conditions the model on observations."""
        pyro.condition(self.model_unconditioned, data={"obs": x})(
            x, ind_x, library, y, batch_index, cat_covs, x_n, distances_n, n_obs, kl_weight
        )

    @auto_move_data
    def model_unconditioned(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        """Full generative model."""
        sparsity_diffusion = pyro.sample(
            "sparsity_diffusion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / self.sparsity_diffusion,
            ),
        ).mean()

        per_gene_background = pyro.sample(
            "per_gene_background",
            Dirichlet(
                concentration=5.0 * self.gene_dummy,
                validate_args=False,
            ).to_event(1),
        )

        prior_proportions = sparsity_diffusion * self.prior_proportions

        true_proportion = pyro.sample(
            "true_proportion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / prior_proportions[0],
            ),
        ).mean()

        background_proportion = pyro.sample(
            "background_proportion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / prior_proportions[2],
            ),
        ).mean()

        diffusion_proportion = pyro.sample(
            "diffusion_proportion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / prior_proportions[1],
            ),
        ).mean()

        diffusion_scale = pyro.sample("diffuse_scale", Exponential(x.new_ones([1])).to_event(1))

        u_prior_logits = pyro.param("u_prior_logits", self.u_prior_logits)
        u_prior_means = pyro.param("u_prior_means", self.u_prior_means)
        u_prior_scales = pyro.param("u_prior_scales", self.u_prior_scales)

        with pyro.plate("obs_plate", size=n_obs or self.n_obs, subsample_size=x.shape[0], dim=-1):
            distances = 30.0 * pyro.deterministic(
                "distances",
                torch.exp(-torch.clamp(diffusion_scale * distances_n / self.median_distance, max=20.0)) + 1e-3,
                event_dim=1,
            )
            px_r = 1 / pyro.sample("px_r_inv", Exponential(torch.ones_like(x)).to_event(1))

            per_neighbor_diffusion = pyro.sample(
                "per_neighbor_diffusion",
                Dirichlet(concentration=distances, validate_args=False),
            )
            with pyro.poutine.scale(scale=5.0):
                mixture_proportions = pyro.sample(
                    "mixture_proportions",
                    Dirichlet(
                        concentration=torch.tensor(
                            [true_proportion, diffusion_proportion, background_proportion],
                            device=x.device,
                        ),
                        validate_args=False,
                    ),
                )

            true_mixture_proportion = pyro.deterministic("true_mixture_proportion", mixture_proportions[..., 0])
            diffusion_mixture_proportion = pyro.deterministic(
                "diffusion_mixture_proportion", mixture_proportions[..., 1]
            )
            background_mixture_proportion = pyro.deterministic(
                "background_mixture_proportion", mixture_proportions[..., 2]
            )

            v = pyro.deterministic(
                "diffusion_proportion_per_neighbor",
                per_neighbor_diffusion * diffusion_mixture_proportion.unsqueeze(-1),
                event_dim=1,
            )

            background = pyro.deterministic(
                "background",
                background_mixture_proportion.unsqueeze(-1)
                * torch.exp(library)
                * torch.matmul(
                    torch.nn.functional.one_hot(batch_index.flatten(), self.n_batch).float(),
                    per_gene_background,
                ),
                event_dim=1,
            )

            if self.semisupervised:
                logits_input = (
                    torch.stack(
                        [
                            torch.nn.functional.one_hot(y_i, self.n_labels)
                            if y_i < self.n_labels
                            else torch.zeros(self.n_labels).to(x.device)
                            for y_i in y
                        ]
                    )
                    .to(x.device)
                    .float()
                )
                u_prior_logits = u_prior_logits + 10 * logits_input
                u_prior_means = u_prior_means.expand(x.shape[0], -1, -1)
                u_prior_scales = u_prior_scales.expand(x.shape[0], -1, -1)

            cats = Categorical(logits=u_prior_logits)
            normal_dists = Independent(
                Normal(u_prior_means, torch.exp(self.u_prior_scales) + 1e-4),
                reinterpreted_batch_ndims=1,
            )

            with pyro.poutine.scale(scale=kl_weight):
                z = pyro.sample("latent", pyro.distributions.MixtureSameFamily(cats, normal_dists))

            if cat_covs is not None:
                categorical_input = list(torch.split(cat_covs, 1, dim=1))
            else:
                categorical_input = ()

            px_scale, _, px_rate, _ = self.decoder(
                self.dispersion,
                z,
                library,
                batch_index,
                *categorical_input,
            )

            if self.semisupervised:
                probs_prediction_ = self.classifier(z)

            px_rate = pyro.deterministic("px_rate", px_rate, event_dim=1)
            pyro.deterministic("px_scale", px_scale, event_dim=1)

            with torch.no_grad():
                if cat_covs is not None:
                    categorical_input = [
                        i.repeat_interleave(self.n_neighbors).unsqueeze(1) for i in torch.split(cat_covs, 1, dim=1)
                    ]
                else:
                    categorical_input = ()
                if cat_covs is not None and self.encode_covariates:
                    categorical_encoder = categorical_input
                else:
                    categorical_encoder = ()

                qz_m_n, qz_v_n, _ = self.z_encoder(
                    torch.reshape(
                        torch.log1p(x_n / torch.mean(x_n, dim=1, keepdim=True)),
                        (x.shape[0] * self.n_neighbors, x.shape[1]),
                    ),
                    batch_index.repeat_interleave(self.n_neighbors).unsqueeze(1),
                    *categorical_encoder,
                )

                if z.ndim == 2:
                    zn = Normal(
                        qz_m_n.reshape(x.shape[0], self.n_neighbors, self.n_latent),
                        torch.sqrt(qz_v_n.reshape(x.shape[0], self.n_neighbors, self.n_latent)),
                    ).sample()
                    _, _, px_rate_n, _ = self.decoder(
                        self.dispersion,
                        zn.reshape([x.shape[0] * self.n_neighbors, self.n_latent]),
                        library.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        batch_index.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        *categorical_input,
                    )
                    px_rate_n = px_rate_n.reshape([x.shape[0], self.n_neighbors, self.n_input])
                else:
                    zn = Normal(
                        qz_m_n.reshape(x.shape[0], self.n_neighbors, self.n_latent),
                        torch.sqrt(qz_v_n.reshape(x.shape[0], self.n_neighbors, self.n_latent)),
                    ).sample([z.shape[0]])
                    _, _, px_rate_n, _ = self.decoder(
                        self.dispersion,
                        zn.reshape([z.shape[0], x.shape[0] * self.n_neighbors, self.n_latent]),
                        library.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        batch_index.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        *categorical_input,
                    )
                    px_rate_n = px_rate_n.reshape([z.shape[0], x.shape[0], self.n_neighbors, self.n_input])

                px_rate_n = pyro.deterministic("px_rate_n", px_rate_n, event_dim=2)

            px_rate_sum = torch.sum(
                torch.cat(
                    [
                        (true_mixture_proportion.unsqueeze(-1) * px_rate).unsqueeze(-2),
                        v.unsqueeze(-1) * px_rate_n,
                    ],
                    dim=-2,
                ),
                dim=-2,
            )

            if self.gene_likelihood == "poisson":
                mean_nb = Delta(px_rate_sum, event_dim=1).rsample()
            else:
                mean_nb = Gamma(concentration=px_r, rate=px_r / (px_rate_sum + self.eps)).to_event(1).rsample()

            mean_poisson = pyro.deterministic(
                "mean_poisson",
                mean_nb + background,
                event_dim=1,
            )

            pyro.sample("obs", Poisson(mean_poisson + 1e-9).to_event(1))

            if self.semisupervised:
                probs_prediction = pyro.deterministic(
                    "probs_prediction",
                    probs_prediction_,
                    event_dim=1,
                )
                is_observed = y != self.n_labels
                valid_data = y.clone()
                valid_data[~is_observed] = 0

                with pyro.poutine.scale(scale=50.0):
                    with pyro.poutine.mask(mask=is_observed):
                        pyro.sample("prediction", Categorical(probs=probs_prediction), obs=valid_data)

    @auto_move_data
    def model_corrected(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        """Model conditioned on zero background and diffusion."""
        pyro.condition(
            self.model_unconditioned,
            data={
                "background_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
                "diffusion_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
                "true_mixture_proportion": torch.ones(x.shape[0], device=x.device),
            },
        )(x, ind_x, library, y, batch_index, cat_covs, x_n, distances_n, n_obs, kl_weight)


class RESOLVAEGuide(PyroModule):
    """A PyroModule that serves as the guide for the RESOLVAE class.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_obs
        Number of total input cells.
    n_neighbors
        Number of spatial neighbors to consider for diffusion.
    z_encoder
        Shared encoder between model and guide.
    n_latent
        Dimensionality of the latent space.
    n_batch
        Number of batches.
    n_layers
        Number of hidden layers.
    n_hidden_encoder
        Number of nodes per hidden layer in the encoder.
    n_cats_per_cov
        Number of categories for each extra categorical covariate.
    dispersion
        Dispersion mode.
    downsample_counts_mean
        Mean of the log-normal distribution used to downsample counts.
    downsample_counts_std
        Standard deviation of the log-normal distribution.
    encode_covariates
        Whether to concatenate covariates to expression in encoder.
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    median_distance
        Kernel size in the RBF kernel.
    diffusion_eps
        Epsilon value for diffusion.
    """

    def __init__(
        self,
        n_input: int,
        n_obs: int,
        n_neighbors: int,
        z_encoder: Encoder,
        n_batch: int = 0,
        n_latent: int = 10,
        n_layers: int = 2,
        n_hidden_encoder: int = 128,
        n_cats_per_cov: Iterable[int] | None = None,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        downsample_counts_mean: int | None = None,
        downsample_counts_std: float = 1.0,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        median_distance: float = 1.0,
        diffusion_eps: float = 0.01,
    ):
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro is required for RESOLVAEGuide. Install with: pip install pyro-ppl")

        super().__init__(_RESOLVAE_PYRO_MODULE_NAME)
        self.dispersion = dispersion
        self.z_encoder = z_encoder
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.encode_covariates = encode_covariates
        self.n_input = n_input
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.median_distance = median_distance
        self.downsample_counts_mean = downsample_counts_mean
        self.downsample_counts_std = downsample_counts_std

        if self.dispersion == "gene":
            init_px_r = torch.full([n_input], 0.01)
        elif self.dispersion == "gene-batch":
            init_px_r = torch.full([n_input, n_batch], 0.01)
        else:
            raise ValueError(f"dispersion must be one of ['gene', 'gene-batch'], but input was {dispersion}.")
        self.register_buffer("px_r", init_px_r)
        self.register_buffer("per_neighbor_diffusion_init", torch.zeros([n_obs, n_neighbors]))
        self.register_buffer("gene_dummy", torch.ones([n_batch, n_input]))
        self.eps = torch.tensor(1e-6)
        self.diffusion_eps = diffusion_eps

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.diffusion_encoder = Encoder(
            n_input,
            3,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_encoder,
            dropout_rate=0.0,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=torch.nn.Softmax(dim=-1),
            var_eps=1e-3,
        )

    @auto_move_data
    def forward(
        self,
        x,
        ind_x,
        library,
        y,
        batch_index,
        cat_covs,
        x_n,
        distances_n,
        n_obs=None,
        kl_weight=1.0,
    ):
        """Variational guide for inference."""
        sparsity_diffusion_est = pyro.param(
            "sparsity_diffusion_est",
            20.0 * x.new_ones([1]),
            constraint=constraints.greater_than(1e-3),
        )
        pyro.sample("sparsity_diffusion", Delta(sparsity_diffusion_est))

        background_concentration = torch.softmax(
            pyro.param("background_concentration", self.gene_dummy, event_dim=2),
            dim=-1,
        )
        pyro.sample("per_gene_background", Delta(background_concentration, event_dim=2))

        background_proportion_est = pyro.param(
            "background_proportion_est",
            0.5 * x.new_ones([1]),
            constraint=constraints.greater_than(1e-6),
        )
        pyro.sample("background_proportion", Delta(background_proportion_est))

        diffusion_proportion_est = pyro.param(
            "diffusion_proportion_est",
            3.0 * x.new_ones([1]),
            constraint=constraints.greater_than(1e-6),
        )
        pyro.sample("diffusion_proportion", Delta(diffusion_proportion_est))

        true_proportion_est = pyro.param(
            "true_proportion_est", 5.0 * x.new_ones([1]), constraint=constraints.greater_than(1e-6)
        )
        pyro.sample("true_proportion", Delta(true_proportion_est))

        diffusion_scale_est = pyro.param(
            "diffuse_scale_est",
            x.new_ones([1]),
            constraint=constraints.greater_than(self.eps),
            event_dim=1,
        )
        pyro.sample("diffuse_scale", Delta(diffusion_scale_est, event_dim=1))

        per_neighbor_diffusion = pyro.param(
            "per_neighbor_diffusion_map",
            self.per_neighbor_diffusion_init,
            constraint=constraints.interval(-10.0, 10.0),
            event_dim=1,
        )

        if self.downsample_counts_mean is not None:
            downsample_counts = (
                int(LogNormal(float(self.downsample_counts_mean), float(self.downsample_counts_std)).sample()) + 10
            )

        with pyro.plate("obs_plate", size=n_obs or self.n_obs, subsample=ind_x, dim=-1):
            px_r_mle = pyro.param(
                "px_r_mle",
                self.px_r,
                constraint=constraints.greater_than(self.eps),
                event_dim=len(self.px_r.shape),
            )

            if self.dispersion == "gene-batch":
                px_r_inv = F.linear(
                    torch.nn.functional.one_hot(batch_index.flatten(), self.n_batch).to(px_r_mle.dtype),
                    px_r_mle,
                )
            elif self.dispersion == "gene":
                px_r_inv = px_r_mle
            pyro.sample("px_r_inv", Delta(px_r_inv, event_dim=1))

            concentration = torch.nn.Softmax(dim=-1)(
                per_neighbor_diffusion[ind_x, :] - torch.clamp(torch.sqrt(distances_n / self.median_distance), max=10.0)
            )
            pyro.sample("per_neighbor_diffusion", Delta(concentration, event_dim=1))

            if cat_covs is not None and self.encode_covariates:
                categorical_input = list(torch.split(cat_covs, 1, dim=1))
            else:
                categorical_input = ()

            with pyro.poutine.scale(scale=5.0):
                _, mixture_proportions_est, _ = self.diffusion_encoder(torch.log1p(x), batch_index, *categorical_input)
                mixture_proportions_est[..., 1] += self.diffusion_eps
                pyro.sample("mixture_proportions", Delta(mixture_proportions_est, event_dim=1))

            with pyro.poutine.scale(scale=kl_weight):
                if self.training and self.downsample_counts_mean is not None:
                    x = Multinomial(total_count=downsample_counts, probs=x).sample()
                qz_m, qz_v, _ = self.z_encoder(
                    torch.log1p(x / torch.mean(x, dim=1, keepdim=True)),
                    batch_index,
                    *categorical_input,
                )
                pyro.sample("latent", Normal(qz_m, torch.sqrt(qz_v)).to_event(1))


class RESOLVAE(PyroBaseModuleClass):
    """Implementation of ResolVI module.

    This is the main module class that combines the model and guide
    for training with Pyro's SVI.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_obs
        Number of total input cells.
    n_neighbors
        Number of spatial neighbors.
    expression_anntorchdata
        AnnTorchDataset with expression data.
    n_batch
        Number of batches.
    n_hidden
        Number of nodes per hidden layer in decoder.
    n_hidden_encoder
        Number of nodes per hidden layer in encoder.
    n_latent
        Dimensionality of the latent space.
    mixture_k
        Number of mixture components.
    n_layers
        Number of hidden layers.
    n_cats_per_cov
        Number of categories for each categorical covariate.
    n_labels
        Number of cell-type labels.
    dropout_rate
        Dropout rate.
    dispersion
        Dispersion mode.
    gene_likelihood
        Gene likelihood distribution.
    semisupervised
        Whether to use semi-supervised mode.
    encode_covariates
        Whether to encode covariates.
    deeply_inject_covariates
        Whether to deeply inject covariates.
    use_batch_norm
        Batch norm mode.
    use_layer_norm
        Layer norm mode.
    var_activation
        Variance activation function.
    classifier_parameters
        Parameters for classifier.
    prior_true_amount
        Prior for true proportion.
    prior_diffusion_amount
        Prior for diffusion proportion.
    sparsity_diffusion
        Sparsity diffusion prior.
    background_ratio
        Background ratio prior.
    prior_proportions_rate
        Prior proportions rate.
    median_distance
        Median distance for kernel.
    downsample_counts_mean
        Downsample counts mean.
    downsample_counts_std
        Downsample counts std.
    diffusion_eps
        Diffusion epsilon.
    latent_distribution
        Placeholder for compatibility.
    """

    def __init__(
        self,
        n_input: int,
        n_obs: int,
        n_neighbors: int,
        expression_anntorchdata: AnnTorchDataset,
        n_batch: int = 0,
        n_hidden: int = 32,
        n_hidden_encoder: int = 128,
        n_latent: int = 10,
        mixture_k: int = 30,
        n_layers: int = 2,
        n_cats_per_cov: Iterable[int] | None = None,
        n_labels: Iterable[int] | None = None,
        dropout_rate: float = 0.05,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        semisupervised: bool = False,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        var_activation: Callable | None = None,
        classifier_parameters: dict | None = None,
        prior_true_amount: float = 1.0,
        prior_diffusion_amount: float = 0.3,
        sparsity_diffusion: float = 3.0,
        background_ratio: float = 0.1,
        prior_proportions_rate: float = 10.0,
        median_distance: float = 1.0,
        downsample_counts_mean: float | None = None,
        downsample_counts_std: float = 1.0,
        diffusion_eps: float = 0.01,
        latent_distribution: str | None = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.mixture_k = mixture_k
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.expression_anntorchdata = expression_anntorchdata
        self.semisupervised = semisupervised
        self.eps = torch.tensor(1e-6)
        self.encode_covariates = encode_covariates

        use_batch_norm_encoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if self.encode_covariates else None

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_encoder,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )

        self._guide = RESOLVAEGuide(
            z_encoder=self.z_encoder,
            n_input=n_input,
            n_obs=n_obs,
            n_neighbors=n_neighbors,
            n_batch=n_batch,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden_encoder=n_hidden_encoder,
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            median_distance=median_distance,
            downsample_counts_mean=downsample_counts_mean,
            downsample_counts_std=downsample_counts_std,
            diffusion_eps=diffusion_eps,
        )

        self._model = RESOLVAEModel(
            n_input=n_input,
            n_obs=n_obs,
            n_neighbors=n_neighbors,
            z_encoder=self.z_encoder,
            expression_anntorchdata=expression_anntorchdata,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            mixture_k=mixture_k,
            n_layers=n_layers,
            n_cats_per_cov=n_cats_per_cov,
            n_labels=n_labels,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            semisupervised=semisupervised,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            classifier_parameters=classifier_parameters,
            prior_true_amount=prior_true_amount,
            prior_diffusion_amount=prior_diffusion_amount,
            sparsity_diffusion=sparsity_diffusion,
            background_ratio=background_ratio,
            prior_proportions_rate=prior_proportions_rate,
            median_distance=median_distance,
        )
        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        """Return the generative model."""
        return self._model

    @property
    def model_corrected(self):
        """Return the corrected model."""
        return self._model.model_corrected

    @property
    def model_unconditioned(self):
        """Return the unconditioned model."""
        return self._model.model_unconditioned

    @property
    def guide(self):
        """Return the variational guide."""
        return self._guide

    @property
    def list_obs_plate_vars(self):
        """Return observation plate variables."""
        return {
            "name": "obs_plate",
            "event_dim": 1,
        }
