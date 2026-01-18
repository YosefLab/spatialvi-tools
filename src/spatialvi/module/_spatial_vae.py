"""Spatial VAE module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, NegativeBinomial
from torch.distributions import kl_divergence as kl

from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial as SCVINB
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
import torch.nn.functional as F

from spatialvi.module._base import BaseSpatialModule
from spatialvi.nn import SpatialEncoder, SpatialDecoder

if TYPE_CHECKING:
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class SpatialVAEModule(BaseSpatialModule):
    """Spatial VAE module for gene expression modeling.

    This module implements a VAE that incorporates spatial context
    for improved latent representations.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_batch
        Number of batches.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        Dispersion parameter type.
    gene_likelihood
        Distribution for gene expression.
    latent_distribution
        Distribution for latent space.
    use_spatial
        Whether to use spatial information.
    spatial_weight
        Weight for spatial regularization.
    library_log_means
        Log means of library sizes.
    library_log_vars
        Log variances of library sizes.
    n_cats_per_cov
        Number of categories per categorical covariate.
    n_continuous_cov
        Number of continuous covariates.
    use_batch_norm
        Whether to use batch normalization.
    use_layer_norm
        Whether to use layer normalization.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_spatial: bool = True,
        spatial_weight: float = 1.0,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        n_cats_per_cov: list[int] | None = None,
        n_continuous_cov: int = 0,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_spatial=use_spatial,
            spatial_weight=spatial_weight,
        )

        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov

        # Dispersion parameters
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(n_input))
        elif dispersion == "gene-batch":
            self.px_r = nn.Parameter(torch.randn(n_input, n_batch))
        elif dispersion == "gene-cell":
            pass  # Will be computed per cell in generative
        else:
            raise ValueError(f"Unknown dispersion: {dispersion}")

        use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        use_layer_norm_decoder = use_layer_norm in ("decoder", "both")

        # Library size prior
        if library_log_means is not None and library_log_vars is not None:
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )
        else:
            self.library_log_means = None
            self.library_log_vars = None

        # Compute input size with covariates
        cat_list = [n_batch] if n_batch > 0 else []
        if n_cats_per_cov is not None:
            cat_list.extend(n_cats_per_cov)

        n_input_encoder = n_input + n_continuous_cov
        encoder_cat_list = cat_list

        # Encoder
        if use_spatial:
            self.z_encoder = SpatialEncoder(
                n_input=n_input_encoder,
                n_output=n_latent,
                n_cat_list=encoder_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
            )
        else:
            self.z_encoder = Encoder(
                n_input=n_input_encoder,
                n_output=n_latent,
                n_cat_list=encoder_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                distribution=latent_distribution,
            )

        # Library encoder
        self.l_encoder = Encoder(
            n_input=n_input_encoder,
            n_output=1,
            n_cat_list=encoder_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        # Decoder
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = DecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Get input for inference network."""
        x = tensors.get("X")
        batch_index = tensors.get("batch", None)
        cont_covs = tensors.get("continuous_covs", None)
        cat_covs = tensors.get("categorical_covs", None)
        spatial_coords = tensors.get("spatial", None)
        neighbor_indices = tensors.get("neighbor_indices", None)

        input_dict = {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "spatial_coords": spatial_coords,
            "neighbor_indices": neighbor_indices,
        }
        return input_dict

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution],
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Get input for generative network."""
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors.get("batch", None)
        cont_covs = tensors.get("continuous_covs", None)
        cat_covs = tensors.get("categorical_covs", None)

        input_dict = {
            "z": z,
            "library": library,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        spatial_coords: torch.Tensor | None = None,
        neighbor_indices: torch.Tensor | None = None,
        n_samples: int = 1,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the inference network.

        Parameters
        ----------
        x
            Gene expression tensor.
        batch_index
            Batch indices.
        cont_covs
            Continuous covariates.
        cat_covs
            Categorical covariates.
        spatial_coords
            Spatial coordinates.
        neighbor_indices
            Neighbor indices.
        n_samples
            Number of samples from latent distribution.

        Returns
        -------
        Dictionary of inference outputs.
        """
        x_ = torch.log1p(x)

        if cont_covs is not None and self.n_continuous_cov > 0:
            encoder_input = torch.cat([x_, cont_covs], dim=-1)
        else:
            encoder_input = x_

        # Build categorical list
        cat_list = []
        if batch_index is not None:
            cat_list.append(batch_index)
        if cat_covs is not None and self.n_cats_per_cov is not None:
            cat_list.append(cat_covs)

        # Encode to latent space
        # Encoder returns (mean, var, sample)
        if self.use_spatial and hasattr(self.z_encoder, "forward_spatial"):
            qz_m, qz_v, z = self.z_encoder.forward_spatial(
                encoder_input,
                cat_list=cat_list if cat_list else None,
                spatial_coords=spatial_coords,
                neighbor_indices=neighbor_indices,
            )
        else:
            qz_m, qz_v, z = self.z_encoder(encoder_input, *cat_list)

        # Encode library size
        ql_m, ql_v, _ = self.l_encoder(encoder_input, *cat_list)

        if self.library_log_means is not None:
            local_l_mean = self.library_log_means[batch_index.squeeze(-1).long()]
            local_l_var = self.library_log_vars[batch_index.squeeze(-1).long()]
        else:
            local_l_mean = torch.zeros_like(ql_m)
            local_l_var = torch.ones_like(ql_v)

        library = torch.exp(ql_m + 0.5 * ql_v)

        return {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "ql_m": ql_m,
            "ql_v": ql_v,
            "library": library,
            "local_l_mean": local_l_mean,
            "local_l_var": local_l_var,
        }

    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Distribution]:
        """Run the generative network.

        Parameters
        ----------
        z
            Latent representation.
        library
            Library size.
        batch_index
            Batch indices.
        cont_covs
            Continuous covariates.
        cat_covs
            Categorical covariates.

        Returns
        -------
        Dictionary of generative outputs.
        """
        # Prepare decoder input
        if cont_covs is not None and self.n_continuous_cov > 0:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
        else:
            decoder_input = z

        # Build categorical list
        cat_list = []
        if batch_index is not None:
            cat_list.append(batch_index)
        if cat_covs is not None and self.n_cats_per_cov is not None:
            cat_list.append(cat_covs)

        # Decode
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            library,
            *cat_list,
        )

        if self.dispersion == "gene":
            px_r = torch.exp(self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = torch.exp(self.px_r[:, batch_index.squeeze(-1).long()]).T

        # Build distribution
        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
            )
        elif self.gene_likelihood == "nb":
            px = SCVINB(
                mu=px_rate,
                theta=px_r,
            )
        elif self.gene_likelihood == "poisson":
            px = torch.distributions.Poisson(rate=px_rate)
        else:
            raise ValueError(f"Unknown gene likelihood: {self.gene_likelihood}")

        return {
            "px": px,
            "px_rate": px_rate,
            "px_r": px_r,
            "px_scale": px_scale,
            "px_dropout": px_dropout,
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution],
        generative_outputs: dict[str, torch.Tensor | Distribution],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the loss.

        Parameters
        ----------
        tensors
            Dictionary of input tensors.
        inference_outputs
            Dictionary of inference outputs.
        generative_outputs
            Dictionary of generative outputs.
        kl_weight
            Weight for KL divergence term.

        Returns
        -------
        LossOutput containing reconstruction loss, KL divergence, etc.
        """
        x = tensors["X"]
        px = generative_outputs["px"]

        # Reconstruction loss
        reconst_loss = -px.log_prob(x).sum(dim=-1)

        # KL divergence for latent z
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(mean, scale),
        ).sum(dim=-1)

        # KL divergence for library
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        local_l_mean = inference_outputs["local_l_mean"]
        local_l_var = inference_outputs["local_l_var"]

        kl_library = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=-1)

        # Spatial regularization
        spatial_loss = self._compute_spatial_loss(
            inference_outputs["z"],
            neighbor_indices=tensors.get("neighbor_indices"),
        )

        # Total loss
        kl_local = kl_z + kl_library
        weighted_kl = kl_weight * kl_local + spatial_loss
        loss = torch.mean(reconst_loss + weighted_kl)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics={"spatial_loss": spatial_loss},
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the model.

        Parameters
        ----------
        tensors
            Dictionary of input tensors.
        n_samples
            Number of samples.

        Returns
        -------
        Sampled gene expression.
        """
        inference_inputs = self._get_inference_input(tensors)
        inference_outputs = self.inference(**inference_inputs)

        generative_inputs = self._get_generative_input(tensors, inference_outputs)

        samples = []
        for _ in range(n_samples):
            generative_outputs = self.generative(**generative_inputs)
            px = generative_outputs["px"]
            samples.append(px.sample())

        return torch.stack(samples, dim=0)
