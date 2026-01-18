"""VAE mixin for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from anndata import AnnData
    from numpy.typing import NDArray
    from torch import Tensor

logger = logging.getLogger(__name__)


class VAEMixin:
    """Universal variational auto-encoder (VAE) methods for spatial models.

    This mixin provides methods common to VAE-based models including:
    - ELBO computation
    - Reconstruction error
    - Marginal log-likelihood estimation
    - Latent representation extraction
    """

    @torch.inference_mode()
    def get_elbo(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        **kwargs,
    ) -> float | NDArray:
        """Compute the evidence lower bound (ELBO) on the data.

        The ELBO is the reconstruction error plus the Kullback-Leibler (KL)
        divergences between the variational distributions and the priors.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of observations to use. If None, defaults to all observations.
        batch_size
            Minibatch size for the forward pass.
        return_mean
            Whether to return the mean of the ELBO or the ELBO for each observation.
        **kwargs
            Additional keyword arguments to pass into the forward method of the module.

        Returns
        -------
        Evidence lower bound (ELBO) of the data. Higher is better.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        elbo_list = []
        for tensors in scdl:
            _, _, losses = self.module(tensors, compute_loss=True, **kwargs)
            elbo = -(losses.reconstruction_loss_sum + losses.kl_local_sum + losses.kl_global_sum)
            elbo_list.append(elbo.cpu().numpy())

        if return_mean:
            return float(np.mean(elbo_list))
        return np.concatenate(elbo_list, axis=0)

    @torch.inference_mode()
    def get_reconstruction_error(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        **kwargs,
    ) -> dict[str, float] | dict[str, NDArray]:
        """Compute the reconstruction error on the data.

        The reconstruction error is the negative log likelihood of the data
        given the latent variables.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of observations to use. If None, defaults to all observations.
        batch_size
            Minibatch size for the forward pass.
        return_mean
            Whether to return the mean reconstruction loss or the reconstruction
            loss for each observation.
        **kwargs
            Additional keyword arguments to pass into the forward method of the module.

        Returns
        -------
        Dictionary of reconstruction errors. Higher (less negative) is better.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        rec_losses: dict[str, list] = {}
        for tensors in scdl:
            _, _, losses = self.module(tensors, compute_loss=True, **kwargs)
            for key, val in losses.reconstruction_loss.items():
                if key not in rec_losses:
                    rec_losses[key] = []
                rec_losses[key].append(val.cpu().numpy())

        if return_mean:
            return {key: float(np.mean(val)) for key, val in rec_losses.items()}
        return {key: np.concatenate(val, axis=0) for key, val in rec_losses.items()}

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: int | None = None,
        return_dist: bool = False,
        **kwargs,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Compute the latent representation of the data.

        This is typically denoted as z_n.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of observations to use. If None, defaults to all observations.
        give_mean
            If True, returns the mean of the latent distribution.
        mc_samples
            Number of Monte Carlo samples for distributions without closed-form mean.
        batch_size
            Minibatch size for the forward pass.
        return_dist
            If True, returns the mean and variance of the latent distribution.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Array of shape (n_obs, n_latent) if return_dist is False. Otherwise,
        tuple of (mean, variance) arrays.
        """
        from torch.distributions import Normal

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        zs: list[Tensor] = []
        qz_means: list[Tensor] = []
        qz_vars: list[Tensor] = []

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs, **kwargs)

            # Handle different output formats
            if "qz" in outputs:
                qz = outputs["qz"]
                qzm = qz.loc
                qzv = qz.scale.square()
            elif "qz_m" in outputs and "qz_v" in outputs:
                qzm = outputs["qz_m"]
                qzv = outputs["qz_v"]
                qz = Normal(qzm, qzv.sqrt())
            else:
                # Fallback for simple latent outputs
                z = outputs.get("z", outputs.get("latent"))
                if z is not None:
                    zs.append(z.cpu())
                    continue
                raise ValueError("Could not find latent representation in inference outputs")

            if return_dist:
                qz_means.append(qzm.cpu())
                qz_vars.append(qzv.cpu())
                continue

            z = qzm if give_mean else outputs.get("z", outputs.get("latent"))
            zs.append(z.cpu())

        if return_dist:
            return torch.cat(qz_means).numpy(), torch.cat(qz_vars).numpy()
        return torch.cat(zs).numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        library_size: float | str = 1.0,
        **kwargs,
    ) -> NDArray:
        """Return normalized gene expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of observations to use. If None, defaults to all observations.
        n_samples
            Number of posterior samples.
        batch_size
            Minibatch size for the forward pass.
        return_mean
            Whether to return the mean of samples.
        library_size
            Library size to use for normalization.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Normalized expression array.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        exprs = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs, **kwargs)

            gen_inputs = self.module._get_generative_input(
                tensors, inference_outputs, **kwargs
            )
            gen_outputs = self.module.generative(**gen_inputs, **kwargs)

            # Get the scale/rate parameter
            if "px_scale" in gen_outputs:
                scale = gen_outputs["px_scale"]
            elif "px" in gen_outputs and hasattr(gen_outputs["px"], "mean"):
                scale = gen_outputs["px"].mean
            else:
                scale = gen_outputs.get("scale", gen_outputs.get("rate"))

            if scale is None:
                raise NotImplementedError(
                    "This model does not support normalized expression retrieval."
                )

            if isinstance(library_size, (int, float)):
                scale = scale * library_size

            exprs.append(scale.cpu().numpy())

        return np.concatenate(exprs, axis=0)
