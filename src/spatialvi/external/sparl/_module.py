"""SPARL module for spatial proteomics analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers
from torch import nn
from torch.distributions import Normal

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class SPARLModule(BaseModuleClass):
    """SPARL module for spatial proteomics representation learning.

    Parameters
    ----------
    n_proteins
        Number of protein channels.
    n_hidden
        Number of hidden units.
    n_latent
        Latent dimension.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    use_spatial
        Whether to use spatial context.
    spatial_dim
        Dimension of spatial features.
    """

    def __init__(
        self,
        n_proteins: int,
        n_hidden: int = 128,
        n_latent: int = 32,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_spatial: bool = True,
        spatial_dim: int = 64,
    ):
        super().__init__()

        self.n_proteins = n_proteins
        self.n_latent = n_latent
        self.use_spatial = use_spatial
        self.spatial_dim = spatial_dim

        # Protein encoder
        encoder_input_dim = n_proteins
        if use_spatial:
            encoder_input_dim += spatial_dim

        self.encoder = FCLayers(
            n_in=encoder_input_dim,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )

        # Latent distribution parameters
        self.z_mean = nn.Linear(n_hidden, n_latent)
        self.z_var = nn.Sequential(
            nn.Linear(n_hidden, n_latent),
            nn.Softplus(),
        )

        # Spatial context encoder
        if use_spatial:
            self.spatial_encoder = FCLayers(
                n_in=2,  # x, y coordinates
                n_out=spatial_dim,
                n_layers=1,
                n_hidden=spatial_dim,
                dropout_rate=0.0,
            )

        # Decoder
        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )

        # Output layers
        self.px_mean = nn.Linear(n_hidden, n_proteins)
        self.px_var = nn.Sequential(
            nn.Linear(n_hidden, n_proteins),
            nn.Softplus(),
        )

    def _get_inference_input(
        self,
        tensors: dict[str, Tensor],
        **kwargs,
    ) -> dict[str, Tensor | None]:
        """Get inference inputs."""
        return {
            "x": tensors.get("X"),
            "spatial": tensors.get("spatial"),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, Tensor],
        inference_outputs: dict[str, Tensor | Distribution],
        **kwargs,
    ) -> dict[str, Tensor | None]:
        """Get generative inputs."""
        return {
            "z": inference_outputs["z"],
        }

    def inference(
        self,
        x: Tensor,
        spatial: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor | Distribution]:
        """Run inference network.

        Parameters
        ----------
        x
            Protein expression tensor.
        spatial
            Spatial coordinates.

        Returns
        -------
        Inference outputs.
        """
        # Log transform input
        x_log = torch.log1p(x)

        # Encode spatial context
        if self.use_spatial:
            if spatial is not None:
                spatial_features = self.spatial_encoder(spatial)
            else:
                # If spatial is None, use zeros
                batch_size = x.shape[0]
                spatial_features = torch.zeros(batch_size, self.spatial_dim, device=x.device, dtype=x.dtype)
            encoder_input = torch.cat([x_log, spatial_features], dim=-1)
        else:
            encoder_input = x_log

        # Encode
        h = self.encoder(encoder_input)

        # Latent distribution
        qz_m = self.z_mean(h)
        qz_v = self.z_var(h) + 1e-4

        qz = Normal(qz_m, qz_v.sqrt())
        z = qz.rsample()

        return {
            "z": z,
            "qz": qz,
            "qz_m": qz_m,
            "qz_v": qz_v,
        }

    def generative(
        self,
        z: Tensor,
        **kwargs,
    ) -> dict[str, Tensor | Distribution]:
        """Run generative network.

        Parameters
        ----------
        z
            Latent representation.

        Returns
        -------
        Generative outputs.
        """
        # Decode
        h = self.decoder(z)

        # Output distribution
        px_m = self.px_mean(h)
        px_v = self.px_var(h) + 1e-4

        px = Normal(px_m, px_v.sqrt())

        return {
            "px": px,
            "px_m": px_m,
            "px_v": px_v,
        }

    def loss(
        self,
        tensors: dict[str, Tensor],
        inference_outputs: dict[str, Tensor | Distribution],
        generative_outputs: dict[str, Tensor | Distribution],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute loss.

        Parameters
        ----------
        tensors
            Input tensors.
        inference_outputs
            Inference outputs.
        generative_outputs
            Generative outputs.
        kl_weight
            KL divergence weight.

        Returns
        -------
        Loss output.
        """
        x = tensors["X"]
        px = generative_outputs["px"]
        qz = inference_outputs["qz"]

        # Reconstruction loss
        reconst_loss = -px.log_prob(torch.log1p(x)).sum(dim=-1)

        # KL divergence
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        kl_z = torch.distributions.kl_divergence(qz, pz).sum(dim=-1)

        # Total loss
        loss = torch.mean(reconst_loss + kl_weight * kl_z)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_z,
        )

    @auto_move_data
    def forward(
        self,
        tensors: dict[str, Tensor],
        get_inference_input_kwargs: dict | None = None,
        get_generative_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        generative_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss: bool = True,
    ) -> tuple[dict, dict] | tuple[dict, dict, LossOutput]:
        """Forward pass."""
        get_inference_input_kwargs = get_inference_input_kwargs or {}
        get_generative_input_kwargs = get_generative_input_kwargs or {}
        inference_kwargs = inference_kwargs or {}
        generative_kwargs = generative_kwargs or {}
        loss_kwargs = loss_kwargs or {}

        inference_inputs = self._get_inference_input(tensors, **get_inference_input_kwargs)
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)

        generative_inputs = self._get_generative_input(tensors, inference_outputs, **get_generative_input_kwargs)
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)

        if compute_loss:
            losses = self.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
            return inference_outputs, generative_outputs, losses

        return inference_outputs, generative_outputs
