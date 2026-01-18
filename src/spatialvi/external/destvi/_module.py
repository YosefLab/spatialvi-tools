"""DestVI module implementation.

This module contains the neural network components for DestVI,
a model for multi-resolution deconvolution of spatial transcriptomics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial
from scvi.module.base import BaseModuleClass, EmbeddingModuleMixin, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    Normal,
)

if TYPE_CHECKING:
    from collections import OrderedDict


def identity(x):
    """Identity function."""
    return x


class MRDeconv(EmbeddingModuleMixin, BaseModuleClass):
    """Model for multi-resolution deconvolution of spatial transcriptomics.

    This module learns to deconvolve spatial transcriptomics data into
    cell type proportions and cell state representations.

    Parameters
    ----------
    n_spots
        Number of input spots.
    n_labels
        Number of cell types.
    n_batch
        Number of batches.
    n_hidden
        Number of neurons in the hidden layers.
    n_layers
        Number of layers used in the encoder networks.
    n_latent
        Number of dimensions used in the latent variables.
    n_genes
        Number of genes used in the decoder.
    decoder_state_dict
        State dict from the decoder of the CondSCVI model.
    px_decoder_state_dict
        State dict from the px_decoder of the CondSCVI model.
    px_r
        Parameters for the px_r tensor in the CondSCVI model.
    per_ct_bias
        Estimates of per cell-type expression bias in the CondSCVI model.
    dropout_decoder
        Dropout rate for the decoder neural network.
    dropout_amortization
        Dropout rate for the amortization neural network.
    augmentation
        Whether to use augmentation during training.
    n_samples_augmentation
        Number of samples used in the augmentation.
    n_states_per_label
        Number of states per cell-type in each spot.
    eps_v
        Epsilon value for each cell-type proportion used during training.
    mean_vprior
        Mean parameter for each component in the empirical prior.
    var_vprior
        Diagonal variance parameter for each component in the empirical prior.
    mp_vprior
        Mixture proportion in cell type sub-clustering of each component.
    amortization
        Mode of amortization.
    prior_mode
        Mode of the prior distribution for the latent space.
    add_celltypes
        Number of additional cell types to add to the model.
    n_latent_amortization
        Number of dimensions used in the latent variables for amortization.
    extra_encoder_kwargs
        Extra keyword arguments passed into encoder.
    extra_decoder_kwargs
        Extra keyword arguments passed into decoder.
    """

    def __init__(
        self,
        n_spots: int,
        n_labels: int,
        n_batch: int,
        n_hidden: int,
        n_layers: int,
        n_latent: int,
        n_genes: int,
        decoder_state_dict: OrderedDict,
        px_decoder_state_dict: OrderedDict,
        px_r: torch.Tensor,
        per_ct_bias: torch.Tensor,
        dropout_decoder: float,
        dropout_amortization: float = 0.03,
        augmentation: bool = True,
        n_samples_augmentation: int = 2,
        n_states_per_label: int = 3,
        eps_v: float = 2e-3,
        mean_vprior: np.ndarray = None,
        var_vprior: np.ndarray = None,
        mp_vprior: np.ndarray = None,
        amortization: Literal["none", "latent", "proportion", "both"] = "both",
        prior_mode: Literal["mog", "normal"] = "mog",
        add_celltypes: int = 1,
        n_latent_amortization: int | None = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
    ):
        super().__init__()
        if prior_mode == "mog":
            assert amortization in ["both", "latent"], (
                "Amortization must be active for latent variables to use mixture of gaussians generation"
            )
        self.n_spots = n_spots
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.augmentation = augmentation
        self.n_samples_augmentation = n_samples_augmentation
        self.dropout_decoder = dropout_decoder
        self.n_states_per_label = n_states_per_label
        self.dropout_amortization = dropout_amortization
        self.n_genes = n_genes
        self.amortization = amortization
        self.prior_mode = prior_mode
        self.add_celltypes = add_celltypes
        self.eps_v = eps_v
        self.n_latent_amortization = n_latent_amortization

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        cat_list = [n_labels]
        self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch)
        batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        n_input_decoder = n_latent + batch_dim

        self.decoder = FCLayers(
            n_in=n_input_decoder,
            n_out=n_hidden,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_decoder,
            use_layer_norm=True,
            use_batch_norm=False,
            **_extra_decoder_kwargs,
        )
        self.decoder.load_state_dict(decoder_state_dict)
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.px_decoder = torch.nn.Linear(n_hidden, n_genes)
        self.px_decoder.load_state_dict(px_decoder_state_dict)
        for param in self.px_decoder.parameters():
            param.requires_grad = False

        self.px_r = torch.nn.Parameter(px_r)
        self.register_buffer("per_ct_bias", per_ct_bias)

        # Cell-type specific factor loadings
        self.V = torch.nn.Parameter(torch.randn(self.n_labels + self.add_celltypes, self.n_spots))

        # Within cell-type factor loadings
        self.gamma = torch.nn.Parameter(torch.randn(n_latent, self.n_labels, self.n_spots))

        if mean_vprior is not None:
            self.register_buffer("mean_vprior", mean_vprior)
            self.register_buffer("var_vprior", var_vprior)
            self.register_buffer("mp_vprior", mp_vprior)
            cats = Categorical(probs=self.mp_vprior)
            normal_dists = Independent(
                Normal(self.mean_vprior, torch.sqrt(self.var_vprior) + 1e-4),
                reinterpreted_batch_ndims=1,
            )
            self.qz_prior = MixtureSameFamily(cats, normal_dists)
        else:
            self.mean_vprior = None
            self.var_vprior = None

        # Noise from data
        self.eta = torch.nn.Parameter(torch.zeros(self.add_celltypes, self.n_genes))

        # Additive gene bias
        self.beta = torch.nn.Parameter(torch.zeros(self.n_genes))

        # Create amortization neural nets
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        if self.prior_mode == "mog":
            return_dist = self.n_states_per_label * n_labels * n_latent + self.n_states_per_label * n_labels
        else:
            return_dist = n_labels * n_latent

        n_input_encoder_amortized_latent = n_genes
        if self.n_latent_amortization is not None:
            self.encoder_z_amortization = Encoder(
                n_input=n_input_encoder_amortized_latent,
                n_output=return_dist,
                n_cat_list=None,
                n_layers=n_layers,
                n_hidden=n_latent_amortization,
                dropout_rate=dropout_amortization,
                use_batch_norm=False,
                use_layer_norm=True,
                activation_fn=torch.nn.LeakyReLU,
                var_eps=1e-4,
                return_dist=False,
                **_extra_encoder_kwargs,
            )
        else:
            self.encoder_z_amortization = FCLayers(
                n_in=n_input_encoder_amortized_latent,
                n_out=return_dist,
                n_cat_list=None,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_amortization,
                use_batch_norm=False,
                use_layer_norm=True,
                activation_fn=torch.nn.LeakyReLU,
                **_extra_encoder_kwargs,
            )

        self.encoder_v_amortization = Encoder(
            n_input=n_input_encoder_amortized_latent,
            n_output=n_labels + self.add_celltypes,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_amortization,
            use_batch_norm=False,
            use_layer_norm=True,
            activation_fn=torch.nn.LeakyReLU,
            var_eps=1e-4,
            return_dist=False,
            **_extra_encoder_kwargs,
        )

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Get input for inference."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        ind_x = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()
        return {"x": x, "ind_x": ind_x}

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get input for generative."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        ind_x = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()
        batch_index = tensors.get(REGISTRY_KEYS.BATCH_KEY, None)

        v = inference_outputs["v"]
        z = inference_outputs["z"]

        return {
            "x": x,
            "ind_x": ind_x,
            "v": v,
            "z": z,
            "batch_index": batch_index,
        }

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference step.

        Parameters
        ----------
        x
            Expression data.
        ind_x
            Spot indices.

        Returns
        -------
        Dictionary with inference outputs including v and z.
        """
        library = torch.log(x.sum(1, keepdim=True))

        # Get amortized cell-type proportions
        if self.amortization in ["proportion", "both"]:
            v, _, _ = self.encoder_v_amortization(torch.log1p(x))
            v = torch.softmax(v, dim=-1)
        else:
            v = torch.softmax(self.V[:, ind_x].T, dim=-1)

        # Get amortized latent states
        if self.amortization in ["latent", "both"]:
            z_params = self.encoder_z_amortization(torch.log1p(x))
            if self.prior_mode == "mog":
                # Mixture of Gaussians
                z_mean = z_params[:, : self.n_states_per_label * self.n_labels * self.n_latent]
                z_mean = z_mean.reshape(-1, self.n_labels, self.n_states_per_label, self.n_latent)
                z_logits = z_params[:, self.n_states_per_label * self.n_labels * self.n_latent :]
                z_logits = z_logits.reshape(-1, self.n_labels, self.n_states_per_label)
                z_probs = torch.softmax(z_logits, dim=-1)
                z = (z_mean * z_probs.unsqueeze(-1)).sum(dim=2)
            else:
                z = z_params.reshape(-1, self.n_labels, self.n_latent)
        else:
            z = self.gamma[:, :, ind_x].permute(2, 1, 0)

        return {
            "v": v,
            "z": z,
            "library": library,
        }

    @auto_move_data
    def generative(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        v: torch.Tensor,
        z: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generative step.

        Parameters
        ----------
        x
            Expression data.
        ind_x
            Spot indices.
        v
            Cell type proportions.
        z
            Latent states.
        batch_index
            Batch indices.

        Returns
        -------
        Dictionary with generative outputs.
        """
        library = torch.log(x.sum(1, keepdim=True))

        if batch_index is not None:
            batch_emb = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
        else:
            batch_emb = torch.zeros(x.shape[0], 1, device=x.device)

        # Decode each cell type
        px_rates = []
        for ct in range(self.n_labels):
            z_ct = z[:, ct, :]
            decoder_input = torch.cat([z_ct, batch_emb], dim=-1)
            ct_tensor = torch.full((x.shape[0], 1), ct, device=x.device, dtype=torch.long)
            h = self.decoder(decoder_input, ct_tensor)
            px_scale = torch.softmax(self.px_decoder(h), dim=-1)
            px_rate = px_scale * torch.exp(library)
            px_rates.append(px_rate)

        px_rates = torch.stack(px_rates, dim=1)

        # Add noise cell types
        if self.add_celltypes > 0:
            noise_rates = torch.softmax(self.eta, dim=-1).unsqueeze(0) * torch.exp(library.unsqueeze(-1))
            px_rates = torch.cat([px_rates, noise_rates.expand(x.shape[0], -1, -1)], dim=1)

        # Combine with proportions
        px_rate = (v.unsqueeze(-1) * px_rates).sum(dim=1)

        # Add gene bias
        px_rate = px_rate + torch.softmax(self.beta, dim=-1) * torch.exp(library)

        px_r = torch.exp(self.px_r)

        return {
            "px_rate": px_rate,
            "px_r": px_r,
            "px_scale": px_rate / px_rate.sum(dim=-1, keepdim=True),
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
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
            KL weight.

        Returns
        -------
        LossOutput with loss components.
        """
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]

        # Reconstruction loss
        px = NegativeBinomial(mu=px_rate, theta=px_r)
        reconst_loss = -px.log_prob(x).sum(dim=-1)

        # KL divergence for proportions
        v = inference_outputs["v"]
        v_prior = torch.ones_like(v) / v.shape[-1]
        kl_v = (v * (torch.log(v + 1e-8) - torch.log(v_prior + 1e-8))).sum(dim=-1)

        # Total loss
        loss = torch.mean(reconst_loss + kl_weight * kl_v)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={"kl_v": kl_v},
        )

    @torch.inference_mode()
    def get_proportions(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get cell type proportions.

        Parameters
        ----------
        x
            Expression data.
        ind_x
            Spot indices.

        Returns
        -------
        Cell type proportions tensor.
        """
        if ind_x is None:
            ind_x = torch.arange(x.shape[0], device=x.device)

        inference_outputs = self.inference(x, ind_x)
        return inference_outputs["v"][:, : self.n_labels]


class DestVIModule(MRDeconv):
    """Alias for MRDeconv module for DestVI compatibility."""

    pass
