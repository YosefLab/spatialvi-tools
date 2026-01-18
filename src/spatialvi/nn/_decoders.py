"""Decoder architectures for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from scvi.nn import FCLayers
from torch import nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpatialDecoder(nn.Module):
    """Decoder that can incorporate spatial context for reconstruction.

    Parameters
    ----------
    n_input
        Dimensionality of the latent space (input to decoder).
    n_output
        Number of output features (genes).
    n_cat_list
        List of number of categories for categorical variables.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of nodes per hidden layer.
    dropout_rate
        Dropout rate.
    use_batch_norm
        Whether to use batch normalization.
    use_layer_norm
        Whether to use layer normalization.
    use_spatial_context
        Whether to use spatial context in decoding.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_spatial_context: bool = False,
    ):
        super().__init__()
        self.use_spatial_context = use_spatial_context

        # Main decoder
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # Spatial context integration
        if use_spatial_context:
            self.spatial_integration = nn.Sequential(
                nn.Linear(n_hidden * 2, n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

        # Output layers
        self.px_scale_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        self.px_r_layer = nn.Linear(n_hidden, n_output)
        self.px_dropout_layer = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: torch.Tensor,
        neighbor_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z
            Latent representation.
        library
            Library size.
        cat_list
            Categorical covariates.
        neighbor_z
            Latent representations of neighbors (for spatial context).

        Returns
        -------
        Tuple of (px_scale, px_r, px_rate, px_dropout).
        """
        h = self.decoder(z, *cat_list)

        # Integrate spatial context if available
        if self.use_spatial_context and neighbor_z is not None:
            # neighbor_z: (batch, n_neighbors, n_latent)
            neighbor_h = self.decoder(neighbor_z.mean(dim=1), *cat_list)
            h = self.spatial_integration(torch.cat([h, neighbor_h], dim=-1))

        # Output
        px_scale = self.px_scale_layer(h)
        px_rate = library * px_scale
        px_r = self.px_r_layer(h)
        px_dropout = self.px_dropout_layer(h)

        return px_scale, px_r, px_rate, px_dropout


class ConditionalDecoder(nn.Module):
    """Decoder conditioned on additional context (e.g., cell type).

    Parameters
    ----------
    n_input
        Dimensionality of the latent space.
    n_output
        Number of output features.
    n_conditions
        Number of conditioning variables (e.g., cell types).
    n_hidden
        Number of hidden nodes.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_conditions: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Condition embedding
        self.condition_embedding = nn.Embedding(n_conditions, n_hidden)

        # FiLM layers (Feature-wise Linear Modulation)
        self.film_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.film_layers.append(
                nn.Linear(n_hidden, n_hidden * 2)  # gamma and beta
            )

        # Main decoder
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))

        self.output_layer = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        z
            Latent representation.
        condition
            Condition indices.

        Returns
        -------
        Decoded output.
        """
        # Get condition embedding
        cond_emb = self.condition_embedding(condition.long().squeeze(-1))

        # Apply FiLM-modulated layers
        h = z
        for i, layer in enumerate(self.layers):
            h = layer(h)

            # FiLM modulation
            film_params = self.film_layers[i](cond_emb)
            gamma, beta = film_params.chunk(2, dim=-1)
            h = gamma * h + beta

            h = torch.relu(h)
            h = self.dropout(h)

        return self.output_layer(h)


class FactorizedDecoder(nn.Module):
    """Decoder with factorized output for cell type deconvolution.

    Parameters
    ----------
    n_input
        Dimensionality of the latent space.
    n_output
        Number of output features (genes).
    n_factors
        Number of factors (cell types).
    n_hidden
        Number of hidden nodes.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_factors: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_factors = n_factors

        # Factor-specific decoders
        self.factor_decoders = nn.ModuleList(
            [
                FCLayers(
                    n_in=n_input,
                    n_out=n_output,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_factors)
            ]
        )

        # Mixing weights decoder
        self.mixing_decoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, n_factors),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z
            Latent representation.

        Returns
        -------
        Tuple of (factor_outputs, mixing_weights).
        """
        # Get factor-specific outputs
        factor_outputs = []
        for decoder in self.factor_decoders:
            factor_outputs.append(decoder(z))
        factor_outputs = torch.stack(factor_outputs, dim=1)  # (batch, n_factors, n_output)

        # Get mixing weights
        mixing_weights = self.mixing_decoder(z)  # (batch, n_factors)

        return factor_outputs, mixing_weights

    def get_mixed_output(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Get the weighted mixture of factor outputs.

        Parameters
        ----------
        z
            Latent representation.

        Returns
        -------
        Mixed output of shape (batch, n_output).
        """
        factor_outputs, mixing_weights = self.forward(z)
        # factor_outputs: (batch, n_factors, n_output)
        # mixing_weights: (batch, n_factors)
        mixed = torch.einsum("bfg,bf->bg", factor_outputs, mixing_weights)
        return mixed
