"""Neural network components for scVIVA module."""

from collections.abc import Iterable

import torch
from scvi.nn import Decoder, FCLayers
from torch import nn
from torch.distributions import Dirichlet, Normal


def _identity(x):
    """Identity function."""
    return x


class Encoder(nn.Module):
    """Encode data into a latent space.

    Uses a fully-connected neural network.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space).
    n_output
        The dimensionality of the output (latent space).
    n_cat_list
        A list containing the number of categories for each categorical covariate.
    n_layers
        The number of fully-connected hidden layers.
    n_hidden
        The number of nodes per hidden layer.
    dropout_rate
        Dropout rate to apply to each of the hidden layers.
    distribution
        Distribution of z (either "normal" or "ln" for logistic normal).
    var_eps
        Minimum value for the variance; used for numerical stability.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.dist_encoder = nn.Linear(n_hidden, 2 * n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward pass.

        Parameters
        ----------
        x
            Tensor with shape (n_input,).
        cat_list
            List of category membership(s) for this sample.

        Returns
        -------
        If return_dist is True, returns (distribution, latent_sample).
        Otherwise returns (mean, variance, latent_sample).
        """
        q = self.encoder(x, *cat_list)
        q_m, q_v = self.dist_encoder(q).chunk(2, dim=-1)
        q_v = torch.nn.Softplus()(q_v) + self.var_eps

        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


class DirichletDecoder(Decoder):
    """Predict the cell type composition from the latent space.

    Returns a Dirichlet distribution over cell type compositions.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space dimension).
    n_output
        The dimensionality of the output (number of cell types).
    n_cat_list
        A list containing the number of categories for each categorical covariate.
    n_layers
        The number of fully-connected hidden layers.
    n_hidden
        The number of nodes per hidden layer.
    concentration_eps
        Minimum value for the concentration; used for numerical stability.
    **kwargs
        Keyword args for :class:`~scvi.nn.Decoder`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        concentration_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            **kwargs,
        )
        self.concentration_eps = concentration_eps

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward pass.

        Parameters
        ----------
        x
            Tensor with shape (n_input,).
        cat_list
            List of category membership(s) for this sample.

        Returns
        -------
        Dirichlet distribution over compositions.
        """
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_m = torch.nn.Softplus()(p_m) + self.concentration_eps
        dist = Dirichlet(p_m)
        return dist


class NicheDecoder(nn.Module):
    """Decode data from latent space to niche embedding space.

    Outputs the mean and variance of a multivariate Gaussian
    for each cell type in the neighborhood.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space).
    n_output
        The dimensionality of the output (data space).
    n_niche_components
        Number of niche components (cell types).
    n_cat_list
        A list containing the number of categories for each categorical covariate.
    n_layers
        The number of fully-connected hidden layers.
    n_hidden
        The number of nodes per hidden layer.
    dropout_rate
        Dropout rate to apply to each of the hidden layers.
    var_eps
        Minimum value for the variance; used for numerical stability.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_niche_components: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.n_niche_components = n_niche_components
        self.n_output = n_output
        self.var_eps = var_eps

        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.dist_decoder = nn.Linear(n_hidden, 2 * n_output * n_niche_components)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward pass.

        Parameters
        ----------
        x
            Tensor with shape (n_input,).
        cat_list
            List of category membership(s) for this sample.

        Returns
        -------
        Tuple of (mean, variance) tensors with shape
        (batch_size, n_niche_components, n_output).
        """
        p = self.decoder(x, *cat_list)
        p_m, p_v = self.dist_decoder(p).chunk(2, dim=-1)
        p_v = torch.nn.Softplus()(p_v) + self.var_eps

        if p.ndim == 2:
            p_m = p_m.view(p_m.shape[0], self.n_niche_components, self.n_output)
            p_v = p_v.view(p_v.shape[0], self.n_niche_components, self.n_output)
        elif p.ndim == 3:
            p_m = p_m.view(-1, p_m.shape[1], self.n_niche_components, self.n_output)
            p_v = p_v.view(-1, p_v.shape[1], self.n_niche_components, self.n_output)

        return p_m, p_v
