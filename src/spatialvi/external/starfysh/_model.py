"""Starfysh model for spatial deconvolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Starfysh:
    """Starfysh model for spatial deconvolution with histology.

    Starfysh deconvolves spatial transcriptomics spots into cell type
    proportions using reference single-cell data. It can optionally
    integrate histology images for improved accuracy.

    Parameters
    ----------
    adata_spatial
        Spatial transcriptomics AnnData.
    adata_ref
        Reference single-cell AnnData with cell type annotations.
    cell_type_key
        Key in adata_ref.obs for cell type labels.
    spatial_key
        Key in adata_spatial.obsm for spatial coordinates.
    n_factors
        Number of archetypal factors per cell type.
    n_hidden
        Number of hidden units.
    use_histology
        Whether to integrate histology features.

    Examples
    --------
    >>> import spatialvi
    >>> adata_sp = spatialvi.data.synthetic_spatial()
    >>> adata_sc = spatialvi.data.synthetic_scrna()
    >>> model = Starfysh(adata_sp, adata_sc, cell_type_key="cell_type")
    >>> model.fit()
    >>> proportions = model.get_proportions()
    """

    def __init__(
        self,
        adata_spatial: AnnData,
        adata_ref: AnnData,
        cell_type_key: str,
        spatial_key: str = "spatial",
        n_factors: int = 5,
        n_hidden: int = 128,
        use_histology: bool = False,
    ):
        self.adata_spatial = adata_spatial
        self.adata_ref = adata_ref
        self.cell_type_key = cell_type_key
        self.spatial_key = spatial_key
        self.n_factors = n_factors
        self.n_hidden = n_hidden
        self.use_histology = use_histology

        # Get cell types
        self.cell_types = adata_ref.obs[cell_type_key].cat.categories.tolist()
        self.n_cell_types = len(self.cell_types)

        # Find common genes
        common_genes = adata_spatial.var_names.intersection(adata_ref.var_names)
        self.common_genes = common_genes
        self.n_genes = len(common_genes)

        logger.info(f"Found {self.n_genes} common genes")
        logger.info(f"Found {self.n_cell_types} cell types")

        self._fitted = False
        self._model = None
        self._proportions = None

    def _build_model(self) -> nn.Module:
        """Build the Starfysh neural network."""
        return StarfyshModule(
            n_genes=self.n_genes,
            n_cell_types=self.n_cell_types,
            n_factors=self.n_factors,
            n_hidden=self.n_hidden,
            use_histology=self.use_histology,
        )

    def _compute_reference_signatures(self) -> np.ndarray:
        """Compute reference expression signatures per cell type."""
        X_ref = self.adata_ref[:, self.common_genes].X
        if hasattr(X_ref, "toarray"):
            X_ref = X_ref.toarray()

        labels = self.adata_ref.obs[self.cell_type_key]

        signatures = np.zeros((self.n_cell_types, self.n_genes))
        for i, ct in enumerate(self.cell_types):
            mask = labels == ct
            signatures[i] = X_ref[mask].mean(axis=0)

        return signatures

    def fit(
        self,
        max_epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 256,
        early_stopping: bool = True,
        patience: int = 20,
        device: str = "auto",
    ) -> "Starfysh":
        """Fit the Starfysh model.

        Parameters
        ----------
        max_epochs
            Maximum number of training epochs.
        lr
            Learning rate.
        batch_size
            Batch size for training.
        early_stopping
            Whether to use early stopping.
        patience
            Patience for early stopping.
        device
            Device to use ("auto", "cpu", "cuda").

        Returns
        -------
        Self.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Prepare data
        X_spatial = self.adata_spatial[:, self.common_genes].X
        if hasattr(X_spatial, "toarray"):
            X_spatial = X_spatial.toarray()

        # Compute reference signatures
        signatures = self._compute_reference_signatures()

        # Create model
        self._model = self._build_model().to(device)

        # Register reference signatures as buffer
        self._model.register_buffer(
            "reference_signatures",
            torch.tensor(signatures, dtype=torch.float32),
        )

        # Create data loader
        X_tensor = torch.tensor(X_spatial, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = optim.Adam(self._model.parameters(), lr=lr)

        # Training loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            epoch_loss = 0.0

            for (batch_x,) in dataloader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()

                # Forward pass
                proportions, reconstruction = self._model(batch_x)

                # Loss
                loss = self._compute_loss(batch_x, proportions, reconstruction)

                # Backward
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)

            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: loss = {epoch_loss:.4f}")

            # Early stopping
            if early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        self._fitted = True

        # Compute final proportions
        self._model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(device)
            proportions, _ = self._model(X_tensor)
            self._proportions = proportions.cpu().numpy()

        return self

    def _compute_loss(
        self,
        x: torch.Tensor,
        proportions: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss."""
        # Reconstruction loss (MSE on log-transformed data)
        x_log = torch.log1p(x)
        recon_log = torch.log1p(reconstruction)
        recon_loss = nn.functional.mse_loss(recon_log, x_log)

        # Sparsity regularization on proportions
        sparsity_loss = 0.01 * torch.mean(proportions * torch.log(proportions + 1e-8))

        # Entropy regularization (encourage diverse proportions)
        entropy_loss = -0.01 * torch.mean(
            torch.sum(proportions * torch.log(proportions + 1e-8), dim=-1)
        )

        return recon_loss + sparsity_loss + entropy_loss

    def get_proportions(
        self,
        return_dataframe: bool = True,
    ) -> NDArray | pd.DataFrame:
        """Get estimated cell type proportions.

        Parameters
        ----------
        return_dataframe
            Whether to return as DataFrame.

        Returns
        -------
        Cell type proportions.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if return_dataframe:
            return pd.DataFrame(
                self._proportions,
                index=self.adata_spatial.obs_names,
                columns=self.cell_types,
            )

        return self._proportions

    def get_reconstruction(self) -> NDArray:
        """Get reconstructed expression.

        Returns
        -------
        Reconstructed expression array.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = next(self._model.parameters()).device

        X_spatial = self.adata_spatial[:, self.common_genes].X
        if hasattr(X_spatial, "toarray"):
            X_spatial = X_spatial.toarray()

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_spatial, dtype=torch.float32, device=device)
            _, reconstruction = self._model(X_tensor)
            return reconstruction.cpu().numpy()

    def to_adata(self) -> None:
        """Store results in spatial AnnData.

        Adds proportions to adata_spatial.obsm["starfysh_proportions"].
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.adata_spatial.obsm["starfysh_proportions"] = self._proportions


class StarfyshModule(nn.Module):
    """Neural network module for Starfysh."""

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_factors: int = 5,
        n_hidden: int = 128,
        use_histology: bool = False,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_factors = n_factors

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )

        # Proportion predictor
        self.proportion_head = nn.Sequential(
            nn.Linear(n_hidden, n_cell_types),
            nn.Softmax(dim=-1),
        )

        # Cell type-specific factor loadings
        self.factor_loadings = nn.Parameter(
            torch.randn(n_cell_types, n_factors, n_genes)
        )

        # Factor weights per spot
        self.factor_encoder = nn.Linear(n_hidden, n_cell_types * n_factors)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x
            Input expression of shape (batch, n_genes).

        Returns
        -------
        Tuple of (proportions, reconstruction).
        """
        # Encode
        h = self.encoder(torch.log1p(x))

        # Predict proportions
        proportions = self.proportion_head(h)

        # Get factor weights
        factor_weights = self.factor_encoder(h)
        factor_weights = factor_weights.view(-1, self.n_cell_types, self.n_factors)
        factor_weights = torch.softmax(factor_weights, dim=-1)

        # Reconstruct using weighted factors
        # factor_loadings: (n_cell_types, n_factors, n_genes)
        # factor_weights: (batch, n_cell_types, n_factors)
        # proportions: (batch, n_cell_types)

        # Weighted factor expression per cell type
        ct_expr = torch.einsum(
            "bcf,cfg->bcg",
            factor_weights,
            torch.softmax(self.factor_loadings, dim=-1),
        )  # (batch, n_cell_types, n_genes)

        # Mix by proportions
        reconstruction = torch.einsum("bc,bcg->bg", proportions, ct_expr)

        # Scale to match input
        scale = x.sum(dim=-1, keepdim=True) / (reconstruction.sum(dim=-1, keepdim=True) + 1e-8)
        reconstruction = reconstruction * scale

        return proportions, reconstruction
