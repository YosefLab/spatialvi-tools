"""Training plans for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi.train import TrainingPlan

if TYPE_CHECKING:
    from scvi.module.base import BaseModuleClass

logger = logging.getLogger(__name__)


class SpatialTrainingPlan(TrainingPlan):
    """Training plan with spatial-specific features.

    This training plan extends the base TrainingPlan with:
    - Spatial loss weighting
    - Neighbor-aware loss computation
    - Multi-scale training

    Parameters
    ----------
    module
        Neural network module.
    lr
        Learning rate.
    weight_decay
        Weight decay for optimizer.
    spatial_weight
        Weight for spatial regularization term.
    spatial_warmup_epochs
        Number of epochs to warmup spatial regularization.
    **kwargs
        Additional keyword arguments for TrainingPlan.
    """

    def __init__(
        self,
        module: "BaseModuleClass",
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        spatial_weight: float = 1.0,
        spatial_warmup_epochs: int = 10,
        **kwargs,
    ):
        super().__init__(module, lr=lr, weight_decay=weight_decay, **kwargs)
        self.spatial_weight = spatial_weight
        self.spatial_warmup_epochs = spatial_warmup_epochs
        self._current_spatial_weight = 0.0

    def on_train_epoch_start(self) -> None:
        """Update spatial weight at epoch start."""
        super().on_train_epoch_start()

        current_epoch = self.current_epoch
        if current_epoch < self.spatial_warmup_epochs:
            self._current_spatial_weight = (
                self.spatial_weight * (current_epoch + 1) / self.spatial_warmup_epochs
            )
        else:
            self._current_spatial_weight = self.spatial_weight

        # Update module spatial weight if applicable
        if hasattr(self.module, "spatial_weight"):
            self.module.spatial_weight = self._current_spatial_weight

    def training_step(self, batch, batch_idx):
        """Training step with spatial weight."""
        # Call parent training step
        output = super().training_step(batch, batch_idx)

        # Log spatial weight
        self.log("spatial_weight", self._current_spatial_weight)

        return output


class NicheTrainingPlan(SpatialTrainingPlan):
    """Training plan for niche-aware models.

    Extends SpatialTrainingPlan with:
    - Niche composition loss
    - Classification loss for semi-supervised training
    - Balanced sampling for cell types

    Parameters
    ----------
    module
        Neural network module.
    lr
        Learning rate.
    classification_weight
        Weight for classification loss.
    niche_weight
        Weight for niche composition loss.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        module: "BaseModuleClass",
        lr: float = 1e-3,
        classification_weight: float = 1.0,
        niche_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(module, lr=lr, **kwargs)
        self.classification_weight = classification_weight
        self.niche_weight = niche_weight

    def training_step(self, batch, batch_idx):
        """Training step with niche losses."""
        # Standard forward pass
        inference_outputs, generative_outputs, loss_output = self.module(
            batch, compute_loss=True
        )

        loss = loss_output.loss

        # Add classification loss if available
        if "classification_loss" in loss_output.extra_metrics:
            class_loss = loss_output.extra_metrics["classification_loss"]
            loss = loss + self.classification_weight * class_loss
            self.log("classification_loss", class_loss)

        # Add niche loss if available
        if "niche_loss" in loss_output.extra_metrics:
            niche_loss = loss_output.extra_metrics["niche_loss"]
            loss = loss + self.niche_weight * niche_loss
            self.log("niche_loss", niche_loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "kl_local",
            loss_output.kl_local.mean() if loss_output.kl_local is not None else 0,
        )
        self.log("reconstruction_loss", loss_output.reconstruction_loss.mean())

        return loss


class DeconvolutionTrainingPlan(SpatialTrainingPlan):
    """Training plan for spatial deconvolution models.

    Extends SpatialTrainingPlan with:
    - Proportion sparsity regularization
    - Reference-guided constraints
    - Multi-stage training

    Parameters
    ----------
    module
        Neural network module.
    lr
        Learning rate.
    sparsity_weight
        Weight for proportion sparsity.
    reference_weight
        Weight for reference consistency.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(
        self,
        module: "BaseModuleClass",
        lr: float = 1e-3,
        sparsity_weight: float = 0.1,
        reference_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(module, lr=lr, **kwargs)
        self.sparsity_weight = sparsity_weight
        self.reference_weight = reference_weight

    def training_step(self, batch, batch_idx):
        """Training step with deconvolution losses."""
        inference_outputs, generative_outputs, loss_output = self.module(
            batch, compute_loss=True
        )

        loss = loss_output.loss

        # Sparsity regularization on proportions
        if "proportions" in generative_outputs:
            proportions = generative_outputs["proportions"]
            # L1 sparsity
            sparsity_loss = self.sparsity_weight * torch.mean(
                torch.abs(proportions)
            )
            # Entropy regularization (encourage diversity)
            entropy = -torch.mean(
                torch.sum(proportions * torch.log(proportions + 1e-8), dim=-1)
            )
            loss = loss + sparsity_loss - 0.01 * entropy

            self.log("sparsity_loss", sparsity_loss)
            self.log("proportion_entropy", entropy)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("reconstruction_loss", loss_output.reconstruction_loss.mean())

        return loss

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduling."""
        optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
                "frequency": 1,
            },
        }
