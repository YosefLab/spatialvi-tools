"""Training callbacks for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

logger = logging.getLogger(__name__)


class SpatialMetricsCallback(Callback):
    """Callback for computing spatial metrics during training.

    Parameters
    ----------
    compute_every_n_epochs
        Frequency of metric computation.
    metrics
        List of metrics to compute.
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 10,
        metrics: list[str] | None = None,
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.metrics = metrics or ["spatial_loss"]

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute metrics at epoch end."""
        if (trainer.current_epoch + 1) % self.compute_every_n_epochs != 0:
            return

        if "spatial_loss" in self.metrics:
            # Log spatial coherence
            if hasattr(pl_module, "module") and hasattr(pl_module.module, "spatial_weight"):
                spatial_weight = pl_module.module.spatial_weight
                pl_module.log("spatial_weight_current", spatial_weight)


class NeighborSamplingCallback(Callback):
    """Callback for neighbor-aware batch sampling.

    This callback modifies the data loader to include
    neighbor information in each batch.

    Parameters
    ----------
    neighbor_key
        Key in obsm for neighbor indices.
    include_neighbor_expr
        Whether to include neighbor expression.
    """

    def __init__(
        self,
        neighbor_key: str = "nn_index",
        include_neighbor_expr: bool = True,
    ):
        super().__init__()
        self.neighbor_key = neighbor_key
        self.include_neighbor_expr = include_neighbor_expr

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Augment batch with neighbor information."""
        # This is a placeholder - actual implementation depends on
        # how the dataloader is structured
        pass


class EarlyStoppingOnSpatialLoss(Callback):
    """Early stopping based on spatial loss convergence.

    Parameters
    ----------
    patience
        Number of epochs to wait for improvement.
    min_delta
        Minimum change to qualify as improvement.
    monitor
        Metric to monitor.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        monitor: str = "spatial_loss",
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float("inf")
        self.wait_count = 0

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Check for improvement."""
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        current = current.item() if hasattr(current, "item") else current

        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered after {trainer.current_epoch} epochs")
                trainer.should_stop = True


class SpatialRegularizationScheduler(Callback):
    """Scheduler for spatial regularization weight.

    Parameters
    ----------
    initial_weight
        Starting weight.
    final_weight
        Final weight.
    warmup_epochs
        Number of epochs to warmup.
    schedule
        Schedule type: "linear", "cosine", "step".
    """

    def __init__(
        self,
        initial_weight: float = 0.0,
        final_weight: float = 1.0,
        warmup_epochs: int = 50,
        schedule: str = "linear",
    ):
        super().__init__()
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule

    def _compute_weight(self, epoch: int) -> float:
        """Compute weight for given epoch."""
        if epoch >= self.warmup_epochs:
            return self.final_weight

        progress = epoch / self.warmup_epochs

        if self.schedule == "linear":
            return self.initial_weight + progress * (self.final_weight - self.initial_weight)
        elif self.schedule == "cosine":
            return self.final_weight - (self.final_weight - self.initial_weight) * (1 + np.cos(np.pi * progress)) / 2
        elif self.schedule == "step":
            # Step at halfway
            if progress < 0.5:
                return self.initial_weight
            else:
                return self.final_weight
        else:
            return self.final_weight

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Update spatial weight."""
        weight = self._compute_weight(trainer.current_epoch)

        if hasattr(pl_module, "module") and hasattr(pl_module.module, "spatial_weight"):
            pl_module.module.spatial_weight = weight

        pl_module.log("scheduled_spatial_weight", weight)
