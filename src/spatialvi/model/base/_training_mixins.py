"""Training mixins for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from lightning.pytorch.callbacks import Callback
from scvi.train import TrainRunner

if TYPE_CHECKING:
    from anndata import AnnData
    from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class SpatialTrainingMixin:
    """Mixin for training spatial models.

    This mixin provides spatial-aware training methods including:
    - Spatial neighbor sampling strategies
    - Spatial regularization callbacks
    - Multi-resolution training schemes
    """

    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        early_stopping_patience: int = 45,
        early_stopping_min_delta: float = 0.0,
        plan_kwargs: dict | None = None,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        datasplitter_kwargs: dict | None = None,
        **trainer_kwargs,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        max_epochs
            Maximum number of epochs to train. If None, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`.
        accelerator
            Supports passing different accelerator types ("cpu", "gpu", "tpu", "auto").
        devices
            The device(s) to use.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of validation set. If None, defaults to 1 - train_size.
        shuffle_set_split
            Whether to shuffle indices before splitting.
        load_sparse_tensor
            Whether to load data as sparse tensors.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping.
        early_stopping_patience
            Number of epochs to wait for improvement before stopping.
        early_stopping_min_delta
            Minimum change in loss to qualify as improvement.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`.
        check_val_every_n_epoch
            Check val every n train epochs.
        n_steps_kl_warmup
            Number of training steps to scale weight on KL divergences.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences.
        datasplitter_kwargs
            Additional keyword arguments for data splitting.
        **trainer_kwargs
            Additional keyword arguments for the Trainer.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if plan_kwargs is not None else {}
        datasplitter_kwargs = datasplitter_kwargs if datasplitter_kwargs is not None else {}

        if n_steps_kl_warmup is not None:
            plan_kwargs["n_steps_kl_warmup"] = n_steps_kl_warmup
        if n_epochs_kl_warmup is not None:
            plan_kwargs["n_epochs_kl_warmup"] = n_epochs_kl_warmup

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            load_sparse_tensor=load_sparse_tensor,
            **datasplitter_kwargs,
        )

        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        if early_stopping:
            callbacks = trainer_kwargs.get("callbacks", [])
            if not isinstance(callbacks, list):
                callbacks = [callbacks]

            early_stopping_callback = self._get_early_stopping_callback(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
            callbacks.append(early_stopping_callback)
            trainer_kwargs["callbacks"] = callbacks

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **trainer_kwargs,
        )
        return runner()

    def _get_early_stopping_callback(
        self,
        patience: int = 45,
        min_delta: float = 0.0,
        monitor: str = "elbo_validation",
    ) -> Callback:
        """Get early stopping callback.

        Parameters
        ----------
        patience
            Number of epochs to wait.
        min_delta
            Minimum change in monitored quantity.
        monitor
            Metric to monitor.

        Returns
        -------
        Early stopping callback.
        """
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping

        return EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode="min",
            verbose=False,
        )


class SpatialSamplerMixin:
    """Mixin for spatial-aware sampling during training.

    This mixin provides methods for:
    - Spatially contiguous batch sampling
    - Multi-scale spatial sampling
    - Neighbor-aware batching
    """

    def _get_spatial_sampler(
        self,
        adata: AnnData,
        batch_size: int,
        mode: Literal["random", "spatial", "niche"] = "random",
    ):
        """Get a spatial-aware sampler.

        Parameters
        ----------
        adata
            AnnData object with spatial information.
        batch_size
            Batch size for sampling.
        mode
            Sampling mode. One of:
            - "random": Standard random sampling
            - "spatial": Spatially contiguous sampling
            - "niche": Sample complete neighborhoods

        Returns
        -------
        Data sampler.
        """
        if mode == "random":
            from torch.utils.data import RandomSampler

            return RandomSampler(range(adata.n_obs))

        elif mode == "spatial":
            return SpatialBatchSampler(
                adata=adata,
                batch_size=batch_size,
                spatial_key="spatial",
            )

        elif mode == "niche":
            return NicheBatchSampler(
                adata=adata,
                batch_size=batch_size,
                neighbor_key="nn_index",
            )

        else:
            raise ValueError(f"Unknown sampling mode: {mode}")


class SpatialBatchSampler:
    """Batch sampler that samples spatially contiguous regions.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    batch_size
        Number of cells per batch.
    spatial_key
        Key in obsm for spatial coordinates.
    n_clusters
        Number of spatial clusters to use for batching.
    """

    def __init__(
        self,
        adata: AnnData,
        batch_size: int,
        spatial_key: str = "spatial",
        n_clusters: int | None = None,
    ):
        self.adata = adata
        self.batch_size = batch_size
        self.spatial_key = spatial_key
        self.n_clusters = n_clusters or max(1, adata.n_obs // batch_size)

        # Precompute spatial clusters
        self._compute_spatial_clusters()

    def _compute_spatial_clusters(self) -> None:
        """Compute spatial clusters for batching."""
        from sklearn.cluster import MiniBatchKMeans

        coords = self.adata.obsm[self.spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0)
        self.cluster_labels = kmeans.fit_predict(coords)

    def __iter__(self):
        """Iterate over batches."""
        # Shuffle clusters
        cluster_order = np.random.permutation(self.n_clusters)

        for cluster_id in cluster_order:
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            np.random.shuffle(cluster_indices)

            for i in range(0, len(cluster_indices), self.batch_size):
                yield cluster_indices[i : i + self.batch_size].tolist()

    def __len__(self) -> int:
        """Return number of batches."""
        return (self.adata.n_obs + self.batch_size - 1) // self.batch_size


class NicheBatchSampler:
    """Batch sampler that includes complete neighborhoods.

    Parameters
    ----------
    adata
        AnnData object with neighbor information.
    batch_size
        Number of seed cells per batch.
    neighbor_key
        Key in obsm for neighbor indices.
    """

    def __init__(
        self,
        adata: AnnData,
        batch_size: int,
        neighbor_key: str = "nn_index",
    ):
        self.adata = adata
        self.batch_size = batch_size
        self.neighbor_key = neighbor_key

    def __iter__(self):
        """Iterate over batches."""
        indices = np.random.permutation(self.adata.n_obs)
        nn_indices = self.adata.obsm[self.neighbor_key]

        for i in range(0, len(indices), self.batch_size):
            seed_cells = indices[i : i + self.batch_size]

            # Include all neighbors
            batch_indices = set(seed_cells.tolist())
            for idx in seed_cells:
                batch_indices.update(nn_indices[idx].tolist())

            yield list(batch_indices)

    def __len__(self) -> int:
        """Return number of batches."""
        return (self.adata.n_obs + self.batch_size - 1) // self.batch_size


class SpatialRegularizationCallback(Callback):
    """Callback for spatial regularization during training.

    Parameters
    ----------
    spatial_weight
        Weight for spatial regularization term.
    spatial_key
        Key in obsm for spatial coordinates.
    warmup_epochs
        Number of epochs to warmup spatial regularization.
    """

    def __init__(
        self,
        spatial_weight: float = 1.0,
        spatial_key: str = "spatial",
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.spatial_key = spatial_key
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Update spatial weight based on epoch."""
        current_epoch = trainer.current_epoch
        if current_epoch < self.warmup_epochs:
            weight = self.spatial_weight * (current_epoch / self.warmup_epochs)
        else:
            weight = self.spatial_weight

        if hasattr(pl_module.module, "spatial_weight"):
            pl_module.module.spatial_weight = weight
