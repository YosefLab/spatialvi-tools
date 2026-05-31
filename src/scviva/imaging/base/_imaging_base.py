"""ImagingBaseModel and image loading utilities."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scviva.model.base._spatial_base import SpatialBaseModel

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


def _load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an image file to a float32 tensor of shape (C, H, W) in [0, 1]."""
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix in ("tif", "tiff"):
        try:
            import tifffile
        except ImportError as e:
            raise ImportError(
                "tifffile is required for TIFF images. "
                "Install with: pip install 'scviva-tools[imaging]'"
            ) from e
        arr = tifffile.imread(path).astype(np.float32)
    else:
        from PIL import Image as PILImage

        arr = np.array(PILImage.open(path), dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)

    max_val = arr.max()
    if max_val > 1.0:
        arr = arr / (255.0 if max_val <= 255.0 else max_val)

    return torch.from_numpy(arr)


class _ImagePathDataset(Dataset):
    """Minimal dataset loading per-cell image crops from disk paths."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _load_image_as_tensor(self.paths[idx])


class ImagingBaseModel(SpatialBaseModel):
    """Base class for image-based spatial models.

    Provides pre-trained backbone loading, AnnData/SpatialData integration,
    and inference via get_latent_representation(). Subclasses implement
    _build_module() and setup_anndata().
    """

    _default_obsm_key: str = "X_imaging"

    def __init__(self, *args, **kwargs) -> None:
        if args or kwargs:
            raise TypeError(
                f"Use {type(self).__name__}.from_pretrained() to load a model, "
                "not the constructor directly."
            )
        # BaseModelClass.__init__ rejects adata=None unless the class name is
        # hard-coded in scvi internals. We initialise the required attributes
        # directly to avoid that restriction.
        self.id = str(uuid4())
        self._adata = None
        self._adata_manager = None
        self._module_init_on_train = True
        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None
        self.get_normalized_function_name_ = "get_normalized_expression"
        self.run_name_ = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.run_id_ = ""
        self.module: nn.Module | None = None
        self.registry_ = {}
        self.summary_stats = {}

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is a pre-trained model — call from_pretrained() "
            "to load weights. Training is not supported via this interface."
        )

    @classmethod
    def _build_module(cls, checkpoint_data: dict) -> nn.Module:
        raise NotImplementedError(f"{cls.__name__} must implement _build_module(checkpoint_data)")

    @classmethod
    def _resolve_path(cls, model_name_or_path: str) -> str:
        """Return local filesystem path; download from HuggingFace if needed."""
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        # Heuristic: HF repo IDs look like "user/model-name" with "/" but no os.sep
        if not os.path.isabs(model_name_or_path) and "/" in model_name_or_path:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "huggingface-hub is required to load from HuggingFace. "
                    "Install with: pip install 'scviva-tools[imaging]'"
                ) from e
            logger.info(f"Downloading {model_name_or_path} from HuggingFace Hub...")
            return hf_hub_download(repo_id=model_name_or_path, filename="model.pt")
        raise FileNotFoundError(
            f"Could not resolve model path: {model_name_or_path!r}. "
            "Provide a local path or a HuggingFace repo ID (e.g. 'YosefLab/sparl-imc-vitb16')."
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> ImagingBaseModel:
        """Load a pre-trained backbone from a local checkpoint or HuggingFace Hub.

        Parameters
        ----------
        model_name_or_path
            Local path to a ``.pt`` checkpoint file, or a HuggingFace repo ID
            (e.g. ``"YosefLab/sparl-imc-vitb16"``).

        Returns
        -------
        Loaded model instance with ``self.module`` set.
        """
        path = cls._resolve_path(model_name_or_path)
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls()
        try:
            obj.module = cls._build_module(checkpoint_data)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build module from checkpoint at {path!r}. "
                "Check that the checkpoint format matches the model class."
            ) from exc
        obj.module.eval()
        return obj

    @classmethod
    def from_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **kwargs,
    ) -> AnnData:
        """Register SpatialData fields and return the extracted AnnData.

        Unlike other scviva models, imaging models are loaded via
        :meth:`from_pretrained`. This method only registers data fields
        and returns the AnnData for passing to :meth:`get_latent_representation`.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the AnnData table.
        region
            Region name to subset.
        **kwargs
            Passed to :meth:`setup_anndata`.

        Returns
        -------
        The extracted and registered AnnData object.
        """
        cls.setup_spatialdata(sdata, table_key=table_key, region=region, **kwargs)
        adata = sdata[table_key]
        if region is not None:
            region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")
            adata = adata[adata.obs[region_key] == region].copy()
        return adata

    def _build_inference_dataloader(self, adata: AnnData, batch_size: int) -> DataLoader:
        img_path_col = adata.uns["scviva_imaging"]["img_path_col"]
        paths = adata.obs[img_path_col].tolist()
        dataset = _ImagePathDataset(paths)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_latent_representation(
        self,
        adata: AnnData,
        batch_size: int = 256,
        obsm_key: str | None = None,
        device: str = "cpu",
        backend: str = "cpu",
    ) -> np.ndarray:
        """Run inference and write CLS-token embeddings to ``adata.obsm[obsm_key]``.

        Parameters
        ----------
        adata
            AnnData registered via :meth:`setup_anndata` or :meth:`from_spatialdata`.
        batch_size
            Number of cells per forward pass.
        obsm_key
            Key to write embeddings into. Defaults to ``cls._default_obsm_key``.
        device
            Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        backend
            ``"cpu"`` returns numpy array. ``"rapids"`` returns cupy array.

        Returns
        -------
        Embedding array of shape ``(n_cells, embed_dim)``.
        """
        if self.module is None:
            raise RuntimeError(
                "No backbone loaded. Call from_pretrained() before get_latent_representation()."
            )
        if obsm_key is None:
            obsm_key = self._default_obsm_key

        self.module.eval()
        self.module.to(device)

        loader = self._build_inference_dataloader(adata, batch_size)
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                emb = self.module(batch)
                all_embeddings.append(emb.cpu().numpy())

        result = np.concatenate(all_embeddings, axis=0)
        adata.obsm[obsm_key] = result

        if backend == "rapids":
            try:
                import cupy as cp

                return cp.asarray(result)
            except ImportError as e:
                raise ImportError(
                    "backend='rapids' requires cupy. "
                    "Install with: pip install 'scviva-tools[rapids]'"
                ) from e

        return result
