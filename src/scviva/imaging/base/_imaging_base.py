"""ImagingBaseModel and image loading utilities."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from uuid import uuid4

import torch
import torch.nn as nn

from scviva.model.base._spatial_base import SpatialBaseModel

logger = logging.getLogger(__name__)


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
        if "/" in model_name_or_path and os.sep not in model_name_or_path:
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
        obj.module = cls._build_module(checkpoint_data)
        obj.module.eval()
        return obj
