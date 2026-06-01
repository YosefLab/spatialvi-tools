"""SPARL model for scviva-tools — inference-only wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scviva.imaging.base._imaging_base import ImagingBaseModel
from scviva.imaging.sparl._module import SPARLModule

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class SPARL(ImagingBaseModel):
    """Spatial Proteomics Analysis with Representation Learning.

    Loads a pre-trained ViT backbone (trained with DINO/iBOT self-supervised
    learning on multi-channel microscopy images) and runs inference to produce
    per-cell CLS-token embeddings stored in ``adata.obsm["X_sparl"]``.

    Examples
    --------
    >>> model = SPARL.from_pretrained("YosefLab/sparl-imc-vitb16")
    >>> SPARL.setup_anndata(adata, img_path_col="crop_path", channel_names=["CD3", "CD8", "DAPI"])
    >>> model.get_latent_representation(adata)
    >>> # adata.obsm["X_sparl"] now contains per-cell embeddings
    """

    _default_obsm_key: str = "X_sparl"

    @classmethod
    def _build_module(cls, checkpoint_data: dict) -> SPARLModule:
        """Build SPARLModule from a SPARL training checkpoint."""
        try:
            from sparl.models.backbones.vision_transformer import DinoVisionTransformer
        except ImportError as e:
            raise ImportError(
                "The sparl package is required for SPARL inference. "
                "Install it from source: pip install /path/to/SPARL"
            ) from e

        config = dict(checkpoint_data["model_config"])
        config.pop("arch", None)
        logger.info(f"Building SPARL backbone: embed_dim={config.get('embed_dim')}")

        backbone = DinoVisionTransformer(**config)
        backbone_state = checkpoint_data["model"]["teacher"]["backbone"]
        backbone.load_state_dict(backbone_state)

        return SPARLModule(backbone)

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        img_path_col: str,
        channel_names: list[str | int] | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Register AnnData fields for SPARL inference.

        Parameters
        ----------
        adata
            AnnData with per-cell image paths in ``obs[img_path_col]``
            and spatial coordinates in ``obsm[spatial_key]``.
        img_path_col
            Column in ``adata.obs`` containing per-cell image file paths.
        channel_names
            Ordered list of channel names/IDs matching the backbone's training
            channels. Stored in ``adata.uns["sparl_channels"]``.
        spatial_key
            Key in ``adata.obsm`` for 2D spatial coordinates.
        """
        from scvi.data import AnnDataManager

        from scviva.data._fields import SpatialCoordsField

        fields = [SpatialCoordsField(obsm_key=spatial_key)]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)

        adata.uns["scviva_imaging"] = {"img_path_col": img_path_col}
        if channel_names is not None:
            adata.uns["sparl_channels"] = list(channel_names)
