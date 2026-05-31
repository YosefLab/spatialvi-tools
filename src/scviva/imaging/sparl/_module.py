"""SPARLModule: thin nn.Module wrapper around SPARL's DinoVisionTransformer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from sparl.models.backbones.vision_transformer import DinoVisionTransformer


class SPARLModule(nn.Module):
    """Wraps a SPARL DinoVisionTransformer for scviva-tools inference.

    Handles the ``channels`` tensor required by DinoVisionTransformer internally,
    so callers only need to pass the image batch ``(B, C, H, W)``.
    """

    def __init__(self, backbone: DinoVisionTransformer) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS-token embeddings for a batch of images.

        Parameters
        ----------
        x
            Image batch ``(B, C, H, W)``, float32.

        Returns
        -------
        CLS-token embeddings ``(B, embed_dim)``.
        """
        B = x.shape[0]
        channels = self.backbone.channel_names.unsqueeze(0).expand(B, -1).to(x.device)
        result = self.backbone(x, channels)
        return result["x_norm_clstoken"]
