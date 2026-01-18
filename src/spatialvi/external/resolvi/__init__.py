"""ResolVI wrapper for spatial denoising.

ResolVI denoises and corrects segmentation errors in cellular-resolved
spatial transcriptomics data (Xenium, MERFISH, CosMx).

This module provides both wrapper access to the scvi-tools implementation
and the underlying module classes for advanced usage.
"""

from __future__ import annotations

from ._model import ResolVI
from ._module import RESOLVAE, RESOLVAEGuide, RESOLVAEModel
from ._utils import (
    compare_denoising_quality,
    compute_background_signal,
    compute_segmentation_confidence,
    compute_signal_to_noise,
    compute_spatial_smoothness,
    filter_low_quality_cells,
    identify_contaminated_cells,
)

__all__ = [
    # Model wrapper
    "ResolVI",
    # Module classes
    "RESOLVAE",
    "RESOLVAEModel",
    "RESOLVAEGuide",
    # Utility functions
    "compare_denoising_quality",
    "compute_background_signal",
    "compute_segmentation_confidence",
    "compute_signal_to_noise",
    "compute_spatial_smoothness",
    "filter_low_quality_cells",
    "identify_contaminated_cells",
]
