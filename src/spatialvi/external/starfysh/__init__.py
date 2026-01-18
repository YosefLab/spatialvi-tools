"""Starfysh: Spatial deconvolution with histology integration.

Starfysh performs spatial deconvolution of bulk spatial transcriptomics
spots into cell type proportions, optionally integrating histology images.
"""

from __future__ import annotations

from ._model import Starfysh
from ._module import StarfyshModule
from ._utils import (
    compute_reference_signatures,
    evaluate_deconvolution,
    find_marker_genes,
    proportions_to_counts,
    validate_input_data,
)

__all__ = [
    "Starfysh",
    "StarfyshModule",
    "compute_reference_signatures",
    "evaluate_deconvolution",
    "find_marker_genes",
    "proportions_to_counts",
    "validate_input_data",
]
