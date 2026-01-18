"""AMICI: Attention-based Multi-scale Interaction for Cell-cell Inference.

AMICI uses cross-attention mechanisms to model cell-cell interactions
in spatial transcriptomics data.
"""

from __future__ import annotations

from ._model import AMICI
from ._module import AMICIModule
from ._utils import (
    build_interaction_matrix,
    compute_interaction_neighbors,
    compute_interaction_strength,
    filter_expressed_pairs,
    get_ligand_receptor_pairs,
)

__all__ = [
    "AMICI",
    "AMICIModule",
    "build_interaction_matrix",
    "compute_interaction_neighbors",
    "compute_interaction_strength",
    "filter_expressed_pairs",
    "get_ligand_receptor_pairs",
]
