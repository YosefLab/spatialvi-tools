"""SPARL model for spatial proteomics representation learning."""

from spatialvi.external.sparl._model import SPARL
from spatialvi.external.sparl._module import SPARLModule
from spatialvi.external.sparl._utils import (
    compute_neighborhood_composition,
    compute_protein_coexpression,
    detect_protein_communities,
    identify_marker_proteins,
    normalize_protein_expression,
    spatial_protein_enrichment,
)

__all__ = [
    "SPARL",
    "SPARLModule",
    "compute_neighborhood_composition",
    "compute_protein_coexpression",
    "detect_protein_communities",
    "identify_marker_proteins",
    "normalize_protein_expression",
    "spatial_protein_enrichment",
]
