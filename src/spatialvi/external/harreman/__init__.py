"""Harreman: Metabolic exchange inference from spatial data.

Harreman infers metabolic exchange between spatially proximal cells
using spatial correlation analysis.
"""

from __future__ import annotations

from ._model import Harreman
from ._module import HarremanModule, MetabolicExchangeScorer
from ._utils import (
    METABOLIC_PATHWAYS,
    annotate_metabolic_genes,
    compute_exchange_network,
    compute_pathway_scores,
    filter_genes_by_expression,
    get_metabolic_genes,
)

__all__ = [
    "Harreman",
    "HarremanModule",
    "METABOLIC_PATHWAYS",
    "MetabolicExchangeScorer",
    "annotate_metabolic_genes",
    "compute_exchange_network",
    "compute_pathway_scores",
    "filter_genes_by_expression",
    "get_metabolic_genes",
]
