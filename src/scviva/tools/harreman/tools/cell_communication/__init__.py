"""Cell-cell communication (CCC) scoring.

Gene-pair filtering, CCC and cell-type-aware CCC scoring, metabolite-level aggregation,
and shared statistics helpers.
"""

from ._ccc_scoring import (
    compute_cell_communication,
    compute_interacting_cell_scores,
    compute_interaction_module_correlation,
    compute_p_int_cell_results_no_ct,
    compute_p_results,
    get_cell_communication_results,
    normalize_values,
    run_cell_communication_analysis,
    select_significant_interactions,
)
from ._ct_ccc_scoring import (
    center_ct_counts_torch,
    compute_ct_cell_communication,
    compute_ct_interacting_cell_scores,
    compute_ct_p_results,
    create_weights_ct_pairs,
    get_ct_cell_communication_results,
    normalize_ct_values,
    run_ct_cell_communication_analysis,
    standardize_ct_counts,
)
from ._gene_filtering import (
    apply_gene_filtering,
    cohens_d,
    compute_gene_pairs,
    de_threshold,
    filter_expr_matrix,
    filter_genes,
    get_interacting_cell_type_pairs,
    perform_feature_elimination,
)
from ._metabolite_scoring import compute_metabolite_cs, compute_metabolite_cs_ct
from ._stats import compute_max_cs, compute_max_cs_gp, flatten

__all__ = [
    "apply_gene_filtering",
    "center_ct_counts_torch",
    "cohens_d",
    "compute_cell_communication",
    "compute_ct_cell_communication",
    "compute_ct_interacting_cell_scores",
    "compute_ct_p_results",
    "compute_gene_pairs",
    "compute_interacting_cell_scores",
    "compute_interaction_module_correlation",
    "compute_max_cs",
    "compute_max_cs_gp",
    "compute_metabolite_cs",
    "compute_metabolite_cs_ct",
    "compute_p_int_cell_results_no_ct",
    "compute_p_results",
    "create_weights_ct_pairs",
    "de_threshold",
    "filter_expr_matrix",
    "filter_genes",
    "flatten",
    "get_cell_communication_results",
    "get_ct_cell_communication_results",
    "get_interacting_cell_type_pairs",
    "normalize_ct_values",
    "normalize_values",
    "perform_feature_elimination",
    "run_cell_communication_analysis",
    "run_ct_cell_communication_analysis",
    "select_significant_interactions",
    "standardize_ct_counts",
]
