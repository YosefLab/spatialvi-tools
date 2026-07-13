# src/scviva/tools/harreman/_constants.py
from __future__ import annotations

# ── adata.uns namespace for run config (see HarremanAnalysis.setup) ──────────
HARREMAN_UNS_KEY = "harreman"
HARREMAN_PARAMS_KEY = "params"

# ── Top-level adata.uns keys written by analysis steps ────────────────────────
# Stored at the top level (not under HARREMAN_UNS_KEY) to match the internal
# read/write conventions of scviva.tools.harreman.tools.cell_communication.
HARREMAN_AUTOCORR_RESULTS_KEY = "gene_autocorrelation_results"
HARREMAN_GENE_PAIRS_RESULTS_KEY = "gene_pairs"
HARREMAN_CCC_RESULTS_KEY = "ccc_results"
HARREMAN_CT_CCC_RESULTS_KEY = "ct_ccc_results"
HARREMAN_ICS_RESULTS_KEY = "interacting_cell_results"
HARREMAN_CT_ICS_RESULTS_KEY = "ct_interacting_cell_results"
HARREMAN_SIG_GP_SUFFIX = "cell_com_df_gp_sig"
HARREMAN_SIG_M_SUFFIX = "cell_com_df_m_sig"

# ── Step names (used in _completed_steps set) ─────────────────────────────────
STEP_SETUP = "setup"
STEP_FILTER = "filter_genes"
STEP_GENE_PAIRS = "compute_gene_pairs"
STEP_CCC = "compute_cell_communication"
STEP_ICS = "compute_interacting_cell_scores"
STEP_SIG = "select_significant_interactions"

# ── Model type strings ────────────────────────────────────────────────────────
MODEL_DESTVI = "DestVI"
MODEL_RESOLVI = "ResolVI"
MODEL_SCVIVA = "SCVIVA"

SUPPORTED_MODELS = (MODEL_DESTVI, MODEL_RESOLVI, MODEL_SCVIVA)

# ── Internal adata keys written during model integration ─────────────────────
HARREMAN_DENOISED_LAYER = "harreman_denoised"
HARREMAN_LATENT_OBSM = "X_harreman_latent"
