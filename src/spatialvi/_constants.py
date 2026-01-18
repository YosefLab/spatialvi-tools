"""Constants and registry keys for spatialvi-tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _REGISTRY_KEYS:
    """Standard registry keys compatible with scvi-tools."""

    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    CAT_COVS_KEY: str = "categorical_covs"
    CONT_COVS_KEY: str = "continuous_covs"
    SIZE_FACTOR_KEY: str = "size_factor"
    INDICES_KEY: str = "indices"


@dataclass(frozen=True)
class _SPATIAL_REGISTRY_KEYS:
    """Spatial-specific registry keys."""

    # Spatial coordinates
    SPATIAL_KEY: str = "spatial"
    COORD_KEY: str = "coordinates"

    # Neighbor information
    NN_INDEX_KEY: str = "nn_index"
    NN_DIST_KEY: str = "nn_dist"
    NUM_NEIGHBORS_KEY: str = "n_neighbors"
    NN_X_KEY: str = "nn_X"

    # Niche/neighborhood composition
    NICHE_COMPOSITION_KEY: str = "niche_composition"
    NICHE_INDEXES_KEY: str = "niche_indexes"
    NICHE_DISTANCES_KEY: str = "niche_distances"
    NICHE_ACTIVATION_KEY: str = "niche_activation"

    # Expression embeddings
    EXPRESSION_EMBEDDING_KEY: str = "expression_embedding"
    Z1_MEAN_KEY: str = "z1_mean"
    Z1_MEAN_CT_KEY: str = "z1_mean_ct"

    # Sample/slide information
    SAMPLE_KEY: str = "sample"
    SLIDE_KEY: str = "slide"

    # Cell type deconvolution
    PROPORTIONS_KEY: str = "proportions"
    GAMMA_KEY: str = "gamma"

    # Background/noise modeling
    BACKGROUND_KEY: str = "background"
    DIFFUSION_KEY: str = "diffusion"


@dataclass(frozen=True)
class _MODULE_KEYS:
    """Module output keys."""

    # Latent space
    Z_KEY: str = "z"
    QZ_KEY: str = "qz"
    QZ_M_KEY: str = "qz_m"
    QZ_V_KEY: str = "qz_v"

    # Library size
    L_KEY: str = "l"
    QL_KEY: str = "ql"
    QL_M_KEY: str = "ql_m"
    QL_V_KEY: str = "ql_v"

    # Gene expression
    PX_KEY: str = "px"
    PX_RATE_KEY: str = "px_rate"
    PX_SCALE_KEY: str = "px_scale"
    PX_R_KEY: str = "px_r"

    # Losses
    KL_L_KEY: str = "kl_l"
    KL_Z_KEY: str = "kl_z"
    RECONSTRUCTION_LOSS_KEY: str = "reconstruction_loss"

    # Niche-specific
    NICHE_COMPOSITION_PRED_KEY: str = "niche_composition_pred"
    NICHE_ACTIVATION_PRED_KEY: str = "niche_activation_pred"


REGISTRY_KEYS = _REGISTRY_KEYS()
SPATIAL_REGISTRY_KEYS = _SPATIAL_REGISTRY_KEYS()
MODULE_KEYS = _MODULE_KEYS()