"""Module constants for spatial transcriptomics models."""

from __future__ import annotations


class MODULE_KEYS:
    """Keys for module outputs and inputs.

    These constants define the standard keys used in dictionaries
    returned by inference and generative methods.
    """

    # Latent variables
    Z_KEY = "z"
    QZ_KEY = "qz"
    QZM_KEY = "qz_m"
    QZV_KEY = "qz_v"

    # Library size
    LIBRARY_KEY = "library"
    QL_KEY = "ql"
    QLM_KEY = "ql_m"
    QLV_KEY = "ql_v"

    # Generative outputs
    PX_KEY = "px"
    PX_SCALE_KEY = "px_scale"
    PX_RATE_KEY = "px_rate"
    PX_DROPOUT_KEY = "px_dropout"
    PX_R_KEY = "px_r"

    # Spatial-specific keys
    SPATIAL_LATENT_KEY = "z_spatial"
    NEIGHBOR_Z_KEY = "neighbor_z"
    SPATIAL_CONTEXT_KEY = "spatial_context"
    NICHE_KEY = "niche"

    # Deconvolution keys
    PROPORTIONS_KEY = "proportions"
    CELLTYPE_SCALE_KEY = "celltype_scale"

    # Attention keys
    ATTENTION_WEIGHTS_KEY = "attention_weights"
    NEIGHBOR_ATTENTION_KEY = "neighbor_attention"


class SPATIAL_MODULE_KEYS:
    """Additional keys specific to spatial modules."""

    # Spatial coordinates and neighbors
    COORDS_KEY = "spatial_coords"
    NEIGHBOR_INDEX_KEY = "neighbor_index"
    NEIGHBOR_DIST_KEY = "neighbor_dist"

    # Niche composition
    NICHE_COMPOSITION_KEY = "niche_composition"
    NICHE_COMPOSITION_PRED_KEY = "niche_composition_pred"

    # Cell-cell interactions
    INTERACTION_KEY = "interactions"
    LIGAND_KEY = "ligand"
    RECEPTOR_KEY = "receptor"

    # Spatial regularization
    SPATIAL_LOSS_KEY = "spatial_loss"
    NEIGHBOR_LOSS_KEY = "neighbor_loss"


class LOSS_KEYS:
    """Keys for loss components."""

    # Standard VAE losses
    RECONSTRUCTION_LOSS = "reconstruction_loss"
    KL_LOCAL = "kl_local"
    KL_GLOBAL = "kl_global"

    # Spatial-specific losses
    SPATIAL_LOSS = "spatial_loss"
    NEIGHBOR_LOSS = "neighbor_loss"
    NICHE_LOSS = "niche_loss"

    # Classification losses
    CLASSIFICATION_LOSS = "classification_loss"
    CELLTYPE_LOSS = "celltype_loss"

    # Regularization
    L1_REG = "l1_reg"
    L2_REG = "l2_reg"
