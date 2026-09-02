from __future__ import annotations

from typing import NamedTuple

from scvi import REGISTRY_KEYS as _SCVI_REGISTRY_KEYS

# Re-export all scvi registry keys so scviva code can import from one place
REGISTRY_KEYS = _SCVI_REGISTRY_KEYS

# Spatial-specific keys used across models
SPATIAL_COORDS_KEY = "spatial_coords"
NEIGHBOR_INDEX_KEY = "index_neighbor"
NEIGHBOR_DISTANCE_KEY = "distance_neighbor"
NICHE_COMPOSITION_KEY = "neighborhood_composition"

# Private batch metadata used by scVIVA's optional contiguity-aware loader.
SCVIVA_SEED_COUNT_KEY = "_scviva_seed_count"
SCVIVA_CONTIGUITY_EDGE_INDEX_KEY = "_scviva_contiguity_edge_index"
SCVIVA_CONTIGUITY_REPLACEMENT_KEY = "_scviva_contiguity_replacement"


# scVIVA-specific registry and module keys
class _SCVIVA_REGISTRY_KEYS_NT(NamedTuple):
    SAMPLE_KEY: str = "sample"
    Z1_MEAN_KEY: str = "latent_mean"
    NICHE_INDEXES_KEY: str = "niche_indexes"
    NICHE_DISTANCES_KEY: str = "niche_distances"
    NICHE_COMPOSITION_KEY: str = "niche_composition"
    Z1_MEAN_CT_KEY: str = "latent_mean_ct_key"
    CELL_COORDINATES_KEY: str = "spatial"


SCVIVA_REGISTRY_KEYS = _SCVIVA_REGISTRY_KEYS_NT()


class _SCVIVA_MODULE_KEYS(NamedTuple):
    # generative model
    NICHE_MEAN: str = "niche_mean"
    NICHE_VARIANCE: str = "niche_variance"
    P_NICHE_COMPOSITION: str = "niche_composition"
    P_NICHE_EXPRESSION: str = "niche_expression"
    NICHE_ATTENTION: str = "niche_attention"
    # loss
    NLL_NICHE_COMPOSITION_KEY: str = "niche_compo"
    NLL_NICHE_EXPRESSION_KEY: str = "niche_reconst"
    SPATIAL_WEIGHT_KEY: str = "spatial_weight"


SCVIVA_MODULE_KEYS = _SCVIVA_MODULE_KEYS()
