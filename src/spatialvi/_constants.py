from __future__ import annotations

from scvi import REGISTRY_KEYS as _SCVI_REGISTRY_KEYS

# Re-export all scvi registry keys so spatialvi code can import from one place
REGISTRY_KEYS = _SCVI_REGISTRY_KEYS

# Spatial-specific keys used across models
SPATIAL_COORDS_KEY = "spatial_coords"
NEIGHBOR_INDEX_KEY = "index_neighbor"
NEIGHBOR_DISTANCE_KEY = "distance_neighbor"
NICHE_COMPOSITION_KEY = "neighborhood_composition"
