"""
Data loading utilities for spatialvi‑tools.

This subpackage contains functions to generate or load example datasets
used throughout the library and in the test suite.  In a real deployment
you would implement dataset‑specific loaders here.
"""

from .datasets import load_dummy_spatial_dataset  # noqa: F401

__all__ = ["load_dummy_spatial_dataset"]