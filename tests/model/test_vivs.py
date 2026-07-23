"""Tests for VIVS model."""

from scviva._constants import VIVS_REGISTRY_KEYS


def test_vivs_registry_keys():
    assert VIVS_REGISTRY_KEYS.Y_KEY == "Y"
