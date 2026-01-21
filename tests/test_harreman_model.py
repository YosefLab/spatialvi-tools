"""Tests for the HarremanModel wrapper."""

import pytest

from spatialvi_tools.data import load_dummy_spatial_dataset
from spatialvi_tools.models import HarremanModel


def test_harreman_placeholder() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = HarremanModel(adata)
    # calling train should return None (no training required)
    assert model.train() is None
    # predict should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        model.predict()