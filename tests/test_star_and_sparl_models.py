"""Tests for StarfyshModel and SparlModel wrappers."""

import pytest

from spatialvi_tools.data import load_dummy_spatial_dataset
from spatialvi_tools.models import StarfyshModel, SparlModel


def test_starfysh_placeholder_methods() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = StarfyshModel(adata)
    # train should raise NotImplementedError if dependency is present; otherwise skip
    try:
        model.train()
    except ImportError:
        pytest.skip("starfysh dependency not installed")
    except NotImplementedError:
        pass
    # predict should also raise NotImplementedError
    with pytest.raises(NotImplementedError):
        model.predict()


def test_sparl_placeholder_methods() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = SparlModel(adata)
    # train should raise NotImplementedError if dependency is present; otherwise skip
    try:
        model.train()
    except ImportError:
        pytest.skip("SPARL dependency not installed")
    except NotImplementedError:
        pass
    with pytest.raises(NotImplementedError):
        model.predict()