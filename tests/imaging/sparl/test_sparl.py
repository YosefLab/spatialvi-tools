"""Tests for SPARL model. Requires sparl package."""

from __future__ import annotations

import pytest
import torch

from tests.imaging.conftest import _MinimalImagingModel  # noqa: F401

sparl = pytest.importorskip("sparl")


def test_sparl_module_output_shape(sparl_tiny_vit):
    from scviva.imaging.sparl._module import SPARLModule

    module = SPARLModule(sparl_tiny_vit)
    module.eval()
    x = torch.randn(4, 3, 16, 16)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (4, 32)


def test_sparl_module_no_nan(sparl_tiny_vit):
    from scviva.imaging.sparl._module import SPARLModule

    module = SPARLModule(sparl_tiny_vit)
    module.eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out = module(x)
    assert not torch.any(torch.isnan(out))
