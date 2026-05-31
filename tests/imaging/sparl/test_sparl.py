"""Tests for SPARL model. Requires sparl package."""

from __future__ import annotations

import pytest
import torch

from tests.imaging.conftest import _MinimalImagingModel  # noqa: F401

sparl = pytest.importorskip("sparl")

from scviva.imaging.sparl._model import SPARL  # noqa: E402
from scviva.imaging.sparl._module import SPARLModule  # noqa: E402


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


def test_sparl_from_pretrained_local(sparl_checkpoint):
    model = SPARL.from_pretrained(sparl_checkpoint)
    assert model.module is not None
    assert isinstance(model.module, SPARLModule)
    assert not model.module.training


def test_sparl_from_pretrained_embed_dim(sparl_checkpoint):
    model = SPARL.from_pretrained(sparl_checkpoint)
    assert model.module.backbone.embed_dim == 32


def test_sparl_from_pretrained_hf(sparl_checkpoint, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: sparl_checkpoint
    )
    model = SPARL.from_pretrained("YosefLab/sparl-imc-vitb16")
    assert isinstance(model.module, SPARLModule)
    # TODO: add real hub test once YosefLab/sparl-* is published on HuggingFace
