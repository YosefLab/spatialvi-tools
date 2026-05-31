"""Tests for ImagingBaseModel."""

from __future__ import annotations

import pytest

from tests.imaging.conftest import _MinimalImagingModel


def test_from_pretrained_local_sets_module(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    assert model.module is not None


def test_from_pretrained_local_module_in_eval(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    assert not model.module.training


def test_from_pretrained_bad_path_raises():
    with pytest.raises(FileNotFoundError, match="Could not resolve model path"):
        _MinimalImagingModel.from_pretrained("/does/not/exist.pt")


def test_train_raises(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    with pytest.raises(NotImplementedError):
        model.train()
