from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.data import synthetic_iid
from scvi.module._constants import MODULE_KEYS

from scviva._constants import SCVIVA_MODULE_KEYS, SCVIVA_REGISTRY_KEYS
from scviva.model._scviva import SCVIVA, _calibrated_lambda
from scviva.model.utils._contiguity import (
    CONTIGUITY_EDGE_INDEX_KEY,
    SEED_COUNT_KEY,
    ContiguityDataLoader,
    build_same_label_edges,
)
from scviva.module._nichevae import _latent_contiguity_loss


@pytest.fixture
def contiguity_adata():
    """Return deterministic registered data with same-label edges in both batches."""
    rng = np.random.default_rng(34)
    adata = synthetic_iid(
        batch_size=16,
        n_genes=20,
        n_batches=2,
        n_labels=2,
        dropout_ratio=0.1,
    )
    adata.obs["labels"] = pd.Categorical(
        ["type_0"] * 8 + ["type_1"] * 8 + ["type_0"] * 8 + ["type_1"] * 8
    )
    adata.obsm["spatial"] = np.vstack(
        [
            rng.normal((0, 0), 0.1, (8, 2)),
            rng.normal((10, 10), 0.1, (8, 2)),
            rng.normal((0, 0), 0.1, (8, 2)),
            rng.normal((10, 10), 0.1, (8, 2)),
        ]
    )
    adata.obsm["X_scVI"] = rng.normal(size=(adata.n_obs, 10))
    raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    # Keep this tiny integration fixture numerically valid for library-size
    # initialization: synthetic_iid can otherwise produce all-zero cells.
    adata.layers["counts"] = np.abs(raw).astype(int) + 1
    setup_kwargs = {
        "sample_key": "batch",
        "labels_key": "labels",
        "cell_coordinates_key": "spatial",
        "expression_embedding_key": "X_scVI",
        "expression_embedding_niche_key": "niche_activation",
        "niche_composition_key": "niche_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }
    SCVIVA.preprocessing_anndata(adata, k_nn=3, **setup_kwargs)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **setup_kwargs)
    return adata


def _make_model(adata, **kwargs):
    return SCVIVA(adata, prior_mixture=False, **kwargs)


def _fixed_split():
    return [np.arange(16), np.arange(16, 24), np.arange(24, 32)]


def _module_loss(module, batch):
    inference_inputs = module._get_inference_input(batch)
    inference_outputs = module.inference(**inference_inputs)
    generative_inputs = module._get_generative_input(batch, inference_outputs)
    generative_outputs = module.generative(**generative_inputs)
    return module.loss(
        batch,
        inference_outputs,
        generative_outputs,
        kl_weight=0.0,
        classification_ratio=50,
    )


def _single_loss_record(record):
    assert len(record) == 1
    return next(iter(record.values()))


def test_nichevae_accepts_numeric_contiguity_lambda(contiguity_adata):
    """Catch a model/module boundary that drops the requested numeric weight."""
    model = _make_model(contiguity_adata, contiguity_lambda=2.5)

    assert model.module.contiguity_lambda == pytest.approx(2.5)


def test_latent_contiguity_loss_known_value():
    """Catch incorrect endpoint, latent-dimension, or edge normalization."""
    latent_mean = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    edges = torch.tensor([[0, 1], [1, 2]])

    observed = _latent_contiguity_loss(latent_mean, edges)

    assert observed.item() == pytest.approx(2.0)


def test_contiguity_loss_uses_edges_but_reports_seed_records(contiguity_adata):
    """Catch auxiliary endpoints leaking into ordinary per-cell loss records."""
    model = _make_model(contiguity_adata, contiguity_lambda=2.5)
    manager = model.adata_manager
    edges = build_same_label_edges(
        manager.get_from_registry(SCVIVA_REGISTRY_KEYS.NICHE_INDEXES_KEY),
        manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY),
    )
    batch = next(
        iter(
            ContiguityDataLoader(
                manager,
                indices=np.arange(contiguity_adata.n_obs),
                eligible_edges=edges,
                batch_size=4,
                edge_budget=8,
                shuffle=False,
                seed=34,
            )
        )
    )

    torch.manual_seed(34)
    loss_output = _module_loss(model.module, batch)
    metrics = loss_output.extra_metrics

    assert int(batch[SEED_COUNT_KEY]) == 4
    assert batch[CONTIGUITY_EDGE_INDEX_KEY].shape == (2, 8)
    assert _single_loss_record(loss_output.reconstruction_loss).shape[0] == 4
    assert loss_output.kl_local[MODULE_KEYS.KL_Z_KEY].shape[0] == 4
    assert _single_loss_record(loss_output.composition_loss).shape[0] == 4
    assert _single_loss_record(loss_output.niche_loss).shape[0] == 4
    assert int(metrics["contiguity_edge_count"]) == 8
    assert metrics["contiguity_edge_count"].is_floating_point()
    torch.testing.assert_close(
        loss_output.loss,
        metrics["ordinary_loss"] + metrics["weighted_contiguity_loss"],
    )
    torch.testing.assert_close(
        metrics["weighted_contiguity_loss"], 2.5 * metrics["contiguity_loss"]
    )


def test_explicit_zero_matches_default_loss(contiguity_adata):
    """Catch any numerical change to ordinary scVIVA when contiguity is disabled."""
    default_model = _make_model(contiguity_adata)
    zero_model = _make_model(contiguity_adata, contiguity_lambda=0.0)
    zero_model.module.load_state_dict(default_model.module.state_dict(), strict=True)
    batch = next(
        iter(
            default_model._make_data_loader(contiguity_adata, indices=np.arange(16), batch_size=16)
        )
    )

    torch.manual_seed(34)
    default_loss = _module_loss(default_model.module, batch)
    torch.manual_seed(34)
    zero_loss = _module_loss(zero_model.module, batch)

    torch.testing.assert_close(default_loss.loss, zero_loss.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        _single_loss_record(default_loss.reconstruction_loss),
        _single_loss_record(zero_loss.reconstruction_loss),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        default_loss.kl_local[MODULE_KEYS.KL_Z_KEY],
        zero_loss.kl_local[MODULE_KEYS.KL_Z_KEY],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        default_loss.extra_metrics[SCVIVA_MODULE_KEYS.NLL_NICHE_COMPOSITION_KEY],
        zero_loss.extra_metrics[SCVIVA_MODULE_KEYS.NLL_NICHE_COMPOSITION_KEY],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("requested", "effective", "needs_calibration"),
    [(0.0, 0.0, False), (2.5, 2.5, False), ("auto", 0.0, True)],
)
def test_public_contiguity_modes_are_normalized(
    contiguity_adata, requested, effective, needs_calibration
):
    """Catch loss of the requested mode or an incorrect initial effective weight."""
    model = _make_model(contiguity_adata, contiguity_lambda=requested)

    assert model.contiguity_lambda_requested_ == requested
    assert model.contiguity_lambda_ == pytest.approx(effective)
    assert model.contiguity_calibration_ is None
    assert model._contiguity_enabled is (requested == "auto" or effective > 0)
    assert model._contiguity_requires_calibration is needs_calibration
    assert model.module.contiguity_lambda == pytest.approx(effective)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, True, "automatic"])
def test_invalid_contiguity_lambda_is_rejected(contiguity_adata, value):
    """Catch invalid public weights reaching the training loop."""
    with pytest.raises((TypeError, ValueError), match="contiguity_lambda"):
        _make_model(contiguity_adata, contiguity_lambda=value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"contiguity_target_fraction": 0.0}, "contiguity_target_fraction"),
        ({"contiguity_target_fraction": np.nan}, "contiguity_target_fraction"),
        ({"contiguity_edge_budget": 0}, "contiguity_edge_budget"),
        ({"contiguity_edge_budget": 1.5}, "contiguity_edge_budget"),
    ],
)
def test_invalid_contiguity_calibration_config_is_rejected(contiguity_adata, kwargs, message):
    """Catch unusable calibration targets or edge budgets at construction time."""
    with pytest.raises((TypeError, ValueError), match=message):
        _make_model(contiguity_adata, **kwargs)


def test_effective_lambda_synchronizes_module_and_serialized_init(contiguity_adata):
    """Catch fitted lambda being lost at the model/module or save/load boundary."""
    model = _make_model(contiguity_adata, contiguity_lambda="auto")

    model._set_effective_contiguity_lambda(12.5)

    assert model.contiguity_lambda_ == pytest.approx(12.5)
    assert model.module.contiguity_lambda == pytest.approx(12.5)
    assert model.init_params_["non_kwargs"]["contiguity_lambda"] == pytest.approx(12.5)


def test_calibrated_lambda_targets_requested_fraction():
    """Catch an inverted ratio or omitted target fraction in auto-calibration."""
    observed = _calibrated_lambda(ordinary_loss=200.0, contiguity_loss=0.5, target=0.05)

    assert observed == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("ordinary", "contiguity", "target"),
    [(0.0, 1.0, 0.05), (1.0, 0.0, 0.05), (np.nan, 1.0, 0.05)],
)
def test_calibrated_lambda_rejects_invalid_measurements(ordinary, contiguity, target):
    """Catch silent zero, non-finite, or invalid automatic weights."""
    with pytest.raises(ValueError, match="calibration"):
        _calibrated_lambda(ordinary, contiguity, target)


def test_auto_calibration_does_not_mutate_parameters(contiguity_adata):
    """Catch calibration accidentally taking an optimizer-like parameter update."""
    model = _make_model(contiguity_adata, contiguity_lambda="auto", contiguity_edge_budget=8)
    splitter = model._make_contiguity_splitter(
        train_size=0.5,
        validation_size=0.25,
        shuffle_set_split=False,
        load_sparse_tensor=False,
        batch_size=4,
        datasplitter_kwargs={"external_indexing": _fixed_split()},
    )
    before = {key: value.detach().clone() for key, value in model.module.state_dict().items()}

    effective = model._calibrate_contiguity_lambda(splitter, classification_ratio=50)

    for key, value in model.module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    record = model.contiguity_calibration_
    assert effective == pytest.approx(0.05 * record["ordinary_loss"] / record["contiguity_loss"])
    assert record["training_cells"] == 16


def test_contiguity_splitter_accepts_unset_scvi_seed(contiguity_adata, monkeypatch):
    """Support normal processes where scvi has no active global seed."""
    monkeypatch.setattr(settings, "seed", None)
    model = _make_model(contiguity_adata, contiguity_lambda=2.5)

    splitter = model._make_contiguity_splitter(
        train_size=0.5,
        validation_size=0.25,
        shuffle_set_split=False,
        load_sparse_tensor=False,
        batch_size=4,
        datasplitter_kwargs={"external_indexing": _fixed_split()},
    )

    assert splitter.loader_seed == 0


def test_auto_contiguity_trains_and_records_calibration(contiguity_adata):
    """Catch failure of the public automatic mode to calibrate and train end to end."""
    model = _make_model(contiguity_adata, contiguity_lambda="auto", contiguity_edge_budget=8)

    model.train(
        max_epochs=1,
        train_size=0.5,
        validation_size=0.25,
        batch_size=4,
        accelerator="cpu",
        devices=1,
        datasplitter_kwargs={"external_indexing": _fixed_split()},
        plan_kwargs={"classification_ratio": 50},
    )

    assert model.is_trained
    assert model.contiguity_lambda_ > 0
    assert model.module.contiguity_lambda == pytest.approx(model.contiguity_lambda_)
    assert model.contiguity_calibration_["target_fraction"] == pytest.approx(0.05)
    assert model._contiguity_requires_calibration is False


def test_auto_devices_remains_single_device_compatible(contiguity_adata):
    """Keep scvi's first-device normalization available to the public default."""
    model = _make_model(contiguity_adata, contiguity_lambda="auto", contiguity_edge_budget=8)

    model.train(
        max_epochs=1,
        train_size=0.5,
        validation_size=0.25,
        batch_size=4,
        accelerator="cpu",
        devices="auto",
        datasplitter_kwargs={"external_indexing": _fixed_split()},
    )

    assert model.is_trained


@pytest.mark.parametrize("devices", ["0,1", "-1"])
def test_multi_device_strings_are_rejected_before_split(contiguity_adata, monkeypatch, devices):
    """Reject Lightning string forms that can select multiple devices."""
    model = _make_model(contiguity_adata, contiguity_lambda=2.5)

    def fail_if_split_is_built(**kwargs):
        raise AssertionError("device validation must run before split construction")

    monkeypatch.setattr(model, "_make_contiguity_splitter", fail_if_split_is_built)

    with pytest.raises(NotImplementedError, match="one device"):
        model.train(max_epochs=1, devices=devices)


def test_distributed_trainer_config_is_rejected_before_split(contiguity_adata, monkeypatch):
    """Inspect strategy supplied through trainer_config as well as direct kwargs."""
    model = _make_model(contiguity_adata, contiguity_lambda=2.5)

    def fail_if_split_is_built(**kwargs):
        raise AssertionError("strategy validation must run before split construction")

    monkeypatch.setattr(model, "_make_contiguity_splitter", fail_if_split_is_built)

    with pytest.raises(NotImplementedError, match="one device"):
        model.train(max_epochs=1, trainer_config={"strategy": "ddp"})


def test_explicit_contiguity_trains_without_calibration(contiguity_adata):
    """Catch explicit weights being overwritten by the automatic calibration path."""
    model = _make_model(contiguity_adata, contiguity_lambda=2.5, contiguity_edge_budget=8)

    model.train(
        max_epochs=1,
        train_size=0.5,
        validation_size=0.25,
        batch_size=4,
        accelerator="cpu",
        devices=1,
        datasplitter_kwargs={"external_indexing": _fixed_split()},
    )

    assert model.is_trained
    assert model.contiguity_lambda_ == pytest.approx(2.5)
    assert model.module.contiguity_lambda == pytest.approx(2.5)
    assert model.contiguity_calibration_ is None


def test_auto_contiguity_save_load_preserves_fitted_lambda(contiguity_adata, tmp_path):
    """Catch fitted calibration metadata being lost across serialization."""
    model = _make_model(contiguity_adata, contiguity_lambda="auto", contiguity_edge_budget=8)
    model.train(
        max_epochs=1,
        train_size=0.5,
        validation_size=0.25,
        batch_size=4,
        accelerator="cpu",
        devices=1,
        datasplitter_kwargs={"external_indexing": _fixed_split()},
    )
    effective_before = model.contiguity_lambda_
    latent_before = model.get_latent_representation()
    path = tmp_path / "auto-contiguity"

    model.save(path, save_anndata=True)
    loaded = SCVIVA.load(path)

    assert loaded.contiguity_lambda_ == pytest.approx(effective_before)
    assert loaded.module.contiguity_lambda == pytest.approx(effective_before)
    assert loaded.contiguity_lambda_requested_ == "auto"
    assert loaded.contiguity_calibration_ == model.contiguity_calibration_
    assert loaded._contiguity_requires_calibration is False
    np.testing.assert_allclose(
        latent_before, loaded.get_latent_representation(), rtol=1e-5, atol=1e-6
    )


def test_default_state_dict_has_no_contiguity_key(contiguity_adata):
    """Keep old strict checkpoints compatible with the opt-in feature."""
    model = _make_model(contiguity_adata)

    assert not any("contiguity" in key for key in model.module.state_dict())
