"""Regression checks for tutorial notebook wiring."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _cells(path: str) -> list[dict]:
    notebook = json.loads((ROOT / path).read_text())
    return notebook["cells"]


def _sources(path: str) -> list[str]:
    return ["".join(cell.get("source", [])) for cell in _cells(path)]


def test_resolvi_legacy_benchmark_runs_after_active_pyro_model_usage():
    """The AnnDataLoader baseline must not overwrite Pyro params before DE/query cells."""
    sources = _sources("docs/tutorials/resolVI_tutorial.ipynb")
    legacy_idx = next(i for i, source in enumerate(sources) if "class ResolVILegacy" in source)
    active_model_uses = [
        i
        for i, source in enumerate(sources)
        if (
            "supervised_resolvi.differential_expression" in source
            or "supervised_resolvi.differential_niche_abundance" in source
            or "supervised_resolvi.sample_posterior" in source
            or "query_resolvi.train" in source
            or "query_resolvi.predict" in source
        )
    ]
    assert active_model_uses
    assert legacy_idx > max(active_model_uses)


def test_tutorial_metric_sections_use_scib_rapids_not_scib_metrics_benchmarker():
    """Tutorials should use scib_rapids metrics while spatial classes are unavailable there."""
    for path in [
        "docs/tutorials/resolVI_tutorial.ipynb",
        "docs/tutorials/scVIVA_tutorial.ipynb",
    ]:
        text = "\n".join(_sources(path))
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in _cells(path)
            if cell.get("cell_type") == "code"
        )
        assert "scib_rapids" in text
        assert "from scib_metrics.benchmark" not in code_text
        assert "CoordinatePreservation(" not in code_text
        assert "NichePreservation(" not in code_text
        assert "DomainBoundary(" not in code_text
