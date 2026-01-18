"""Comprehensive tests for ResolVI utilities."""

import numpy as np
import pytest


class TestResolVIUtils:
    """Tests for ResolVI utility functions."""

    def test_compute_background_signal_local_median(self, small_spatial_adata):
        """Test background signal estimation with local median."""
        from spatialvi.external.resolvi import compute_background_signal

        adata = small_spatial_adata.copy()

        background = compute_background_signal(
            adata,
            spatial_key="spatial",
            n_neighbors=5,
            method="local_median",
        )

        assert background.shape == adata.X.shape

    def test_compute_background_signal_local_min(self, small_spatial_adata):
        """Test background signal estimation with local min."""
        from spatialvi.external.resolvi import compute_background_signal

        adata = small_spatial_adata.copy()

        background = compute_background_signal(
            adata,
            spatial_key="spatial",
            n_neighbors=5,
            method="local_min",
        )

        assert background.shape == adata.X.shape

    def test_compute_background_signal_percentile(self, small_spatial_adata):
        """Test background signal estimation with percentile."""
        from spatialvi.external.resolvi import compute_background_signal

        adata = small_spatial_adata.copy()

        background = compute_background_signal(
            adata,
            spatial_key="spatial",
            n_neighbors=5,
            method="percentile",
        )

        assert background.shape == adata.X.shape

    def test_compute_segmentation_confidence(self, small_spatial_adata):
        """Test segmentation confidence computation."""
        from spatialvi.external.resolvi import compute_segmentation_confidence

        adata = small_spatial_adata.copy()
        adata.obs["cell_area"] = np.random.uniform(50, 500, adata.n_obs)
        adata.obs["n_transcripts"] = np.random.randint(10, 1000, adata.n_obs)

        confidence = compute_segmentation_confidence(
            adata,
            area_key="cell_area",
            n_transcripts_key="n_transcripts",
        )

        assert len(confidence) == adata.n_obs
        assert all(0 <= c <= 1 for c in confidence)

    def test_identify_contaminated_cells(self, small_spatial_adata):
        """Test contaminated cell identification."""
        from spatialvi.external.resolvi import identify_contaminated_cells

        adata = small_spatial_adata.copy()
        background_fraction = np.random.uniform(0, 1, adata.n_obs)

        contaminated = identify_contaminated_cells(
            adata, background_fraction, threshold=0.5
        )

        assert len(contaminated) == adata.n_obs
        assert contaminated.dtype == bool
        assert contaminated.sum() == (background_fraction > 0.5).sum()

    def test_compute_signal_to_noise(self):
        """Test signal-to-noise computation."""
        from spatialvi.external.resolvi import compute_signal_to_noise

        raw = np.random.rand(100, 50) * 10
        denoised = raw + np.random.rand(100, 50) * 0.1  # Small noise added

        snr = compute_signal_to_noise(raw, denoised)

        assert len(snr) == 50  # Per gene

    def test_filter_low_quality_cells(self, small_spatial_adata):
        """Test low quality cell filtering."""
        from spatialvi.external.resolvi import filter_low_quality_cells

        adata = small_spatial_adata.copy()
        adata.obs["background_fraction"] = np.random.uniform(0, 1, adata.n_obs)

        keep = filter_low_quality_cells(
            adata,
            min_counts=10,
            min_genes=5,
            max_background=0.7,
        )

        assert len(keep) == adata.n_obs
        assert keep.dtype == bool

    def test_compute_spatial_smoothness(self, small_spatial_adata):
        """Test spatial smoothness computation."""
        from spatialvi.external.resolvi import compute_spatial_smoothness

        adata = small_spatial_adata.copy()
        expression = adata.X
        if hasattr(expression, "toarray"):
            expression = expression.toarray()

        smoothness = compute_spatial_smoothness(
            adata, expression, spatial_key="spatial", n_neighbors=5
        )

        assert isinstance(smoothness, float)
        assert smoothness >= 0

    def test_compare_denoising_quality(self, small_spatial_adata):
        """Test denoising quality comparison."""
        from spatialvi.external.resolvi import compare_denoising_quality

        adata = small_spatial_adata.copy()
        raw = adata.X
        if hasattr(raw, "toarray"):
            raw = raw.toarray()
        denoised = raw * 0.9  # Simulated denoised

        metrics = compare_denoising_quality(
            raw, denoised, adata, spatial_key="spatial"
        )

        assert "raw_sparsity" in metrics
        assert "denoised_sparsity" in metrics
        assert "raw_mean" in metrics
        assert "denoised_mean" in metrics
        assert "raw_mean_cv" in metrics
        assert "denoised_mean_cv" in metrics
        assert "raw_spatial_var" in metrics
        assert "denoised_spatial_var" in metrics


class TestResolVIModelWrapper:
    """Tests for ResolVI model wrapper."""

    def test_import(self):
        """Test that ResolVI wrapper can be imported."""
        from spatialvi.external.resolvi import ResolVI

        assert ResolVI is not None

    def test_utils_import(self):
        """Test that all utils can be imported."""
        from spatialvi.external.resolvi import (
            compare_denoising_quality,
            compute_background_signal,
            compute_segmentation_confidence,
            compute_signal_to_noise,
            compute_spatial_smoothness,
            filter_low_quality_cells,
            identify_contaminated_cells,
        )

        assert compare_denoising_quality is not None
        assert compute_background_signal is not None
