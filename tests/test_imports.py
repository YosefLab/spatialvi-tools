"""Tests for package imports and API exports."""

import pytest


class TestTopLevelImports:
    """Test top-level package imports."""

    def test_version_available(self):
        """Test that version is accessible."""
        import spatialvi

        assert hasattr(spatialvi, "__version__")
        assert isinstance(spatialvi.__version__, str)

    def test_module_imports(self):
        """Test that all modules can be imported."""
        from spatialvi import data
        from spatialvi import model
        from spatialvi import module
        from spatialvi import nn
        from spatialvi import train
        from spatialvi import utils
        from spatialvi import external

        assert data is not None
        assert model is not None
        assert module is not None
        assert nn is not None
        assert train is not None
        assert utils is not None
        assert external is not None

    def test_registry_keys_available(self):
        """Test that registry keys are available."""
        from spatialvi import REGISTRY_KEYS, SPATIAL_REGISTRY_KEYS

        assert REGISTRY_KEYS is not None
        assert SPATIAL_REGISTRY_KEYS is not None

    def test_settings_available(self):
        """Test that settings object is available."""
        from spatialvi import settings

        assert settings is not None


class TestModuleImports:
    """Test module class imports."""

    def test_base_module(self):
        """Test BaseSpatialModule import."""
        from spatialvi.module import BaseSpatialModule

        assert BaseSpatialModule is not None

    def test_spatial_vae_module(self):
        """Test SpatialVAEModule import."""
        from spatialvi.module import SpatialVAEModule

        assert SpatialVAEModule is not None

    def test_deconv_module(self):
        """Test DeconvolutionModule import."""
        from spatialvi.module import DeconvolutionModule

        assert DeconvolutionModule is not None

    def test_niche_module(self):
        """Test NicheModule import."""
        from spatialvi.module import NicheModule

        assert NicheModule is not None


class TestNNImports:
    """Test neural network component imports."""

    def test_encoder_imports(self):
        """Test encoder imports."""
        from spatialvi.nn import SpatialEncoder, GraphEncoder, AttentionEncoder

        assert SpatialEncoder is not None
        assert GraphEncoder is not None
        assert AttentionEncoder is not None

    def test_decoder_imports(self):
        """Test decoder imports."""
        from spatialvi.nn import SpatialDecoder

        assert SpatialDecoder is not None

    def test_attention_imports(self):
        """Test attention mechanism imports."""
        from spatialvi.nn import (
            SpatialAttention,
            CrossAttention,
            NeighborAttention,
            GATLayer,
        )

        assert SpatialAttention is not None
        assert CrossAttention is not None
        assert NeighborAttention is not None
        assert GATLayer is not None

    def test_layer_imports(self):
        """Test custom layer imports."""
        from spatialvi.nn import SpatialConv, GraphConv, PositionalEncoding

        assert SpatialConv is not None
        assert GraphConv is not None
        assert PositionalEncoding is not None


class TestDataImports:
    """Test data utility imports."""

    def test_preprocessing_imports(self):
        """Test preprocessing function imports."""
        from spatialvi.data import (
            compute_spatial_neighbors,
            compute_niche_composition,
            normalize_spatial,
            filter_by_spatial_density,
            add_spatial_noise,
            get_neighbor_expression,
        )

        assert compute_spatial_neighbors is not None
        assert compute_niche_composition is not None
        assert normalize_spatial is not None
        assert filter_by_spatial_density is not None
        assert add_spatial_noise is not None
        assert get_neighbor_expression is not None

    def test_dataset_imports(self):
        """Test dataset function imports."""
        from spatialvi.data import synthetic_spatial, synthetic_scrna

        assert synthetic_spatial is not None
        assert synthetic_scrna is not None

    def test_field_imports(self):
        """Test data field imports."""
        from spatialvi.data import (
            SpatialCoordinatesField,
            NeighborIndexField,
            NeighborDistanceField,
            NicheCompositionField,
        )

        assert SpatialCoordinatesField is not None
        assert NeighborIndexField is not None
        assert NeighborDistanceField is not None
        assert NicheCompositionField is not None


class TestTrainImports:
    """Test training utility imports."""

    def test_training_plan_imports(self):
        """Test training plan imports."""
        from spatialvi.train import (
            SpatialTrainingPlan,
            NicheTrainingPlan,
            DeconvolutionTrainingPlan,
        )

        assert SpatialTrainingPlan is not None
        assert NicheTrainingPlan is not None
        assert DeconvolutionTrainingPlan is not None

    def test_callback_imports(self):
        """Test callback imports."""
        from spatialvi.train import (
            SpatialMetricsCallback,
            NeighborSamplingCallback,
            EarlyStoppingOnSpatialLoss,
            SpatialRegularizationScheduler,
        )

        assert SpatialMetricsCallback is not None
        assert NeighborSamplingCallback is not None
        assert EarlyStoppingOnSpatialLoss is not None
        assert SpatialRegularizationScheduler is not None


class TestExternalImports:
    """Test external model imports."""

    def test_custom_model_imports(self):
        """Test custom external model imports."""
        from spatialvi.external import (
            AMICI,
            VIVS,
            Starfysh,
            Harreman,
            Nolan,
            Lambda,
            PPIInference,
            SPARL,
        )

        assert AMICI is not None
        assert VIVS is not None
        assert Starfysh is not None
        assert Harreman is not None
        assert Nolan is not None
        assert Lambda is not None
        assert PPIInference is not None
        assert SPARL is not None

    def test_scvi_wrapper_imports(self):
        """Test scvi-tools wrapper imports."""
        from spatialvi.external import scVIVA, ResolVI, DestVI

        assert scVIVA is not None
        assert ResolVI is not None
        assert DestVI is not None


class TestUtilsImports:
    """Test utility function imports."""

    def test_metric_imports(self):
        """Test metric function imports."""
        from spatialvi.utils import (
            spatial_autocorrelation,
            compute_morans_i,
            silhouette_spatial,
        )

        assert spatial_autocorrelation is not None
        assert compute_morans_i is not None
        assert silhouette_spatial is not None

    def test_visualization_imports(self):
        """Test visualization function imports."""
        from spatialvi.utils import (
            plot_spatial,
            plot_proportions,
            plot_interactions,
        )

        assert plot_spatial is not None
        assert plot_proportions is not None
        assert plot_interactions is not None
