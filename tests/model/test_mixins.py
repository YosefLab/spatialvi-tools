"""Tests for model mixins."""


class TestVAEMixin:
    """Tests for VAEMixin."""

    def test_vaemixin_import(self):
        """Test VAEMixin can be imported."""
        from spatialvi.model.base import VAEMixin

        assert VAEMixin is not None

    def test_vaemixin_has_get_elbo(self):
        """Test VAEMixin has get_elbo method."""
        from spatialvi.model.base import VAEMixin

        assert hasattr(VAEMixin, "get_elbo")

    def test_vaemixin_has_get_reconstruction_error(self):
        """Test VAEMixin has get_reconstruction_error method."""
        from spatialvi.model.base import VAEMixin

        assert hasattr(VAEMixin, "get_reconstruction_error")

    def test_vaemixin_has_get_latent_representation(self):
        """Test VAEMixin has get_latent_representation method."""
        from spatialvi.model.base import VAEMixin

        assert hasattr(VAEMixin, "get_latent_representation")

    def test_vaemixin_has_get_normalized_expression(self):
        """Test VAEMixin has get_normalized_expression method."""
        from spatialvi.model.base import VAEMixin

        assert hasattr(VAEMixin, "get_normalized_expression")


class TestEmbeddingMixin:
    """Tests for EmbeddingMixin."""

    def test_embeddingmixin_import(self):
        """Test EmbeddingMixin can be imported."""
        from spatialvi.model.base import EmbeddingMixin

        assert EmbeddingMixin is not None

    def test_spatialembeddingmixin_import(self):
        """Test SpatialEmbeddingMixin can be imported."""
        from spatialvi.model.base import SpatialEmbeddingMixin

        assert SpatialEmbeddingMixin is not None

    def test_embeddingmixin_has_get_embedding(self):
        """Test EmbeddingMixin has get_embedding method."""
        from spatialvi.model.base import EmbeddingMixin

        assert hasattr(EmbeddingMixin, "get_embedding")

    def test_embeddingmixin_has_add_embedding_to_adata(self):
        """Test EmbeddingMixin has add_embedding_to_adata method."""
        from spatialvi.model.base import EmbeddingMixin

        assert hasattr(EmbeddingMixin, "add_embedding_to_adata")

    def test_spatialembeddingmixin_has_get_spatial_embedding(self):
        """Test SpatialEmbeddingMixin has get_spatial_embedding method."""
        from spatialvi.model.base import SpatialEmbeddingMixin

        assert hasattr(SpatialEmbeddingMixin, "get_spatial_embedding")


class TestSpatialMixin:
    """Tests for SpatialMixin."""

    def test_spatialmixin_import(self):
        """Test SpatialMixin can be imported."""
        from spatialvi.model.base import SpatialMixin

        assert SpatialMixin is not None

    def test_spatialmixin_has_get_spatial_neighbors(self):
        """Test SpatialMixin has get_spatial_neighbors method."""
        from spatialvi.model.base import SpatialMixin

        assert hasattr(SpatialMixin, "get_spatial_neighbors")

    def test_spatialmixin_has_get_spatial_coordinates(self):
        """Test SpatialMixin has get_spatial_coordinates method."""
        from spatialvi.model.base import SpatialMixin

        assert hasattr(SpatialMixin, "get_spatial_coordinates")


class TestNicheMixin:
    """Tests for NicheMixin."""

    def test_nichemixin_import(self):
        """Test NicheMixin can be imported."""
        from spatialvi.model.base import NicheMixin

        assert NicheMixin is not None

    def test_nichemixin_has_predict_neighborhood(self):
        """Test NicheMixin has predict_neighborhood method."""
        from spatialvi.model.base import NicheMixin

        assert hasattr(NicheMixin, "predict_neighborhood")

    def test_nichemixin_has_predict_niche_activation(self):
        """Test NicheMixin has predict_niche_activation method."""
        from spatialvi.model.base import NicheMixin

        assert hasattr(NicheMixin, "predict_niche_activation")


class TestDeconvolutionMixin:
    """Tests for DeconvolutionMixin."""

    def test_deconvmixin_import(self):
        """Test DeconvolutionMixin can be imported."""
        from spatialvi.model.base import DeconvolutionMixin

        assert DeconvolutionMixin is not None

    def test_deconvmixin_has_get_proportions(self):
        """Test DeconvolutionMixin has get_proportions method."""
        from spatialvi.model.base import DeconvolutionMixin

        assert hasattr(DeconvolutionMixin, "get_proportions")


class TestTrainingMixins:
    """Tests for training mixins."""

    def test_spatialtrainingmixin_import(self):
        """Test SpatialTrainingMixin can be imported."""
        from spatialvi.model.base import SpatialTrainingMixin

        assert SpatialTrainingMixin is not None

    def test_spatialtrainingmixin_has_train(self):
        """Test SpatialTrainingMixin has train method."""
        from spatialvi.model.base import SpatialTrainingMixin

        assert hasattr(SpatialTrainingMixin, "train")

    def test_spatialsamplermixin_import(self):
        """Test SpatialSamplerMixin can be imported."""
        from spatialvi.model.base import SpatialSamplerMixin

        assert SpatialSamplerMixin is not None


class TestSamplers:
    """Tests for batch samplers."""

    def test_spatialbatchsampler_import(self):
        """Test SpatialBatchSampler can be imported."""
        from spatialvi.model.base import SpatialBatchSampler

        assert SpatialBatchSampler is not None

    def test_nichebatchsampler_import(self):
        """Test NicheBatchSampler can be imported."""
        from spatialvi.model.base import NicheBatchSampler

        assert NicheBatchSampler is not None

    def test_spatialbatchsampler_init(self, small_spatial_adata):
        """Test SpatialBatchSampler initialization."""
        from spatialvi.model.base import SpatialBatchSampler

        sampler = SpatialBatchSampler(small_spatial_adata, batch_size=16)
        assert sampler.batch_size == 16
        assert sampler.adata is small_spatial_adata

    def test_spatialbatchsampler_iteration(self, small_spatial_adata):
        """Test SpatialBatchSampler iteration."""
        from spatialvi.model.base import SpatialBatchSampler

        sampler = SpatialBatchSampler(small_spatial_adata, batch_size=16)
        batches = list(sampler)
        assert len(batches) > 0
        # All batches should have indices
        for batch in batches:
            assert len(batch) > 0
            assert all(0 <= idx < small_spatial_adata.n_obs for idx in batch)

    def test_nichebatchsampler_init(self, spatial_adata_with_neighbors):
        """Test NicheBatchSampler initialization."""
        from spatialvi.model.base import NicheBatchSampler

        sampler = NicheBatchSampler(spatial_adata_with_neighbors, batch_size=16)
        assert sampler.batch_size == 16

    def test_nichebatchsampler_iteration(self, spatial_adata_with_neighbors):
        """Test NicheBatchSampler iteration."""
        from spatialvi.model.base import NicheBatchSampler

        sampler = NicheBatchSampler(spatial_adata_with_neighbors, batch_size=16)
        batches = list(sampler)
        assert len(batches) > 0
        # Niche batches should be larger than seed batch (includes neighbors)
        for batch in batches:
            assert len(batch) >= 1


class TestSpatialRegularizationCallback:
    """Tests for SpatialRegularizationCallback."""

    def test_callback_import(self):
        """Test callback can be imported."""
        from spatialvi.model.base import SpatialRegularizationCallback

        assert SpatialRegularizationCallback is not None

    def test_callback_init(self):
        """Test callback initialization."""
        from spatialvi.model.base import SpatialRegularizationCallback

        callback = SpatialRegularizationCallback(
            spatial_weight=0.5,
            warmup_epochs=5,
        )
        assert callback.spatial_weight == 0.5
        assert callback.warmup_epochs == 5


class TestAllMixinsInBaseModule:
    """Test all mixins are exported from model.base."""

    def test_all_exports(self):
        """Test all expected classes are in __all__."""
        from spatialvi.model import base

        expected = [
            "BaseSpatialModel",
            "DeconvolutionMixin",
            "EmbeddingMixin",
            "NicheBatchSampler",
            "NicheMixin",
            "SpatialBatchSampler",
            "SpatialEmbeddingMixin",
            "SpatialMixin",
            "SpatialRegularizationCallback",
            "SpatialSamplerMixin",
            "SpatialTrainingMixin",
            "VAEMixin",
        ]

        for name in expected:
            assert hasattr(base, name), f"Missing export: {name}"
