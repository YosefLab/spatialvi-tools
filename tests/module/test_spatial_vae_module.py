"""Comprehensive tests for SpatialVAEModule."""

import pytest
import torch

from spatialvi.module import SpatialVAEModule


class TestSpatialVAEModuleInitialization:
    """Tests for SpatialVAEModule initialization."""

    def test_basic_initialization(self, n_genes, n_latent):
        """Test basic module initialization."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_layers=1,
        )

        assert module.n_input == n_genes
        assert module.n_latent == n_latent
        assert module.n_batch == 2

    def test_initialization_without_batch(self, n_genes, n_latent):
        """Test initialization without batch information."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=0,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_batch == 0

    @pytest.mark.parametrize("dispersion", ["gene", "gene-batch"])
    def test_dispersion_types(self, n_genes, n_latent, dispersion):
        """Test different dispersion parameter types."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            dispersion=dispersion,
        )

        assert module.dispersion == dispersion
        if dispersion == "gene":
            assert module.px_r.shape == (n_genes,)
        elif dispersion == "gene-batch":
            assert module.px_r.shape == (n_genes, 2)

    @pytest.mark.parametrize("gene_likelihood", ["zinb", "nb", "poisson"])
    def test_gene_likelihoods(self, n_genes, n_latent, gene_likelihood):
        """Test different gene likelihood distributions."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
        )

        assert module.gene_likelihood == gene_likelihood

    def test_spatial_encoder_enabled(self, n_genes, n_latent):
        """Test initialization with spatial encoder."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=True,
        )

        assert module.use_spatial is True
        assert hasattr(module.z_encoder, "forward_spatial")

    def test_standard_encoder_fallback(self, n_genes, n_latent):
        """Test initialization with standard encoder."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        )

        assert module.use_spatial is False

    def test_with_continuous_covariates(self, n_genes, n_latent):
        """Test initialization with continuous covariates."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_continuous_cov=3,
        )

        assert module.n_continuous_cov == 3

    def test_with_categorical_covariates(self, n_genes, n_latent):
        """Test initialization with categorical covariates."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_cats_per_cov=[3, 5],
        )

        assert module.n_cats_per_cov == [3, 5]


class TestSpatialVAEModuleForward:
    """Tests for SpatialVAEModule forward pass."""

    def test_forward_without_spatial(self, batch_size, n_genes, n_latent, device):
        """Test forward pass without spatial context."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_layers=1,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        inference_outputs, generative_outputs, loss = module(tensors, compute_loss=True)

        assert "z" in inference_outputs
        assert "qz_m" in inference_outputs
        assert "qz_v" in inference_outputs
        assert "library" in inference_outputs
        assert "px" in generative_outputs
        assert "px_rate" in generative_outputs
        assert loss.loss.numel() == 1

    def test_forward_with_spatial(self, spatial_vae_tensors, n_genes, n_latent, device):
        """Test forward pass with spatial context."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_layers=1,
            use_spatial=True,
        ).to(device)

        inference_outputs, generative_outputs, loss = module(spatial_vae_tensors, compute_loss=True)

        assert "z" in inference_outputs
        assert inference_outputs["z"].shape[-1] == n_latent
        assert loss.loss.numel() == 1

    def test_forward_no_loss(self, batch_size, n_genes, n_latent, device):
        """Test forward pass without loss computation."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        result = module(tensors, compute_loss=False)

        assert len(result) == 2
        inference_outputs, generative_outputs = result
        assert "z" in inference_outputs
        assert "px" in generative_outputs


class TestSpatialVAEModuleInference:
    """Tests for SpatialVAEModule inference method."""

    def test_inference_output_shapes(self, batch_size, n_genes, n_latent, device):
        """Test inference output shapes."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        outputs = module.inference(x=x, batch_index=batch_index)

        assert outputs["z"].shape == (batch_size, n_latent)
        assert outputs["qz_m"].shape == (batch_size, n_latent)
        assert outputs["qz_v"].shape == (batch_size, n_latent)
        assert outputs["library"].shape == (batch_size, 1)

    def test_inference_with_spatial_context(self, batch_size, n_genes, n_latent, n_neighbors, device):
        """Test inference with spatial context."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=True,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)
        neighbor_indices = torch.randint(0, batch_size, (batch_size, n_neighbors)).to(device)

        outputs = module.inference(
            x=x,
            batch_index=batch_index,
            neighbor_indices=neighbor_indices,
            neighbor_expr=neighbor_expr,
        )

        assert outputs["z"].shape == (batch_size, n_latent)


class TestSpatialVAEModuleGenerative:
    """Tests for SpatialVAEModule generative method."""

    def test_generative_output_shapes(self, batch_size, n_genes, n_latent, device):
        """Test generative output shapes."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        z = torch.randn(batch_size, n_latent).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        outputs = module.generative(z=z, library=library, batch_index=batch_index)

        assert outputs["px_rate"].shape == (batch_size, n_genes)
        assert outputs["px_scale"].shape == (batch_size, n_genes)
        assert "px" in outputs

    @pytest.mark.parametrize("gene_likelihood", ["zinb", "nb", "poisson"])
    def test_generative_distributions(self, batch_size, n_genes, n_latent, gene_likelihood, device):
        """Test generative produces correct distributions."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
        ).to(device)

        z = torch.randn(batch_size, n_latent).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        outputs = module.generative(z=z, library=library, batch_index=batch_index)

        # Check we can sample from the distribution
        sample = outputs["px"].sample()
        assert sample.shape == (batch_size, n_genes)


class TestSpatialVAEModuleLoss:
    """Tests for SpatialVAEModule loss computation."""

    def test_loss_components(self, batch_size, n_genes, n_latent, device):
        """Test loss has all expected components."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        _, _, loss = module(tensors, compute_loss=True)

        assert hasattr(loss, "loss")
        assert hasattr(loss, "reconstruction_loss")
        assert hasattr(loss, "kl_local")
        # scvi LossOutput wraps tensors in dicts
        reconst = loss.reconstruction_loss["reconstruction_loss"]
        kl = loss.kl_local["kl_local"]
        assert reconst.shape == (batch_size,)
        assert kl.shape == (batch_size,)

    def test_loss_with_kl_weight(self, batch_size, n_genes, n_latent, device):
        """Test loss with different KL weights."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        _, _, loss_w1 = module(tensors, compute_loss=True, loss_kwargs={"kl_weight": 1.0})
        _, _, loss_w0 = module(tensors, compute_loss=True, loss_kwargs={"kl_weight": 0.0})

        # Loss with kl_weight=0 should be smaller (only reconstruction)
        assert loss_w0.loss < loss_w1.loss

    def test_spatial_loss_component(self, spatial_vae_tensors, n_genes, n_latent, device):
        """Test spatial loss is included when using spatial context."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=True,
            spatial_weight=1.0,
        ).to(device)

        _, _, loss = module(spatial_vae_tensors, compute_loss=True)

        assert "spatial_loss" in loss.extra_metrics


class TestSpatialVAEModuleSampling:
    """Tests for SpatialVAEModule sampling."""

    def test_sample_output_shape(self, batch_size, n_genes, n_latent, device):
        """Test sample method output shape."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        samples = module.sample(tensors, n_samples=5)

        assert samples.shape == (5, batch_size, n_genes)

    def test_sample_non_negative(self, batch_size, n_genes, n_latent, device):
        """Test that samples are non-negative (count data)."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            gene_likelihood="nb",
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        samples = module.sample(tensors, n_samples=3)

        assert (samples >= 0).all()


class TestSpatialVAEModuleGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self, batch_size, n_genes, n_latent, device):
        """Test that gradients flow through the model."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        _, _, loss = module(tensors, compute_loss=True)
        loss.loss.backward()

        # Check gradients exist for key parameters
        assert module.px_r.grad is not None
        # Check decoder has gradients (DecoderSCVI has px_decoder.fc_layers structure)
        has_decoder_grads = any(p.grad is not None for name, p in module.decoder.named_parameters() if p.requires_grad)
        assert has_decoder_grads, "No gradients for decoder parameters"

    def test_gradients_with_spatial(self, spatial_vae_tensors, n_genes, n_latent, device):
        """Test gradients flow with spatial encoder."""
        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_spatial=True,
        ).to(device)

        _, _, loss = module(spatial_vae_tensors, compute_loss=True)
        loss.loss.backward()

        # Check spatial encoder gradients
        assert module.z_encoder.encoder.fc_layers[0][0].weight.grad is not None
