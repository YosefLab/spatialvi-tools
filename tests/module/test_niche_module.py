"""Comprehensive tests for NicheModule."""

import pytest
import torch

from spatialvi.module import NicheModule


class TestNicheModuleInitialization:
    """Tests for NicheModule initialization."""

    def test_basic_initialization(self, n_genes, n_latent, n_labels):
        """Test basic module initialization."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_input == n_genes
        assert module.n_labels == n_labels
        assert module.n_latent == n_latent

    def test_initialization_without_batch(self, n_genes, n_latent, n_labels):
        """Test initialization without batch information."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=0,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_batch == 0

    def test_with_attention(self, n_genes, n_latent, n_labels):
        """Test initialization with attention mechanism."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            use_attention=True,
        )

        assert module.use_attention is True
        assert hasattr(module, "neighbor_attention")

    def test_without_attention(self, n_genes, n_latent, n_labels):
        """Test initialization without attention mechanism."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            use_attention=False,
        )

        assert module.use_attention is False

    def test_niche_factors_configuration(self, n_genes, n_latent, n_labels):
        """Test niche factors configuration."""
        n_niche_factors = 10
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            n_niche_factors=n_niche_factors,
        )

        assert module.n_niche_factors == n_niche_factors

    def test_classifier_shape(self, n_genes, n_latent, n_labels):
        """Test classifier output shape."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.classifier.out_features == n_labels
        assert module.classifier.in_features == n_latent



class TestNicheModuleForward:
    """Tests for NicheModule forward pass."""

    def test_forward_with_niche_composition(
        self, niche_tensors, n_genes, n_latent, n_labels, device
    ):
        """Test forward pass with niche composition."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        inference_outputs, generative_outputs, loss = module(
            niche_tensors, compute_loss=True
        )

        assert "z" in inference_outputs
        assert "niche_z" in inference_outputs
        assert "logits" in inference_outputs
        assert loss.loss.numel() == 1

    def test_forward_without_niche_composition(
        self, batch_size, n_genes, n_latent, n_labels, device
    ):
        """Test forward pass without niche composition."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_attention=False,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        inference_outputs, generative_outputs, loss = module(tensors, compute_loss=True)

        assert "z" in inference_outputs
        assert "niche_z" in inference_outputs
        # niche_z should be zeros when no niche composition is provided
        assert inference_outputs["niche_z"].shape[0] == batch_size

    def test_forward_no_loss(self, niche_tensors, n_genes, n_latent, n_labels, device):
        """Test forward pass without loss computation."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        result = module(niche_tensors, compute_loss=False)

        assert len(result) == 2
        inference_outputs, generative_outputs = result
        assert "z" in inference_outputs
        assert "niche_z" in inference_outputs



class TestNicheModuleInference:
    """Tests for NicheModule inference method."""

    def test_inference_output_shapes(
        self, batch_size, n_genes, n_latent, n_labels, device
    ):
        """Test inference output shapes."""
        n_niche_factors = 5
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_niche_factors=n_niche_factors,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)
        niche_comp = torch.softmax(torch.randn(batch_size, n_labels), dim=-1).to(device)

        outputs = module.inference(
            x=x, batch_index=batch_index, niche_composition=niche_comp
        )

        assert outputs["z"].shape == (batch_size, n_latent)
        assert outputs["niche_z"].shape == (batch_size, n_niche_factors)
        assert outputs["logits"].shape == (batch_size, n_labels)
        assert outputs["library"].shape == (batch_size, 1)

    def test_inference_with_attention_neighbors(
        self, batch_size, n_genes, n_latent, n_labels, n_neighbors, n_hidden, device
    ):
        """Test inference with attention over neighbors."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            use_attention=True,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

        outputs = module.inference(x=x, neighbor_expr=neighbor_expr)

        assert outputs["z"].shape == (batch_size, n_latent)
        assert "niche_z" in outputs


class TestNicheModuleGenerative:
    """Tests for NicheModule generative method."""

    def test_generative_output_shapes(
        self, batch_size, n_genes, n_latent, n_labels, device
    ):
        """Test generative output shapes."""
        n_niche_factors = 5
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            n_niche_factors=n_niche_factors,
        ).to(device)

        z = torch.randn(batch_size, n_latent).to(device)
        niche_z = torch.randn(batch_size, n_niche_factors).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)

        outputs = module.generative(z=z, niche_z=niche_z, library=library)

        assert outputs["px_rate"].shape == (batch_size, n_genes)
        assert outputs["px_scale"].shape == (batch_size, n_genes)

    def test_px_rate_non_negative(
        self, batch_size, n_genes, n_latent, n_labels, device
    ):
        """Test that px_rate is non-negative."""
        n_niche_factors = 5
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            n_niche_factors=n_niche_factors,
        ).to(device)

        z = torch.randn(batch_size, n_latent).to(device)
        niche_z = torch.randn(batch_size, n_niche_factors).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)

        outputs = module.generative(z=z, niche_z=niche_z, library=library)

        assert (outputs["px_rate"] >= 0).all()



class TestNicheModuleLoss:
    """Tests for NicheModule loss computation."""

    def test_loss_components(self, niche_tensors, n_genes, n_latent, n_labels, device):
        """Test loss has all expected components."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(niche_tensors, compute_loss=True)

        assert hasattr(loss, "loss")
        assert hasattr(loss, "reconstruction_loss")
        assert hasattr(loss, "kl_local")
        assert "classification_loss" in loss.extra_metrics

    def test_loss_with_labels(self, niche_tensors, n_genes, n_latent, n_labels, device):
        """Test loss includes classification when labels provided."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(niche_tensors, compute_loss=True)

        # Classification loss should be non-zero when labels are provided
        assert "classification_loss" in loss.extra_metrics

    def test_loss_without_labels(
        self, batch_size, n_genes, n_latent, n_labels, device
    ):
        """Test loss without classification labels."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)
        niche_comp = torch.softmax(torch.randn(batch_size, n_labels), dim=-1).to(device)

        tensors = {"X": x, "batch": batch_index, "niche_composition": niche_comp}

        _, _, loss = module(tensors, compute_loss=True)

        assert torch.isfinite(loss.loss)

    def test_loss_finite(self, niche_tensors, n_genes, n_latent, n_labels, device):
        """Test that loss is finite."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(niche_tensors, compute_loss=True)

        assert torch.isfinite(loss.loss)



class TestNicheModuleCellTypePrediction:
    """Tests for cell type prediction functionality."""

    def test_logits_shape(self, niche_tensors, n_genes, n_latent, n_labels, batch_size, device):
        """Test cell type logits shape."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        inference_outputs, _, _ = module(niche_tensors, compute_loss=True)

        assert inference_outputs["logits"].shape == (batch_size, n_labels)

    def test_predicted_probabilities(
        self, niche_tensors, n_genes, n_latent, n_labels, batch_size, device
    ):
        """Test that logits can be converted to probabilities."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        inference_outputs, _, _ = module(niche_tensors, compute_loss=True)

        probs = torch.softmax(inference_outputs["logits"], dim=-1)

        assert probs.shape == (batch_size, n_labels)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, device=device))
        assert (probs >= 0).all()



class TestNicheModuleGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self, niche_tensors, n_genes, n_latent, n_labels, device):
        """Test that gradients flow through the model."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(niche_tensors, compute_loss=True)
        loss.loss.backward()

        # Check gradients exist for key parameters
        assert module.px_r.grad is not None
        assert module.classifier.weight.grad is not None

    def test_gradients_with_attention(
        self, batch_size, n_genes, n_latent, n_labels, n_neighbors, n_hidden, device
    ):
        """Test gradients flow with attention mechanism."""
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=n_hidden,
            n_latent=n_latent,
            use_attention=True,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

        tensors = {"X": x, "batch": batch_index, "neighbor_expr": neighbor_expr}

        _, _, loss = module(tensors, compute_loss=True)
        loss.loss.backward()

        # Check attention parameters have gradients
        assert module.neighbor_attention.in_proj_weight.grad is not None
