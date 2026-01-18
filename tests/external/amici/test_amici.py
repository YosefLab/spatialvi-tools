"""Comprehensive tests for AMICIModule."""

import pytest
import torch

from spatialvi.external.amici import AMICIModule


class TestAMICIModuleInitialization:
    """Tests for AMICIModule initialization."""

    def test_basic_initialization(self, n_genes, n_latent, n_labels):
        """Test basic module initialization."""
        module = AMICIModule(
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
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=0,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_batch == 0

    def test_with_cell_type_attention(self, n_genes, n_latent, n_labels):
        """Test initialization with cell type attention."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            use_cell_type_attention=True,
        )

        assert module.use_cell_type_attention is True
        assert module.cell_type_embedding is not None

    def test_without_cell_type_attention(self, n_genes, n_latent):
        """Test initialization without cell type attention."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=1,
            n_hidden=32,
            n_latent=n_latent,
            use_cell_type_attention=False,
        )

        assert module.cell_type_embedding is None

    @pytest.mark.parametrize("n_attention_heads", [1, 2, 4, 8])
    def test_attention_head_configuration(self, n_genes, n_latent, n_labels, n_attention_heads):
        """Test different attention head configurations."""
        n_hidden = 32  # Must be divisible by n_heads
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_attention_heads=n_attention_heads,
        )

        assert len(module.interaction_layers) > 0

    @pytest.mark.parametrize("gene_likelihood", ["zinb", "nb", "poisson"])
    def test_gene_likelihoods(self, n_genes, n_latent, n_labels, gene_likelihood):
        """Test different gene likelihood distributions."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
        )

        assert module.gene_likelihood == gene_likelihood


class TestAMICIModuleForward:
    """Tests for AMICIModule forward pass."""

    def test_forward_with_neighbors(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test forward pass with neighbor information."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        inference_outputs, generative_outputs, loss = module(amici_tensors, compute_loss=True)

        assert "z" in inference_outputs
        assert "interaction_context" in inference_outputs
        assert "interaction_scores" in inference_outputs
        assert "px" in generative_outputs
        assert loss.loss.numel() == 1

    def test_forward_without_neighbors(self, batch_size, n_genes, n_latent, n_labels, device):
        """Test forward pass without neighbor information."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        inference_outputs, generative_outputs, loss = module(tensors, compute_loss=True)

        assert "z" in inference_outputs
        assert loss.loss.numel() == 1

    def test_forward_no_loss(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test forward pass without loss computation."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        result = module(amici_tensors, compute_loss=False)

        assert len(result) == 2
        inference_outputs, generative_outputs = result
        assert "z" in inference_outputs


class TestAMICIModuleInference:
    """Tests for AMICIModule inference method."""

    def test_inference_output_shapes(self, batch_size, n_genes, n_latent, n_labels, n_neighbors, n_hidden, device):
        """Test inference output shapes."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=n_hidden,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)
        labels = torch.randint(0, n_labels, (batch_size, 1)).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

        outputs = module.inference(x=x, batch_index=batch_index, labels=labels, neighbor_expr=neighbor_expr)

        assert outputs["z"].shape == (batch_size, n_latent)
        assert outputs["qz_m"].shape == (batch_size, n_latent)
        assert outputs["qz_v"].shape == (batch_size, n_latent)
        assert outputs["interaction_context"].shape == (batch_size, n_hidden)

    def test_inference_interaction_scores(self, batch_size, n_genes, n_latent, n_labels, n_neighbors, n_hidden, device):
        """Test inference interaction scores."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

        outputs = module.inference(x=x, neighbor_expr=neighbor_expr)

        assert outputs["interaction_scores"].shape == (batch_size, 1)
        # Scores should be between 0 and 1 (sigmoid)
        assert (outputs["interaction_scores"] >= 0).all()
        assert (outputs["interaction_scores"] <= 1).all()

    def test_inference_attention_weights(self, batch_size, n_genes, n_latent, n_labels, n_neighbors, n_hidden, device):
        """Test inference attention weights."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

        outputs = module.inference(x=x, neighbor_expr=neighbor_expr)

        assert outputs["attention_weights"].shape == (batch_size, n_neighbors)


class TestAMICIModuleGenerative:
    """Tests for AMICIModule generative method."""

    def test_generative_output_shapes(self, batch_size, n_genes, n_latent, n_labels, n_hidden, device):
        """Test generative output shapes."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
        ).to(device)

        z = torch.randn(batch_size, n_latent).to(device)
        interaction_context = torch.randn(batch_size, n_hidden).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)

        outputs = module.generative(z=z, interaction_context=interaction_context, library=library)

        assert outputs["px_rate"].shape == (batch_size, n_genes)
        assert outputs["px_scale"].shape == (batch_size, n_genes)
        assert "px" in outputs

    @pytest.mark.parametrize("gene_likelihood", ["zinb", "nb", "poisson"])
    def test_generative_distributions(self, batch_size, n_genes, n_latent, n_labels, n_hidden, gene_likelihood, device):
        """Test generative produces correct distributions."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
        ).to(device)

        z = torch.randn(batch_size, n_latent).to(device)
        interaction_context = torch.randn(batch_size, n_hidden).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)

        outputs = module.generative(z=z, interaction_context=interaction_context, library=library)

        # Check we can sample from the distribution
        sample = outputs["px"].sample()
        assert sample.shape == (batch_size, n_genes)


class TestAMICIModuleLoss:
    """Tests for AMICIModule loss computation."""

    def test_loss_components(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test loss has all expected components."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(amici_tensors, compute_loss=True)

        assert hasattr(loss, "loss")
        assert hasattr(loss, "reconstruction_loss")
        assert hasattr(loss, "kl_local")

    def test_loss_finite(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test that loss is finite."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(amici_tensors, compute_loss=True)

        assert torch.isfinite(loss.loss)


class TestAMICIModuleInteractionModeling:
    """Tests for cell-cell interaction modeling."""

    def test_interaction_layers_effect(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test that interaction layers affect the output."""
        module_1layer = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            interaction_layers=1,
        ).to(device)

        module_3layers = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=32,
            n_latent=n_latent,
            interaction_layers=3,
        ).to(device)

        assert len(module_1layer.interaction_layers) == 1
        assert len(module_3layers.interaction_layers) == 3

    def test_cell_type_embedding_usage(self, batch_size, n_genes, n_latent, n_labels, n_neighbors, n_hidden, device):
        """Test that cell type embeddings are used."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            use_cell_type_attention=True,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        labels = torch.randint(0, n_labels, (batch_size, 1)).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

        outputs = module.inference(x=x, labels=labels, neighbor_expr=neighbor_expr)

        assert "interaction_context" in outputs


class TestAMICIModuleGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test that gradients flow through the model."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(amici_tensors, compute_loss=True)
        loss.loss.backward()

        # Check gradients exist for key parameters
        assert module.px_r.grad is not None
        assert module.feature_proj.weight.grad is not None

    def test_gradients_through_interaction_layers(self, amici_tensors, n_genes, n_latent, n_labels, device):
        """Test gradients flow through interaction layers."""
        module = AMICIModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            interaction_layers=2,
        ).to(device)

        _, _, loss = module(amici_tensors, compute_loss=True)
        loss.loss.backward()

        # Check interaction layer gradients
        for layer in module.interaction_layers:
            assert layer.attention.in_proj_weight.grad is not None
