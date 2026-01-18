"""Tests for AMICI module."""

import pytest
import torch

from spatialvi.external.amici import AMICIModule


class TestAMICIModule:
    """Tests for AMICIModule."""

    @pytest.fixture
    def module(self, n_genes):
        """Create an AMICIModule instance."""
        return AMICIModule(
            n_input=n_genes,
            n_labels=5,
            n_batch=3,
            n_hidden=64,
            n_latent=10,
            n_layers=2,
            n_attention_heads=4,
            dropout_rate=0.1,
            gene_likelihood="nb",
            use_cell_type_attention=True,
            interaction_layers=2,
        )

    def test_initialization(self, module, n_genes):
        """Test that AMICIModule initializes correctly."""
        assert module.n_input == n_genes
        assert module.n_labels == 5
        assert module.n_batch == 3
        assert module.n_latent == 10
        assert module.gene_likelihood == "nb"

    def test_inference_without_neighbors(self, module, random_expression, random_batch_index, random_labels):
        """Test inference without neighbor information."""
        outputs = module.inference(
            x=random_expression,
            batch_index=random_batch_index,
            labels=random_labels,
            neighbor_indices=None,
            neighbor_expr=None,
        )

        assert "z" in outputs
        assert "qz_m" in outputs
        assert "qz_v" in outputs
        assert "library" in outputs
        assert "interaction_context" in outputs
        assert outputs["z"].shape[1] == module.n_latent

    def test_inference_with_neighbors(
        self,
        module,
        random_expression,
        random_batch_index,
        random_labels,
        random_neighbor_expression,
    ):
        """Test inference with neighbor information."""
        outputs = module.inference(
            x=random_expression,
            batch_index=random_batch_index,
            labels=random_labels,
            neighbor_indices=None,
            neighbor_expr=random_neighbor_expression,
        )

        assert "z" in outputs
        assert "interaction_scores" in outputs
        assert "attention_weights" in outputs

    def test_generative(self, module, batch_size, n_latent, embed_dim=64):
        """Test generative network."""
        z = torch.randn(batch_size, n_latent)
        interaction_context = torch.randn(batch_size, 64)  # n_hidden
        library = torch.rand(batch_size, 1).abs() + 1

        outputs = module.generative(
            z=z,
            interaction_context=interaction_context,
            library=library,
        )

        assert "px" in outputs
        assert "px_rate" in outputs
        assert "px_scale" in outputs

    def test_forward(
        self,
        module,
        random_expression,
        random_batch_index,
        random_labels,
    ):
        """Test full forward pass."""
        tensors = {
            "X": random_expression,
            "batch": random_batch_index,
            "labels": random_labels,
        }

        inference_outputs, generative_outputs, losses = module(tensors, compute_loss=True)

        assert losses.loss is not None
        assert losses.reconstruction_loss is not None
        assert losses.kl_local is not None

    def test_loss_computation(
        self,
        module,
        random_expression,
        random_batch_index,
        random_labels,
    ):
        """Test loss computation."""
        tensors = {
            "X": random_expression,
            "batch": random_batch_index,
            "labels": random_labels,
        }

        inference_outputs, generative_outputs, losses = module(tensors, compute_loss=True)

        # Loss should be a scalar
        assert losses.loss.dim() == 0
        # Loss should be positive
        assert losses.loss.item() > 0

    def test_different_gene_likelihoods(self, n_genes):
        """Test different gene likelihood distributions."""
        for likelihood in ["nb", "zinb", "poisson"]:
            module = AMICIModule(
                n_input=n_genes,
                gene_likelihood=likelihood,
            )
            assert module.gene_likelihood == likelihood


class TestAMICIModuleGradients:
    """Test gradient flow in AMICIModule."""

    @pytest.fixture
    def module(self, n_genes):
        """Create an AMICIModule instance."""
        return AMICIModule(n_input=n_genes, n_latent=10)

    def test_gradients_flow(self, module, random_expression):
        """Test that gradients flow through the model."""
        tensors = {"X": random_expression}
        random_expression.requires_grad = False

        _, _, losses = module(tensors, compute_loss=True)
        losses.loss.backward()

        # Check that gradients exist for key parameters (decoder and z_encoder)
        # Note: l_encoder.var_encoder may not get gradients as library variance
        # is not always used in the loss path
        params_with_grads = []
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grads.append(name)

        # Key components should have gradients
        assert any("decoder" in name for name in params_with_grads), "No gradient for decoder"
        assert any("z_encoder" in name for name in params_with_grads), "No gradient for z_encoder"
