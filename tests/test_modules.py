"""Legacy tests for neural network modules.

Note: Comprehensive module tests are now in tests/module/ directory:
- tests/module/test_spatial_vae_module.py
- tests/module/test_deconv_module.py
- tests/module/test_niche_module.py

This file contains basic smoke tests for quick validation.
"""

import torch


class TestSpatialVAEModule:
    """Basic tests for SpatialVAE module."""

    def test_initialization(self, n_genes, n_latent):
        """Test module initialization."""
        from spatialvi.module import SpatialVAEModule

        module = SpatialVAEModule(
            n_input=n_genes,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            n_layers=1,
        )

        assert module.n_input == n_genes
        assert module.n_latent == n_latent

    def test_forward(self, batch_size, n_genes, n_latent, device):
        """Test forward pass."""
        from spatialvi.module import SpatialVAEModule

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
        assert "px" in generative_outputs
        assert loss.loss.numel() == 1


class TestNicheModule:
    """Basic tests for Niche module."""

    def test_initialization(self, n_genes, n_latent):
        """Test module initialization."""
        from spatialvi.module import NicheModule

        module = NicheModule(
            n_input=n_genes,
            n_labels=5,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_labels == 5
        assert module.n_latent == n_latent

    def test_forward(self, batch_size, n_genes, n_latent, device):
        """Test forward pass."""
        from spatialvi.module import NicheModule

        n_labels = 5
        module = NicheModule(
            n_input=n_genes,
            n_labels=n_labels,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)
        niche_composition = torch.softmax(torch.randn(batch_size, n_labels), dim=-1).to(device)

        tensors = {
            "X": x,
            "batch": batch_index,
            "niche_composition": niche_composition,
        }

        inference_outputs, generative_outputs, loss = module(tensors, compute_loss=True)

        assert "z" in inference_outputs
        assert "niche_z" in inference_outputs
        assert loss.loss.numel() == 1


class TestDeconvolutionModule:
    """Basic tests for Deconvolution module."""

    def test_initialization(self, n_genes, n_latent):
        """Test module initialization."""
        from spatialvi.module import DeconvolutionModule

        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=5,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_cell_types == 5

    def test_forward(self, batch_size, n_genes, n_latent, device):
        """Test forward pass."""
        from spatialvi.module import DeconvolutionModule

        n_cell_types = 5
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        tensors = {"X": x, "batch": batch_index}

        inference_outputs, generative_outputs, loss = module(tensors, compute_loss=True)

        assert "proportions" in inference_outputs
        assert inference_outputs["proportions"].shape == (batch_size, n_cell_types)
        # Proportions should sum to 1
        prop_sums = inference_outputs["proportions"].sum(dim=-1)
        assert torch.allclose(prop_sums, torch.ones_like(prop_sums), atol=1e-4)
