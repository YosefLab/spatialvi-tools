"""Comprehensive tests for DeconvolutionModule."""

import torch

from spatialvi.module import DeconvolutionModule


class TestDeconvolutionModuleInitialization:
    """Tests for DeconvolutionModule initialization."""

    def test_basic_initialization(self, n_genes, n_latent, n_cell_types):
        """Test basic module initialization."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_input == n_genes
        assert module.n_cell_types == n_cell_types

    def test_initialization_without_batch(self, n_genes, n_latent, n_cell_types):
        """Test initialization without batch information."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=0,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.n_batch == 0

    def test_cell_type_profiles_shape(self, n_genes, n_latent, n_cell_types):
        """Test cell type profiles parameter shape."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
        )

        assert module.cell_type_profiles.shape == (n_cell_types, n_genes)

    def test_with_subcell_variation(self, n_genes, n_latent, n_cell_types):
        """Test initialization with subcell variation modeling."""
        n_subcell_factors = 5
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
            use_subcell_variation=True,
            n_subcell_factors=n_subcell_factors,
        )

        assert module.use_subcell_variation is True
        assert module.n_subcell_factors == n_subcell_factors
        assert len(module.subcell_decoder) == n_cell_types

    def test_without_subcell_variation(self, n_genes, n_latent, n_cell_types):
        """Test initialization without subcell variation."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
            use_subcell_variation=False,
        )

        assert module.use_subcell_variation is False
        assert not hasattr(module, "subcell_encoder") or module.subcell_encoder is None


class TestDeconvolutionModuleForward:
    """Tests for DeconvolutionModule forward pass."""

    def test_forward_basic(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test basic forward pass."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        inference_outputs, generative_outputs, loss = module(deconv_tensors, compute_loss=True)

        assert "proportions" in inference_outputs
        assert "px_rate" in generative_outputs
        assert loss.loss.numel() == 1

    def test_forward_no_loss(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test forward pass without loss computation."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        result = module(deconv_tensors, compute_loss=False)

        assert len(result) == 2
        inference_outputs, generative_outputs = result
        assert "proportions" in inference_outputs

    def test_forward_with_subcell_variation(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test forward pass with subcell variation."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_subcell_variation=True,
            n_subcell_factors=5,
        ).to(device)

        inference_outputs, generative_outputs, loss = module(deconv_tensors, compute_loss=True)

        assert "subcell_factors" in inference_outputs
        assert loss.loss.numel() == 1


class TestDeconvolutionModuleInference:
    """Tests for DeconvolutionModule inference method."""

    def test_inference_output_shapes(self, batch_size, n_genes, n_latent, n_cell_types, device):
        """Test inference output shapes."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        outputs = module.inference(x=x, batch_index=batch_index)

        assert outputs["proportions"].shape == (batch_size, n_cell_types)
        assert outputs["library"].shape == (batch_size, 1)
        assert "qc" in outputs  # Dirichlet distribution

    def test_proportions_sum_to_one(self, batch_size, n_genes, n_latent, n_cell_types, device):
        """Test that inferred proportions sum to 1."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)

        outputs = module.inference(x=x)

        prop_sums = outputs["proportions"].sum(dim=-1)
        assert torch.allclose(prop_sums, torch.ones_like(prop_sums), atol=1e-5)

    def test_proportions_non_negative(self, batch_size, n_genes, n_latent, n_cell_types, device):
        """Test that inferred proportions are non-negative."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        x = torch.randn(batch_size, n_genes).abs().to(device)

        outputs = module.inference(x=x)

        assert (outputs["proportions"] >= 0).all()


class TestDeconvolutionModuleGenerative:
    """Tests for DeconvolutionModule generative method."""

    def test_generative_output_shapes(self, batch_size, n_genes, n_latent, n_cell_types, device):
        """Test generative output shapes."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        proportions = torch.softmax(torch.randn(batch_size, n_cell_types), dim=-1).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)

        outputs = module.generative(proportions=proportions, library=library)

        assert outputs["px_rate"].shape == (batch_size, n_genes)
        assert outputs["px_scale"].shape == (batch_size, n_genes)
        assert outputs["cell_type_profiles"].shape == (batch_size, n_cell_types, n_genes)

    def test_px_rate_non_negative(self, batch_size, n_genes, n_latent, n_cell_types, device):
        """Test that px_rate is non-negative."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        proportions = torch.softmax(torch.randn(batch_size, n_cell_types), dim=-1).to(device)
        library = torch.randn(batch_size, 1).exp().to(device)

        outputs = module.generative(proportions=proportions, library=library)

        assert (outputs["px_rate"] >= 0).all()


class TestDeconvolutionModuleLoss:
    """Tests for DeconvolutionModule loss computation."""

    def test_loss_components(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test loss has all expected components."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(deconv_tensors, compute_loss=True)

        assert hasattr(loss, "loss")
        assert hasattr(loss, "reconstruction_loss")
        assert hasattr(loss, "kl_local")  # KL for Dirichlet

    def test_loss_finite(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test that loss is finite."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(deconv_tensors, compute_loss=True)

        assert torch.isfinite(loss.loss)


class TestDeconvolutionModuleCellTypeExpression:
    """Tests for cell type-specific expression retrieval."""

    def test_get_cell_type_expression_shape(self, deconv_tensors, n_genes, n_latent, n_cell_types, batch_size, device):
        """Test cell type expression output shape."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        ct_expr = module.get_cell_type_expression(deconv_tensors)

        assert ct_expr.shape == (batch_size, n_cell_types, n_genes)

    def test_get_cell_type_expression_non_negative(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test that cell type expression is non-negative."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        ct_expr = module.get_cell_type_expression(deconv_tensors)

        assert (ct_expr >= 0).all()


class TestDeconvolutionModuleGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test that gradients flow through the model."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
        ).to(device)

        _, _, loss = module(deconv_tensors, compute_loss=True)
        loss.loss.backward()

        # Check gradients exist for key parameters
        assert module.cell_type_profiles.grad is not None
        assert module.px_r.grad is not None

    def test_gradients_with_subcell_variation(self, deconv_tensors, n_genes, n_latent, n_cell_types, device):
        """Test gradients flow with subcell variation."""
        module = DeconvolutionModule(
            n_input=n_genes,
            n_cell_types=n_cell_types,
            n_batch=2,
            n_hidden=32,
            n_latent=n_latent,
            use_subcell_variation=True,
        ).to(device)

        _, _, loss = module(deconv_tensors, compute_loss=True)
        loss.loss.backward()

        # Check subcell decoder gradients
        assert module.subcell_decoder[0].weight.grad is not None
