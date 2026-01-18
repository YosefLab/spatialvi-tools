"""Tests for Harreman module components (HarremanModule, MetabolicExchangeScorer)."""

import torch


class TestHarremanModuleInitialization:
    """Tests for HarremanModule initialization."""

    def test_basic_initialization(self):
        """Test basic module initialization with defaults."""
        from spatialvi.external.harreman import HarremanModule

        module = HarremanModule(n_genes=100)

        assert module.n_genes == 100
        assert module.n_neighbors == 20  # Default
        assert module.correlation_method == "pearson"  # Default

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        from spatialvi.external.harreman import HarremanModule

        module = HarremanModule(
            n_genes=50,
            n_neighbors=15,
            correlation_method="spearman",
        )

        assert module.n_genes == 50
        assert module.n_neighbors == 15
        assert module.correlation_method == "spearman"

    def test_is_torch_module(self):
        """Test that HarremanModule is a PyTorch module."""
        from spatialvi.external.harreman import HarremanModule

        module = HarremanModule(n_genes=100)

        assert isinstance(module, torch.nn.Module)


class TestHarremanModuleForward:
    """Tests for HarremanModule forward pass."""

    def test_compute_neighbor_expression(self):
        """Test neighbor expression computation."""
        from spatialvi.external.harreman import HarremanModule

        n_cells = 32
        n_genes = 20
        n_neighbors = 5

        module = HarremanModule(n_genes=n_genes, n_neighbors=n_neighbors)

        X = torch.randn(n_cells, n_genes)
        neighbor_indices = torch.randint(0, n_cells, (n_cells, n_neighbors))

        neighbor_expr = module.compute_neighbor_expression(X, neighbor_indices)

        assert neighbor_expr.shape == (n_cells, n_genes)

    def test_compute_spatial_correlation_pearson(self):
        """Test Pearson spatial correlation computation."""
        from spatialvi.external.harreman import HarremanModule

        n_cells = 100
        n_genes = 20

        module = HarremanModule(n_genes=n_genes, correlation_method="pearson")

        X = torch.randn(n_cells, n_genes)
        neighbor_mean = torch.randn(n_cells, n_genes)

        corr = module.compute_spatial_correlation(X, neighbor_mean)

        assert corr.shape == (n_genes,)
        # Correlations should be between -1 and 1
        assert (corr >= -1.0).all()
        assert (corr <= 1.0).all()

    def test_compute_spatial_correlation_spearman(self):
        """Test Spearman spatial correlation computation."""
        from spatialvi.external.harreman import HarremanModule

        n_cells = 100
        n_genes = 20

        module = HarremanModule(n_genes=n_genes, correlation_method="spearman")

        X = torch.randn(n_cells, n_genes)
        neighbor_mean = torch.randn(n_cells, n_genes)

        corr = module.compute_spatial_correlation(X, neighbor_mean)

        assert corr.shape == (n_genes,)
        # Correlations should be between -1 and 1
        assert (corr >= -1.0).all()
        assert (corr <= 1.0).all()

    def test_forward_pass(self):
        """Test complete forward pass."""
        from spatialvi.external.harreman import HarremanModule

        n_cells = 32
        n_genes = 20
        n_neighbors = 5

        module = HarremanModule(n_genes=n_genes, n_neighbors=n_neighbors)

        X = torch.randn(n_cells, n_genes)
        neighbor_indices = torch.randint(0, n_cells, (n_cells, n_neighbors))

        outputs = module(X, neighbor_indices)

        assert "spatial_correlation" in outputs
        assert "neighbor_expression" in outputs
        assert outputs["spatial_correlation"].shape == (n_genes,)
        assert outputs["neighbor_expression"].shape == (n_cells, n_genes)

    def test_forward_with_perfect_correlation(self):
        """Test forward pass with perfectly correlated data."""
        from spatialvi.external.harreman import HarremanModule

        n_cells = 50
        n_genes = 10
        n_neighbors = 3

        module = HarremanModule(n_genes=n_genes, n_neighbors=n_neighbors)

        # Create data where each cell equals mean of neighbors (perfect correlation)
        X = torch.randn(n_cells, n_genes)
        # Use indices that include the cell itself - approximates high correlation
        neighbor_indices = torch.arange(n_cells).unsqueeze(1).repeat(1, n_neighbors)

        outputs = module(X, neighbor_indices)

        # Should get high correlations
        assert outputs["spatial_correlation"].shape == (n_genes,)


class TestMetabolicExchangeScorer:
    """Tests for MetabolicExchangeScorer class."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        scorer = MetabolicExchangeScorer(n_genes=100, n_cell_types=5)

        assert scorer.n_genes == 100
        assert scorer.n_cell_types == 5

    def test_custom_hidden_units(self):
        """Test initialization with custom hidden units."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        scorer = MetabolicExchangeScorer(
            n_genes=100,
            n_cell_types=5,
            n_hidden=128,
        )

        assert scorer.n_genes == 100
        assert scorer.n_cell_types == 5

    def test_is_torch_module(self):
        """Test that MetabolicExchangeScorer is a PyTorch module."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        scorer = MetabolicExchangeScorer(n_genes=100, n_cell_types=5)

        assert isinstance(scorer, torch.nn.Module)

    def test_has_required_components(self):
        """Test that module has required components."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        scorer = MetabolicExchangeScorer(n_genes=100, n_cell_types=5)

        assert hasattr(scorer, "exchange_encoder")
        assert hasattr(scorer, "interaction_weights")
        assert hasattr(scorer, "score_head")

    def test_forward_pass(self):
        """Test forward pass with exchange scores."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        n_genes = 20
        n_cell_types = 5
        batch_size = 32

        scorer = MetabolicExchangeScorer(n_genes=n_genes, n_cell_types=n_cell_types)

        sender_expr = torch.randn(batch_size, n_genes)
        receiver_expr = torch.randn(batch_size, n_genes)
        sender_type = torch.randint(0, n_cell_types, (batch_size,))
        receiver_type = torch.randint(0, n_cell_types, (batch_size,))

        scores = scorer(sender_expr, receiver_expr, sender_type, receiver_type)

        assert scores.shape == (batch_size, n_genes)
        # Scores should be between 0 and 1 (sigmoid output)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the scorer."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        n_genes = 20
        n_cell_types = 5
        batch_size = 16

        scorer = MetabolicExchangeScorer(n_genes=n_genes, n_cell_types=n_cell_types)

        sender_expr = torch.randn(batch_size, n_genes, requires_grad=True)
        receiver_expr = torch.randn(batch_size, n_genes, requires_grad=True)
        sender_type = torch.randint(0, n_cell_types, (batch_size,))
        receiver_type = torch.randint(0, n_cell_types, (batch_size,))

        scores = scorer(sender_expr, receiver_expr, sender_type, receiver_type)
        loss = scores.mean()
        loss.backward()

        assert sender_expr.grad is not None
        assert receiver_expr.grad is not None


class TestHarremanModuleImports:
    """Tests for module imports."""

    def test_all_exports(self):
        """Test all expected components are exported from harreman."""
        from spatialvi.external import harreman

        # Module components
        assert hasattr(harreman, "HarremanModule")
        assert hasattr(harreman, "MetabolicExchangeScorer")

        # Utility functions
        assert hasattr(harreman, "METABOLIC_PATHWAYS")
        assert hasattr(harreman, "get_metabolic_genes")
        assert hasattr(harreman, "filter_genes_by_expression")
        assert hasattr(harreman, "compute_pathway_scores")
        assert hasattr(harreman, "compute_exchange_network")
        assert hasattr(harreman, "annotate_metabolic_genes")

    def test_direct_imports(self):
        """Test direct imports work."""
        from spatialvi.external.harreman import (
            HarremanModule,
            MetabolicExchangeScorer,
        )

        assert HarremanModule is not None
        assert MetabolicExchangeScorer is not None


class TestHarremanModuleEdgeCases:
    """Tests for edge cases in Harreman modules."""

    def test_single_cell(self):
        """Test HarremanModule with single cell."""
        from spatialvi.external.harreman import HarremanModule

        n_genes = 10
        n_neighbors = 1

        module = HarremanModule(n_genes=n_genes, n_neighbors=n_neighbors)

        X = torch.randn(1, n_genes)
        neighbor_indices = torch.zeros(1, n_neighbors, dtype=torch.long)

        outputs = module(X, neighbor_indices)

        assert "spatial_correlation" in outputs
        assert "neighbor_expression" in outputs

    def test_single_gene(self):
        """Test HarremanModule with single gene."""
        from spatialvi.external.harreman import HarremanModule

        n_cells = 32
        n_genes = 1
        n_neighbors = 5

        module = HarremanModule(n_genes=n_genes, n_neighbors=n_neighbors)

        X = torch.randn(n_cells, n_genes)
        neighbor_indices = torch.randint(0, n_cells, (n_cells, n_neighbors))

        outputs = module(X, neighbor_indices)

        assert outputs["spatial_correlation"].shape == (n_genes,)

    def test_scorer_single_cell_type(self):
        """Test MetabolicExchangeScorer with single cell type."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        n_genes = 10
        n_cell_types = 1
        batch_size = 16

        scorer = MetabolicExchangeScorer(n_genes=n_genes, n_cell_types=n_cell_types)

        sender_expr = torch.randn(batch_size, n_genes)
        receiver_expr = torch.randn(batch_size, n_genes)
        sender_type = torch.zeros(batch_size, dtype=torch.long)
        receiver_type = torch.zeros(batch_size, dtype=torch.long)

        scores = scorer(sender_expr, receiver_expr, sender_type, receiver_type)

        assert scores.shape == (batch_size, n_genes)
