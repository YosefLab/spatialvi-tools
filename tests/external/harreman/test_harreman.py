"""Comprehensive tests for Harreman model, module, and utilities."""

import torch


class TestHarremanModuleInitialization:
    """Tests for HarremanModule initialization."""

    def test_basic_initialization(self):
        """Test basic module initialization."""
        from spatialvi.external.harreman import HarremanModule

        module = HarremanModule(n_genes=100)

        assert module.n_genes == 100
        assert module.n_neighbors == 20  # Default
        assert module.correlation_method == "pearson"  # Default

    def test_with_custom_neighbors(self):
        """Test initialization with custom neighbor count."""
        from spatialvi.external.harreman import HarremanModule

        module = HarremanModule(
            n_genes=100,
            n_neighbors=15,
        )

        assert module.n_neighbors == 15

    def test_with_spearman_correlation(self):
        """Test initialization with Spearman correlation."""
        from spatialvi.external.harreman import HarremanModule

        module = HarremanModule(
            n_genes=100,
            correlation_method="spearman",
        )

        assert module.correlation_method == "spearman"


class TestHarremanModuleForward:
    """Tests for HarremanModule forward pass."""

    def test_forward_pass(self):
        """Test forward pass returns expected outputs."""
        from spatialvi.external.harreman import HarremanModule

        n_genes = 20
        n_cells = 32
        n_neighbors = 5

        module = HarremanModule(n_genes=n_genes, n_neighbors=n_neighbors)

        X = torch.randn(n_cells, n_genes).abs()
        neighbor_indices = torch.randint(0, n_cells, (n_cells, n_neighbors))

        outputs = module(X, neighbor_indices)

        assert "spatial_correlation" in outputs
        assert "neighbor_expression" in outputs
        assert outputs["spatial_correlation"].shape == (n_genes,)
        assert outputs["neighbor_expression"].shape == (n_cells, n_genes)


class TestMetabolicExchangeScorer:
    """Tests for MetabolicExchangeScorer class."""

    def test_initialization(self):
        """Test MetabolicExchangeScorer initialization."""
        from spatialvi.external.harreman import MetabolicExchangeScorer

        scorer = MetabolicExchangeScorer(n_genes=100, n_cell_types=5)

        assert scorer.n_genes == 100
        assert scorer.n_cell_types == 5

    def test_scoring(self):
        """Test metabolic scoring."""
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


class TestHarremanUtils:
    """Tests for Harreman utility functions."""

    def test_metabolic_pathways_dict(self):
        """Test metabolic pathways dictionary."""
        from spatialvi.external.harreman import METABOLIC_PATHWAYS

        assert isinstance(METABOLIC_PATHWAYS, dict)
        assert len(METABOLIC_PATHWAYS) > 0

        # Check structure
        for pathway, genes in METABOLIC_PATHWAYS.items():
            assert isinstance(pathway, str)
            assert isinstance(genes, list)

    def test_get_metabolic_genes_all(self):
        """Test metabolic gene retrieval for all pathways."""
        from spatialvi.external.harreman import get_metabolic_genes

        genes = get_metabolic_genes()

        assert isinstance(genes, list)
        assert len(genes) > 0

    def test_get_metabolic_genes_specific(self):
        """Test metabolic gene retrieval for specific pathways."""
        from spatialvi.external.harreman import get_metabolic_genes

        genes = get_metabolic_genes(pathways=["glycolysis"])

        assert isinstance(genes, list)
        assert len(genes) > 0
        # Should include known glycolysis genes
        assert any(g in ["GAPDH", "PKM", "LDHA"] for g in genes)

    def test_get_metabolic_genes_mouse(self):
        """Test metabolic gene retrieval for mouse."""
        from spatialvi.external.harreman import get_metabolic_genes

        genes = get_metabolic_genes(species="mouse")

        assert isinstance(genes, list)
        # Mouse genes should be capitalized
        assert all(g[0].isupper() for g in genes)

    def test_filter_genes_by_expression(self, small_spatial_adata):
        """Test gene filtering by expression."""
        from spatialvi.external.harreman import filter_genes_by_expression

        adata = small_spatial_adata.copy()
        genes = adata.var_names.tolist()

        filtered = filter_genes_by_expression(adata, genes=genes, min_cells=1, min_counts=1)

        assert isinstance(filtered, list)
        assert len(filtered) <= len(genes)

    def test_compute_pathway_scores(self, small_spatial_adata):
        """Test pathway score computation."""
        from spatialvi.external.harreman import compute_pathway_scores

        adata = small_spatial_adata.copy()
        # Use adata genes as fake pathway genes
        pathway_genes = {"set1": adata.var_names[:10].tolist()}

        scores = compute_pathway_scores(adata, pathway_genes=pathway_genes)

        assert isinstance(scores, dict)
        assert "set1" in scores
        assert len(scores["set1"]) == adata.n_obs

    def test_annotate_metabolic_genes(self):
        """Test metabolic gene annotation."""
        from spatialvi.external.harreman import annotate_metabolic_genes

        genes = ["GAPDH", "LDHA", "PKM", "UNKNOWN_GENE"]

        annotations = annotate_metabolic_genes(genes)

        assert isinstance(annotations, dict)
        assert annotations["GAPDH"] == "glycolysis"
        assert annotations["UNKNOWN_GENE"] == "unknown"


class TestHarremanModelWrapper:
    """Tests for Harreman model wrapper."""

    def test_import(self):
        """Test that Harreman wrapper can be imported."""
        from spatialvi.external.harreman import Harreman

        assert Harreman is not None

    def test_initialization(self, small_spatial_adata):
        """Test Harreman model initialization."""
        from spatialvi.external.harreman import Harreman

        adata = small_spatial_adata.copy()

        model = Harreman(adata, spatial_key="spatial")

        assert model.adata is adata
