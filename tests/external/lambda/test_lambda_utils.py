"""Comprehensive tests for Lambda (LLM annotation) utilities."""

import numpy as np


class TestLambdaUtils:
    """Tests for Lambda utility functions."""

    def test_compute_marker_genes(self, small_spatial_adata):
        """Test marker gene computation."""
        from spatialvi.external.lambda_model import compute_marker_genes

        adata = small_spatial_adata.copy()
        adata.obs["cluster"] = np.random.choice(["Cluster1", "Cluster2", "Cluster3"], adata.n_obs)

        markers = compute_marker_genes(adata, groupby="cluster", n_genes=10)

        assert isinstance(markers, dict)
        assert len(markers) == 3
        for _cluster, genes in markers.items():
            assert isinstance(genes, list)
            assert len(genes) <= 10

    def test_compute_marker_genes_without_groupby(self, small_spatial_adata):
        """Test marker gene computation without groupby (returns HVG)."""
        from spatialvi.external.lambda_model import compute_marker_genes

        adata = small_spatial_adata.copy()

        markers = compute_marker_genes(adata, n_genes=10)

        assert isinstance(markers, list)
        assert len(markers) <= 10

    def test_format_gene_list_for_llm(self):
        """Test gene list formatting for LLM."""
        from spatialvi.external.lambda_model import format_gene_list_for_llm

        genes = ["CD4", "CD8A", "CD3D", "FOXP3", "IL2RA"]
        expression_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        formatted = format_gene_list_for_llm(genes, expression_values)

        assert isinstance(formatted, str)
        assert "CD4" in formatted
        assert "0.90" in formatted  # Expression is formatted to 2 decimals

    def test_format_gene_list_without_scores(self):
        """Test gene list formatting without scores."""
        from spatialvi.external.lambda_model import format_gene_list_for_llm

        genes = ["CD4", "CD8A", "CD3D"]

        formatted = format_gene_list_for_llm(genes)

        assert isinstance(formatted, str)
        assert "CD4" in formatted
        # Without scores, should be comma-separated
        assert "," in formatted or "CD4" in formatted

    def test_create_annotation_prompt(self):
        """Test annotation prompt creation."""
        from spatialvi.external.lambda_model import create_annotation_prompt

        genes = ["CD4", "FOXP3", "IL2RA", "CD8A", "CD8B", "GZMB"]

        prompt = create_annotation_prompt(genes, tissue="PBMC")

        assert isinstance(prompt, str)
        assert "CD4" in prompt
        assert "PBMC" in prompt

    def test_create_annotation_prompt_without_context(self):
        """Test annotation prompt without tissue context."""
        from spatialvi.external.lambda_model import create_annotation_prompt

        genes = ["CD4", "FOXP3"]

        prompt = create_annotation_prompt(genes)

        assert isinstance(prompt, str)
        assert "CD4" in prompt

    def test_create_annotation_prompt_with_organism(self):
        """Test annotation prompt with organism."""
        from spatialvi.external.lambda_model import create_annotation_prompt

        genes = ["Cd4", "Foxp3"]

        prompt = create_annotation_prompt(genes, organism="mouse")

        assert isinstance(prompt, str)
        assert "mouse" in prompt

    def test_parse_llm_response(self):
        """Test LLM response parsing."""
        from spatialvi.external.lambda_model import parse_llm_response

        response = """
        Cell type: CD4+ T regulatory cells (Tregs)
        Confidence: high
        Reasoning: Strong expression of CD4, FOXP3, and IL2RA
        """

        result = parse_llm_response(response)

        assert isinstance(result, dict)
        assert "cell_type" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert "Treg" in result["cell_type"] or "T reg" in result["cell_type"].lower()

    def test_parse_llm_response_incomplete(self):
        """Test parsing incomplete LLM response."""
        from spatialvi.external.lambda_model import parse_llm_response

        response = "Cell type: Unknown cell"

        result = parse_llm_response(response)

        assert isinstance(result, dict)
        assert result["cell_type"] == "Unknown cell"

    def test_aggregate_cluster_annotations_majority(self):
        """Test cluster annotation aggregation with majority vote."""
        from spatialvi.external.lambda_model import aggregate_cluster_annotations

        annotations = [
            {"cell_type": "T cells", "confidence": "high"},
            {"cell_type": "T cells", "confidence": "medium"},
            {"cell_type": "B cells", "confidence": "low"},
        ]

        result = aggregate_cluster_annotations(annotations, method="majority")

        assert isinstance(result, dict)
        assert result["cell_type"] == "T cells"

    def test_aggregate_cluster_annotations_weighted(self):
        """Test cluster annotation aggregation with weighted vote."""
        from spatialvi.external.lambda_model import aggregate_cluster_annotations

        annotations = [
            {"cell_type": "T cells", "confidence": "high"},  # weight 3
            {"cell_type": "B cells", "confidence": "high"},  # weight 3
            {"cell_type": "B cells", "confidence": "medium"},  # weight 2
        ]

        result = aggregate_cluster_annotations(annotations, method="weighted")

        assert isinstance(result, dict)
        # B cells should win: 3+2=5 vs T cells: 3
        assert result["cell_type"] == "B cells"

    def test_validate_cell_type_name(self):
        """Test cell type name validation/standardization."""
        from spatialvi.external.lambda_model import validate_cell_type_name

        # Test basic cleaning
        result = validate_cell_type_name("  T_cells  ")
        assert result == "T cells"

    def test_validate_cell_type_name_with_valid_types(self):
        """Test cell type name validation with valid types list."""
        from spatialvi.external.lambda_model import validate_cell_type_name

        valid_types = ["T cells", "B cells", "NK cells", "Macrophages"]

        # Should match to T cells
        result = validate_cell_type_name("t cell", valid_types=valid_types)
        assert result == "T cells"


class TestLambdaModelWrapper:
    """Tests for Lambda model wrapper."""

    def test_import(self):
        """Test that Lambda wrapper can be imported."""
        from spatialvi.external.lambda_model import Lambda

        assert Lambda is not None

    def test_utils_import(self):
        """Test that all utils can be imported."""
        from spatialvi.external.lambda_model import (
            aggregate_cluster_annotations,
            compute_marker_genes,
            create_annotation_prompt,
            format_gene_list_for_llm,
            parse_llm_response,
            validate_cell_type_name,
        )

        assert compute_marker_genes is not None
        assert create_annotation_prompt is not None
        assert format_gene_list_for_llm is not None
        assert parse_llm_response is not None
        assert aggregate_cluster_annotations is not None
        assert validate_cell_type_name is not None
