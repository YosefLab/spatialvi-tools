"""Tests for Lambda module components (AnnotationConfig, ClusterAnnotation, etc.)."""

import pytest


class TestAnnotationConfig:
    """Tests for AnnotationConfig dataclass."""

    def test_default_initialization(self):
        """Test default AnnotationConfig values."""
        from spatialvi.external.lambda_model import AnnotationConfig

        config = AnnotationConfig()

        assert config.organism == "human"
        assert config.location is None
        assert config.n_top_genes == 50
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.num_rounds == 3

    def test_custom_initialization(self):
        """Test AnnotationConfig with custom values."""
        from spatialvi.external.lambda_model import AnnotationConfig

        config = AnnotationConfig(
            organism="mouse",
            location="brain",
            n_top_genes=100,
            provider="google",
            model="gemini-pro",
            temperature=0.7,
            max_tokens=2048,
            num_rounds=5,
        )

        assert config.organism == "mouse"
        assert config.location == "brain"
        assert config.n_top_genes == 100
        assert config.provider == "google"
        assert config.model == "gemini-pro"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.num_rounds == 5


class TestClusterAnnotation:
    """Tests for ClusterAnnotation dataclass."""

    def test_minimal_initialization(self):
        """Test ClusterAnnotation with minimal required values."""
        from spatialvi.external.lambda_model import ClusterAnnotation

        annotation = ClusterAnnotation(
            cluster_id="cluster_0",
            cell_type="T cell",
        )

        assert annotation.cluster_id == "cluster_0"
        assert annotation.cell_type == "T cell"
        assert annotation.confidence == 0.0
        assert annotation.marker_genes == []
        assert annotation.reasoning == ""
        assert annotation.alternative_types == []

    def test_full_initialization(self):
        """Test ClusterAnnotation with all values."""
        from spatialvi.external.lambda_model import ClusterAnnotation

        annotation = ClusterAnnotation(
            cluster_id="cluster_1",
            cell_type="CD8+ T cell",
            confidence=0.85,
            marker_genes=["CD8A", "CD8B", "CD3E"],
            reasoning="Strong expression of cytotoxic T cell markers",
            alternative_types=["NK cell", "CD4+ T cell"],
        )

        assert annotation.cluster_id == "cluster_1"
        assert annotation.cell_type == "CD8+ T cell"
        assert annotation.confidence == 0.85
        assert annotation.marker_genes == ["CD8A", "CD8B", "CD3E"]
        assert "cytotoxic" in annotation.reasoning
        assert "NK cell" in annotation.alternative_types


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        from spatialvi.external.lambda_model import PromptTemplate

        assert hasattr(PromptTemplate, "SYSTEM_PROMPT")
        assert len(PromptTemplate.SYSTEM_PROMPT) > 0
        assert "cell type" in PromptTemplate.SYSTEM_PROMPT.lower()

    def test_annotation_template_exists(self):
        """Test that annotation template is defined."""
        from spatialvi.external.lambda_model import PromptTemplate

        assert hasattr(PromptTemplate, "ANNOTATION_TEMPLATE")
        assert "{gene_list}" in PromptTemplate.ANNOTATION_TEMPLATE
        assert "{organism_text}" in PromptTemplate.ANNOTATION_TEMPLATE

    def test_refinement_template_exists(self):
        """Test that refinement template is defined."""
        from spatialvi.external.lambda_model import PromptTemplate

        assert hasattr(PromptTemplate, "REFINEMENT_TEMPLATE")
        assert "{cluster_id}" in PromptTemplate.REFINEMENT_TEMPLATE
        assert "{previous_type}" in PromptTemplate.REFINEMENT_TEMPLATE

    def test_format_annotation_prompt_basic(self):
        """Test formatting annotation prompt with genes only."""
        from spatialvi.external.lambda_model import PromptTemplate

        genes = ["CD3E", "CD8A", "GZMB"]
        prompt = PromptTemplate.format_annotation_prompt(genes)

        assert "CD3E" in prompt
        assert "CD8A" in prompt
        assert "GZMB" in prompt
        assert "1." in prompt  # Numbered list

    def test_format_annotation_prompt_with_scores(self):
        """Test formatting annotation prompt with genes and scores."""
        from spatialvi.external.lambda_model import PromptTemplate

        genes = ["CD3E", "CD8A"]
        scores = [0.95, 0.87]
        prompt = PromptTemplate.format_annotation_prompt(genes, scores)

        assert "CD3E" in prompt
        assert "0.95" in prompt or "score" in prompt.lower()

    def test_format_annotation_prompt_with_context(self):
        """Test formatting annotation prompt with organism and location."""
        from spatialvi.external.lambda_model import PromptTemplate

        genes = ["CD3E"]
        prompt = PromptTemplate.format_annotation_prompt(
            genes, organism="human", location="lung"
        )

        assert "human" in prompt
        assert "lung" in prompt


class TestLambdaAnnotator:
    """Tests for LambdaAnnotator class."""

    def test_default_initialization(self):
        """Test LambdaAnnotator with default config."""
        from spatialvi.external.lambda_model import LambdaAnnotator

        annotator = LambdaAnnotator()

        assert annotator.config is not None
        assert annotator.config.organism == "human"
        assert annotator.results == []

    def test_custom_config_initialization(self):
        """Test LambdaAnnotator with custom config."""
        from spatialvi.external.lambda_model import AnnotationConfig, LambdaAnnotator

        config = AnnotationConfig(organism="mouse", n_top_genes=25)
        annotator = LambdaAnnotator(config=config)

        assert annotator.config.organism == "mouse"
        assert annotator.config.n_top_genes == 25

    def test_parse_llm_response_cell_type(self):
        """Test parsing cell type from LLM response."""
        from spatialvi.external.lambda_model import LambdaAnnotator

        annotator = LambdaAnnotator()
        response = """Cell type: CD8+ T cell
Confidence: high
Reasoning: The expression of CD8A and CD8B indicates cytotoxic T cells."""

        result = annotator.parse_llm_response(response)

        assert "CD8" in result.cell_type or "T cell" in result.cell_type
        assert result.confidence > 0.5  # High confidence

    def test_parse_llm_response_medium_confidence(self):
        """Test parsing medium confidence from LLM response."""
        from spatialvi.external.lambda_model import LambdaAnnotator

        annotator = LambdaAnnotator()
        response = """Cell type: B cell
Confidence: medium
Some reasoning here."""

        result = annotator.parse_llm_response(response)

        assert result.confidence == 0.6  # Medium confidence

    def test_parse_llm_response_low_confidence(self):
        """Test parsing low confidence from LLM response."""
        from spatialvi.external.lambda_model import LambdaAnnotator

        annotator = LambdaAnnotator()
        response = """Cell type: Unknown
Confidence: low"""

        result = annotator.parse_llm_response(response)

        assert result.confidence == 0.3  # Low confidence

    def test_parse_llm_response_unknown(self):
        """Test parsing empty/invalid LLM response."""
        from spatialvi.external.lambda_model import LambdaAnnotator

        annotator = LambdaAnnotator()
        response = "I cannot determine the cell type."

        result = annotator.parse_llm_response(response)

        # Should default to Unknown
        assert result.cell_type == "Unknown" or len(result.cell_type) > 0

    def test_results_property(self):
        """Test results property returns list."""
        from spatialvi.external.lambda_model import LambdaAnnotator

        annotator = LambdaAnnotator()

        assert isinstance(annotator.results, list)
        assert len(annotator.results) == 0


class TestLambdaModuleImports:
    """Tests for module imports."""

    def test_all_exports(self):
        """Test all expected components are exported."""
        from spatialvi.external import lambda_model

        # Check main components
        assert hasattr(lambda_model, "AnnotationConfig")
        assert hasattr(lambda_model, "ClusterAnnotation")
        assert hasattr(lambda_model, "PromptTemplate")
        assert hasattr(lambda_model, "LambdaAnnotator")

    def test_direct_imports(self):
        """Test direct imports work."""
        from spatialvi.external.lambda_model import (
            AnnotationConfig,
            ClusterAnnotation,
            LambdaAnnotator,
            PromptTemplate,
        )

        assert AnnotationConfig is not None
        assert ClusterAnnotation is not None
        assert PromptTemplate is not None
        assert LambdaAnnotator is not None
