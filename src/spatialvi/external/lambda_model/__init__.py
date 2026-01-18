"""LAMBDA model for LLM-based cell type annotation.

Lambda uses large language models to annotate cell types
based on marker gene signatures.

This module provides the Lambda wrapper class and supporting components
for LLM-based annotation pipelines.
"""

from __future__ import annotations

from ._model import Lambda
from ._module import (
    AnnotationConfig,
    ClusterAnnotation,
    LambdaAnnotator,
    PromptTemplate,
)
from ._utils import (
    aggregate_cluster_annotations,
    compute_marker_genes,
    create_annotation_prompt,
    format_gene_list_for_llm,
    parse_llm_response,
    validate_cell_type_name,
)

__all__ = [
    # Model wrapper
    "Lambda",
    # Module components
    "AnnotationConfig",
    "ClusterAnnotation",
    "PromptTemplate",
    "LambdaAnnotator",
    # Utility functions
    "aggregate_cluster_annotations",
    "compute_marker_genes",
    "create_annotation_prompt",
    "format_gene_list_for_llm",
    "parse_llm_response",
    "validate_cell_type_name",
]
