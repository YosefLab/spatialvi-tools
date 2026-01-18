"""Lambda module components.

Lambda is an LLM-based cell type annotation framework that does not use
neural network modules in the traditional sense. Instead, it leverages
large language models (like GPT) to annotate cell clusters based on
differentially expressed marker genes.

This module provides supporting components for the Lambda annotation pipeline
including text processing utilities and prompt templates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


@dataclass
class AnnotationConfig:
    """Configuration for Lambda annotation.

    Parameters
    ----------
    organism
        Species name (e.g., "human", "mouse").
    location
        Anatomical location of the sample (e.g., "brain", "liver").
    n_top_genes
        Number of top differentially expressed genes to use.
    provider
        LLM provider to use (e.g., "openai", "google").
    model
        Specific model name to use.
    temperature
        Temperature for LLM generation.
    max_tokens
        Maximum number of tokens in response.
    num_rounds
        Number of reasoning rounds for refinement.
    """

    organism: str = "human"
    location: str | None = None
    n_top_genes: int = 50
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 1024
    num_rounds: int = 3


@dataclass
class ClusterAnnotation:
    """Result of annotating a single cluster.

    Parameters
    ----------
    cluster_id
        Identifier of the cluster.
    cell_type
        Assigned cell type name.
    confidence
        Confidence score (0-1).
    marker_genes
        List of marker genes used for annotation.
    reasoning
        LLM's reasoning for the annotation.
    alternative_types
        List of alternative cell type names considered.
    """

    cluster_id: str
    cell_type: str
    confidence: float = 0.0
    marker_genes: list[str] = field(default_factory=list)
    reasoning: str = ""
    alternative_types: list[str] = field(default_factory=list)


class PromptTemplate:
    """Templates for LLM prompts in Lambda annotation.

    This class provides prompt templates for various stages of the
    Lambda annotation pipeline.
    """

    SYSTEM_PROMPT = """You are an expert biologist specializing in single-cell genomics
and cell type annotation. Your task is to identify cell types based on
differentially expressed marker genes."""

    ANNOTATION_TEMPLATE = """Given the following marker genes for a cell cluster{organism_text}{location_text}:

Top marker genes (ranked by differential expression):
{gene_list}

Please identify the most likely cell type for this cluster.

Respond with:
1. Cell type name
2. Confidence level (high/medium/low)
3. Brief reasoning based on the marker genes
4. Alternative cell types that could match (if any)"""

    REFINEMENT_TEMPLATE = """Based on your previous annotation of cluster {cluster_id}:

Previous annotation: {previous_type}
Previous reasoning: {previous_reasoning}

Additional context:
- Similar clusters were annotated as: {similar_annotations}
- Tissue context: {tissue_context}

Please refine your annotation if needed, or confirm the previous one."""

    @classmethod
    def format_annotation_prompt(
        cls,
        genes: list[str],
        scores: list[float] | None = None,
        organism: str | None = None,
        location: str | None = None,
    ) -> str:
        """Format the annotation prompt.

        Parameters
        ----------
        genes
            List of marker gene names.
        scores
            Optional list of scores for each gene.
        organism
            Species name.
        location
            Anatomical location.

        Returns
        -------
        Formatted prompt string.
        """
        organism_text = f" from {organism}" if organism else ""
        location_text = f" in {location}" if location else ""

        if scores is not None:
            gene_list = "\n".join(
                f"{i+1}. {g} (score: {s:.3f})"
                for i, (g, s) in enumerate(zip(genes, scores))
            )
        else:
            gene_list = "\n".join(f"{i+1}. {g}" for i, g in enumerate(genes))

        return cls.ANNOTATION_TEMPLATE.format(
            organism_text=organism_text,
            location_text=location_text,
            gene_list=gene_list,
        )


class LambdaAnnotator:
    """Core annotation logic for Lambda.

    This class handles the annotation pipeline including marker gene
    extraction, prompt generation, and result parsing.

    Parameters
    ----------
    config
        Annotation configuration.
    """

    def __init__(self, config: AnnotationConfig | None = None):
        self.config = config or AnnotationConfig()
        self._results: list[ClusterAnnotation] = []

    def extract_marker_genes(
        self,
        adata: "AnnData",
        cluster_key: str,
        cluster_id: str,
    ) -> tuple[list[str], list[float]]:
        """Extract marker genes for a cluster.

        Parameters
        ----------
        adata
            Annotated data matrix.
        cluster_key
            Key in obs for cluster assignments.
        cluster_id
            Cluster to extract markers for.

        Returns
        -------
        Tuple of (gene names, scores).
        """
        import scanpy as sc

        # Check if rank_genes_groups exists
        if "rank_genes_groups" not in adata.uns:
            sc.tl.rank_genes_groups(adata, cluster_key, method="wilcoxon")

        # Get genes and scores for the cluster
        try:
            genes = adata.uns["rank_genes_groups"]["names"][cluster_id]
            scores = adata.uns["rank_genes_groups"]["scores"][cluster_id]
            return (
                list(genes[: self.config.n_top_genes]),
                list(scores[: self.config.n_top_genes]),
            )
        except (KeyError, IndexError):
            logger.warning(f"Could not extract markers for cluster {cluster_id}")
            return [], []

    def parse_llm_response(self, response: str) -> ClusterAnnotation:
        """Parse LLM response into structured annotation.

        Parameters
        ----------
        response
            Raw response from the LLM.

        Returns
        -------
        Parsed annotation result.
        """
        # Simple parsing - can be made more sophisticated
        lines = response.strip().split("\n")
        cell_type = ""
        confidence = 0.5
        reasoning = ""

        for line in lines:
            line_lower = line.lower()
            if "cell type" in line_lower or (not cell_type and ":" in line):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    cell_type = parts[1].strip()
            elif "confidence" in line_lower:
                if "high" in line_lower:
                    confidence = 0.9
                elif "medium" in line_lower:
                    confidence = 0.6
                elif "low" in line_lower:
                    confidence = 0.3
            elif "reasoning" in line_lower or "because" in line_lower:
                reasoning = line

        return ClusterAnnotation(
            cluster_id="",
            cell_type=cell_type or "Unknown",
            confidence=confidence,
            reasoning=reasoning,
        )

    @property
    def results(self) -> list[ClusterAnnotation]:
        """Get annotation results."""
        return self._results
