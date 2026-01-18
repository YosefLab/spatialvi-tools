"""Utility functions for Lambda LLM-based annotation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_marker_genes(
    adata: AnnData,
    groupby: str | None = None,
    n_genes: int = 50,
    method: str = "wilcoxon",
) -> dict[str, list[str]] | list[str]:
    """Compute marker genes for annotation.

    Parameters
    ----------
    adata
        AnnData object.
    groupby
        Key in obs for grouping (e.g., cluster labels).
    n_genes
        Number of top genes per group.
    method
        Method for marker detection.

    Returns
    -------
    Dictionary of marker genes per group, or list of highly variable genes.
    """
    import scanpy as sc

    if groupby is not None and groupby in adata.obs:
        # Compute markers per group
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)

        markers = {}
        for group in adata.obs[groupby].unique():
            markers[str(group)] = (
                adata.uns["rank_genes_groups"]["names"][group][:n_genes].tolist()
            )
        return markers
    else:
        # Return highly variable genes
        if "highly_variable" not in adata.var:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
        return adata.var_names[adata.var["highly_variable"]].tolist()[:n_genes]


def format_gene_list_for_llm(
    genes: list[str],
    expression_values: NDArray | None = None,
    max_genes: int = 100,
) -> str:
    """Format gene list for LLM prompt.

    Parameters
    ----------
    genes
        List of gene names.
    expression_values
        Optional expression values.
    max_genes
        Maximum genes to include.

    Returns
    -------
    Formatted string for LLM prompt.
    """
    genes = genes[:max_genes]

    if expression_values is not None:
        expression_values = expression_values[:max_genes]
        # Sort by expression
        sorted_idx = np.argsort(expression_values)[::-1]
        genes = [genes[i] for i in sorted_idx]
        expression_values = expression_values[sorted_idx]

        lines = []
        for gene, expr in zip(genes, expression_values):
            lines.append(f"- {gene} (expression: {expr:.2f})")
        return "\n".join(lines)
    else:
        return ", ".join(genes)


def create_annotation_prompt(
    genes: list[str],
    tissue: str | None = None,
    organism: str = "human",
    context: str | None = None,
) -> str:
    """Create annotation prompt for LLM.

    Parameters
    ----------
    genes
        List of marker genes.
    tissue
        Tissue type (e.g., "brain", "liver").
    organism
        Organism ("human" or "mouse").
    context
        Additional context.

    Returns
    -------
    Formatted prompt string.
    """
    prompt = f"""You are an expert in single-cell biology and cell type annotation.

Based on the following marker genes, identify the most likely cell type.

Marker genes: {format_gene_list_for_llm(genes, max_genes=50)}

"""
    if tissue:
        prompt += f"Tissue context: {tissue}\n"
    if organism:
        prompt += f"Organism: {organism}\n"
    if context:
        prompt += f"Additional context: {context}\n"

    prompt += """
Please provide:
1. The most likely cell type name
2. Confidence level (high/medium/low)
3. Brief reasoning based on the marker genes

Format your response as:
Cell type: [name]
Confidence: [level]
Reasoning: [brief explanation]
"""
    return prompt


def parse_llm_response(
    response: str,
) -> dict[str, str]:
    """Parse LLM annotation response.

    Parameters
    ----------
    response
        Raw LLM response text.

    Returns
    -------
    Dictionary with cell_type, confidence, and reasoning.
    """
    result = {
        "cell_type": "Unknown",
        "confidence": "low",
        "reasoning": "",
    }

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("cell type:"):
            result["cell_type"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("reasoning:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result


def aggregate_cluster_annotations(
    annotations: list[dict[str, str]],
    method: str = "majority",
) -> dict[str, str]:
    """Aggregate multiple annotations for a cluster.

    Parameters
    ----------
    annotations
        List of annotation dictionaries.
    method
        Aggregation method ("majority" or "weighted").

    Returns
    -------
    Aggregated annotation.
    """
    from collections import Counter

    cell_types = [a["cell_type"] for a in annotations]

    if method == "majority":
        type_counts = Counter(cell_types)
        most_common = type_counts.most_common(1)[0][0]
        agreement = type_counts[most_common] / len(annotations)

        return {
            "cell_type": most_common,
            "confidence": "high" if agreement > 0.8 else "medium" if agreement > 0.5 else "low",
            "reasoning": f"Majority vote ({agreement:.0%} agreement)",
        }
    else:
        # Weighted by confidence
        confidence_weights = {"high": 3, "medium": 2, "low": 1}
        weighted_counts = {}

        for a in annotations:
            ct = a["cell_type"]
            weight = confidence_weights.get(a.get("confidence", "low"), 1)
            weighted_counts[ct] = weighted_counts.get(ct, 0) + weight

        best_type = max(weighted_counts.items(), key=lambda x: x[1])[0]
        total_weight = sum(weighted_counts.values())

        return {
            "cell_type": best_type,
            "confidence": "high" if weighted_counts[best_type] / total_weight > 0.6 else "medium",
            "reasoning": "Weighted confidence vote",
        }


def validate_cell_type_name(
    cell_type: str,
    valid_types: list[str] | None = None,
) -> str:
    """Validate and standardize cell type name.

    Parameters
    ----------
    cell_type
        Raw cell type name.
    valid_types
        Optional list of valid cell types.

    Returns
    -------
    Standardized cell type name.
    """
    # Basic cleaning
    cell_type = cell_type.strip()
    cell_type = cell_type.replace("_", " ")

    if valid_types:
        # Find closest match
        from difflib import get_close_matches

        matches = get_close_matches(cell_type.lower(), [t.lower() for t in valid_types], n=1, cutoff=0.6)
        if matches:
            # Return original case version
            idx = [t.lower() for t in valid_types].index(matches[0])
            return valid_types[idx]

    return cell_type
