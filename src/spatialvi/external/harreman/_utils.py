"""Utility functions for Harreman metabolic exchange analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Default metabolic gene sets by pathway
METABOLIC_PATHWAYS = {
    "glycolysis": [
        "HK1",
        "HK2",
        "GPI",
        "PFKL",
        "PFKM",
        "PFKP",
        "ALDOA",
        "ALDOB",
        "ALDOC",
        "GAPDH",
        "PGK1",
        "PGAM1",
        "PGAM2",
        "ENO1",
        "ENO2",
        "ENO3",
        "PKM",
        "PKLR",
        "LDHA",
        "LDHB",
        "LDHC",
    ],
    "tca_cycle": [
        "CS",
        "ACO1",
        "ACO2",
        "IDH1",
        "IDH2",
        "IDH3A",
        "IDH3B",
        "IDH3G",
        "OGDH",
        "OGDHL",
        "SUCLA2",
        "SUCLG1",
        "SUCLG2",
        "SDHA",
        "SDHB",
        "SDHC",
        "SDHD",
        "FH",
        "MDH1",
        "MDH2",
    ],
    "oxidative_phosphorylation": [
        "NDUFA1",
        "NDUFA2",
        "NDUFB1",
        "NDUFB2",
        "NDUFS1",
        "NDUFS2",
        "NDUFS3",
        "UQCRC1",
        "UQCRC2",
        "UQCRFS1",
        "UQCRQ",
        "COX4I1",
        "COX4I2",
        "COX5A",
        "COX5B",
        "COX6A1",
        "COX6B1",
        "COX7A1",
        "ATP5A1",
        "ATP5B",
        "ATP5C1",
        "ATP5D",
        "ATP5E",
        "ATP5F1",
    ],
    "amino_acid_metabolism": [
        "GOT1",
        "GOT2",
        "GPT",
        "GPT2",
        "GLS",
        "GLS2",
        "GLUD1",
        "GLUD2",
        "BCAT1",
        "BCAT2",
        "ASNS",
        "ASS1",
        "ASL",
    ],
    "lipid_metabolism": [
        "FASN",
        "ACACA",
        "ACACB",
        "SCD",
        "SCD5",
        "FADS1",
        "FADS2",
        "ACLY",
        "ACSS2",
        "ELOVL1",
        "ELOVL5",
        "ELOVL6",
    ],
    "transporters": [
        "SLC2A1",
        "SLC2A2",
        "SLC2A3",
        "SLC2A4",
        "SLC2A5",
        "SLC16A1",
        "SLC16A3",
        "SLC16A7",
        "SLC1A5",
        "SLC7A5",
        "SLC7A11",
        "SLC25A1",
        "SLC25A10",
        "SLC25A11",
    ],
}


def get_metabolic_genes(
    pathways: list[str] | None = None,
    species: str = "human",
) -> list[str]:
    """Get metabolic genes for specified pathways.

    Parameters
    ----------
    pathways
        List of pathway names. If None, returns all.
    species
        Species for gene names ("human" or "mouse").

    Returns
    -------
    List of metabolic gene names.
    """
    if pathways is None:
        pathways = list(METABOLIC_PATHWAYS.keys())

    genes = []
    for pathway in pathways:
        if pathway in METABOLIC_PATHWAYS:
            genes.extend(METABOLIC_PATHWAYS[pathway])

    genes = list(set(genes))  # Remove duplicates

    # Convert to mouse if needed
    if species == "mouse":
        genes = [g.capitalize() for g in genes]

    return genes


def filter_genes_by_expression(
    adata: AnnData,
    genes: list[str],
    min_cells: int = 10,
    min_counts: int = 1,
) -> list[str]:
    """Filter genes by expression levels.

    Parameters
    ----------
    adata
        AnnData object.
    genes
        List of genes to filter.
    min_cells
        Minimum number of cells expressing gene.
    min_counts
        Minimum total counts per gene.

    Returns
    -------
    Filtered list of genes.
    """
    # Get genes present in data
    present_genes = [g for g in genes if g in adata.var_names]

    if len(present_genes) == 0:
        return []

    # Get expression data
    X = adata[:, present_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Filter by expression
    cells_expressing = (X > 0).sum(axis=0)
    total_counts = X.sum(axis=0)

    mask = (cells_expressing >= min_cells) & (total_counts >= min_counts)
    filtered_genes = [g for g, m in zip(present_genes, mask, strict=False) if m]

    logger.info(f"Filtered from {len(present_genes)} to {len(filtered_genes)} genes")

    return filtered_genes


def compute_pathway_scores(
    adata: AnnData,
    pathway_genes: dict[str, list[str]] | None = None,
    layer: str | None = None,
) -> dict[str, NDArray]:
    """Compute pathway activity scores.

    Parameters
    ----------
    adata
        AnnData object.
    pathway_genes
        Dictionary mapping pathway names to gene lists.
    layer
        Layer to use for expression.

    Returns
    -------
    Dictionary mapping pathway names to score arrays.
    """
    if pathway_genes is None:
        pathway_genes = METABOLIC_PATHWAYS

    scores = {}

    for pathway, genes in pathway_genes.items():
        present_genes = [g for g in genes if g in adata.var_names]
        if len(present_genes) == 0:
            continue

        if layer is not None:
            X = adata[:, present_genes].layers[layer]
        else:
            X = adata[:, present_genes].X

        if hasattr(X, "toarray"):
            X = X.toarray()

        # Mean expression as pathway score
        scores[pathway] = X.mean(axis=1)

    return scores


def compute_exchange_network(
    exchange_df,
    min_score: float = 0.0,
    min_cells: int = 10,
) -> dict:
    """Convert exchange results to network format.

    Parameters
    ----------
    exchange_df
        DataFrame with exchange results.
    min_score
        Minimum exchange score threshold.
    min_cells
        Minimum number of cells.

    Returns
    -------
    Dictionary with nodes and edges.
    """
    # Filter
    df = exchange_df[(exchange_df["mean_expression"] >= min_score) & (exchange_df["n_cells"] >= min_cells)]

    # Get unique cell types
    cell_types = list(set(df["sender"].unique()) | set(df["receiver"].unique()))

    # Create edges
    edges = []
    for _, row in df.iterrows():
        edges.append(
            {
                "source": row["sender"],
                "target": row["receiver"],
                "gene": row["gene"],
                "weight": row["mean_expression"],
            }
        )

    return {
        "nodes": cell_types,
        "edges": edges,
    }


def annotate_metabolic_genes(
    genes: list[str],
) -> dict[str, str]:
    """Annotate genes with their metabolic pathway.

    Parameters
    ----------
    genes
        List of gene names.

    Returns
    -------
    Dictionary mapping genes to pathway names.
    """
    annotations = {}

    for gene in genes:
        for pathway, pathway_genes in METABOLIC_PATHWAYS.items():
            if gene.upper() in [g.upper() for g in pathway_genes]:
                annotations[gene] = pathway
                break
        else:
            annotations[gene] = "unknown"

    return annotations
