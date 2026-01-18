"""Synthetic datasets for testing and examples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def synthetic_spatial(
    n_cells: int = 1000,
    n_genes: int = 200,
    n_cell_types: int = 5,
    n_batches: int = 1,
    spatial_dims: int = 2,
    spatial_scale: float = 100.0,
    library_size_mean: float = 10000,
    library_size_std: float = 3000,
    dropout_rate: float = 0.3,
    seed: int | None = None,
) -> AnnData:
    """Generate synthetic spatial transcriptomics data.

    Creates an AnnData object with:
    - Count matrix with cell type-specific expression patterns
    - Spatial coordinates
    - Cell type labels
    - Optional batch labels

    Parameters
    ----------
    n_cells
        Number of cells to generate.
    n_genes
        Number of genes.
    n_cell_types
        Number of cell types.
    n_batches
        Number of batches.
    spatial_dims
        Number of spatial dimensions (2 or 3).
    spatial_scale
        Scale of spatial coordinates.
    library_size_mean
        Mean library size.
    library_size_std
        Standard deviation of library size.
    dropout_rate
        Rate of dropout (zero inflation).
    seed
        Random seed for reproducibility.

    Returns
    -------
    AnnData object with synthetic spatial data.
    """
    rng = np.random.default_rng(seed)

    logger.info(
        f"Generating synthetic spatial data: "
        f"{n_cells} cells, {n_genes} genes, {n_cell_types} cell types"
    )

    # Generate cell type assignments
    cell_types = rng.integers(0, n_cell_types, size=n_cells)

    # Generate cell type-specific expression profiles
    # Each cell type has different marker genes
    cell_type_profiles = np.zeros((n_cell_types, n_genes))
    genes_per_type = n_genes // n_cell_types

    for ct in range(n_cell_types):
        # Marker genes for this cell type
        start_gene = ct * genes_per_type
        end_gene = min(start_gene + genes_per_type, n_genes)
        cell_type_profiles[ct, start_gene:end_gene] = rng.exponential(2, size=end_gene - start_gene)

        # Background expression for all genes
        cell_type_profiles[ct] += rng.exponential(0.5, size=n_genes)

    # Normalize profiles
    cell_type_profiles = cell_type_profiles / cell_type_profiles.sum(axis=1, keepdims=True)

    # Generate library sizes
    library_sizes = rng.normal(library_size_mean, library_size_std, size=n_cells)
    library_sizes = np.maximum(library_sizes, 1000)  # Minimum library size

    # Generate counts
    expression_rates = cell_type_profiles[cell_types]  # (n_cells, n_genes)
    expected_counts = expression_rates * library_sizes[:, np.newaxis]

    # Sample from negative binomial
    counts = rng.negative_binomial(
        n=10,  # dispersion
        p=10 / (10 + expected_counts),
    )

    # Add dropout
    dropout_mask = rng.random(counts.shape) < dropout_rate
    counts[dropout_mask] = 0

    # Generate spatial coordinates
    # Create clusters in space for each cell type
    spatial_coords = np.zeros((n_cells, spatial_dims))
    for ct in range(n_cell_types):
        ct_mask = cell_types == ct
        n_ct_cells = ct_mask.sum()

        # Random cluster center
        center = rng.uniform(0, spatial_scale, size=spatial_dims)

        # Cells around center
        spatial_coords[ct_mask] = (
            center + rng.normal(0, spatial_scale / 10, size=(n_ct_cells, spatial_dims))
        )

    # Clip to [0, spatial_scale]
    spatial_coords = np.clip(spatial_coords, 0, spatial_scale)

    # Generate batch labels
    if n_batches > 1:
        batch_labels = rng.integers(0, n_batches, size=n_cells)
    else:
        batch_labels = np.zeros(n_cells, dtype=int)

    # Create AnnData
    adata = AnnData(
        X=csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(
                [f"CellType_{i}" for i in cell_types],
                categories=[f"CellType_{i}" for i in range(n_cell_types)],
            ),
            "batch": pd.Categorical(
                [f"Batch_{i}" for i in batch_labels],
                categories=[f"Batch_{i}" for i in range(n_batches)],
            ),
            "library_size": library_sizes,
        }),
        var=pd.DataFrame(
            index=[f"Gene_{i}" for i in range(n_genes)]
        ),
        obsm={
            "spatial": spatial_coords.astype(np.float32),
        },
    )

    # Add marker gene information
    adata.var["marker_for"] = "None"
    for ct in range(n_cell_types):
        start_gene = ct * genes_per_type
        end_gene = min(start_gene + genes_per_type, n_genes)
        adata.var.iloc[start_gene:end_gene, adata.var.columns.get_loc("marker_for")] = f"CellType_{ct}"

    logger.info("Generated synthetic spatial data")

    return adata


def synthetic_scrna(
    n_cells: int = 2000,
    n_genes: int = 200,
    n_cell_types: int = 5,
    n_batches: int = 1,
    library_size_mean: float = 10000,
    library_size_std: float = 3000,
    dropout_rate: float = 0.3,
    seed: int | None = None,
) -> AnnData:
    """Generate synthetic single-cell RNA-seq data.

    Creates an AnnData object suitable for use as a reference
    for spatial deconvolution methods.

    Parameters
    ----------
    n_cells
        Number of cells to generate.
    n_genes
        Number of genes.
    n_cell_types
        Number of cell types.
    n_batches
        Number of batches.
    library_size_mean
        Mean library size.
    library_size_std
        Standard deviation of library size.
    dropout_rate
        Rate of dropout (zero inflation).
    seed
        Random seed for reproducibility.

    Returns
    -------
    AnnData object with synthetic scRNA-seq data.
    """
    rng = np.random.default_rng(seed)

    logger.info(
        f"Generating synthetic scRNA-seq data: "
        f"{n_cells} cells, {n_genes} genes, {n_cell_types} cell types"
    )

    # Generate cell type assignments with roughly equal proportions
    cell_types = rng.integers(0, n_cell_types, size=n_cells)

    # Generate cell type-specific expression profiles
    cell_type_profiles = np.zeros((n_cell_types, n_genes))
    genes_per_type = n_genes // n_cell_types

    for ct in range(n_cell_types):
        # Marker genes for this cell type
        start_gene = ct * genes_per_type
        end_gene = min(start_gene + genes_per_type, n_genes)
        cell_type_profiles[ct, start_gene:end_gene] = rng.exponential(3, size=end_gene - start_gene)

        # Background expression
        cell_type_profiles[ct] += rng.exponential(0.3, size=n_genes)

    # Normalize profiles
    cell_type_profiles = cell_type_profiles / cell_type_profiles.sum(axis=1, keepdims=True)

    # Generate library sizes
    library_sizes = rng.normal(library_size_mean, library_size_std, size=n_cells)
    library_sizes = np.maximum(library_sizes, 1000)

    # Generate counts
    expression_rates = cell_type_profiles[cell_types]
    expected_counts = expression_rates * library_sizes[:, np.newaxis]

    # Sample from negative binomial
    counts = rng.negative_binomial(
        n=10,
        p=10 / (10 + expected_counts),
    )

    # Add dropout
    dropout_mask = rng.random(counts.shape) < dropout_rate
    counts[dropout_mask] = 0

    # Generate batch labels
    if n_batches > 1:
        batch_labels = rng.integers(0, n_batches, size=n_cells)
    else:
        batch_labels = np.zeros(n_cells, dtype=int)

    # Create AnnData
    adata = AnnData(
        X=csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(
                [f"CellType_{i}" for i in cell_types],
                categories=[f"CellType_{i}" for i in range(n_cell_types)],
            ),
            "batch": pd.Categorical(
                [f"Batch_{i}" for i in batch_labels],
                categories=[f"Batch_{i}" for i in range(n_batches)],
            ),
            "library_size": library_sizes,
        }),
        var=pd.DataFrame(
            index=[f"Gene_{i}" for i in range(n_genes)]
        ),
    )

    # Add marker gene information
    adata.var["marker_for"] = "None"
    for ct in range(n_cell_types):
        start_gene = ct * genes_per_type
        end_gene = min(start_gene + genes_per_type, n_genes)
        adata.var.iloc[start_gene:end_gene, adata.var.columns.get_loc("marker_for")] = f"CellType_{ct}"

    logger.info("Generated synthetic scRNA-seq data")

    return adata


def synthetic_visium(
    n_spots: int = 500,
    n_genes: int = 200,
    n_cell_types: int = 5,
    cells_per_spot_mean: float = 10,
    cells_per_spot_std: float = 3,
    grid_rows: int = 25,
    grid_cols: int = 20,
    seed: int | None = None,
) -> AnnData:
    """Generate synthetic Visium-like spatial data.

    Creates aggregated spot-level data with known cell type compositions.

    Parameters
    ----------
    n_spots
        Number of spots to generate.
    n_genes
        Number of genes.
    n_cell_types
        Number of cell types.
    cells_per_spot_mean
        Mean number of cells per spot.
    cells_per_spot_std
        Std of cells per spot.
    grid_rows
        Number of rows in spatial grid.
    grid_cols
        Number of columns in spatial grid.
    seed
        Random seed for reproducibility.

    Returns
    -------
    AnnData object with synthetic Visium data.
    """
    rng = np.random.default_rng(seed)

    logger.info(f"Generating synthetic Visium data: {n_spots} spots")

    # Generate cell type profiles
    cell_type_profiles = np.zeros((n_cell_types, n_genes))
    genes_per_type = n_genes // n_cell_types

    for ct in range(n_cell_types):
        start_gene = ct * genes_per_type
        end_gene = min(start_gene + genes_per_type, n_genes)
        cell_type_profiles[ct, start_gene:end_gene] = rng.exponential(3, size=end_gene - start_gene)
        cell_type_profiles[ct] += rng.exponential(0.3, size=n_genes)

    cell_type_profiles = cell_type_profiles / cell_type_profiles.sum(axis=1, keepdims=True)

    # Generate spot locations on a grid
    row_indices = np.arange(n_spots) // grid_cols
    col_indices = np.arange(n_spots) % grid_cols
    spatial_coords = np.column_stack([col_indices * 100, row_indices * 100])

    # Generate cell type proportions per spot
    # Use Dirichlet distribution with spatial smoothness
    alpha = np.ones(n_cell_types)
    proportions = rng.dirichlet(alpha, size=n_spots)

    # Generate cells per spot
    cells_per_spot = rng.normal(cells_per_spot_mean, cells_per_spot_std, size=n_spots)
    cells_per_spot = np.maximum(cells_per_spot, 1).astype(int)

    # Generate counts by mixing cell type profiles
    spot_profiles = proportions @ cell_type_profiles  # (n_spots, n_genes)
    library_sizes = cells_per_spot * 5000  # Rough library size

    expected_counts = spot_profiles * library_sizes[:, np.newaxis]
    counts = rng.negative_binomial(
        n=5,
        p=5 / (5 + expected_counts),
    )

    # Create AnnData
    adata = AnnData(
        X=csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame({
            "n_cells": cells_per_spot,
            "library_size": library_sizes,
            "row": row_indices,
            "col": col_indices,
        }),
        var=pd.DataFrame(
            index=[f"Gene_{i}" for i in range(n_genes)]
        ),
        obsm={
            "spatial": spatial_coords.astype(np.float32),
            "true_proportions": proportions.astype(np.float32),
        },
    )

    # Store cell type names
    adata.uns["cell_type_names"] = [f"CellType_{i}" for i in range(n_cell_types)]

    logger.info("Generated synthetic Visium data")

    return adata
