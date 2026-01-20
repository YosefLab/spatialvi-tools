"""Shared test fixtures for spatialvi-tools."""

import numpy as np
import pytest
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

# =============================================================================
# Device and Basic Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get the device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 32


@pytest.fixture
def n_genes():
    """Default number of genes."""
    return 100


@pytest.fixture
def n_cells():
    """Default number of cells."""
    return 256


@pytest.fixture
def n_neighbors():
    """Default number of neighbors."""
    return 10


@pytest.fixture
def embed_dim():
    """Default embedding dimension."""
    return 64


@pytest.fixture
def n_latent():
    """Default latent dimension."""
    return 10


@pytest.fixture
def n_hidden():
    """Default hidden dimension."""
    return 32


@pytest.fixture
def n_batches():
    """Default number of batches."""
    return 2


@pytest.fixture
def n_labels():
    """Default number of cell type labels."""
    return 5


@pytest.fixture
def n_cell_types():
    """Default number of cell types."""
    return 5


# =============================================================================
# Random Data Fixtures
# =============================================================================


@pytest.fixture
def random_expression(batch_size, n_genes):
    """Generate random gene expression data."""
    return torch.randn(batch_size, n_genes).abs()


@pytest.fixture
def random_neighbor_expression(batch_size, n_neighbors, n_genes):
    """Generate random neighbor expression data."""
    return torch.randn(batch_size, n_neighbors, n_genes).abs()


@pytest.fixture
def random_distances(batch_size, n_neighbors):
    """Generate random distances to neighbors."""
    return torch.rand(batch_size, n_neighbors)


@pytest.fixture
def random_batch_index(batch_size, n_batches):
    """Generate random batch indices."""
    return torch.randint(0, n_batches, (batch_size, 1))


@pytest.fixture
def random_labels(batch_size, n_labels):
    """Generate random cell type labels."""
    return torch.randint(0, n_labels, (batch_size, 1))


@pytest.fixture
def random_spatial_coords(batch_size):
    """Generate random spatial coordinates."""
    return torch.rand(batch_size, 2) * 100


@pytest.fixture
def random_neighbor_indices(batch_size, n_neighbors):
    """Generate random neighbor indices."""
    return torch.randint(0, batch_size, (batch_size, n_neighbors))


@pytest.fixture
def random_niche_composition(batch_size, n_labels):
    """Generate random niche composition."""
    comp = torch.rand(batch_size, n_labels)
    return comp / comp.sum(dim=-1, keepdim=True)


# =============================================================================
# Module Input Tensors
# =============================================================================


@pytest.fixture
def spatial_vae_tensors(batch_size, n_genes, n_batches, n_neighbors, device):
    """Create tensors for SpatialVAEModule testing."""
    x = torch.randn(batch_size, n_genes).abs().to(device)
    batch_index = torch.randint(0, n_batches, (batch_size, 1)).to(device)
    neighbor_indices = torch.randint(0, batch_size, (batch_size, n_neighbors)).to(device)
    neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)
    spatial_coords = torch.rand(batch_size, 2).to(device) * 100

    return {
        "X": x,
        "batch": batch_index,
        "neighbor_indices": neighbor_indices,
        "neighbor_expr": neighbor_expr,
        "spatial": spatial_coords,
    }


@pytest.fixture
def deconv_tensors(batch_size, n_genes, n_batches, device):
    """Create tensors for DeconvolutionModule testing."""
    x = torch.randn(batch_size, n_genes).abs().to(device)
    batch_index = torch.randint(0, n_batches, (batch_size, 1)).to(device)

    return {
        "X": x,
        "batch": batch_index,
    }


@pytest.fixture
def niche_tensors(batch_size, n_genes, n_batches, n_labels, n_neighbors, device):
    """Create tensors for NicheModule testing."""
    x = torch.randn(batch_size, n_genes).abs().to(device)
    batch_index = torch.randint(0, n_batches, (batch_size, 1)).to(device)
    labels = torch.randint(0, n_labels, (batch_size, 1)).to(device)
    niche_comp = torch.rand(batch_size, n_labels).to(device)
    niche_comp = niche_comp / niche_comp.sum(dim=-1, keepdim=True)
    neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).abs().to(device)

    return {
        "X": x,
        "batch": batch_index,
        "labels": labels,
        "niche_composition": niche_comp,
        "neighbor_expr": neighbor_expr,
    }


# =============================================================================
# AnnData Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def small_spatial_adata():
    """Create small spatial AnnData for fast tests."""
    np.random.seed(42)
    n_cells = 50
    n_genes = 20

    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    spatial = np.random.uniform(0, 100, size=(n_cells, 2))

    adata = AnnData(
        X=csr_matrix(X.astype(np.float32)),
        obsm={"spatial": spatial.astype(np.float32)},
    )
    adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], size=n_cells)
    adata.obs["batch"] = np.random.choice(["batch1", "batch2"], size=n_cells)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    return adata


@pytest.fixture(scope="function")
def medium_spatial_adata():
    """Create medium-sized spatial AnnData for integration tests."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    spatial = np.random.uniform(0, 1000, size=(n_cells, 2))

    adata = AnnData(
        X=csr_matrix(X.astype(np.float32)),
        obsm={"spatial": spatial.astype(np.float32)},
    )
    adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB", "TypeC", "TypeD"], size=n_cells)
    adata.obs["batch"] = np.random.choice(["batch1", "batch2", "batch3"], size=n_cells)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    return adata


# =============================================================================
# Reference scRNA-seq Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def reference_scrna_adata():
    """Create reference scRNA-seq AnnData for deconvolution."""
    np.random.seed(42)
    n_cells = 500
    n_genes = 100
    n_cell_types = 5

    # Generate cell type-specific expression patterns
    cell_types = np.random.choice([f"CT_{i}" for i in range(n_cell_types)], size=n_cells)
    X = np.zeros((n_cells, n_genes))

    for i, ct in enumerate(np.unique(cell_types)):
        mask = cell_types == ct
        # Each cell type has higher expression in a subset of genes
        marker_genes = slice(i * 20, (i + 1) * 20)
        X[mask, marker_genes] = np.random.negative_binomial(10, 0.3, size=(mask.sum(), 20))
        X[mask] += np.random.negative_binomial(2, 0.5, size=(mask.sum(), n_genes))

    adata = AnnData(X=csr_matrix(X.astype(np.float32)))
    adata.obs["cell_type"] = cell_types
    adata.obs["batch"] = np.random.choice(["ref_batch1", "ref_batch2"], size=n_cells)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"RefCell_{i}" for i in range(n_cells)]

    return adata


# =============================================================================
# Graph Data Fixtures
# =============================================================================


@pytest.fixture
def edge_index(batch_size, n_neighbors):
    """Create edge index for graph-based operations."""
    src = torch.arange(batch_size).repeat_interleave(n_neighbors)
    dst = torch.randint(0, batch_size, (batch_size * n_neighbors,))
    return torch.stack([src, dst], dim=0)


@pytest.fixture
def spatial_graph(batch_size, n_neighbors):
    """Create spatial graph with coordinates and edges."""
    coords = torch.rand(batch_size, 2) * 100
    src = torch.arange(batch_size).repeat_interleave(n_neighbors)
    dst = torch.randint(0, batch_size, (batch_size * n_neighbors,))
    edge_index = torch.stack([src, dst], dim=0)

    # Compute edge distances
    edge_dist = torch.norm(coords[src] - coords[dst], dim=-1)

    return {
        "coords": coords,
        "edge_index": edge_index,
        "edge_dist": edge_dist,
    }


# =============================================================================
# Training Fixtures
# =============================================================================


@pytest.fixture
def training_config():
    """Default training configuration."""
    return {
        "max_epochs": 2,
        "batch_size": 32,
        "lr": 1e-3,
        "n_samples_per_label": 10,
        "check_val_every_n_epoch": 1,
        "accelerator": "cpu",
    }


@pytest.fixture
def kl_weight():
    """Default KL weight for loss computation."""
    return 1.0


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_val = 42
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    return seed_val


@pytest.fixture(autouse=True)
def set_seed():
    """Automatically set seed for all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
