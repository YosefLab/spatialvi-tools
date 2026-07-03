"""Shared statistics helpers used by both cell-type-agnostic and cell-type-aware CCC scoring."""

import numpy as np
import torch
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def z_to_pval_fdr(Z, method: str = "fdr_bh") -> tuple:
    """Convert an array of Z-scores to (p-values, FDR-corrected p-values).

    Z is expected to already be a numpy array (callers convert torch tensors
    to numpy via ``.detach().cpu().numpy()`` before calling this function,
    matching every existing call site's current behavior).
    """
    pvals = norm.sf(Z)
    flat_pvals = np.asarray(pvals).flatten()
    fdr_flat = multipletests(flat_pvals, method=method)[1]
    fdr = fdr_flat.reshape(np.asarray(pvals).shape)
    return pvals, fdr


def flatten(nested_list):
    """Yield scalar values from a nested list or tuple structure."""
    for item in nested_list:
        if isinstance(item, list | tuple):
            yield from flatten(item)
        else:
            yield item


def compute_max_cs_gp(vals, node_degrees):
    """Compute max communication score for a single gene (vector)."""
    return 0.5 * torch.sum(node_degrees * vals**2)


def compute_max_cs(node_degrees, counts, gene_pairs_ind):
    """Compute max communication scores per gene pair."""
    result = torch.empty(len(gene_pairs_ind), dtype=counts.dtype, device=counts.device)

    for i, (g1, _) in enumerate(gene_pairs_ind):
        if isinstance(g1, list):
            vals = counts[g1].mean(dim=0)
        else:
            vals = counts[g1]
        result[i] = compute_max_cs_gp(vals, node_degrees)

    return result
