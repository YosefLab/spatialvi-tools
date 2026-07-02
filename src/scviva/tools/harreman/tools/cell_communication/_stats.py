"""Shared statistics helpers used by both cell-type-agnostic and cell-type-aware CCC scoring."""

import torch


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
