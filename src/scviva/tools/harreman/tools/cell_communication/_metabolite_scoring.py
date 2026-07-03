import numpy as np
import sparse
import torch


def compute_metabolite_cs(
    cs_gp: torch.Tensor, gene_pair_dict: dict, interacting_cell_scores: bool = False
) -> torch.Tensor:
    """
    Computes metabolite-level communication scores from gene-pair scores.

    Parameters
    ----------
    cs_gp : torch.Tensor
        - If interacting_cell_scores is False: shape (gene_pairs,)
        - If interacting_cell_scores is True: shape (cells, gene_pairs)
    gene_pair_dict : dict
        Maps metabolite names to a list of indices (ints) referring to gene-pairs.
    interacting_cell_scores : bool, optional
        Whether cs_gp contains per-cell scores.

    Returns
    -------
    cs_m : torch.Tensor
        - If interacting_cell_scores is False: shape (num_metabolites,)
        - If interacting_cell_scores is True: shape (cells, num_metabolites)
    """
    device = cs_gp.device
    scores = []

    for indices in gene_pair_dict.values():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        if interacting_cell_scores:
            summed = cs_gp[:, idx_tensor].sum(dim=1)  # shape: (cells,)
        else:
            summed = cs_gp[idx_tensor].sum()  # scalar
        scores.append(summed)

    if interacting_cell_scores:
        cs_m = torch.stack(scores, dim=1)  # shape: (cells, metabolites)
    else:
        cs_m = torch.stack(scores)  # shape: (metabolites,)

    return cs_m


def compute_metabolite_cs_ct(
    cs_gp,
    cell_type_key,
    gene_pair_dict,
    gene_pairs_per_ct_pair_ind=None,
    ct_specific_gene_pairs=None,
    interacting_cell_scores=False,
):
    """Compute metabolite scores with optional cell type-pair masking."""
    if cell_type_key and ct_specific_gene_pairs:
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_ind.keys()):
            if i not in ct_specific_gene_pairs:
                continue
            mask_dim = 2 if interacting_cell_scores else 1
            mask = np.ones(cs_gp.shape[mask_dim], dtype=bool)
            mask[gene_pairs_per_ct_pair_ind[ct_pair]] = False
            if interacting_cell_scores:
                cs_gp[i, :, mask] = 0
            else:
                cs_gp[i, mask] = 0

    device = cs_gp.device
    scores = []

    for indices in gene_pair_dict.values():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        if interacting_cell_scores:
            summed = cs_gp[:, :, idx_tensor].sum(dim=2)  # shape: (cells,)
        else:
            summed = cs_gp[:, idx_tensor].sum(dim=1)  # scalar
        scores.append(summed)

    if interacting_cell_scores:
        cs_m = torch.stack(scores, dim=2)  # shape: (cells, metabolites)
    else:
        cs_m = torch.stack(scores, dim=1)

    return cs_m


def compute_metabolite_cs_old(
    cs_gp,
    cell_type_key,
    gene_pair_dict,
    gene_pairs_per_ct_pair_ind=None,
    ct_specific_gene_pairs=None,
    interacting_cell_scores=False,
):
    """Compute metabolite scores using the legacy NumPy implementation."""
    if cell_type_key and ct_specific_gene_pairs:
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_ind.keys()):
            if i not in ct_specific_gene_pairs:
                continue
            mask = np.ones(cs_gp.shape[1], dtype=bool)
            mask[gene_pairs_per_ct_pair_ind[ct_pair]] = False
            cs_gp[i, mask] = 0

    cells_metabolites = []
    for _metabolite, gene_pair_indices in gene_pair_dict.items():
        if interacting_cell_scores:
            summed_values = (
                cs_gp[:, :, gene_pair_indices].sum(axis=2)
                if cell_type_key
                else cs_gp[:, gene_pair_indices].sum(axis=1)
            )
            cells_metabolites.append(summed_values)
        else:
            summed_values = (
                cs_gp[:, gene_pair_indices].sum(axis=1)
                if cell_type_key
                else cs_gp[gene_pair_indices].sum(axis=0)
            )
            cells_metabolites.append(summed_values)
    if interacting_cell_scores:
        axis = 2 if cell_type_key else 1
    else:
        axis = 1 if cell_type_key else 0
    cs_m = np.stack(cells_metabolites, axis=axis)

    return cs_m


def ensure_tuple(x):
    """Convert nested lists in a pair-like object to tuples."""
    return tuple(tuple(i) if isinstance(i, list) else i for i in x)


def compute_CCC_scores(
    counts_1: np.array,
    counts_2: np.array,
    weights: sparse.COO,
    gene_pairs: list,
):
    """Compute aggregate cell-cell communication scores."""
    if len(weights.shape) == 3:
        scores = (counts_1.T * np.tensordot(weights, counts_2.T, axes=([2], [0]))).sum(axis=1)
    else:
        same_gene_mask = np.array([pair1 == pair2 for pair1, pair2 in gene_pairs])
        scores = (counts_1.T * (weights @ counts_2.T)).sum(axis=0) + (
            counts_1.T * (weights.T @ counts_2.T)
        ).sum(axis=0)
        scores[same_gene_mask] = scores[same_gene_mask] / 2

    return scores


def compute_int_CCC_scores(
    counts_1: np.array,
    counts_2: np.array,
    weights: sparse.COO,
    gene_pairs: list,
):
    """Compute per-cell interacting cell communication scores."""
    if len(weights.shape) == 3:
        scores = counts_1.T * np.tensordot(weights, counts_2.T, axes=([2], [0]))
    else:
        same_gene_mask = np.array([pair1 == pair2 for pair1, pair2 in gene_pairs])
        scores = (counts_1.T * (weights @ counts_2.T)) + (counts_1.T * (weights.T @ counts_2.T))
        scores[:, same_gene_mask] = scores[:, same_gene_mask] / 2

    return scores
