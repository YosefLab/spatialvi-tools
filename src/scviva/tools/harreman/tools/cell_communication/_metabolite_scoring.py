import numpy as np
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
    cs_gp: torch.Tensor,
    cell_type_key: str | None,
    gene_pair_dict: dict,
    gene_pairs_per_ct_pair_ind: dict | None = None,
    ct_specific_gene_pairs: list[int] | None = None,
    interacting_cell_scores: bool = False,
) -> torch.Tensor:
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
