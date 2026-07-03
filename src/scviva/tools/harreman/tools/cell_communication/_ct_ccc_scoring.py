import itertools
import time
import warnings
from collections import defaultdict
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import sparse
import torch
from anndata import AnnData
from numba import jit, njit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from scviva.tools.harreman.hotspot import models
from scviva.tools.harreman.preprocessing.anndata import counts_from_anndata
from scviva.utils import resolve_device

from ._metabolite_scoring import (
    compute_metabolite_cs,
    compute_metabolite_cs_ct,
    compute_metabolite_cs_old,
)
from ._stats import compute_max_cs, flatten


def compute_ct_cell_communication(
    adata: AnnData,
    layer_key_p_test: Literal["use_raw"] | str | None = None,
    layer_key_np_test: Literal["use_raw"] | str | None = None,
    model: str = None,
    cell_type_key: str | None = None,
    center_counts_for_np_test: bool | None = False,
    subset_gene_pairs: list | None = None,
    subset_metabolites: list | None = None,
    fix_gp: bool | None = False,
    M: int | None = 1000,
    seed: int | None = 42,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None = "both",
    mean: Literal["algebraic"] | Literal["geometric"] | None = "algebraic",
    check_analytic_null: bool | None = False,
    device: torch.device | str = "auto",
    verbose: bool | None = False,
):
    """Compute cell type-aware cell-cell communication scores.

    Communication is stratified by interacting cell type pairs and supports parametric and
    non-parametric statistical inference.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Required fields include:
            - `uns["gene_pairs"]`: gene pairs involved in communication.
            - `uns["gene_pairs_per_metabolite"]`: maps metabolites to gene pairs.
            - `uns["gene_pairs_per_ct_pair"]`: gene pairs per cell type pair.
            - `obsp["weights"]`: sparse cell-cell proximity matrix.
            - `obs[cell_type_key]`: categorical cell type annotations.
            - `uns["cell_type_pairs"]`: list of interacting cell type pairs.
            - (Optional) `uns["LR_database"]`: for metabolite/pathway annotation.
    layer_key_p_test : str or "use_raw", optional
        Data layer to use for parametric test.
    layer_key_np_test : str or "use_raw", optional
        Data layer to use for non-parametric test.
    model : str, optional
        Normalization model to use for centering gene expression. Options include "none",
        "normal", "bernoulli", or "danb".
    cell_type_key : str, optional
        Key in `adata.obs` corresponding to cell type annotations. Required if not stored in `uns`.
    center_counts_for_np_test : bool, optional (default: False)
        Whether to center expression counts using the specified model before non-parametric
        testing.
    subset_gene_pairs : list, optional
        Subset of gene pairs to consider. If None, uses all pairs.
    subset_metabolites : list, optional
        Subset of metabolites to include in the analysis.
    fix_gp : bool, optional (default: False)
        If True, keeps gene pair identity fixed during permutation testing, randomizing cell types
        only.
    M : int, optional (default: 1000)
        Number of permutations to use if `permutation_test` is True.
    seed : int, optional (default: 42)
        Random seed for permutation reproducibility.
    test : {'parametric', 'non-parametric', 'both'}, optional (default: 'both')
        Specifies which statistical test(s) to run.
    mean : {'algebraic', 'geometric'}, optional (default: 'algebraic')
        Averaging method for multi-gene modules.
    check_analytic_null : bool, optional (default: False)
        Whether to compute Z-scores and p-values under the null distribution for the permutation
        test.
    device : torch.device, optional
        PyTorch device to run computations on. Defaults to CUDA if available.
    verbose : bool, optional (default: False)
        Whether to print progress and status messages.

    Returns
    -------
    None
        Results are stored in the following `adata.uns` fields:
            - `ct_ccc_results["p"]`: parametric results per gene pair, metabolite, and cell type.
            - `ct_ccc_results["np"]`: non-parametric scores, empirical p-values, and FDRs.
            - `gene_pair_dict`: dictionary mapping metabolites to relevant gene pairs.
            - `gene_pairs_ind`, `gene_pairs_ind_per_ct_pair`: index-referenced gene pair
              representations.
            - `D`: spatial node degree for each cell per cell type pair.
            - `cells`, `genes`: ordered list of cells and genes used in analysis.
            - (optional) `ct_ccc_results["np"]["analytic_null"]`: permutation null outputs.
    """
    start = time.time()
    device = resolve_device(device)
    if verbose:
        print("Starting cell type-aware cell-cell communication analysis...")

    adata.uns["ct_ccc_results"] = {}

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError(
            'The "test" variable should be one of ["both", "parametric", "non-parametric"].'
        )

    if mean not in ["algebraic", "geometric"]:
        raise ValueError('The "mean" variable should be one of ["algebraic", "geometric"].')

    if "cell_type_key" in adata.uns and cell_type_key is None:
        cell_type_key = adata.uns["cell_type_key"]
    elif "cell_type_key" not in adata.uns and cell_type_key is None:
        raise ValueError('Please provide the "cell_type_key" argument.')

    adata.uns["layer_key_p_test"] = layer_key_p_test
    adata.uns["layer_key_np_test"] = layer_key_np_test
    adata.uns["model"] = model
    adata.uns["cell_type_key"] = cell_type_key
    adata.uns["center_counts_for_np_test"] = center_counts_for_np_test
    adata.uns["mean"] = mean

    run_ct_cell_communication_analysis(
        adata,
        layer_key_p_test,
        layer_key_np_test,
        model,
        cell_type_key,
        center_counts_for_np_test,
        subset_gene_pairs,
        subset_metabolites,
        fix_gp,
        M,
        seed,
        test,
        mean,
        check_analytic_null,
        device,
        verbose,
    )

    if verbose:
        print("Obtaining cell type-aware communication results...")
    get_ct_cell_communication_results(
        adata,
        adata.uns["genes"],
        adata.uns["cells"],
        layer_key_p_test,
        layer_key_np_test,
        model,
        adata.obs[cell_type_key],
        adata.uns["cell_type_pairs"],
        adata.uns["D"],
        test,
        device,
    )

    if verbose:
        print(
            "Finished computing cell type-aware cell-cell communication analysis in %.3f seconds"
            % (time.time() - start)
        )

    return


def run_ct_cell_communication_analysis(
    adata,
    layer_key_p_test,
    layer_key_np_test,
    model,
    cell_type_key,
    center_counts_for_np_test,
    subset_gene_pairs,
    subset_metabolites,
    fix_gp,
    M,
    seed,
    test,
    mean,
    check_analytic_null,
    device,
    verbose,
):
    """Run the cell type-aware CCC score and significance workflow."""
    use_raw = (layer_key_p_test == "use_raw") & (layer_key_np_test == "use_raw")
    obs = adata.raw.obs if use_raw else adata.obs
    cells = (
        adata.raw.obs.index.values.astype(str) if use_raw else adata.obs_names.values.astype(str)
    )

    sample_specific = "sample_key" in adata.uns

    fix_ct = True if adata.uns["fix_ct"] else False

    gene_pairs = adata.uns["gene_pairs"] if subset_gene_pairs is None else subset_gene_pairs
    genes = list(np.unique(list(flatten(adata.uns["gene_pairs"]))))
    adata.uns["genes"] = genes

    cell_types = obs[cell_type_key]
    cell_type_pairs = adata.uns.get("cell_type_pairs")
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", {})

    weights = adata.obsp["weights"]

    used_ct_pairs = list({ct for cell_type_pair in cell_type_pairs for ct in cell_type_pair})
    all_cell_types = set(cell_types.unique())
    used_ct_pairs_set = set(used_ct_pairs)
    if used_ct_pairs_set < all_cell_types:
        keep_mask = cell_types[cells].isin(used_ct_pairs).values
        keep_indices = np.where(keep_mask)[0]
        weights = weights[keep_indices][:, keep_indices]
        cells = cells[keep_indices]
        cell_types = cell_types.loc[cells]

    adata.uns["cells"] = cells

    weights_ct_pairs = create_weights_ct_pairs(
        weights.tocoo(), cell_types, cell_type_pairs, device
    )

    row_degrees = torch.sparse.sum(weights_ct_pairs, dim=2).to_dense()
    col_degrees = torch.sparse.sum(weights_ct_pairs, dim=1).to_dense()
    D = row_degrees + col_degrees
    if used_ct_pairs_set < all_cell_types:
        D_full = torch.zeros(
            (len(cell_type_pairs), adata.shape[0]),
            device=weights_ct_pairs.device,
            dtype=weights_ct_pairs.dtype,
        )
        D_full[:, keep_indices] = D
        adata.uns["D"] = D_full.cpu().numpy()
    else:
        adata.uns["D"] = D.cpu().numpy()

    # Map gene_pairs to index
    gene_pairs_ind = []
    for pair in gene_pairs:
        idx1 = (
            [genes.index(g) for g in pair[0]]
            if isinstance(pair[0], list)
            else genes.index(pair[0])
        )
        idx2 = (
            [genes.index(g) for g in pair[1]]
            if isinstance(pair[1], list)
            else genes.index(pair[1])
        )
        gene_pairs_ind.append((idx1, idx2))
    adata.uns["gene_pairs_ind"] = gene_pairs_ind

    # Cell-type pair-specific indices
    gene_pairs_ind_per_ct_pair = defaultdict(list)
    gene_pairs_per_ct_pair_ind = defaultdict(list)
    for ct_pair, gpairs in gene_pairs_per_ct_pair.items():
        for pair in gpairs:
            if pair not in gene_pairs:
                continue
            idx = gene_pairs.index(pair)
            gene_pairs_ind_per_ct_pair[ct_pair].append(gene_pairs_ind[idx])
            gene_pairs_per_ct_pair_ind[ct_pair].append(idx)

    adata.uns["gene_pairs_ind_per_ct_pair"] = dict(gene_pairs_ind_per_ct_pair)
    adata.uns["gene_pairs_per_ct_pair_ind"] = dict(gene_pairs_per_ct_pair_ind)

    def make_hashable(pair):
        return tuple(tuple(x) if isinstance(x, list) else x for x in pair)

    gene_pairs_ind_set = {make_hashable(pair) for pair in gene_pairs_ind}
    ct_specific_gene_pairs = [
        i
        for i, pairs in enumerate(gene_pairs_ind_per_ct_pair.values())
        if {make_hashable(pair) for pair in pairs} < gene_pairs_ind_set
    ]

    # Metabolite-gene pair preparation
    gp_metab = adata.uns["gene_pairs_per_metabolite"]
    metabolite_gene_pair_df = (
        pd.DataFrame.from_dict(gp_metab, orient="index")
        .rename_axis("metabolite")
        .explode(["gene_pair", "gene_type"])
        .reset_index()
    )

    if "LR_database" in adata.uns:
        merged = metabolite_gene_pair_df.merge(
            adata.uns["LR_database"], left_on="metabolite", right_on="interaction_name", how="left"
        )
        LR_df = merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df.loc[
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite), "metabolite"
        ] = LR_df["pathway_name"].values

    if subset_metabolites:
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df.metabolite.isin(subset_metabolites)
        ]

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = (
            group["gene_pair"]
            .apply(lambda gp: gene_pairs.index(gp) if gp in gene_pairs else None)
            .dropna()
            .tolist()
        )
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    adata.uns["gene_pair_dict"] = gene_pair_dict

    if test in ["parametric", "both"]:
        if verbose:
            print("Running the parametric test...")

        adata.uns["ct_ccc_results"]["p"] = {"gp": {}, "m": {}}

        weights_sq_data = weights_ct_pairs.values() ** 2
        weights_sq = torch.sparse_coo_tensor(
            weights_ct_pairs.indices(),
            weights_sq_data,
            weights_ct_pairs.shape,
            device=weights_ct_pairs.device,
        )
        Wtot2 = torch.sparse.sum(weights_sq, dim=(1, 2)).to_dense()

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        counts_1 = standardize_ct_counts(
            adata, counts_1, model, num_umi, sample_specific, cell_types
        )
        counts_2 = standardize_ct_counts(
            adata, counts_2, model, num_umi, sample_specific, cell_types
        )

        # Compute CCC scores
        cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
        for ct_pair in range(len(cell_type_pairs)):
            W = weights_ct_pairs[ct_pair].coalesce()
            WX2t = torch.sparse.mm(W, counts_2.T)
            cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
        adata.uns["ct_ccc_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs_ct(
            cs_gp,
            cell_type_key,
            gene_pair_dict,
            gene_pairs_per_ct_pair_ind,
            ct_specific_gene_pairs,
            interacting_cell_scores=False,
        )
        adata.uns["ct_ccc_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        EG2_gp = torch.zeros_like(cs_gp) if fix_ct or fix_gp else Wtot2
        if fix_ct:
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                W_sq_data = W.values() ** 2
                W_sq = torch.sparse_coo_tensor(W.indices(), W_sq_data, W.shape, device=W.device)
                X1_sq = counts_1**2
                EG2_gp[ct_pair] = torch.sparse.mm(W_sq, X1_sq.T).sum(0)
        elif fix_gp:
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                W_sq_data = W.values() ** 2
                W_sq = torch.sparse_coo_tensor(W.indices(), W_sq_data, W.shape, device=W.device)
                X1_sq = counts_1**2
                X2_sq = counts_2**2
                EG2_gp[ct_pair] = (X1_sq.T * torch.sparse.mm(W_sq, X2_sq.T)).sum(0)

        Z_gp, Z_m = compute_ct_p_results(
            cs_gp,
            cs_m,
            gene_pairs_per_ct_pair_ind,
            ct_specific_gene_pairs,
            EG2_gp,
            cell_type_key,
            gene_pair_dict,
        )

        # Convert tensors to numpy for statsmodels and pandas
        Z_gp_np = Z_gp.detach().cpu().numpy()
        Z_m_np = Z_m.detach().cpu().numpy()
        # Compute p-values and FDRs
        Z_pvals_gp = norm.sf(Z_gp_np)
        Z_pvals_m = norm.sf(Z_m_np)
        FDR_gp = multipletests(Z_pvals_gp.flatten(), method="fdr_bh")[1].reshape(Z_pvals_gp.shape)
        FDR_m = multipletests(Z_pvals_m.flatten(), method="fdr_bh")[1].reshape(Z_pvals_m.shape)

        # Store in AnnData
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z"] = Z_gp_np
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z_pval"] = Z_pvals_gp
        adata.uns["ct_ccc_results"]["p"]["gp"]["Z_FDR"] = FDR_gp
        adata.uns["ct_ccc_results"]["p"]["m"]["Z"] = Z_m_np
        adata.uns["ct_ccc_results"]["p"]["m"]["Z_pval"] = Z_pvals_m
        adata.uns["ct_ccc_results"]["p"]["m"]["Z_FDR"] = FDR_m

    if test in ["non-parametric", "both"]:
        if verbose:
            print("Running the non-parametric test...")

        adata.uns["ct_ccc_results"]["np"] = {"gp": {}, "m": {}}

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_ind:
            if isinstance(idx1, list):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            counts_1 = standardize_ct_counts(
                adata, counts_1, model, num_umi, sample_specific, cell_types
            )
            counts_2 = standardize_ct_counts(
                adata, counts_2, model, num_umi, sample_specific, cell_types
            )

        if center_counts_for_np_test and test == "both":
            adata.uns["ct_ccc_results"]["np"]["gp"]["cs"] = np.array(
                adata.uns["ct_ccc_results"]["p"]["gp"]["cs"]
            )
            adata.uns["ct_ccc_results"]["np"]["m"]["cs"] = np.array(
                adata.uns["ct_ccc_results"]["p"]["m"]["cs"]
            )
        else:
            cs_gp = torch.zeros((len(cell_type_pairs), counts_1.shape[0]), device=counts_1.device)
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, counts_2.T)
                cs_gp[ct_pair] = (counts_1.T * WX2t).sum(0)
            adata.uns["ct_ccc_results"]["np"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()
            cs_m = compute_metabolite_cs_ct(
                cs_gp,
                cell_type_key,
                gene_pair_dict,
                gene_pairs_per_ct_pair_ind,
                ct_specific_gene_pairs,
                interacting_cell_scores=False,
            )
            adata.uns["ct_ccc_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        perm_cs_gp = torch.zeros(
            (len(cell_type_pairs), counts_1.shape[0], M), dtype=torch.float64, device=device
        )
        perm_cs_m = torch.zeros(
            (len(cell_type_pairs), len(gene_pair_dict), M), dtype=torch.float64, device=device
        )

        if check_analytic_null:
            gp_zs_perm_array = torch.zeros_like(perm_cs_gp)
            gp_pvals_perm_array = torch.zeros_like(perm_cs_gp)
            m_zs_perm_array = torch.zeros_like(perm_cs_m)
            m_pvals_perm_array = torch.zeros_like(perm_cs_m)

        if fix_gp:
            c1_perm = counts_1
            c2_perm = counts_2

        torch.manual_seed(seed)
        for i in tqdm(range(M), desc="Permutation test"):
            if fix_gp:
                indices = torch.randperm(len(cell_types)).numpy()
                shuffled_cell_types = cell_types.iloc[indices].reset_index(drop=True)
                weights_ct_pairs = create_weights_ct_pairs(
                    weights.tocoo(), shuffled_cell_types, cell_type_pairs, device
                )
            else:
                cell_type_labels = torch.tensor(
                    cell_types.astype("category").cat.codes.values, device=counts_1.device
                )
                idx = torch.empty_like(cell_type_labels, dtype=torch.int64)

                for ct in torch.unique(cell_type_labels):
                    ct_mask = cell_type_labels == ct
                    ct_indices = torch.nonzero(ct_mask, as_tuple=True)[0]
                    permuted_indices = ct_indices[torch.randperm(len(ct_indices))]
                    idx[ct_indices] = permuted_indices

                c1_perm = counts_1 if fix_ct else counts_1[:, idx.long()]
                c2_perm = counts_2[:, idx.long()]

            cs_gp = torch.zeros((len(cell_type_pairs), c1_perm.shape[0]), device=c1_perm.device)
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, c2_perm.T)
                cs_gp[ct_pair] = (c1_perm.T * WX2t).sum(0)
            perm_cs_gp[:, :, i] = cs_gp

            cs_m = compute_metabolite_cs_ct(
                cs_gp,
                cell_type_key,
                gene_pair_dict,
                gene_pairs_per_ct_pair_ind,
                ct_specific_gene_pairs,
                interacting_cell_scores=False,
            )
            perm_cs_m[:, :, i] = cs_m

            if check_analytic_null:
                Z_gp_perm, Z_m_perm = compute_ct_p_results(
                    cs_gp,
                    cs_m,
                    gene_pairs_per_ct_pair_ind,
                    ct_specific_gene_pairs,
                    EG2_gp,
                    cell_type_key,
                    gene_pair_dict,
                )
                gp_zs_perm_array[:, :, i] = Z_gp_perm
                gp_pvals_perm_array[:, :, i] = torch.tensor(
                    norm.sf(Z_gp_perm.cpu().numpy()), device=device
                )
                m_zs_perm_array[:, :, i] = Z_m_perm
                m_pvals_perm_array[:, :, i] = torch.tensor(
                    norm.sf(Z_m_perm.cpu().numpy()), device=device
                )

        adata.uns["ct_ccc_results"]["np"]["gp"]["perm_cs"] = perm_cs_gp.detach().cpu().numpy()
        adata.uns["ct_ccc_results"]["np"]["m"]["perm_cs"] = perm_cs_m.detach().cpu().numpy()

        x_gp = np.sum(
            adata.uns["ct_ccc_results"]["np"]["gp"]["perm_cs"]
            > adata.uns["ct_ccc_results"]["np"]["gp"]["cs"][:, :, np.newaxis],
            axis=2,
        )
        x_m = np.sum(
            adata.uns["ct_ccc_results"]["np"]["m"]["perm_cs"]
            > adata.uns["ct_ccc_results"]["np"]["m"]["cs"][:, :, np.newaxis],
            axis=2,
        )

        pvals_gp = (x_gp + 1) / (M + 1)
        pvals_m = (x_m + 1) / (M + 1)

        adata.uns["ct_ccc_results"]["np"]["gp"]["pval"] = pvals_gp
        adata.uns["ct_ccc_results"]["np"]["gp"]["FDR"] = multipletests(
            pvals_gp.flatten(), method="fdr_bh"
        )[1].reshape(pvals_gp.shape)
        adata.uns["ct_ccc_results"]["np"]["m"]["pval"] = pvals_m
        adata.uns["ct_ccc_results"]["np"]["m"]["FDR"] = multipletests(
            pvals_m.flatten(), method="fdr_bh"
        )[1].reshape(pvals_m.shape)

        if check_analytic_null:
            adata.uns["ct_ccc_results"]["np"]["analytic_null"] = {
                "gp_zs_perm": gp_zs_perm_array.detach().cpu().numpy(),
                "gp_pvals_perm": gp_pvals_perm_array.detach().cpu().numpy(),
                "m_zs_perm": m_zs_perm_array.detach().cpu().numpy(),
                "m_pvals_perm": m_pvals_perm_array.detach().cpu().numpy(),
            }

    adata.uns["cell_types"] = cell_types.tolist() if cell_type_key else None

    if verbose:
        print("Non-parametric test finished.")

    return


def standardize_ct_counts(adata, counts, model, num_umi, sample_specific, cell_types):
    """Standardize counts within cell types and optional samples."""
    if sample_specific:
        sample_key = adata.uns["sample_key"]
        for sample in adata.obs[sample_key].unique():
            subset = np.where(adata.obs[sample_key] == sample)[0]
            counts[:, subset] = center_ct_counts_torch(
                counts[:, subset], num_umi[subset], model, cell_types[subset]
            )
    else:
        counts = center_ct_counts_torch(counts, num_umi, model, cell_types)

    return counts


def create_weights_ct_pairs(weights, cell_types, cell_type_pairs, device):
    """Create sparse weight tensors for each cell type pair."""
    indices = torch.tensor([weights.row, weights.col], dtype=torch.long, device=device)
    values = torch.tensor(weights.data, dtype=torch.float64, device=device)
    shape = weights.shape

    cell_type_cats = cell_types.astype("category")
    cell_type_codes = torch.tensor(
        cell_type_cats.cat.codes.values, dtype=torch.long, device=device
    )
    ct_name_to_code = {name: code for code, name in enumerate(cell_type_cats.cat.categories)}

    row_idx, col_idx = indices
    sender_types = cell_type_codes[row_idx]
    receiver_types = cell_type_codes[col_idx]

    weights_list = []
    coord_list = []

    for i, (ct1, ct2) in enumerate(cell_type_pairs):
        code1 = ct_name_to_code[ct1]
        code2 = ct_name_to_code[ct2]

        pair_mask = (sender_types == code1) & (receiver_types == code2)
        if pair_mask.sum() == 0:
            continue

        pair_values = values[pair_mask]
        pair_coords = torch.stack(
            [
                torch.full((pair_values.shape[0],), i, dtype=torch.long, device=device),
                row_idx[pair_mask],
                col_idx[pair_mask],
            ],
            dim=0,
        )

        weights_list.append(pair_values)
        coord_list.append(pair_coords)

    all_values = torch.cat(weights_list)
    all_coords = torch.cat(coord_list, dim=1)
    weights_ct_pairs = torch.sparse_coo_tensor(
        all_coords, all_values, (len(cell_type_pairs), shape[0], shape[1]), device=device
    )
    weights_ct_pairs = weights_ct_pairs.coalesce()

    return weights_ct_pairs


def compute_ct_interacting_cell_scores(
    adata: str | AnnData,
    center_counts_for_np_test: bool | None = False,
    test: Literal["parametric"] | Literal["non-parametric"] | Literal["both"] | None = "both",
    restrict_significance: Literal["gene pairs"]
    | Literal["metabolites"]
    | Literal["both"]
    | None = "both",
    device: torch.device | str = "auto",
    verbose: bool | None = False,
):
    """Compute cell-type-aware interacting cell scores for gene pairs and metabolites.

    Parameters
    ----------
    adata : AnnData or str
        Must contain:
        - ``uns['model']``, ``uns['mean']``
        - ``uns['cell_type_key']`` and ``obs[cell_type_key]`` for cell types
        - ``uns['gene_pairs']``, ``uns['gene_pairs_per_ct_pair']``
        - ``uns['gene_pairs_per_metabolite']``
        - ``uns['ct_ccc_results']`` with significance information
        - ``obsp['weights']`` (spatial proximity matrix)
    center_counts_for_np_test : bool, default False
        Whether to standardize counts before the non-parametric test.
    test : {"parametric", "non-parametric", "both"}
        Which statistical test(s) to run.
    restrict_significance : {"gene pairs", "metabolites", "both"}
        Only use cell-type-pair interactions that were significant in cell-type-aware CCC results.
    device : torch.device
        CPU or GPU device for PyTorch computations.
    verbose : bool, default False
        Print detailed progress messages.
    """
    start = time.time()
    if verbose:
        print("Computing cell type-aware gene pair and metabolite scores...")

    adata.uns["ct_interacting_cell_results"] = {}

    model = adata.uns["model"]
    mean = adata.uns["mean"]

    if test not in ["both", "parametric", "non-parametric"]:
        raise ValueError(
            'The "test" variable should be one of ["both", "parametric", "non-parametric"].'
        )

    if restrict_significance not in ["both", "gene pairs", "metabolites"]:
        raise ValueError(
            'The "restrict_significance" variable should be one of '
            '["both", "gene pairs", "metabolites"].'
        )

    sample_specific = "sample_key" in adata.uns

    layer_key_p_test = adata.uns.get("layer_key_p_test", None)
    layer_key_np_test = adata.uns.get("layer_key_np_test", None)
    use_raw = (layer_key_p_test == "use_raw") and (layer_key_np_test == "use_raw")

    obs = adata.raw.obs if use_raw else adata.obs
    cells = (
        adata.raw.obs.index.values.astype(str) if use_raw else adata.obs_names.values.astype(str)
    )

    gene_pairs = adata.uns.get("gene_pairs", None)
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", None)

    gp_metab = adata.uns["gene_pairs_per_metabolite"]
    metabolite_gene_pair_df = (
        pd.DataFrame.from_dict(gp_metab, orient="index")
        .rename_axis("metabolite")
        .explode(["gene_pair", "gene_type"])
        .reset_index()
    )

    if "LR_database" in adata.uns:
        merged = metabolite_gene_pair_df.merge(
            adata.uns["LR_database"], left_on="metabolite", right_on="interaction_name", how="left"
        )
        LR_df = merged.dropna(subset=["pathway_name"])
        metabolite_gene_pair_df.loc[
            metabolite_gene_pair_df.metabolite.isin(LR_df.metabolite), "metabolite"
        ] = LR_df["pathway_name"].values

    cell_type_pairs = adata.uns.get("cell_type_pairs")
    cell_type_pairs = [tuple(x) for x in cell_type_pairs]

    cell_com_gp_df = adata.uns["ct_ccc_results"]["cell_com_df_gp_sig"].copy()
    if restrict_significance in ["both", "gene pairs"]:
        ct_pairs_gp_set = {tuple(x) for x in cell_com_gp_df[["Cell Type 1", "Cell Type 2"]].values}
        cell_type_pairs = [ct_pair for ct_pair in cell_type_pairs if ct_pair in ct_pairs_gp_set]

        cell_com_gp_df[["Gene 1", "Gene 2"]] = cell_com_gp_df[["Gene 1", "Gene 2"]].map(
            lambda x: tuple(x) if isinstance(x, list) else x
        )

        gene_pairs_set = {tuple(x) for x in cell_com_gp_df[["Gene 1", "Gene 2"]].values}
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df["gene_pair"].isin(gene_pairs_set)
        ]

    cell_com_m_df = adata.uns["ct_ccc_results"]["cell_com_df_m_sig"].copy()
    if restrict_significance in ["both", "metabolites"]:
        ct_pairs_m_set = {tuple(x) for x in cell_com_m_df[["Cell Type 1", "Cell Type 2"]].values}
        missing_ct_pairs = [
            ct_pair for ct_pair in ct_pairs_m_set if ct_pair not in cell_type_pairs
        ]
        if len(missing_ct_pairs) > 0:
            warnings.warn(
                "The following cell type pairs are not included in the "
                f'"cell_type_pairs" set: {missing_ct_pairs}',
                stacklevel=2,
            )

        metabolite_set = set(cell_com_m_df["metabolite"].values)
        metabolite_gene_pair_df = metabolite_gene_pair_df[
            metabolite_gene_pair_df["metabolite"].isin(metabolite_set)
        ]

    if metabolite_gene_pair_df.empty:
        if restrict_significance == "both":
            raise ValueError(
                "There are no significant gene pairs that belong to a significant metabolite."
            )
        if restrict_significance == "gene pairs":
            raise ValueError("There are no significant gene pairs.")
        if restrict_significance == "metabolites":
            raise ValueError("There are no significant metabolites.")

    genes = adata.uns["genes"]
    gene_pairs_sig = []
    if gene_pairs:
        for g1, g2 in gene_pairs:
            g1 = tuple(g1) if isinstance(g1, list) else g1
            g2 = tuple(g2) if isinstance(g2, list) else g2
            if not metabolite_gene_pair_df[metabolite_gene_pair_df["gene_pair"] == (g1, g2)].empty:
                gene_pairs_sig.append((g1, g2))

    gene_pairs_sig_ind = []
    for pair in gene_pairs_sig:
        idx1 = (
            [genes.index(g) for g in pair[0]]
            if isinstance(pair[0], list)
            else genes.index(pair[0])
        )
        idx2 = (
            [genes.index(g) for g in pair[1]]
            if isinstance(pair[1], list)
            else genes.index(pair[1])
        )
        gene_pairs_sig_ind.append((idx1, idx2))

    cell_type_key = adata.uns.get("cell_type_key")
    cell_types = obs[cell_type_key]
    gene_pairs_per_ct_pair = adata.uns.get("gene_pairs_per_ct_pair", {})

    weights = adata.obsp["weights"]

    used_ct_pairs = list({ct for cell_type_pair in cell_type_pairs for ct in cell_type_pair})
    all_cell_types = set(cell_types.unique())
    used_ct_pairs_set = set(used_ct_pairs)
    if used_ct_pairs_set < all_cell_types:
        keep_mask = cell_types[cells].isin(used_ct_pairs).values
        keep_indices = np.where(keep_mask)[0]
        weights = weights[keep_indices][:, keep_indices]
        cells = cells[keep_indices]
        cell_types = cell_types.loc[
            cells
        ]  # Eventually only keep the cell type pairs with at least one significant gene pair

    weights_ct_pairs = create_weights_ct_pairs(
        weights.tocoo(), cell_types, cell_type_pairs, device
    )

    gene_pairs_per_ct_pair_sig = {}
    for ct_pair in gene_pairs_per_ct_pair.keys():
        if ct_pair not in cell_type_pairs:
            continue
        cell_com_df_ct_pair = cell_com_gp_df[
            (cell_com_gp_df["Cell Type 1"] == ct_pair[0])
            & (cell_com_gp_df["Cell Type 2"] == ct_pair[1])
        ]
        gene_pairs_per_ct_pair_sig[ct_pair] = [
            tuple(x) for x in cell_com_df_ct_pair[["Gene 1", "Gene 2"]].values
        ]

    # Cell-type pair-specific indices
    gene_pairs_ind_per_ct_pair_sig = defaultdict(list)
    gene_pairs_per_ct_pair_sig_ind = defaultdict(list)
    for ct_pair, gpairs in gene_pairs_per_ct_pair_sig.items():
        for pair in gpairs:
            if pair not in gene_pairs_sig:
                continue
            idx = gene_pairs_sig.index(pair)
            gene_pairs_ind_per_ct_pair_sig[ct_pair].append(gene_pairs_sig_ind[idx])
            gene_pairs_per_ct_pair_sig_ind[ct_pair].append(idx)

    adata.uns["gene_pairs_ind_per_ct_pair_sig"] = dict(gene_pairs_ind_per_ct_pair_sig)
    adata.uns["gene_pairs_per_ct_pair_sig_ind"] = dict(gene_pairs_per_ct_pair_sig_ind)

    def make_hashable(pair):
        return tuple(tuple(x) if isinstance(x, list) else x for x in pair)

    gene_pairs_sig_ind_set = {make_hashable(pair) for pair in gene_pairs_sig_ind}
    ct_specific_gene_pairs = [
        i
        for i, pairs in enumerate(gene_pairs_ind_per_ct_pair_sig.values())
        if {make_hashable(pair) for pair in pairs} < gene_pairs_sig_ind_set
    ]

    gene_pair_dict = {}
    for metabolite, group in metabolite_gene_pair_df.groupby("metabolite"):
        idxs = (
            group["gene_pair"]
            .apply(lambda gp: gene_pairs_sig.index(gp) if gp in gene_pairs_sig else None)
            .dropna()
            .tolist()
        )
        idxs = [int(ind) for ind in idxs if ind is not None]
        if idxs:
            gene_pair_dict[metabolite] = idxs
    metabolites = list(gene_pair_dict.keys())

    gene_pair_to_metabolite_indices = defaultdict(set)
    for met_idx, met in enumerate(metabolites):
        for gene_pair_idx in gene_pair_dict.get(met, []):
            gene_pair_to_metabolite_indices[gene_pair_idx].add(met_idx)

    ct_pair_to_metabolite_indices = {}
    for ct_pair, gene_pair_indices in gene_pairs_per_ct_pair_sig_ind.items():
        met_indices = set()
        for gi in gene_pair_indices:
            met_indices.update(gene_pair_to_metabolite_indices.get(gi, []))
        ct_pair_to_metabolite_indices[ct_pair] = sorted(met_indices)

    def concat_tuple_elements(t, sep="_"):
        return sep.join(
            s for item in t for s in (item if isinstance(item, list | tuple) else [item])
        )

    if test in ["parametric", "both"]:
        if verbose:
            print("Running the parametric test...")

        adata.uns["ct_interacting_cell_results"]["p"] = {"gp": {}, "m": {}}

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_sig_ind:
            if isinstance(idx1, list):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, list):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        counts_1 = standardize_ct_counts(
            adata[cells, :], counts_1, model, num_umi, sample_specific, cell_types
        )
        counts_2 = standardize_ct_counts(
            adata[cells, :], counts_2, model, num_umi, sample_specific, cell_types
        )

        cs_gp = torch.zeros(
            (len(cell_type_pairs), counts_1.shape[1], counts_1.shape[0]), device=counts_1.device
        )
        for ct_pair in range(len(cell_type_pairs)):
            W = weights_ct_pairs[ct_pair].coalesce()
            WX2t = torch.sparse.mm(W, counts_2.T)
            cs_gp[ct_pair] = counts_1.T * WX2t
        adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"] = cs_gp.detach().cpu().numpy()

        cs_m = compute_metabolite_cs_ct(
            cs_gp,
            cell_type_key,
            gene_pair_dict,
            gene_pairs_per_ct_pair_sig_ind,
            ct_specific_gene_pairs,
            interacting_cell_scores=True,
        )
        adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        column_names = []
        scores = []
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_sig_ind.keys()):
            gp_list = gene_pairs_per_ct_pair_sig_ind[ct_pair]
            for gp in gp_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"][i, :, gp] == 0
                ):
                    continue
                column_names.append(
                    f"{' - '.join(ct_pair)}: {concat_tuple_elements(gene_pairs_sig[gp])}"
                )
                scores.append(adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"][i, :, gp])

        cs_gp_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_gp_df = cs_gp_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_p_gp_cs_df"] = cs_gp_df

        column_names = []
        scores = []
        for i, ct_pair in enumerate(ct_pair_to_metabolite_indices.keys()):
            metab_list = ct_pair_to_metabolite_indices[ct_pair]
            for metab in metab_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"][i, :, metab] == 0
                ):
                    continue
                column_names.append(f"{' - '.join(ct_pair)}: {metabolites[metab]}")
                scores.append(
                    adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"][i, :, metab]
                )

        cs_m_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_m_df = cs_m_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_p_m_cs_df"] = cs_m_df

        if verbose:
            print("Parametric test finished.")

    if test in ["non-parametric", "both"]:
        if verbose:
            print("Running the non-parametric test...")

        adata.uns["ct_interacting_cell_results"]["np"] = {"gp": {}, "m": {}}

        # Load counts
        counts = counts_from_anndata(adata[cells, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)

        # Prepare counts_1 and counts_2
        counts_1 = []
        counts_2 = []
        for idx1, idx2 in gene_pairs_sig_ind:
            if isinstance(idx1, tuple):
                c1 = (
                    counts[idx1, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx1, :] + 1e-8).mean(dim=0))
                )
            else:
                c1 = counts[idx1, :]
            if isinstance(idx2, tuple):
                c2 = (
                    counts[idx2, :].mean(dim=0)
                    if mean == "algebraic"
                    else torch.exp(torch.log(counts[idx2, :] + 1e-8).mean(dim=0))
                )
            else:
                c2 = counts[idx2, :]
            counts_1.append(c1)
            counts_2.append(c2)

        counts_1 = torch.stack(counts_1)
        counts_2 = torch.stack(counts_2)

        if center_counts_for_np_test:
            num_umi = counts.sum(dim=0)
            from scviva.tools.harreman.hotspot.local_autocorrelation import standardize_counts

            counts_1 = standardize_counts(adata, counts_1, model, num_umi, sample_specific)
            counts_2 = standardize_counts(adata, counts_2, model, num_umi, sample_specific)

        if center_counts_for_np_test and test == "both":
            adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"] = np.array(
                adata.uns["ct_interacting_cell_results"]["p"]["gp"]["cs"]
            )
            adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"] = np.array(
                adata.uns["ct_interacting_cell_results"]["p"]["m"]["cs"]
            )
        else:
            cs_gp = torch.zeros(
                (len(cell_type_pairs), counts_1.shape[1], counts_1.shape[0]),
                device=counts_1.device,
            )
            for ct_pair in range(len(cell_type_pairs)):
                W = weights_ct_pairs[ct_pair].coalesce()
                WX2t = torch.sparse.mm(W, counts_2.T)
                cs_gp[ct_pair] = counts_1.T * WX2t
            adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"] = (
                cs_gp.detach().cpu().numpy()
            )
            cs_m = compute_metabolite_cs_ct(
                cs_gp,
                cell_type_key,
                gene_pair_dict,
                gene_pairs_per_ct_pair_sig_ind,
                ct_specific_gene_pairs,
                interacting_cell_scores=True,
            )
            adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"] = cs_m.detach().cpu().numpy()

        column_names = []
        scores = []
        for i, ct_pair in enumerate(gene_pairs_per_ct_pair_sig_ind.keys()):
            gp_list = gene_pairs_per_ct_pair_sig_ind[ct_pair]
            for gp in gp_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"][i, :, gp] == 0
                ):
                    continue
                column_names.append(
                    f"{' - '.join(ct_pair)}: {concat_tuple_elements(gene_pairs_sig[gp])}"
                )
                scores.append(adata.uns["ct_interacting_cell_results"]["np"]["gp"]["cs"][i, :, gp])

        cs_gp_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_gp_df = cs_gp_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_np_gp_cs_df"] = cs_gp_df

        column_names = []
        scores = []
        for i, ct_pair in enumerate(ct_pair_to_metabolite_indices.keys()):
            metab_list = ct_pair_to_metabolite_indices[ct_pair]
            for metab in metab_list:
                if np.all(
                    adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"][i, :, metab] == 0
                ):
                    continue
                column_names.append(f"{' - '.join(ct_pair)}: {metabolites[metab]}")
                scores.append(
                    adata.uns["ct_interacting_cell_results"]["np"]["m"]["cs"][i, :, metab]
                )

        cs_m_df = pd.DataFrame(
            {column_names[i]: array for i, array in enumerate(scores)}, index=cells
        )
        if used_ct_pairs_set < all_cell_types:
            cs_m_df = cs_m_df.reindex(adata.obs_names, fill_value=0)
        adata.obsm["ct_interacting_cell_results_np_m_cs_df"] = cs_m_df

        if verbose:
            print("Non-parametric test finished.")

    if verbose:
        print(
            "Finished computing cell type-aware gene pair and metabolite scores in %.3f seconds"
            % (time.time() - start)
        )

    return


def get_ct_pair_weights(weights, cell_type_pairs, cell_types):
    """Extract weights for all requested cell type pairs."""
    w_nrow, w_ncol = weights.shape
    n_ct_pairs = len(cell_type_pairs)

    extract_weights_results = partial(
        extract_ct_pair_weights,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
    )
    results = list(map(extract_weights_results, cell_type_pairs))

    w_new_data_all = [x[0] for x in results]
    w_new_coords_3d_all = [x[1] for x in results]
    w_new_coords_3d_all = np.hstack(w_new_coords_3d_all)
    w_new_data_all = np.concatenate(w_new_data_all)

    weights_ct_pairs = sparse.COO(
        w_new_coords_3d_all, w_new_data_all, shape=(n_ct_pairs, w_nrow, w_ncol)
    )

    return weights_ct_pairs


def extract_ct_pair_weights(ct_pair, weights, cell_type_pairs, cell_types):
    """Extract spatial weights for one cell type pair."""
    i = cell_type_pairs.index(ct_pair)

    ct_t, ct_u = cell_type_pairs[i]
    ct_t_mask = cell_types.values == ct_t
    ct_t_mask_coords = np.argwhere(ct_t_mask)
    ct_u_mask = cell_types.values == ct_u
    ct_u_mask_coords = np.argwhere(ct_u_mask)

    w_old_coords = weights.coords

    w_row_coords, w_col_coords = np.meshgrid(ct_t_mask_coords, ct_u_mask_coords, indexing="ij")
    w_row_coords = w_row_coords.ravel()
    w_col_coords = w_col_coords.ravel()
    w_new_coords = np.vstack((w_row_coords, w_col_coords))

    # w_matching_indices = np.where(np.all(np.isin(w_old_coords.T, w_new_coords.T), axis=1))[0]
    w_matching_indices = np.isin(w_old_coords[0], ct_t_mask_coords.flatten()) & np.isin(
        w_old_coords[1], ct_u_mask_coords.flatten()
    )
    w_new_data = weights.data[w_matching_indices]
    w_new_coords = w_old_coords[:, w_matching_indices]

    w_coord_3d = np.full(w_new_coords.shape[1], fill_value=i)
    w_new_coords_3d = np.vstack((w_coord_3d, w_new_coords))

    return (w_new_data, w_new_coords_3d)


def conditional_eg2_cellcom_gp(counts_1, counts_2, weights):
    """Compute conditional second moments for gene-pair communication scores."""
    if len(weights.shape) == 3:
        counts_1_sq = counts_1**2
        counts_2_sq = counts_2**2
        weights_sq_data = weights.data**2
        weights_sq = sparse.COO(weights.coords, weights_sq_data, shape=weights.shape)
        out_eg2_a = np.tensordot(counts_1_sq, weights_sq, axes=([1], [1])).sum(axis=2).T
        out_eg2_b = np.tensordot(weights_sq, counts_2_sq, axes=([2], [1])).sum(axis=1)
        out_eg2s = (out_eg2_a, out_eg2_b)
    else:
        out_eg2_a = (((weights + weights.T) @ counts_1.T) ** 2).sum(axis=0)
        out_eg2_b = (((weights + weights.T) @ counts_2.T) ** 2).sum(axis=0)
        out_eg2s = (out_eg2_a, out_eg2_b)

    return out_eg2s


def conditional_eg2_gp_score(counts, weights):
    """Compute conditional second moments for gene scores."""
    counts_sq = counts**2
    if len(weights.shape) == 3:
        # weights_t = weights.transpose(axes=(0, 2, 1))
        # weights = weights + weights_t
        weights_sq_data = weights.data**2
        weights_sq = sparse.COO(weights.coords, weights_sq_data, shape=weights.shape)
        out_eg2_a = np.transpose(np.tensordot(counts_sq, weights_sq, axes=([1], [1])), (1, 2, 0))
        out_eg2_b = np.tensordot(weights_sq, counts_sq, axes=([2], [1]))
        out_eg2s = (out_eg2_a, out_eg2_b)
    else:
        out_eg2s = ((weights + weights.T) @ counts.T) ** 2

    return out_eg2s


def compute_ct_p_results(
    C_gp,
    C_m,
    gene_pairs_per_ct_pair_ind,
    ct_specific_gene_pairs,
    EG2_gp,
    cell_type_key,
    gene_pair_dict,
):
    """Compute cell type-aware parametric Z-scores."""
    EG2_gp = EG2_gp.unsqueeze(1).expand(-1, C_gp.shape[1]) if len(EG2_gp.shape) == 1 else EG2_gp

    stdG = torch.sqrt(EG2_gp)
    stdG[stdG == 0] = 1

    Z_gp = C_gp / stdG

    EG2_m = compute_metabolite_cs_ct(
        EG2_gp,
        cell_type_key,
        gene_pair_dict,
        gene_pairs_per_ct_pair_ind,
        ct_specific_gene_pairs,
        interacting_cell_scores=False,
    )
    if not isinstance(EG2_m, torch.Tensor):
        device = EG2_gp.device
        EG2_m = torch.tensor(EG2_m, device=device, dtype=torch.float64)

    stdG_m = torch.sqrt(EG2_m)
    stdG_m[stdG_m == 0] = 1

    Z_m = C_m / stdG_m

    return Z_gp, Z_m


def compute_p_results_old(
    ct_pair,
    cell_type_pairs,
    cs_gp,
    cs_m,
    gene_pairs_ind,
    gene_pairs_ind_per_ct_pair,
    Wtot2,
    eg2s_gp,
    cell_type_key,
    gene_pair_dict,
):
    """Compute parametric results using the legacy NumPy implementation."""
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = (
        (cs_gp[0][i, :], cs_gp[1][i, :]) if isinstance(cs_gp, tuple) else cs_gp[i, :]
    )
    C_m = (cs_m[0][i, :], cs_m[1][i, :]) if isinstance(cs_m, tuple) else cs_m[i, :]
    # If all gene pairs are considered irrespective of the cell type pair, use
    # `gene_pairs_ind` directly.
    gene_pairs_ind_ct_pair = gene_pairs_ind_per_ct_pair[ct_pair]

    eg2s_a, eg2s_b = eg2s_gp
    C_gp = []
    EG2_a = []
    EG2_b = []
    for gene_pair_ind_ct_pair in gene_pairs_ind_ct_pair:
        idx = gene_pairs_ind.index(gene_pair_ind_ct_pair)
        g1_ind, g2_ind = gene_pair_ind_ct_pair
        lc_gp = (
            (gene_pair_cor_ct[0][idx], gene_pair_cor_ct[1][idx])
            if isinstance(gene_pair_cor_ct, tuple)
            else gene_pair_cor_ct[idx]
        )
        eg2_a = eg2_b = Wtot2[i]
        # if g1_ind == g2_ind:
        #     eg2_a = eg2_b = Wtot2[i]
        # else:
        #     eg2_a = eg2s_a[i, idx]
        #     eg2_b = eg2s_b[i, idx]
        C_gp.append(lc_gp)
        EG2_a.append(eg2_a)
        EG2_b.append(eg2_b)

    # Gene pairs

    EG = [0 for i in range(len(gene_pairs_ind_ct_pair))]

    stdG_a = [(EG2_a[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind_ct_pair))]
    stdG_a = [1 if stdG_a[i] == 0 else stdG_a[i] for i in range(len(stdG_a))]

    stdG_b = [(EG2_b[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind_ct_pair))]
    stdG_b = [1 if stdG_b[i] == 0 else stdG_b[i] for i in range(len(stdG_b))]

    if isinstance(C_gp[0], tuple):
        Z_gp_a = [(C_gp[i][0] - EG[i]) / stdG_a[i] for i in range(len(gene_pairs_ind_ct_pair))]
        Z_gp_b = [(C_gp[i][1] - EG[i]) / stdG_b[i] for i in range(len(gene_pairs_ind_ct_pair))]
    else:
        Z_gp_a = [(C_gp[i] - EG[i]) / stdG_a[i] for i in range(len(gene_pairs_ind_ct_pair))]
        Z_gp_b = [(C_gp[i] - EG[i]) / stdG_b[i] for i in range(len(gene_pairs_ind_ct_pair))]

    EG2_m_a = compute_metabolite_cs_old(
        np.array(EG2_a),
        cell_type_key=None,
        gene_pair_dict=gene_pair_dict,
        interacting_cell_scores=False,
    )
    EG2_m_b = compute_metabolite_cs_old(
        np.array(EG2_b),
        cell_type_key=None,
        gene_pair_dict=gene_pair_dict,
        interacting_cell_scores=False,
    )

    # Metabolites

    EG = [0 for i in range(len(gene_pair_dict.keys()))]

    stdG_a = [(EG2_m_a[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pair_dict.keys()))]
    stdG_a = [1 if stdG_a[i] == 0 else stdG_a[i] for i in range(len(stdG_a))]

    stdG_b = [(EG2_m_b[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pair_dict.keys()))]
    stdG_b = [1 if stdG_b[i] == 0 else stdG_b[i] for i in range(len(stdG_b))]

    if isinstance(C_m, tuple):
        Z_m_a = [(C_m[0][i] - EG[i]) / stdG_a[i] for i in range(len(gene_pair_dict.keys()))]
        Z_m_b = [(C_m[1][i] - EG[i]) / stdG_b[i] for i in range(len(gene_pair_dict.keys()))]
    else:
        Z_m_a = [(C_m[i] - EG[i]) / stdG_a[i] for i in range(len(gene_pair_dict.keys()))]
        Z_m_b = [(C_m[i] - EG[i]) / stdG_b[i] for i in range(len(gene_pair_dict.keys()))]

    return (C_gp, Z_gp_a, Z_gp_b, C_m, Z_m_a, Z_m_b)


def compute_p_int_cell_results(
    ct_pair,
    cell_type_pairs,
    cs_gp,
    cs_m,
    gene_pairs_ind,
    gene_pairs_ind_per_ct_pair,
    Wtot2,
    eg2s_gp,
    cell_type_key,
    gene_pair_dict,
):
    """Compute interacting-cell parametric results for one cell type pair."""
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = cs_gp[i, :, :]
    C_m = cs_m[i, :, :]
    # If all gene pairs are considered irrespective of the cell type pair, use
    # `gene_pairs_ind` directly.
    gene_pairs_ind_ct_pair = gene_pairs_ind_per_ct_pair[ct_pair]

    eg2s_a, eg2s_b = eg2s_gp
    C_gp = []
    EG2_a = []
    EG2_b = []
    for gene_pair_ind_ct_pair in gene_pairs_ind_ct_pair:
        idx = gene_pairs_ind.index(gene_pair_ind_ct_pair)
        g1_ind, g2_ind = gene_pair_ind_ct_pair
        lc_gp = gene_pair_cor_ct[:, idx]
        if g1_ind == g2_ind:
            eg2_a = eg2_b = Wtot2[i, :]
        else:
            eg2_a = (
                eg2s_a[i, :, g1_ind]
                if type(g1_ind) is not list
                else np.max(eg2s_a[i, :, g1_ind], axis=0)
            )
            eg2_b = (
                eg2s_b[i, :, g2_ind]
                if type(g2_ind) is not list
                else np.max(eg2s_b[i, :, g2_ind], axis=0)
            )
        C_gp.append(lc_gp)
        EG2_a.append(eg2_a)
        EG2_b.append(eg2_b)
    C_gp = np.column_stack(C_gp)
    EG2_a = np.column_stack(EG2_a)
    EG2_b = np.column_stack(EG2_b)

    # Gene pairs

    EG = np.zeros(C_gp.shape)

    stdG_a = (EG2_a - EG**2) ** 0.5
    stdG_a[stdG_a == 0] = 1

    stdG_b = (EG2_b - EG**2) ** 0.5
    stdG_b[stdG_b == 0] = 1

    Z_gp = np.where(
        np.abs((C_gp - EG) / stdG_a) < np.abs((C_gp - EG) / stdG_b),
        (C_gp - EG) / stdG_a,
        (C_gp - EG) / stdG_b,
    )

    EG2_gp = np.where(np.abs((C_gp - EG) / stdG_a) < np.abs((C_gp - EG) / stdG_b), EG2_a, EG2_b)

    EG2_m = compute_metabolite_cs(
        EG2_gp, cell_type_key=None, gene_pair_dict=gene_pair_dict, interacting_cell_scores=True
    )

    # Metabolites

    EG = np.zeros(C_m.shape)

    stdG = (EG2_m - EG**2) ** 0.5
    stdG[stdG == 0] = 1

    Z_m = (C_m - EG) / stdG

    return (C_gp, Z_gp, C_m, Z_m)


def compute_np_results(
    ct_pair,
    cell_type_pairs,
    cs_gp,
    cs_m,
    pvals_gp,
    pvals_m,
    gene_pair_dict,
    gene_pairs_ind,
    gene_pairs_ind_per_ct_pair,
):
    """Extract non-parametric results for one cell type pair."""
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_gp_ct = cs_gp[i, :]
    C_m = cs_m[i, :]
    pvals_gp_a, pvals_gp_b = pvals_gp
    pvals_gp_a_ct = pvals_gp_a[i, :]
    pvals_gp_b_ct = pvals_gp_b[i, :]
    pvals_m_a, pvals_m_b = pvals_m
    p_values_m_a = pvals_m_a[i, :]
    # If all gene pairs are considered irrespective of the cell type pair, use
    # `gene_pairs_ind` directly.
    gene_pairs_ind_ct_pair = gene_pairs_ind_per_ct_pair[ct_pair]

    C_gp = []
    p_values_gp_a = []
    p_values_gp_b = []
    for gene_pair_ind_ct_pair in gene_pairs_ind_ct_pair:
        idx = gene_pairs_ind.index(gene_pair_ind_ct_pair)
        lc_gp = gene_pair_cor_gp_ct[idx]
        p_value_gp_a = pvals_gp_a_ct[idx]
        p_value_gp_b = pvals_gp_b_ct[idx]

        C_gp.append(lc_gp.reshape(1))
        p_values_gp_a.append(p_value_gp_a.reshape(1))
        p_values_gp_b.append(p_value_gp_b.reshape(1))

    C_gp = list(np.concatenate(C_gp))
    p_values_gp_a = list(np.concatenate(p_values_gp_a))
    p_values_gp_b = list(np.concatenate(p_values_gp_b))

    # C_m = compute_metabolite_cs(
    #     np.array(C_gp), cell_type_key=None, gene_pair_dict=gene_pair_dict,
    #     interacting_cell_scores=False
    # )

    return (C_gp, p_values_gp_a, p_values_gp_b, C_m, p_values_m_a, p_values_m_a)


def get_ct_cell_communication_results(
    adata,
    genes,
    cells,
    layer_key_p_test,
    layer_key_np_test,
    model,
    cell_types,
    cell_type_pairs,
    D,
    test,
    device,
):
    """Assemble cell type-aware communication result dataframes."""
    gene_pairs_ind_per_ct_pair = adata.uns["gene_pairs_ind_per_ct_pair"]
    gene_pair_dict = adata.uns["gene_pair_dict"]
    genes = adata.uns["genes"]

    sample_specific = "sample_key" in adata.uns

    if isinstance(D, np.ndarray):
        D = torch.tensor(D, dtype=torch.float64, device=device)

    def idx_to_gene(idx):
        return [genes[i] for i in idx] if isinstance(idx, list) else genes[idx]

    records = [
        {
            "Cell Type 1": ct1,
            "Cell Type 2": ct2,
            "Gene 1": idx_to_gene(gp[0]),
            "Gene 2": idx_to_gene(gp[1]),
        }
        for (ct1, ct2), gp_list in gene_pairs_ind_per_ct_pair.items()
        for gp in gp_list
    ]
    cell_com_df_gp = pd.DataFrame.from_records(records)

    # Generate metabolite interaction table
    ct_pairs = list(gene_pairs_ind_per_ct_pair.keys())
    metabolites = list(gene_pair_dict.keys())
    cell_com_df_m = pd.DataFrame(
        [
            {"Cell Type 1": ct1, "Cell Type 2": ct2, "metabolite": m}
            for (ct1, ct2), m in itertools.product(ct_pairs, metabolites)
        ]
    )

    if test in ["parametric", "both"]:
        suffix = "p"
        # Gene pair
        c_values = adata.uns["ct_ccc_results"][suffix]["gp"]["cs"]
        z_values = adata.uns["ct_ccc_results"][suffix]["gp"]["Z"]
        p_values = adata.uns["ct_ccc_results"][suffix]["gp"]["Z_pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["gp"]["Z_FDR"]
        cell_com_df_gp[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_gp["Z"] = z_values.flatten()
        cell_com_df_gp["Z_pval"] = p_values.flatten()
        cell_com_df_gp["Z_FDR"] = fdr_values.flatten()

        counts = counts_from_anndata(adata[:, genes], layer_key_p_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        num_umi = counts.sum(dim=0)
        counts_std = standardize_ct_counts(
            adata, counts, model, num_umi, sample_specific, cell_types
        )

        c_values_norm = normalize_ct_values(
            counts_std, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D
        )
        adata.uns["ct_ccc_results"][suffix]["gp"]["cs_norm"] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f"C_norm_{suffix}"] = c_values_norm.cpu().numpy().flatten()

        # Metabolite
        c_values = adata.uns["ct_ccc_results"][suffix]["m"]["cs"]
        z_values = adata.uns["ct_ccc_results"][suffix]["m"]["Z"]
        p_values = adata.uns["ct_ccc_results"][suffix]["m"]["Z_pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["m"]["Z_FDR"]
        cell_com_df_m[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_m["Z"] = z_values.flatten()
        cell_com_df_m["Z_pval"] = p_values.flatten()
        cell_com_df_m["Z_FDR"] = fdr_values.flatten()

    if test in ["non-parametric", "both"]:
        suffix = "np"
        # Gene pair
        c_values = adata.uns["ct_ccc_results"][suffix]["gp"]["cs"]
        p_values = adata.uns["ct_ccc_results"][suffix]["gp"]["pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["gp"]["FDR"]
        cell_com_df_gp[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_gp[f"pval_{suffix}"] = p_values.flatten()
        cell_com_df_gp[f"FDR_{suffix}"] = fdr_values.flatten()

        counts = counts_from_anndata(adata[:, genes], layer_key_np_test, dense=True)
        counts = torch.tensor(counts, dtype=torch.float64, device=device)
        if adata.uns.get("center_counts_for_np_test", False):
            num_umi = counts.sum(dim=0)
            counts = standardize_ct_counts(
                adata, counts, model, num_umi, sample_specific, cell_types
            )

        c_values_norm = normalize_ct_values(
            counts, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D
        )
        adata.uns["ct_ccc_results"][suffix]["gp"]["cs_norm"] = c_values_norm.cpu().numpy()
        cell_com_df_gp[f"C_norm_{suffix}"] = c_values_norm.cpu().numpy().flatten()

        # Metabolite
        c_values = adata.uns["ct_ccc_results"][suffix]["m"]["cs"]
        p_values = adata.uns["ct_ccc_results"][suffix]["m"]["pval"]
        fdr_values = adata.uns["ct_ccc_results"][suffix]["m"]["FDR"]
        cell_com_df_m[f"C_{suffix}"] = c_values.flatten()
        cell_com_df_m[f"pval_{suffix}"] = p_values.flatten()
        cell_com_df_m[f"FDR_{suffix}"] = fdr_values.flatten()

    adata.uns["ct_ccc_results"]["cell_com_df_gp"] = cell_com_df_gp
    adata.uns["ct_ccc_results"]["cell_com_df_m"] = cell_com_df_m

    return


def get_cell_communication_results_old(
    adata,
    genes,
    cells,
    layer_key_p_test,
    layer_key_np_test,
    model,
    cell_types,
    cell_type_pairs,
    D,
    test,
):
    """Assemble legacy cell type-aware communication result dataframes."""
    gene_pairs_ind_per_ct_pair = adata.uns["gene_pairs_ind_per_ct_pair"]
    gene_pair_dict = adata.uns["gene_pair_dict"]
    genes = adata.uns["genes"]

    def map_to_genes(x):
        if isinstance(x, list):
            return [genes[i] for i in x]
        else:
            return genes[x]

    cell_com_df_gp = (
        pd.DataFrame.from_dict(gene_pairs_ind_per_ct_pair, orient="index")
        .stack()
        .to_frame()
        .reset_index()
    )
    cell_com_df_gp = cell_com_df_gp.drop(["level_1"], axis=1)
    cell_com_df_gp = cell_com_df_gp.rename(columns={"level_0": "cell_type_pair", 0: "gene_pair"})
    cell_com_df_gp["Cell Type 1"], cell_com_df_gp["Cell Type 2"] = zip(
        *cell_com_df_gp["cell_type_pair"], strict=False
    )
    cell_com_df_gp["Gene 1"], cell_com_df_gp["Gene 2"] = zip(
        *cell_com_df_gp["gene_pair"], strict=False
    )
    cell_com_df_gp["Gene 1"] = cell_com_df_gp["Gene 1"].apply(map_to_genes)
    cell_com_df_gp["Gene 2"] = cell_com_df_gp["Gene 2"].apply(map_to_genes)
    cell_com_df_gp = cell_com_df_gp.drop(["cell_type_pair", "gene_pair"], axis=1)

    ct_pair_metab = list(
        itertools.product(gene_pairs_ind_per_ct_pair.keys(), gene_pair_dict.keys())
    )
    cell_com_df_m = pd.DataFrame(ct_pair_metab, columns=["cell_type_pair", "metabolite"])
    cell_com_df_m["Cell Type 1"], cell_com_df_m["Cell Type 2"] = zip(
        *cell_com_df_m["cell_type_pair"], strict=False
    )
    cell_com_df_m = cell_com_df_m.drop(["cell_type_pair"], axis=1)

    if test in ["parametric", "both"]:
        # Gene pair
        c_values = adata.uns["ct_ccc_results"]["p"]["gp"]["cs"]
        cell_com_df_gp["C_p"] = c_values.flatten()
        z_values_a = adata.uns["ct_ccc_results"]["p"]["gp"]["Z_a"]
        cell_com_df_gp["Z_a"] = z_values_a.flatten()
        z_values_b = adata.uns["ct_ccc_results"]["p"]["gp"]["Z_b"]
        cell_com_df_gp["Z_b"] = z_values_b.flatten()
        p_values_a = adata.uns["ct_ccc_results"]["p"]["gp"]["Z_pval_a"]
        cell_com_df_gp["Z_pval_a"] = p_values_a.flatten()
        p_values_b = adata.uns["ct_ccc_results"]["p"]["gp"]["Z_pval_b"]
        cell_com_df_gp["Z_pval_b"] = p_values_b.flatten()
        FDR_values_a = adata.uns["ct_ccc_results"]["p"]["gp"]["Z_FDR_a"]
        cell_com_df_gp["Z_FDR_a"] = FDR_values_a.flatten()
        FDR_values_b = adata.uns["ct_ccc_results"]["p"]["gp"]["Z_FDR_b"]
        cell_com_df_gp["Z_FDR_b"] = FDR_values_b.flatten()

        counts = counts_from_anndata(adata[cells, genes], layer_key_p_test, dense=True)
        num_umi = counts.sum(axis=0)
        counts_std = counts_std = create_centered_counts_ct(counts, model, num_umi, cell_types)
        counts_std = np.nan_to_num(counts_std)

        c_values_norm = normalize_values_old(
            counts_std, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D
        )
        adata.uns["ct_ccc_results"]["p"]["gp"]["cs_norm"] = c_values_norm
        cell_com_df_gp["C_norm_p"] = c_values_norm.flatten()

        # Metabolite
        c_values = adata.uns["ct_ccc_results"]["p"]["m"]["cs"]
        cell_com_df_m["C_p"] = c_values.flatten()
        z_values_a = adata.uns["ct_ccc_results"]["p"]["m"]["Z_a"]
        cell_com_df_m["Z_a"] = z_values_a.flatten()
        z_values_b = adata.uns["ct_ccc_results"]["p"]["m"]["Z_b"]
        cell_com_df_m["Z_b"] = z_values_b.flatten()
        p_values_a = adata.uns["ct_ccc_results"]["p"]["m"]["Z_pval_a"]
        cell_com_df_m["Z_pval_a"] = p_values_a.flatten()
        p_values_b = adata.uns["ct_ccc_results"]["p"]["m"]["Z_pval_b"]
        cell_com_df_m["Z_pval_b"] = p_values_b.flatten()
        FDR_values_a = adata.uns["ct_ccc_results"]["p"]["m"]["Z_FDR_a"]
        cell_com_df_m["Z_FDR_a"] = FDR_values_a.flatten()
        FDR_values_b = adata.uns["ct_ccc_results"]["p"]["m"]["Z_FDR_b"]
        cell_com_df_m["Z_FDR_b"] = FDR_values_b.flatten()

    if test in ["non-parametric", "both"]:
        # Gene pair
        c_values = adata.uns["ct_ccc_results"]["np"]["gp"]["cs"]
        cell_com_df_gp["C_np"] = c_values.flatten()
        p_values_a = adata.uns["ct_ccc_results"]["np"]["gp"]["pval_a"]
        cell_com_df_gp["pval_np_a"] = p_values_a.flatten()
        p_values_b = adata.uns["ct_ccc_results"]["np"]["gp"]["pval_b"]
        cell_com_df_gp["pval_np_b"] = p_values_b.flatten()
        FDR_values_a = adata.uns["ct_ccc_results"]["np"]["gp"]["FDR_a"]
        cell_com_df_gp["FDR_np_a"] = FDR_values_a.flatten()
        FDR_values_b = adata.uns["ct_ccc_results"]["np"]["gp"]["FDR_b"]
        cell_com_df_gp["FDR_np_b"] = FDR_values_b.flatten()

        counts = counts_from_anndata(adata[:, genes], layer_key_np_test, dense=True)
        if adata.uns["center_counts_for_np_test"]:
            num_umi = counts.sum(axis=0)
            from . import create_centered_counts

            counts = create_centered_counts(counts, model, num_umi)
            counts = np.nan_to_num(counts)
        c_values_norm = normalize_values_old(
            counts, cell_types, cell_type_pairs, gene_pairs_ind_per_ct_pair, c_values, D
        )
        adata.uns["ct_ccc_results"]["np"]["gp"]["cs_norm"] = c_values_norm
        cell_com_df_gp["C_norm_np"] = c_values_norm.flatten()

        # Metabolite
        c_values = adata.uns["ct_ccc_results"]["np"]["m"]["cs"]
        cell_com_df_m["C_np"] = c_values.flatten()
        p_values_a = adata.uns["ct_ccc_results"]["np"]["m"]["pval_a"]
        cell_com_df_m["pval_np_a"] = p_values_a.flatten()
        p_values_b = adata.uns["ct_ccc_results"]["np"]["m"]["pval_b"]
        cell_com_df_m["pval_np_b"] = p_values_b.flatten()
        FDR_values_a = adata.uns["ct_ccc_results"]["np"]["m"]["FDR_a"]
        cell_com_df_m["FDR_np_a"] = FDR_values_a.flatten()
        FDR_values_b = adata.uns["ct_ccc_results"]["np"]["m"]["FDR_b"]
        cell_com_df_m["FDR_np_b"] = FDR_values_b.flatten()

    adata.uns["ct_ccc_results"]["cell_com_df_gp"] = cell_com_df_gp
    adata.uns["ct_ccc_results"]["cell_com_df_m"] = cell_com_df_m

    return


def normalize_ct_values(
    counts,
    cell_types,
    cell_type_pairs,
    gene_pairs_per_ct_pair_ind,
    lcs,
    D,
):
    """Normalize communication scores within each cell type pair."""
    if isinstance(cell_types, pd.Series):
        cell_types = cell_types.values

    if isinstance(lcs, np.ndarray):
        lcs = torch.tensor(lcs, dtype=counts.dtype, device=counts.device)

    c_values_norm = torch.empty_like(lcs, dtype=counts.dtype, device=counts.device)

    for i, ct_pair in enumerate(cell_type_pairs):
        ct_t, _ = ct_pair

        ct_mask = cell_types == ct_t
        if isinstance(ct_mask, np.ndarray):
            ct_mask = torch.tensor(ct_mask, device=counts.device)

        counts_ct = counts[:, ct_mask]
        D_ct = D[i][ct_mask]
        gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

        lc_maxs = compute_max_cs(D_ct, counts_ct, gene_pairs_ind)
        lc_maxs = torch.where(lc_maxs == 0, torch.tensor(1.0, device=counts.device), lc_maxs)

        c_values = lcs[i] if lcs.ndim == 2 else lcs[i : i + 1]  # allow 1D or 2D lcs
        c_values_norm[i] = c_values / lc_maxs
        c_values_norm[i] = torch.where(
            torch.isinf(c_values_norm[i]),
            torch.tensor(1.0, device=counts.device),
            c_values_norm[i],
        )

    return c_values_norm


def normalize_values_old(
    counts,
    cell_types,
    cell_type_pairs,
    gene_pairs_per_ct_pair_ind,
    lcs,
    D,
):
    """Normalize legacy cell type-aware communication scores."""
    c_values_norm = np.zeros(lcs.shape)
    for i in range(len(cell_type_pairs)):
        ct_pair = cell_type_pairs[i]
        ct_t, ct_u = ct_pair
        cell_type_t_mask = [ct == ct_t for ct in cell_types]
        counts_ct_t = counts[:, cell_type_t_mask]
        D_ct_t = D[i][cell_type_t_mask]
        gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]
        lc_maxs = compute_max_cs_old(D_ct_t, counts_ct_t, gene_pairs_ind)
        lc_maxs[lc_maxs == 0] = 1
        c_values_norm[i] = lcs[i] / lc_maxs
        c_values_norm[i][np.isinf(c_values_norm[i])] = 1

    return c_values_norm


def compute_max_cs_old(node_degrees, counts, gene_pairs_ind):
    """Compute max communication scores per gene pair using NumPy."""
    result = np.zeros(len(gene_pairs_ind))
    for i, gene_pair_ind in enumerate(gene_pairs_ind):
        vals = (
            counts[gene_pair_ind[0]]
            if type(gene_pair_ind[0]) is not list
            else np.mean(counts[gene_pair_ind[0]], axis=0)
        )
        result[i] = compute_max_cs_gp_old(vals, node_degrees)

    return result


@jit(nopython=True)
def compute_max_cs_gp_old(vals, node_degrees):
    """Compute max communication score for one gene vector using NumPy."""
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2


def center_ct_counts_torch(counts, num_umi, model, cell_types):
    """Center counts within cell types.

    counts: Tensor [genes, cells]
    num_umi: Tensor [cells]
    model: 'bernoulli', 'danb', 'normal', or 'none'

    Returns
    -------
        Centered counts within cell types: Tensor [genes, cells]
    """
    # Binarize if using Bernoulli
    if model == "bernoulli":
        counts = (counts > 0).double()
        mu, var, _ = models.apply_model_per_cell_type(
            models.bernoulli_model_torch, counts, num_umi, cell_types
        )
    elif model == "danb":
        mu, var, _ = models.apply_model_per_cell_type(
            models.danb_model_torch, counts, num_umi, cell_types
        )
    elif model == "normal":
        mu, var, _ = models.apply_model_per_cell_type(
            models.normal_model_torch, counts, num_umi, cell_types
        )
    elif model == "none":
        mu, var, _ = models.apply_model_per_cell_type(
            models.none_model_torch, counts, num_umi, cell_types
        )
    else:
        raise ValueError(f"Unsupported model type: {model}")

    # Avoid division by zero
    std = torch.sqrt(var)
    std[std == 0] = 1.0

    centered = (counts - mu) / std
    centered[centered == 0] = 0  # Optional: to match old behavior

    return centered


def create_centered_counts_ct(counts, model, num_umi, cell_types):
    """Create centered or standardized counts within cell types."""
    out = np.zeros_like(counts, dtype="double")

    for i in range(out.shape[0]):
        vals_x = counts[i]

        out_x = create_centered_counts_row_ct(vals_x, model, num_umi, cell_types)

        out[i] = out_x

    return out


def create_centered_counts_row_ct(vals_x, model, num_umi, cell_types):
    """Center one count vector within cell types."""
    if model == "bernoulli":
        vals_x = (vals_x > 0).astype("double")
        mu_x, var_x, x2_x = models.bernoulli_model(vals_x, num_umi)

    elif model == "danb":
        mu_x, var_x, x2_x = models.ct_danb_model(vals_x, num_umi, cell_types)

    elif model == "normal":
        mu_x, var_x, x2_x = models.normal_model(vals_x, num_umi)

    elif model == "none":
        mu_x, var_x, x2_x = models.none_model(vals_x, num_umi)

    else:
        raise Exception(f"Invalid Model: {model}")

    var_x[var_x == 0] = 1
    out_x = (vals_x - mu_x) / (var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


def compute_Z_scores_cellcom_p(
    ct_pair, cell_type_pairs, gene_pair_cor, gene_pairs_per_ct_pair_ind, Wtot2, eg2s
):
    """Compute parametric cell communication Z-scores for one cell type pair."""
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = gene_pair_cor[i, :, :]
    # If all gene pairs are considered irrespective of the cell type pair, use
    # `gene_pairs_ind` directly.
    gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

    C = []
    EG2 = []
    for gene_pair_ind in gene_pairs_ind:
        g1_ind, g2_ind = gene_pair_ind
        lc = gene_pair_cor_ct[g1_ind, g2_ind]
        if g1_ind == g2_ind:
            eg2 = Wtot2[i]
        else:
            eg2 = eg2s[i][g1_ind]
        C.append(lc)
        EG2.append(eg2)

    EG = [0 for i in range(len(gene_pairs_ind))]

    stdG = [(EG2[i] - EG[i] ** 2) ** 0.5 for i in range(len(gene_pairs_ind))]
    stdG = [1 if stdG[i] == 0 else stdG[i] for i in range(len(stdG))]

    Z = [(C[i] - EG[i]) / stdG[i] for i in range(len(gene_pairs_ind))]

    return (C, Z)


def extract_results_cellcom_np(
    ct_pair, cell_type_pairs, gene_pair_cor, pvals, gene_pairs_per_ct_pair_ind
):
    """Extract non-parametric cell communication results for one cell type pair."""
    i = cell_type_pairs.index(ct_pair)
    gene_pair_cor_ct = gene_pair_cor[i, :, :]
    pvals_ct = pvals[i, :, :]
    # If all gene pairs are considered irrespective of the cell type pair, use
    # `gene_pairs_ind` directly.
    gene_pairs_ind = gene_pairs_per_ct_pair_ind[ct_pair]

    C = []
    p_values = []
    for gene_pair_ind in gene_pairs_ind:
        g1_ind, g2_ind = gene_pair_ind
        lc = gene_pair_cor_ct[g1_ind, g2_ind]
        p_value = pvals_ct[g1_ind, g2_ind]

        C.append(lc.reshape(1))
        p_values.append(p_value.reshape(1))

    C = list(np.concatenate(C))
    p_values = list(np.concatenate(p_values))

    return (C, p_values)


@njit
def expand_ct_pairs_cellcom(pairs, vals, N):
    """Expand sparse cell type-pair values into dense matrices."""
    out = [np.zeros((N, N)) for k in range(len(vals))]

    for k in range(len(out)):
        for i in range(len(pairs)):
            x = pairs[i, 0]
            y = pairs[i, 1]
            v = vals[k][i]

            out[k][x, y] = v

    return out


def get_ct_pair_counts_and_weights(
    counts, weights, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind
):
    """Extract count and weight tensors for all cell type pairs."""
    c_nrow, c_ncol = counts.shape
    w_nrow, w_ncol = weights.shape
    n_ct_pairs = len(cell_type_pairs)

    extract_counts_weights_results = partial(
        extract_ct_pair_counts_weights,
        counts=counts,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
        gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
    )
    results = list(map(extract_counts_weights_results, cell_type_pairs))

    c_new_data_t_all = [x[0] for x in results]
    c_new_coords_3d_t_all = [x[1] for x in results]
    c_new_coords_3d_t_all = np.hstack(c_new_coords_3d_t_all)
    c_new_data_t_all = np.concatenate(c_new_data_t_all)

    c_new_data_u_all = [x[2] for x in results]
    c_new_coords_3d_u_all = [x[3] for x in results]
    c_new_coords_3d_u_all = np.hstack(c_new_coords_3d_u_all)
    c_new_data_u_all = np.concatenate(c_new_data_u_all)

    w_new_data_all = [x[4] for x in results]
    w_new_coords_3d_all = [x[5] for x in results]
    w_new_coords_3d_all = np.hstack(w_new_coords_3d_all)
    w_new_data_all = np.concatenate(w_new_data_all)

    counts_ct_pairs_t = sparse.COO(
        c_new_coords_3d_t_all, c_new_data_t_all, shape=(n_ct_pairs, c_nrow, c_ncol)
    )
    counts_ct_pairs_u = sparse.COO(
        c_new_coords_3d_u_all, c_new_data_u_all, shape=(n_ct_pairs, c_nrow, c_ncol)
    )
    weights_ct_pairs = sparse.COO(
        w_new_coords_3d_all, w_new_data_all, shape=(n_ct_pairs, w_nrow, w_ncol)
    )

    return counts_ct_pairs_t, counts_ct_pairs_u, weights_ct_pairs


def get_ct_pair_counts_and_weights_null(
    counts_ct_pairs_t, counts_ct_pairs_u, weights, cell_type_pairs, cell_types, M
):
    """Build null count and weight tensors for cell type pairs."""
    n_cells = counts_ct_pairs_t.shape[2]
    cell_permutations = np.vstack([np.random.permutation(n_cells) for _ in range(M)])

    n_ct_pairs, c_nrow, c_ncol = counts_ct_pairs_t.shape
    w_nrow, w_ncol = weights.shape

    extract_counts_weights_results_null = partial(
        extract_ct_pair_counts_weights_null,
        permutations=cell_permutations,
        counts_ct_pairs_t=counts_ct_pairs_t,
        counts_ct_pairs_u=counts_ct_pairs_u,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types,
    )
    results_null = list(map(extract_counts_weights_results_null, cell_permutations))

    c_null_data_t_all = [x[0] for x in results_null]
    c_null_coords_4d_t_all = [x[1] for x in results_null]
    c_null_coords_4d_t_all = np.hstack(c_null_coords_4d_t_all)
    c_null_data_t_all = np.concatenate(c_null_data_t_all)

    c_null_data_u_all = [x[2] for x in results_null]
    c_null_coords_4d_u_all = [x[3] for x in results_null]
    c_null_coords_4d_u_all = np.hstack(c_null_coords_4d_u_all)
    c_null_data_u_all = np.concatenate(c_null_data_u_all)

    w_null_data_all = [x[4] for x in results_null]
    w_null_coords_4d_all = [x[5] for x in results_null]
    w_null_coords_4d_all = np.hstack(w_null_coords_4d_all)
    w_null_data_all = np.concatenate(w_null_data_all)

    counts_ct_pairs_t_null = sparse.COO(
        c_null_coords_4d_t_all, c_null_data_t_all, shape=(M, n_ct_pairs, c_nrow, c_ncol)
    )
    counts_ct_pairs_u_null = sparse.COO(
        c_null_coords_4d_u_all, c_null_data_u_all, shape=(M, n_ct_pairs, c_nrow, c_ncol)
    )
    weights_ct_pairs_null = sparse.COO(
        w_null_coords_4d_all, w_null_data_all, shape=(M, n_ct_pairs, w_nrow, w_ncol)
    )

    return counts_ct_pairs_t_null, counts_ct_pairs_u_null, weights_ct_pairs_null


def extract_ct_pair_counts_weights(
    ct_pair, counts, weights, cell_type_pairs, cell_types, gene_pairs_per_ct_pair_ind
):
    """Extract counts and weights for one cell type pair."""
    i = cell_type_pairs.index(ct_pair)

    ct_t, ct_u = cell_type_pairs[i]
    gene_pairs_per_ct_pair_ind_i = gene_pairs_per_ct_pair_ind[(ct_t, ct_u)]
    ct_t_mask = cell_types.values == ct_t
    ct_t_mask_coords = np.argwhere(ct_t_mask)
    ct_u_mask = cell_types.values == ct_u
    ct_u_mask_coords = np.argwhere(ct_u_mask)

    ct_t_genes = np.unique([t[0] for t in gene_pairs_per_ct_pair_ind_i])
    ct_u_genes = np.unique([t[1] for t in gene_pairs_per_ct_pair_ind_i])

    c_old_coords = counts.coords
    w_old_coords = weights.coords

    # Counts

    c_row_coords_t, c_col_coords_t = np.meshgrid(ct_t_genes, ct_t_mask_coords, indexing="ij")
    c_row_coords_t = c_row_coords_t.ravel()
    c_col_coords_t = c_col_coords_t.ravel()
    c_new_coords_t = np.vstack((c_row_coords_t, c_col_coords_t))

    c_row_coords_u, c_col_coords_u = np.meshgrid(ct_u_genes, ct_u_mask_coords, indexing="ij")
    c_row_coords_u = c_row_coords_u.ravel()
    c_col_coords_u = c_col_coords_u.ravel()
    c_new_coords_u = np.vstack((c_row_coords_u, c_col_coords_u))

    c_matching_indices_t = np.where(np.all(np.isin(c_old_coords.T, c_new_coords_t.T), axis=1))[0]
    c_new_data_t = counts.data[c_matching_indices_t]
    c_new_coords_t = c_old_coords[:, c_matching_indices_t]

    c_matching_indices_u = np.where(np.all(np.isin(c_old_coords.T, c_new_coords_u.T), axis=1))[0]
    c_new_data_u = counts.data[c_matching_indices_u]
    c_new_coords_u = c_old_coords[:, c_matching_indices_u]

    c_coord_3d_t = np.full(c_new_coords_t.shape[1], fill_value=i)
    c_new_coords_3d_t = np.vstack((c_coord_3d_t, c_new_coords_t))

    c_coord_3d_u = np.full(c_new_coords_u.shape[1], fill_value=i)
    c_new_coords_3d_u = np.vstack((c_coord_3d_u, c_new_coords_u))

    # Weights

    w_row_coords, w_col_coords = np.meshgrid(ct_t_mask_coords, ct_u_mask_coords, indexing="ij")
    w_row_coords = w_row_coords.ravel()
    w_col_coords = w_col_coords.ravel()
    w_new_coords = np.vstack((w_row_coords, w_col_coords))

    w_matching_indices = np.where(np.all(np.isin(w_old_coords.T, w_new_coords.T), axis=1))[0]
    w_new_data = weights.data[w_matching_indices]
    w_new_coords = w_old_coords[:, w_matching_indices]

    w_coord_3d = np.full(w_new_coords.shape[1], fill_value=i)
    w_new_coords_3d = np.vstack((w_coord_3d, w_new_coords))

    return (
        c_new_data_t,
        c_new_coords_3d_t,
        c_new_data_u,
        c_new_coords_3d_u,
        w_new_data,
        w_new_coords_3d,
    )


def extract_ct_pair_counts_weights_null(
    permutation,
    permutations,
    counts_ct_pairs_t,
    counts_ct_pairs_u,
    weights,
    cell_type_pairs,
    cell_types,
):
    """Extract null counts and weights for one permutation."""
    i = np.where(np.all(permutations == permutation, axis=1))[0]
    cell_types_perm = pd.Series(cell_types[permutation])

    counts_ct_pairs_t_perm = counts_ct_pairs_t[:, :, permutation]
    c_perm_coords_t = counts_ct_pairs_t_perm.coords
    c_perm_data_t = counts_ct_pairs_t_perm.data
    counts_ct_pairs_u_perm = counts_ct_pairs_u[:, :, permutation]
    c_perm_coords_u = counts_ct_pairs_u_perm.coords
    c_perm_data_u = counts_ct_pairs_u_perm.data

    extract_weights_results = partial(
        extract_ct_pair_weights,
        weights=weights,
        cell_type_pairs=cell_type_pairs,
        cell_types=cell_types_perm,
    )
    weights_results = list(map(extract_weights_results, cell_type_pairs))

    w_perm_data_all = [x[0] for x in weights_results]
    w_perm_coords_3d_all = [x[1] for x in weights_results]
    w_perm_coords_3d_all = np.hstack(w_perm_coords_3d_all)
    w_perm_data_all = np.concatenate(w_perm_data_all)

    c_coord_4d_t = np.full(c_perm_coords_t.shape[1], fill_value=i)
    c_perm_coords_4d_t = np.vstack((c_coord_4d_t, c_perm_coords_t))

    c_coord_4d_u = np.full(c_perm_coords_u.shape[1], fill_value=i)
    c_perm_coords_4d_u = np.vstack((c_coord_4d_u, c_perm_coords_u))

    w_coord_4d_u = np.full(w_perm_coords_3d_all.shape[1], fill_value=i)
    w_perm_coords_4d_u = np.vstack((w_coord_4d_u, w_perm_coords_3d_all))

    return (
        c_perm_data_t,
        c_perm_coords_4d_t,
        c_perm_data_u,
        c_perm_coords_4d_u,
        w_perm_data_all,
        w_perm_coords_4d_u,
    )
