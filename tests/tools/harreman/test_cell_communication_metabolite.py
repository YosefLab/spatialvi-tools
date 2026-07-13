import torch

from scviva.tools.harreman.tools.cell_communication import (
    compute_metabolite_cs,
    compute_metabolite_cs_ct,
)


def test_compute_metabolite_cs_scalar_per_metabolite():
    cs_gp = torch.tensor([1.0, 2.0, 3.0, 4.0])
    gene_pair_dict = {"metabolite_a": [0, 1], "metabolite_b": [2, 3]}
    result = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
    assert torch.equal(result, torch.tensor([3.0, 7.0]))


def test_compute_metabolite_cs_per_cell():
    # shape (cells=2, gene_pairs=4)
    cs_gp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    gene_pair_dict = {"metabolite_a": [0, 1], "metabolite_b": [2, 3]}
    result = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=True)
    assert torch.equal(result, torch.tensor([[3.0, 7.0], [30.0, 70.0]]))


def test_compute_metabolite_cs_single_metabolite_all_pairs():
    cs_gp = torch.tensor([1.0, 2.0, 3.0])
    gene_pair_dict = {"only_metabolite": [0, 1, 2]}
    result = compute_metabolite_cs(cs_gp, gene_pair_dict, interacting_cell_scores=False)
    assert torch.equal(result, torch.tensor([6.0]))


def test_compute_metabolite_cs_ct_scalar_no_masking():
    # shape (ct_pairs=2, gene_pairs=4)
    cs_gp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    gene_pair_dict = {"metabolite_a": [0, 1], "metabolite_b": [2, 3]}
    result = compute_metabolite_cs_ct(
        cs_gp, cell_type_key=None, gene_pair_dict=gene_pair_dict, interacting_cell_scores=False
    )
    assert torch.equal(result, torch.tensor([[3.0, 7.0], [30.0, 70.0]]))


def test_compute_metabolite_cs_ct_per_cell_no_masking():
    # shape (ct_pairs=2, cells=2, gene_pairs=3)
    cs_gp = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    gene_pair_dict = {"only_metabolite": [0, 1, 2]}
    result = compute_metabolite_cs_ct(
        cs_gp, cell_type_key=None, gene_pair_dict=gene_pair_dict, interacting_cell_scores=True
    )
    assert torch.equal(result, torch.tensor([[[6.0], [15.0]], [[24.0], [33.0]]]))


def test_compute_metabolite_cs_ct_masks_non_specific_ct_pairs():
    # shape (ct_pairs=2, gene_pairs=4); ct_pair index 1 gets masked to only gene pairs [2, 3]
    cs_gp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    gene_pair_dict = {"metabolite_a": [0, 1], "metabolite_b": [2, 3]}
    gene_pairs_per_ct_pair_ind = {"ct_pair_0": [0, 1], "ct_pair_1": [2, 3]}
    result = compute_metabolite_cs_ct(
        cs_gp.clone(),
        cell_type_key="cell_type",
        gene_pair_dict=gene_pair_dict,
        gene_pairs_per_ct_pair_ind=gene_pairs_per_ct_pair_ind,
        ct_specific_gene_pairs=[1],
        interacting_cell_scores=False,
    )
    # ct_pair_0 (row 0) untouched: metabolite_a=1+2=3, metabolite_b=3+4=7
    # ct_pair_1 (row 1) masked to indices [2,3]: metabolite_a=0 (both indices zeroed),
    # metabolite_b=30+40=70
    assert torch.equal(result, torch.tensor([[3.0, 7.0], [0.0, 70.0]]))
