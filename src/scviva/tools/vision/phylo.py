"""PhyloVision utilities: Fitch-Hartigan parsimony, plasticity scores, and tree clustering.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).

Mirrors R VISION's ``computePlasticityScores`` /
``computeFitchHartiganParsimonyPerNode`` from ``FitchParsimony.R`` and
``AnalysisFunctions.R``, as well as ``maxSizeCladewiseTreeCluster`` from
``Microclusters.R``.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitch-Hartigan algorithm helpers
# ---------------------------------------------------------------------------


def _build_parent_map(root) -> dict[int, object]:
    """Return {id(clade): parent_clade} for every node in the tree.

    Parameters
    ----------
    root
        Root clade of a ``Bio.Phylo`` tree.

    Returns
    -------
    dict of int to object
        Mapping from ``id(clade)`` to its parent clade (``None`` for the root).
    """
    parent: dict[int, object] = {}

    def _dfs(clade, par) -> None:
        parent[id(clade)] = par
        for child in clade.clades:
            _dfs(child, clade)

    _dfs(root, None)
    return parent


def _fitch_bottom_up(
    root,
    metadata: dict[str, str],
) -> dict[int, frozenset[str]]:
    """Bottom-up Fitch-Hartigan pass for the subtree rooted at *root*.

    For each leaf: possible = {metadata[leaf]}.
    For each internal node: possible = mode of children's possible sets
    (union of most-frequent labels across children).

    Matches R VISION's ``bottomUpFitchHartigan``.

    Parameters
    ----------
    root
        Root clade of the subtree to process.
    metadata : dict of str to str
        Mapping from leaf name to categorical label.

    Returns
    -------
    dict of int to frozenset of str
        Mapping from ``id(node)`` to the set of most-parsimonious labels
        for that node.
    """
    possible: dict[int, frozenset[str]] = {}

    for node in root.find_clades(order="postorder"):
        if node.is_terminal():
            label = metadata.get(node.name)
            possible[id(node)] = frozenset([label] if label is not None else [])
        else:
            counter: Counter = Counter()
            for child in node.clades:
                for label in possible.get(id(child), frozenset()):
                    counter[label] += 1
            if counter:
                max_freq = max(counter.values())
                possible[id(node)] = frozenset(
                    lbl for lbl, freq in counter.items() if freq == max_freq
                )
            else:
                possible[id(node)] = frozenset()

    return possible


def _fitch_top_down(
    root,
    possible: dict[int, frozenset[str]],
) -> dict[int, str | None]:
    """Top-down Fitch-Hartigan pass; returns a maximum-parsimony assignment.

    Tie-breaking is deterministic (lexicographic minimum), unlike R VISION
    which uses ``sample()`` (random).  The parsimony score is identical for
    any valid max-parsimony assignment.

    Matches R VISION's ``topDownFitchHartigan``.

    Parameters
    ----------
    root
        Root clade of the subtree to process.
    possible : dict of int to frozenset of str
        Per-node candidate label sets, as returned by :func:`_fitch_bottom_up`.

    Returns
    -------
    dict of int to str or None
        Mapping from ``id(node)`` to its assigned label.
    """
    assignments: dict[int, str | None] = {}

    def _assign(node, parent_label: str | None) -> None:
        poss = possible.get(id(node), frozenset())
        if not poss:
            assignments[id(node)] = parent_label
        elif parent_label is not None and parent_label in poss:
            assignments[id(node)] = parent_label
        else:
            assignments[id(node)] = min(poss)  # deterministic tie-break
        for child in node.clades:
            _assign(child, assignments[id(node)])

    _assign(root, None)
    return assignments


def _score_parsimony(root, assignments: dict[int, str | None]) -> int:
    """Count label transitions across all edges in the subtree.

    Matches R VISION's ``scoreParsimony``.

    Parameters
    ----------
    root
        Root clade of the subtree to score.
    assignments : dict of int to str or None
        Per-node label assignment, as returned by :func:`_fitch_top_down`.

    Returns
    -------
    int
        Number of edges whose endpoints have different labels.
    """
    parsimony = 0
    for node in root.find_clades():
        par_label = assignments.get(id(node))
        for child in node.clades:
            if par_label != assignments.get(id(child)):
                parsimony += 1
    return parsimony


def _normalized_parsimony(root, metadata: dict[str, str]) -> float:
    """Normalized Fitch-Hartigan parsimony for the subtree rooted at *root*.

    = parsimony / (n_nodes - 1), where n_nodes is the number of nodes
    in the subtree (leaves + internal).

    Matches R VISION's ``computeNormalizedFitchHartiganParsimony(tree,
    metaData, source=node)``.

    Parameters
    ----------
    root
        Root clade of the subtree to score.
    metadata : dict of str to str
        Mapping from leaf name to categorical label.

    Returns
    -------
    float
        Normalized parsimony score in ``[0, 1]``, or 0.0 for single-node
        subtrees.
    """
    possible = _fitch_bottom_up(root, metadata)
    assignments = _fitch_top_down(root, possible)
    parsimony = _score_parsimony(root, assignments)
    n_nodes = sum(1 for _ in root.find_clades())
    return parsimony / (n_nodes - 1) if n_nodes > 1 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_plasticity_scores(
    adata: AnnData,
    newick: str,
) -> None:
    """Compute per-cell Fitch-Hartigan parsimony plasticity scores.

    For every categorical variable in ``adata.obs``, a ``{var}_plasticity``
    column is added to ``adata.obs``.  The score for cell *c* is the average
    normalised Fitch-Hartigan parsimony across all internal nodes on the path
    from the tree root to *c*.

    - **High plasticity**: the metadata label switches often along the
      lineage path → the trait is labile / plastic.
    - **Low plasticity**: the label is conserved along the lineage path.

    Mirrors R VISION's ``computePlasticityScores()``.

    Parameters
    ----------
    adata : AnnData
        Must have ``obs_names`` matching tree leaf labels.
    newick : str
        Newick-format tree string (or the ``adata.uns["vision_tree"]`` value).

    Returns
    -------
    None
        Modifies *adata* in-place, adding one ``{var}_plasticity`` column to
        ``adata.obs`` for each categorical variable found.
    """
    import io

    from Bio import Phylo

    tree = Phylo.read(io.StringIO(newick), "newick")
    terminals = tree.get_terminals()

    name_to_term = {t.name: t for t in terminals}
    parent = _build_parent_map(tree.root)

    # Identify categorical / object columns
    cat_cols: list[str] = [
        col
        for col in adata.obs.columns
        if pd.api.types.is_categorical_dtype(adata.obs[col]) or adata.obs[col].dtype == object
    ]

    if not cat_cols:
        logger.info("No categorical obs columns found; skipping plasticity scores.")
        return

    # Collect all internal nodes (needed for per-node parsimony computation)
    internal_nodes = [c for c in tree.find_clades() if not c.is_terminal()]
    logger.info(
        "Computing plasticity scores for %d categorical variable(s) across %d internal nodes.",
        len(cat_cols),
        len(internal_nodes),
    )

    for col in cat_cols:
        # Build metadata dict: leaf name → label string
        metadata: dict[str, str] = {
            name: str(adata.obs.at[name, col]) for name in adata.obs_names if name in name_to_term
        }
        if not metadata:
            continue

        # Compute normalised parsimony score for each internal node's subtree
        node_score: dict[int, float] = {
            id(node): _normalized_parsimony(node, metadata) for node in internal_nodes
        }

        # Per-leaf plasticity = average node score along root-to-leaf path
        plasticity: dict[str, float] = {}
        for leaf_name in adata.obs_names:
            if leaf_name not in name_to_term:
                plasticity[leaf_name] = np.nan
                continue

            leaf = name_to_term[leaf_name]
            ancestor_scores: list[float] = []

            # Walk from leaf's parent up to (and including) root
            node = leaf
            while parent[id(node)] is not None:
                node = parent[id(node)]
                s = node_score.get(id(node))
                if s is not None:
                    ancestor_scores.append(s)

            plasticity[leaf_name] = float(np.mean(ancestor_scores)) if ancestor_scores else 0.0

        out_col = f"{col}_plasticity"
        adata.obs[out_col] = [plasticity.get(name, np.nan) for name in adata.obs_names]
        logger.info("Plasticity scores stored in adata.obs['%s'].", out_col)


# ---------------------------------------------------------------------------
# Tree-based cell clustering  (maxSizeCladewiseTreeCluster)
# ---------------------------------------------------------------------------


def _n_leaves_map(root) -> dict[int, int]:
    """Return {id(clade): n_leaf_descendants} computed bottom-up.

    Parameters
    ----------
    root
        Root clade of the tree.

    Returns
    -------
    dict of int to int
        Mapping from ``id(clade)`` to its number of leaf descendants.
    """
    n: dict[int, int] = {}
    for clade in root.find_clades(order="postorder"):
        if clade.is_terminal():
            n[id(clade)] = 1
        else:
            n[id(clade)] = sum(n[id(c)] for c in clade.clades)
    return n


def _max_child_size(clade, n_leaves: dict[int, int]) -> int:
    """Max leaf count among direct children; returns 1 for leaves.

    Parameters
    ----------
    clade
        Clade whose children are inspected.
    n_leaves : dict of int to int
        Leaf-count map, as returned by :func:`_n_leaves_map`.

    Returns
    -------
    int
        Maximum leaf count among *clade*'s direct children (1 if terminal).
    """
    if clade.is_terminal():
        return 1
    return max(n_leaves[id(c)] for c in clade.clades)


def _trivial_dist(leaf1_path_to_root: list[int], leaf2_path_set: set) -> int:
    """Steps from leaf1 to LCA(leaf1, leaf2).

    Mirrors R VISION's ``trivial_dist``:  ``which(path1 == mrca)`` (1-indexed),
    i.e. the position of the MRCA node in the root-to-leaf path starting at
    leaf1.  Only relative ordering matters for the merge step.

    Parameters
    ----------
    leaf1_path_to_root : list of int
        Node ids from leaf1 up to (and including) the root.
    leaf2_path_set : set
        Node ids on the path from leaf2 up to the root.

    Returns
    -------
    int
        1-indexed position of the lowest common ancestor in *leaf1_path_to_root*.
    """
    for i, nid in enumerate(leaf1_path_to_root):
        if nid in leaf2_path_set:
            return i + 1  # 1-indexed to match R
    return len(leaf1_path_to_root) + 1


def _path_to_root(leaf, parent: dict[int, object]) -> list[int]:
    """Sequence of node ids from *leaf* up to (and including) the root.

    Parameters
    ----------
    leaf
        Leaf clade to start from.
    parent : dict of int to object
        Parent map, as returned by :func:`_build_parent_map`.

    Returns
    -------
    list of int
        Node ids from *leaf* to the root, inclusive.
    """
    path: list[int] = []
    node = leaf
    while node is not None:
        path.append(id(node))
        node = parent.get(id(node))
    return path


def _max_size_cladewise_cluster(root, target: int = 10) -> list[list[str]]:
    """Pure-tree implementation of R VISION's ``maxSizeCladewiseTreeCluster``.

    Algorithm
    ---------
    1. Start with the root as the sole cluster.
    2. Greedily expand the cluster whose **largest child sub-clade** has the
       most leaves, replacing it with its direct children.  Repeat until
       ``√n_tips`` clusters exist.
    3. If more than *target* clusters remain, iteratively merge the smallest
       cluster into its nearest neighbour by ``trivial_dist`` (steps from the
       smaller cluster's first leaf to their LCA).

    Parameters
    ----------
    root
        Root clade of the tree to cluster.
    target : int, optional
        Desired number of clusters, by default 10.

    Returns
    -------
    list of list of str
        Leaf names grouped by cluster.
    """
    n_leaves = _n_leaves_map(root)
    n_tips = n_leaves[id(root)]
    parent = _build_parent_map(root)

    # Step 1 & 2: greedy expansion
    # active: id → clade for nodes that define the current partition
    active: dict[int, object] = {id(root): root}
    sqrt_n = math.sqrt(n_tips)

    while len(active) < sqrt_n:
        # Pick the node with the largest max-child clade
        best_id = max(active, key=lambda nid: _max_child_size(active[nid], n_leaves))
        node = active.pop(best_id)

        if node.is_terminal():
            # Can't split a leaf; restore and stop
            active[id(node)] = node
            break

        for child in node.clades:
            active[id(child)] = child

    # Convert to lists of leaf names
    clusters: list[list[str]] = [
        [t.name for t in clade.get_terminals()] for clade in active.values()
    ]

    # Precompute leaf → path-to-root for trivial_dist
    leaf_by_name = {t.name: t for t in root.get_terminals()}

    def _path(name: str) -> list[int]:
        return _path_to_root(leaf_by_name[name], parent)

    # Step 3: merge smallest clusters down to target
    while len(clusters) > target:
        sizes = [len(c) for c in clusters]
        smallest_i = int(np.argmin(sizes))
        tip1_name = clusters[smallest_i][0]
        path1 = _path(tip1_name)

        dists: list[int] = []
        for i, cl in enumerate(clusters):
            if i == smallest_i:
                dists.append(n_tips + 1)  # sentinel: never choose self
            else:
                path2_set = set(_path(cl[0]))
                dists.append(_trivial_dist(path1, path2_set))

        closest_i = int(np.argmin(dists))

        merge_i = min(smallest_i, closest_i)
        other_i = max(smallest_i, closest_i)
        clusters[merge_i] = clusters[smallest_i] + clusters[closest_i]
        del clusters[other_i]

    return clusters


def cluster_cells_tree(
    adata: AnnData,
    newick: str,
    target: int = 10,
    obs_key: str = "VISION_Clusters_Tree",
) -> None:
    """Cluster cells by their position in a phylogenetic tree.

    Mirrors R VISION's tree-based clustering path in ``clusterCells()``
    (``methods-Vision.R``), which calls ``maxSizeCladewiseTreeCluster``
    (``Microclusters.R``) when a tree is available.

    The algorithm greedily splits the largest sub-clade until ``√n_cells``
    partitions exist, then merges small partitions back down to *target* using
    tree-distance (trivial_dist) as the merge criterion.  Results are stored
    as a categorical column in ``adata.obs[obs_key]``.

    Parameters
    ----------
    adata : AnnData
        AnnData whose ``obs_names`` must match leaf labels in *newick*.
    newick : str
        Newick-format tree string (or the value of ``adata.uns["vision_tree"]``).
    target : int, optional
        Desired number of clusters.  Default 10, matching R VISION's default.
    obs_key : str, optional
        Column name written to ``adata.obs``.  Default ``"VISION_Clusters_Tree"``.

    Returns
    -------
    None
        Modifies *adata* in-place, adding a categorical ``adata.obs[obs_key]``
        column.
    """
    import io

    from Bio import Phylo

    tree = Phylo.read(io.StringIO(newick), "newick")

    obs_set = set(adata.obs_names)
    # Warn if tree has leaves absent from adata (silently drop them)
    tree_leaves = {t.name for t in tree.get_terminals()}
    missing_in_adata = tree_leaves - obs_set
    if missing_in_adata:
        logger.warning(
            "%d tree leaves absent from adata.obs_names and will be ignored.",
            len(missing_in_adata),
        )

    clusters = _max_size_cladewise_cluster(tree.root, target=target)

    # Build obs_name → cluster_label mapping
    label_map: dict[str, str] = {}
    for cluster_idx, cell_names in enumerate(clusters):
        label = f"Cluster {cluster_idx + 1}"
        for name in cell_names:
            if name in obs_set:
                label_map[name] = label

    labels = [label_map.get(name, "Unassigned") for name in adata.obs_names]
    adata.obs[obs_key] = pd.Categorical(labels)
    logger.info(
        "Tree clustering stored in adata.obs['%s'] — %d clusters.",
        obs_key,
        len(clusters),
    )
