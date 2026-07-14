from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scviva.plotting._utils import prettify_axis


def plot_signature_for_selection(
    adata, signature, coords_obsm_key, s, vmin, vmax, figsize, cmap, colorbar
):
    """Plot signature scores on spatial coordinates for selection."""
    scores = adata.obsm["vision_signatures"]

    if isinstance(adata.obsm[coords_obsm_key], pd.DataFrame):
        coords = adata.obsm[coords_obsm_key].values
    else:
        coords = adata.obsm[coords_obsm_key]

    points = np.column_stack([coords[:, 0], coords[:, 1]])

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    prettify_axis(ax, spatial=True)
    p = plt.scatter(
        coords[:, 0], coords[:, 1], c=scores[signature], cmap=cmap, s=s, vmin=vmin, vmax=vmax
    )
    plt.title(signature)
    if colorbar:
        plt.colorbar()

    return p, ax, points


def plot_selection_histplot(adata, signature, group):
    """Plot histogram of signature scores for selection vs remainder."""
    adata.obs["selected"] = pd.Series(
        np.where(np.asarray(group) == 1, "Selection", "Remainder"), index=adata.obs_names
    )

    if signature not in adata.obs:
        adata.obs[signature] = adata.obsm["vision_signatures"][signature]

    sns.histplot(
        data=adata.obs,
        x=signature,
        hue=adata.obs["selected"].tolist(),
        bins=30,
        palette={"Selection": "#FF7F00", "Remainder": "#1F78B4"},
    )
    plt.show()

    return


def plot_vision_autocorrelation(
    adata,
    type: Literal["observations"] | Literal["signatures"] | None = None,
    center: int | None = 0.5,
    figsize: tuple | None = (1, 10),
    cmap: str | None = "coolwarm",
    cbar: bool | None = True,
):
    """Plot vision autocorrelation results."""
    if type not in ["observations", "signatures"]:
        raise ValueError('The "type" variable should be one of ["observations", "signatures"].')

    type_str = "vision_obs_df_scores" if type == "observations" else "vision_signature_scores"

    masked_data = adata.uns[type_str][["c_prime"]].where(
        (adata.uns[type_str][["fdr"]] < 0.05).values
    )
    masked_data = masked_data.sort_values("c_prime", ascending=False)
    masked_data.columns = ["Consistency"]

    plt.figure(figsize=figsize)
    sns.heatmap(masked_data, annot=masked_data, cmap=cmap, fmt=".2f", cbar=cbar, center=center)
    plt.show()

    return


def plot_vision_de_results(
    adata,
    type: Literal["observations"] | Literal["signatures"] | None = None,
    var: str = None,
    center: int | None = 0.5,
    figsize: tuple | None = (3, 10),
    cmap: str | None = "coolwarm",
    cbar: bool | None = True,
):
    """Plot vision differential expression results."""
    if var is None:
        raise ValueError('The "var" variable should be a categorical variable to plot.')

    if type not in ["observations", "signatures"]:
        raise ValueError('The "type" variable should be one of ["observations", "signatures"].')

    type_score_str = (
        f"one_vs_all_obs_cols_{var}_scores"
        if type == "observations"
        else f"one_vs_all_signatures_{var}_scores"
    )
    type_pval_str = (
        f"one_vs_all_obs_cols_{var}_pvals"
        if type == "observations"
        else f"one_vs_all_signatures_{var}_padj"
    )

    mask = adata.uns[type_pval_str] < 0.05

    plt.figure(figsize=figsize)
    sns.heatmap(
        adata.uns[type_score_str],
        mask=~mask,
        cmap=cmap,
        annot=mask.map(lambda x: "*" if x else ""),
        fmt="",
        cbar=cbar,
        center=center,
    )
    plt.show()

    return
