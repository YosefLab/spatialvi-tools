"""Visualization utilities for spatial analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def plot_spatial(
    adata: AnnData,
    color: str | list[str] | None = None,
    spatial_key: str = "spatial",
    size: float = 1.0,
    alpha: float = 1.0,
    cmap: str = "viridis",
    ax: Axes | None = None,
    show: bool = True,
    **kwargs,
) -> Axes | None:
    """Plot cells in spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object.
    color
        Key in obs or var_names to color by.
    spatial_key
        Key in obsm for spatial coordinates.
    size
        Point size.
    alpha
        Point transparency.
    cmap
        Colormap for continuous values.
    ax
        Matplotlib axes to use.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments for scatter.

    Returns
    -------
    Matplotlib axes if show=False.
    """
    import matplotlib.pyplot as plt

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if color is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=size, alpha=alpha, **kwargs)
    elif color in adata.obs.columns:
        values = adata.obs[color]
        # Check if categorical or string-like
        if hasattr(values, "cat"):
            # Pandas Categorical
            codes = values.cat.codes.values
            categories = values.cat.categories.tolist()
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=size, alpha=alpha, cmap="tab20", **kwargs)
            # Add legend
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=plt.cm.tab20(i / len(categories)),
                    markersize=8,
                    label=cat,
                )
                for i, cat in enumerate(categories)
            ]
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
        elif values.dtype == object or values.dtype.name == "category":
            # Object dtype or uncategorized strings - convert to codes
            import pandas as pd

            cat_values = pd.Categorical(values)
            codes = cat_values.codes
            categories = cat_values.categories.tolist()
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=size, alpha=alpha, cmap="tab20", **kwargs)
            # Add legend
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=plt.cm.tab20(i / len(categories)),
                    markersize=8,
                    label=cat,
                )
                for i, cat in enumerate(categories)
            ]
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            # Numeric values
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values.values, s=size, alpha=alpha, cmap=cmap, **kwargs)
            plt.colorbar(scatter, ax=ax, label=color)
    elif color in adata.var_names:
        gene_idx = adata.var_names.get_loc(color)
        values = adata.X[:, gene_idx]
        if hasattr(values, "toarray"):
            values = values.toarray().flatten()
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, alpha=alpha, cmap=cmap, **kwargs)
        plt.colorbar(scatter, ax=ax, label=color)
    else:
        raise ValueError(f"'{color}' not found in obs or var_names")

    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_aspect("equal")

    if show:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_proportions(
    adata: AnnData,
    proportions_key: str = "proportions",
    spatial_key: str = "spatial",
    cell_types: list[str] | None = None,
    ncols: int = 3,
    size: float = 10,
    cmap: str = "Reds",
    show: bool = True,
    **kwargs,
) -> Figure | None:
    """Plot cell type proportions in spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object.
    proportions_key
        Key in obsm for proportions.
    spatial_key
        Key in obsm for spatial coordinates.
    cell_types
        Cell types to plot. If None, plots all.
    ncols
        Number of columns in subplot grid.
    size
        Point size.
    cmap
        Colormap.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments for scatter.

    Returns
    -------
    Matplotlib figure if show=False.
    """
    import matplotlib.pyplot as plt

    if proportions_key not in adata.obsm:
        raise ValueError(f"Proportions not found at obsm['{proportions_key}']")

    proportions = adata.obsm[proportions_key]
    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    # Get cell type names
    if "cell_type_names" in adata.uns:
        all_cell_types = adata.uns["cell_type_names"]
    else:
        all_cell_types = [f"CellType_{i}" for i in range(proportions.shape[1])]

    if cell_types is None:
        cell_types = all_cell_types
        prop_indices = list(range(len(all_cell_types)))
    else:
        prop_indices = [all_cell_types.index(ct) for ct in cell_types]

    n_types = len(cell_types)
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if n_types > 1 else [axes]

    for i, (ct, ct_idx) in enumerate(zip(cell_types, prop_indices, strict=False)):
        ax = axes[i]
        values = proportions[:, ct_idx]

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, cmap=cmap, vmin=0, vmax=1, **kwargs)
        plt.colorbar(scatter, ax=ax)
        ax.set_title(ct)
        ax.set_aspect("equal")

    # Hide empty subplots
    for i in range(n_types, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if show:
        plt.show()
        return None

    return fig


def plot_interactions(
    interaction_matrix: NDArray,
    cell_types: list[str],
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs,
) -> Axes | None:
    """Plot cell-cell interaction matrix.

    Parameters
    ----------
    interaction_matrix
        Interaction scores matrix of shape (n_types, n_types).
    cell_types
        List of cell type names.
    cmap
        Colormap.
    vmin
        Minimum value for colormap.
    vmax
        Maximum value for colormap.
    annot
        Whether to annotate cells with values.
    ax
        Matplotlib axes to use.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments for heatmap.

    Returns
    -------
    Matplotlib axes if show=False.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Use seaborn if available
    try:
        import seaborn as sns

        sns.heatmap(
            interaction_matrix,
            xticklabels=cell_types,
            yticklabels=cell_types,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=".2f" if annot else None,
            ax=ax,
            **kwargs,
        )
    except ImportError:
        im = ax.imshow(interaction_matrix, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_xticks(range(len(cell_types)))
        ax.set_yticks(range(len(cell_types)))
        ax.set_xticklabels(cell_types, rotation=45, ha="right")
        ax.set_yticklabels(cell_types)
        plt.colorbar(im, ax=ax)

        if annot:
            for i in range(len(cell_types)):
                for j in range(len(cell_types)):
                    ax.text(j, i, f"{interaction_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title("Cell-Cell Interaction Scores")

    if show:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_niche_composition(
    adata: AnnData,
    composition_key: str = "niche_composition",
    spatial_key: str = "spatial",
    method: Literal["pie", "stacked", "dominant"] = "dominant",
    cell_types: list[str] | None = None,
    size: float = 50,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs,
) -> Axes | None:
    """Plot niche composition in spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object.
    composition_key
        Key in obsm for niche composition.
    spatial_key
        Key in obsm for spatial coordinates.
    method
        Visualization method:
        - "dominant": Color by dominant cell type
        - "stacked": Stacked bar at each location (for grid data)
        - "pie": Pie chart at each location (for sparse data)
    cell_types
        Cell type names.
    size
        Point/marker size.
    ax
        Matplotlib axes to use.
    show
        Whether to show the plot.
    **kwargs
        Additional arguments.

    Returns
    -------
    Matplotlib axes if show=False.
    """
    import matplotlib.pyplot as plt

    if composition_key not in adata.obsm:
        raise ValueError(f"Composition not found at obsm['{composition_key}']")

    composition = adata.obsm[composition_key]
    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    n_types = composition.shape[1]

    if cell_types is None:
        if "cell_type_names" in adata.uns:
            cell_types = adata.uns["cell_type_names"]
        else:
            cell_types = [f"CellType_{i}" for i in range(n_types)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if method == "dominant":
        dominant = np.argmax(composition, axis=1)
        ax.scatter(coords[:, 0], coords[:, 1], c=dominant, s=size, cmap="tab20", **kwargs)
        # Legend
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab20(i / n_types), markersize=8, label=ct
            )
            for i, ct in enumerate(cell_types)
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

    elif method == "pie":
        # Small pie charts at each location (works best for sparse grids)
        from matplotlib.patches import Wedge

        # Subsample if too many points
        if len(coords) > 100:
            logger.warning("Too many points for pie charts, subsampling to 100")
            indices = np.random.choice(len(coords), 100, replace=False)
            coords = coords[indices]
            composition = composition[indices]

        colors = plt.cm.tab20(np.linspace(0, 1, n_types))
        radius = size / 10

        for i in range(len(coords)):
            x, y = coords[i]
            props = composition[i]

            start_angle = 0
            for j, prop in enumerate(props):
                if prop > 0.01:  # Skip very small slices
                    end_angle = start_angle + prop * 360
                    wedge = Wedge(
                        (x, y), radius, start_angle, end_angle, facecolor=colors[j], edgecolor="white", linewidth=0.5
                    )
                    ax.add_patch(wedge)
                    start_angle = end_angle

        ax.set_xlim(coords[:, 0].min() - radius * 2, coords[:, 0].max() + radius * 2)
        ax.set_ylim(coords[:, 1].min() - radius * 2, coords[:, 1].max() + radius * 2)

    else:
        raise ValueError(f"Unknown method: {method}")

    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.set_aspect("equal")
    ax.set_title("Niche Composition")

    if show:
        plt.tight_layout()
        plt.show()
        return None

    return ax
