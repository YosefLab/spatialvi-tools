from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from rich import print
from sklearn.gaussian_process import GaussianProcessClassifier


@dataclass
class DifferentialExpressionResults:
    """Dataclass for storing the results of the differential expression analysis,
    including the GP classifier
    """  # noqa: D205

    gpc: GaussianProcessClassifier
    g1_g2: pd.DataFrame
    g1_n1: pd.DataFrame
    n1_g2: pd.DataFrame
    n1_n2: Union[pd.DataFrame, None] = field(default=None)  # noqa: UP007
    n1_index: Union[np.array, None] = field(default=None)  # noqa: UP007
    n2_index: Union[np.array, None] = field(default=None)  # noqa: UP007

    def gpc_info(self):
        """Print the log marginal likelihood value and the kernel
        of the Gaussian Process Classifier

        """  # noqa: D205
        print("Training score: ", self.gpc.train_score_)
        print("Marginal likelihood: ", self.gpc.log_marginal_likelihood_value_)
        print("Kernel: ", self.gpc.kernel_)

    def plot(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        filter: Iterable | None = None,
        background_filter: Iterable | None = None,
        markersize: int = 50,
        fontsize: int = 10,
        chosen_colormap: str = "seismic",
        path_to_save: str | None = None,
        show_plot: bool = True,
        dpi: int = 1000,
        margin: float = 0.1,
        manual_limits: tuple | None = None,
        legend_loc: str = "upper right",
    ) -> None:
        """Plot the results of the differential expression analysis.

        See :func:`scviva.plotting.scviva_de.plot_niche_de_decision_boundary` for parameter
        descriptions.
        """
        from scviva.plotting.scviva_de import plot_niche_de_decision_boundary

        plot_niche_de_decision_boundary(
            self,
            X=X,
            y=y,
            filter=filter,
            background_filter=background_filter,
            markersize=markersize,
            fontsize=fontsize,
            chosen_colormap=chosen_colormap,
            path_to_save=path_to_save,
            show_plot=show_plot,
            dpi=dpi,
            margin=margin,
            manual_limits=manual_limits,
            legend_loc=legend_loc,
        )
