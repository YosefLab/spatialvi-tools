from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scviva.tools.harreman._constants import (
    HARREMAN_AUTOCORR_RESULTS_KEY,
    HARREMAN_CCC_RESULTS_KEY,
    HARREMAN_CT_CCC_RESULTS_KEY,
    HARREMAN_CT_ICS_RESULTS_KEY,
    HARREMAN_GENE_PAIRS_RESULTS_KEY,
    HARREMAN_ICS_RESULTS_KEY,
    HARREMAN_PARAMS_KEY,
    HARREMAN_SIG_GP_SUFFIX,
    HARREMAN_SIG_M_SUFFIX,
    HARREMAN_UNS_KEY,
)

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


@dataclass
class HarremanResults:
    """Typed view over the top-level ``adata.uns`` keys written by analysis steps.

    The compute steps write their outputs at the top level of ``adata.uns`` (not under
    ``adata.uns['harreman']``) to match the internal read/write conventions of
    :mod:`scviva.tools.harreman.tools.cell_communication`. All fields are direct
    references to the stored objects — no copy.
    """

    autocorrelation: pd.DataFrame | None
    gene_pairs: list[str] | None
    cell_communication: dict | None
    ct_cell_communication: dict | None
    interacting_cell_scores: Any | None
    significant_interactions: dict | None
    params: dict

    @classmethod
    def from_adata_uns(cls, uns: dict, ccc_mode: str | None) -> HarremanResults:
        """Construct from ``adata.uns`` (the full top-level mapping).

        Parameters
        ----------
        uns
            The full ``adata.uns`` mapping.
        ccc_mode
            ``"standard"`` or ``"cell_type"``, matching the mode last used with
            ``compute_cell_communication``/``compute_interacting_cell_scores``. ``None``
            if cell communication has not been computed yet. Determines which top-level
            key backs ``interacting_cell_scores`` and ``significant_interactions``.
        """
        ics_key = (
            HARREMAN_CT_ICS_RESULTS_KEY if ccc_mode == "cell_type" else HARREMAN_ICS_RESULTS_KEY
        )
        sig_source_key = (
            HARREMAN_CT_CCC_RESULTS_KEY if ccc_mode == "cell_type" else HARREMAN_CCC_RESULTS_KEY
        )
        sig_source = uns.get(sig_source_key) or {}
        significant_interactions = None
        if HARREMAN_SIG_GP_SUFFIX in sig_source or HARREMAN_SIG_M_SUFFIX in sig_source:
            significant_interactions = {
                "gp": sig_source.get(HARREMAN_SIG_GP_SUFFIX),
                "m": sig_source.get(HARREMAN_SIG_M_SUFFIX),
            }

        return cls(
            autocorrelation=uns.get(HARREMAN_AUTOCORR_RESULTS_KEY),
            gene_pairs=uns.get(HARREMAN_GENE_PAIRS_RESULTS_KEY),
            cell_communication=uns.get(HARREMAN_CCC_RESULTS_KEY),
            ct_cell_communication=uns.get(HARREMAN_CT_CCC_RESULTS_KEY),
            interacting_cell_scores=uns.get(ics_key),
            significant_interactions=significant_interactions,
            params=uns[HARREMAN_UNS_KEY][HARREMAN_PARAMS_KEY],
        )
