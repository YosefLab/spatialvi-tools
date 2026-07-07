from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scviva.tools.vision._constants import (
    DENDROGRAM_UNS_KEY,
    GENE_IMPORTANCE_UNS_KEY,
    META_DIFFERENTIAL_UNS_KEY,
    OBS_DF_SCORES_UNS_KEY,
    PROTEIN_AUTOCORR_UNS_KEY,
    PROTEIN_DIFFERENTIAL_UNS_KEY,
    SIG_CLUSTERS_UNS_KEY,
    SIGNATURE_DIFFERENTIAL_UNS_KEY,
    SIGNATURE_SCORES_UNS_KEY,
    SIGNATURES_OBSM_KEY,
)

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


@dataclass
class VisionResults:
    """Typed view over the ``adata.uns``/``adata.obsm`` keys written by analysis steps.

    All fields are direct references to the stored objects — no copy.
    """

    signature_scores: pd.DataFrame | None
    obs_df_scores: pd.DataFrame | None
    signature_autocorrelation: pd.DataFrame | None
    signature_differential: dict | None
    meta_differential: dict | None
    gene_importance: dict | None
    signature_clusters: dict | None
    signature_dendrogram: str | None
    protein_autocorrelation: Any | None
    protein_differential: str | None

    @classmethod
    def from_adata_uns(cls, uns: dict, obsm: dict) -> VisionResults:
        """Construct from ``adata.uns`` and ``adata.obsm`` (the full top-level mappings).

        Parameters
        ----------
        uns
            The full ``adata.uns`` mapping.
        obsm
            The full ``adata.obsm`` mapping.
        """
        return cls(
            signature_scores=obsm.get(SIGNATURES_OBSM_KEY),
            obs_df_scores=uns.get(OBS_DF_SCORES_UNS_KEY),
            signature_autocorrelation=uns.get(SIGNATURE_SCORES_UNS_KEY),
            signature_differential=uns.get(SIGNATURE_DIFFERENTIAL_UNS_KEY),
            meta_differential=uns.get(META_DIFFERENTIAL_UNS_KEY),
            gene_importance=uns.get(GENE_IMPORTANCE_UNS_KEY),
            signature_clusters=uns.get(SIG_CLUSTERS_UNS_KEY),
            signature_dendrogram=uns.get(DENDROGRAM_UNS_KEY),
            protein_autocorrelation=uns.get(PROTEIN_AUTOCORR_UNS_KEY),
            protein_differential=uns.get(PROTEIN_DIFFERENTIAL_UNS_KEY),
        )
