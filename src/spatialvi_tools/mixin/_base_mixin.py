"""Reusable mixins for handling AnnData objects.

This module defines a simple mixin to validate and cache AnnData input.  It
mimics the behaviour of the mixin classes in `scvi-tools` but remains
lightweight.  Classes that expect an AnnData input can inherit from
``AnnDataMixin`` to automatically store the supplied ``AnnData`` and
validate that required keys are present.
"""

from __future__ import annotations

from typing import Any
import anndata as ad

class AnnDataMixin:
    """Mixin providing AnnData validation and caching.

    Parameters
    ----------
    adata:
        An AnnData object containing counts and optional metadata.
    """

    _REQUIRED_KEYS: dict[str, str] = {}

    def __init__(self, adata: ad.AnnData) -> None:
        if not isinstance(adata, ad.AnnData):
            raise TypeError(
                f"Expected `adata` to be an AnnData, got {type(adata).__name__}."
            )
        # Validate presence of required observations
        for key, message in self._REQUIRED_KEYS.items():
            if key not in adata.obsm and key not in adata.layers and key not in adata.obsm_keys():
                raise ValueError(
                    f"AnnData is missing required key '{key}': {message}."
                )
        self.adata = adata.copy()

    def get_from_adata(self, key: str) -> Any:
        """Helper to retrieve data from various AnnData attributes.

        The search order is ``obs``, ``var``, ``obsm``, ``layers``.

        Parameters
        ----------
        key:
            Name of the field to retrieve.

        Returns
        -------
        Any
            The corresponding field from the AnnData if found, otherwise raises
            ``KeyError``.
        """
        if key in self.adata.obs:
            return self.adata.obs[key]
        if key in self.adata.var:
            return self.adata.var[key]
        if key in self.adata.obsm:
            return self.adata.obsm[key]
        if key in self.adata.layers:
            return self.adata.layers[key]
        raise KeyError(f"Key '{key}' not found in AnnData.")