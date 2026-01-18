"""Wrapper for the LAMBDA cell type annotation framework.

The :class:`Lambda` implements a simple interface to the
language-model based cell type annotation algorithm known as LAMBDA.
LAMBDA annotates clusters of cells without relying on reference datasets
by leveraging large language models to propose cell type names and
refining them via multiple rounds of reasoning.

Unlike most other models in this package, LAMBDA is a post-processing
algorithm rather than a trainable generative model. It does not learn
parameters from data; instead, it clusters the input data, extracts
differentially expressed genes and interacts with a language model API
to assign cell type names. Therefore the :meth:`train` method of
this wrapper simply stores configuration and instantiates an underlying
LAMBDA `Agent` object, whereas the :meth:`predict` method carries out
the annotation process.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from anndata import AnnData

logger = logging.getLogger(__name__)

try:
    from LAMBDA import Agent
except ImportError:
    Agent = None


class Lambda:
    """Language-model based cell type annotation.

    LAMBDA uses large language models to automatically annotate cell
    clusters based on differentially expressed marker genes.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression data; gene symbols must be
        present in ``adata.var.index``.
    location
        Optional text describing the anatomical location of the sample
        (e.g. "cerebral cortex"). Passed to the LLM to guide annotation.
    organism
        Optional species name (e.g. "human" or "mouse"). Passed to the LLM.
    provider
        Backend used for the language model. Possible values include
        "openai" and "google". Ensure that the corresponding API key is
        set in the environment.
    num_parallel
        Number of clusters to annotate concurrently.
    n_top_genes
        Number of differentially expressed genes used for annotation.
    resolution
        Resolution parameter for Leiden clustering when LAMBDA performs
        automatic clustering.
    **kwargs
        Additional arguments passed directly to ``Agent``.

    Examples
    --------
    >>> import scanpy as sc
    >>> from spatialvi.external import Lambda
    >>> adata = sc.read("data.h5ad")
    >>> model = Lambda(adata, organism="human", location="brain")
    >>> model.train()  # Prepares the agent
    >>> adata = model.predict()  # Annotates clusters
    >>> print(adata.obs["lambda_0"].value_counts())
    """

    def __init__(
        self,
        adata: AnnData,
        location: str | None = None,
        organism: str | None = None,
        provider: str = "openai",
        num_parallel: int = 10,
        n_top_genes: int = 50,
        resolution: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.adata = adata
        self.location = location
        self.organism = organism
        self.provider = provider
        self.num_parallel = num_parallel
        self.n_top_genes = n_top_genes
        self.resolution = resolution
        self.kwargs = kwargs
        self._agent: Any = None
        self._is_trained = False

    @staticmethod
    def _require_agent() -> None:
        if Agent is None:
            raise ImportError(
                "The 'LAMBDA' package is not installed. "
                "Install it with ``pip install LAMBDA``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Prepare the LAMBDA agent.

        LAMBDA does not have a trainable model in the usual sense. This
        method instantiates an ``Agent`` and stores it for subsequent
        predictions.
        """
        self._require_agent()

        # Prepare optional arguments
        agent_kwargs = {
            "location": self.location,
            "organism": self.organism,
            "provider": self.provider,
            "num_parallel": self.num_parallel,
            "num_it": 3,
            "max_depth": 2,
            "n_top_genes": self.n_top_genes,
            "resolution": self.resolution,
        }
        agent_kwargs.update(self.kwargs)

        # Remove None values
        agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}

        # Instantiate underlying LAMBDA agent
        self._agent = Agent(self.adata, **agent_kwargs)
        self._is_trained = True
        logger.info("LAMBDA agent initialized.")

    def predict(
        self,
        group_key: str | Iterable[str] | None = "leiden",
        groups: str | Iterable[str] | None = None,
        key_is_hierarchical: bool = False,
        level: int = 0,
        store_key_prefix: str = "lambda",
        **kwargs: Any,
    ) -> AnnData:
        """Annotate clusters of the dataset with cell type names.

        Parameters
        ----------
        group_key
            Key(s) in ``adata.obs`` defining the clusters to annotate. If
            ``None``, the agent will perform its own clustering. If a
            sequence of strings is provided, it encodes a hierarchy of levels.
        groups
            Subset of groups to annotate. ``None`` means annotate all groups.
        key_is_hierarchical
            Whether ``group_key`` encodes a hierarchy.
        level
            If a hierarchy is present, index of the level to return results for.
        store_key_prefix
            Prefix under which to store the annotations in ``adata.obs``. Two
            columns will be created: ``f"{store_key_prefix}_{level}"`` for the
            raw label and ``f"{store_key_prefix}_score_{level}"`` for the
            confidence score.
        **kwargs
            Additional keyword arguments forwarded to ``Agent.annotate``.

        Returns
        -------
        The input AnnData with annotation columns added to ``obs``.
        """
        self._require_agent()
        if self._agent is None:
            # Create agent lazily if train() wasn't called
            self.train()

        # Call the annotation method on the Agent
        res = self._agent.annotate(
            group_key=group_key,
            groups=groups,
            key_is_hierarchical=key_is_hierarchical,
            **kwargs,
        )

        # The LAMBDA Agent stores results internally; extract and store in obs
        try:
            annotation_dict = self._agent.data.annotation
        except AttributeError:
            # Fallback: nothing to store
            logger.warning("Could not retrieve annotations from LAMBDA agent.")
            return self.adata

        # Determine annotation keys for the requested level
        label_key = f"{store_key_prefix}_{level}"
        score_key = f"{store_key_prefix}_score_{level}"

        if isinstance(annotation_dict, dict) and level in annotation_dict:
            df = annotation_dict[level]
            self.adata.obs[label_key] = df.iloc[:, 0]
            self.adata.obs[score_key] = df.iloc[:, 1]
            logger.info(f"Annotations stored in obs['{label_key}'] and obs['{score_key}']")

        return self.adata

    def get_annotations(self, level: int = 0) -> dict[str, Any]:
        """Get the annotation results for a specific level.

        Parameters
        ----------
        level
            The hierarchical level to retrieve.

        Returns
        -------
        Dictionary containing annotation results.
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized. Call train() or predict() first.")

        try:
            return self._agent.data.annotation.get(level, {})
        except AttributeError:
            return {}
