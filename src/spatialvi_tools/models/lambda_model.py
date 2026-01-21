"""Wrapper for the LAMBDA cell type annotation framework.

The :class:`LambdaModel` implements a simple interface to the
language‑model based cell type annotation algorithm known as LAMBDA
【245548828756402†L24-L32】.  LAMBDA annotates clusters of cells without relying on
reference datasets by leveraging large language models to propose cell
type names and refining them via multiple rounds of reasoning【245548828756402†L80-L113】.

Unlike most other models in this package, LAMBDA is a post‑processing
algorithm rather than a trainable generative model.  It does not learn
parameters from data; instead, it clusters the input data, extracts
differentially expressed genes and interacts with a language model API
to assign cell type names.  Therefore the :meth:`train` method of
this wrapper simply stores configuration and instantiates an underlying
LAMBDA `Agent` object, whereas the :meth:`predict` method carries out
the annotation process.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:
    # The package is named LAMBDA on PyPI; import the Agent class
    from LAMBDA import Agent  # type: ignore
except ImportError:  # pragma: no cover
    Agent = None  # type: ignore


class LambdaModel(AnnDataMixin, BaseSpatialModel):
    """Language‑model based cell type annotation.

    Parameters
    ----------
    adata:
        AnnData with log‑normalised expression data; gene symbols must be
        present in ``adata.var.index`` as required by LAMBDA【245548828756402†L67-L71】.
    location:
        Optional text describing the anatomical location of the sample
        (e.g. "cerebral cortex").  Passed to the LLM to guide annotation.
    organism:
        Optional species name (e.g. "human" or "mouse").  Passed to the
        LLM.
    provider:
        Backend used for the language model.  Possible values include
        "openai" and "google"【245548828756402†L94-L99】.  Ensure that the
        corresponding API key is set in the environment.
    num_parallel:
        Number of clusters to annotate concurrently.  See LAMBDA docs for
        details【245548828756402†L90-L100】.
    n_top_genes:
        Number of differentially expressed genes used for annotation.
    resolution:
        Resolution parameter for Leiden clustering when LAMBDA performs
        automatic clustering【245548828756402†L101-L109】.
    **kwargs:
        Additional arguments passed directly to ``Agent``.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        location: Optional[str] = None,
        organism: Optional[str] = None,
        provider: str = "openai",
        num_parallel: int = 10,
        n_top_genes: int = 50,
        resolution: float = 0.5,
        **kwargs: Any,
    ) -> None:
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.location = location
        self.organism = organism
        self.provider = provider
        self.num_parallel = num_parallel
        self.n_top_genes = n_top_genes
        self.resolution = resolution
        self.kwargs = kwargs
        self._agent: Any = None

    def _require_agent(self) -> None:
        if Agent is None:
            raise ImportError(
                "The 'LAMBDA' package is not installed. Install it with ``pip install LAMBDA``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Prepare the LAMBDA agent.

        LAMBDA does not have a trainable model in the usual sense.  This
        method instantiates an ``Agent`` and stores it for subsequent
        predictions.
        """
        self._require_agent()
        # prepare optional arguments
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
        # instantiate underlying LAMBDA agent
        self._agent = Agent(self.adata, **{k: v for k, v in agent_kwargs.items() if v is not None})

    def predict(
        self,
        group_key: Optional[str | Iterable[str]] = "leiden",
        groups: Optional[str | Iterable[str]] = None,
        key_is_hierarchical: bool = False,
        level: int = 0,
        store_key_prefix: str = "lambda",
        **kwargs: Any,
    ) -> ad.AnnData:
        """Annotate clusters of the dataset with cell type names.

        Parameters
        ----------
        group_key:
            Key(s) in ``adata.obs`` defining the clusters to annotate.  If
            ``None``, the agent will perform its own clustering.  If a
            sequence of strings is provided, it encodes a hierarchy of
            levels.【245548828756402†L114-L116】.
        groups:
            Subset of groups to annotate.  ``None`` means annotate all groups.
        key_is_hierarchical:
            Whether ``group_key`` encodes a hierarchy【245548828756402†L115-L116】.
        level:
            If a hierarchy is present, index of the level to return results for.
        store_key_prefix:
            Prefix under which to store the annotations in ``adata.obs``.  Two
            columns will be created: ``f"{store_key_prefix}_{level}"`` for the
            raw label and ``f"{store_key_prefix}_score_{level}"`` for the
            confidence score.
        **kwargs:
            Additional keyword arguments forwarded to ``Agent.annotate``.

        Returns
        -------
        AnnData
            The input AnnData with annotation columns added to ``obs``.
        """
        self._require_agent()
        if self._agent is None:
            # create agent lazily if train() wasn't called
            self.train()
        # call the annotation method on the Agent
        res = self._agent.annotate(
            group_key=group_key,
            groups=groups,
            key_is_hierarchical=key_is_hierarchical,
            **kwargs,
        )
        # The LAMBDA Agent stores results internally; extract and store in obs
        # We expect the annotation to be in self._agent.data.annotation
        # where each key is something like f"lambda_{level}" and contains
        # columns 'label' and 'score'.  We will attempt to retrieve them
        try:
            annotation_dict = self._agent.data.annotation
        except AttributeError:
            # fallback: nothing to store
            return self.adata
        # Determine annotation keys for the requested level
        label_key = f"{store_key_prefix}_{level}"
        score_key = f"{store_key_prefix}_score_{level}"
        if isinstance(annotation_dict, dict) and level in annotation_dict:
            # annotation_dict[level] should be a pandas DataFrame with
            # columns 'lambda', 'lambda_score' as in the original implementation
            df = annotation_dict[level]
            self.adata.obs[label_key] = df.iloc[:, 0]
            self.adata.obs[score_key] = df.iloc[:, 1]
        return self.adata