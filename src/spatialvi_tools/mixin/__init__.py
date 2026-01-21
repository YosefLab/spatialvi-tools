"""
Mixin classes for common functionality across models.

Mixins provide reusable building blocks that can be combined via multiple
inheritance.  The base mixin defined here validates and stores an AnnData
object, similar to the mixins used in `scvi-tools`.
"""

from ._base_mixin import AnnDataMixin  # noqa: F401

__all__ = ["AnnDataMixin"]