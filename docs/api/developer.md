# Developer API

Internal base classes and mixins used to build new spatialvi models.

## Base classes

```{eval-rst}
.. currentmodule:: spatialvi.model.base

.. autosummary::
   :toctree: reference/
   :nosignatures:

   SpatialBaseModel
   SpatialNeighborhoodMixin
   SpatialDeconvolutionMixin
   ResolVIPredictiveMixin
```

## Data fields

```{eval-rst}
.. currentmodule:: spatialvi.data

.. autosummary::
   :toctree: reference/
   :nosignatures:

   SpatialCoordsField
   NeighborhoodGraphField
```

## Modules (neural networks)

```{eval-rst}
.. currentmodule:: spatialvi.module

.. autosummary::
   :toctree: reference/
   :nosignatures:

   nicheVAE
   RESOLVAE
   MRDeconv
   JVAE
```
