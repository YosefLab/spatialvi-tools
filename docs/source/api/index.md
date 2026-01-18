# API Reference

## Models

```{eval-rst}
.. currentmodule:: spatialvi
```

### Core Models

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   model.SpatialVAE
   model.BaseSpatialModel
```

### External Models

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   external.AMICI
   external.VIVS
   external.Starfysh
   external.Harreman
   external.Nolan
   external.Lambda
   external.PPIInference
   external.SPARL
   external.scVIVA
   external.ResolVI
   external.DestVI
```

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   module.SpatialVAEModule
   module.NicheModule
   module.DeconvolutionModule
```

## Data

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   data.compute_spatial_neighbors
   data.compute_niche_composition
   data.synthetic_spatial
   data.synthetic_scrna
```

## Neural Network Components

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   nn.SpatialEncoder
   nn.SpatialDecoder
   nn.SpatialAttention
   nn.NeighborAttention
```

## Training

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   train.SpatialTrainingPlan
   train.NicheTrainingPlan
   train.DeconvolutionTrainingPlan
```

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   utils.spatial_autocorrelation
   utils.plot_spatial
   utils.plot_proportions
```
