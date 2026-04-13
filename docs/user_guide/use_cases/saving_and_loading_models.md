# Saving and Loading Models

## Saving

After training, save the full model state to a directory:

```python
model.save("my_model/", overwrite=True)
```

This saves model weights, the AnnData manager registry, and `init_params_` so the model
can be reconstructed exactly.

## Loading

```python
from spatialvi import SCVIVA

model = SCVIVA.load("my_model/", adata=adata)
```

The `adata` argument must have the same variables (genes) as the training data.

## Transferring to a new dataset (scArches / ARCHES)

SCVIVA and ResolVI inherit scvi-tools' `ArchesMixin`, enabling query-to-reference mapping:

```python
query_model = SCVIVA.load_query_data(
    adata_query,
    reference_model="my_model/",
)
query_model.train(max_epochs=100)
```
