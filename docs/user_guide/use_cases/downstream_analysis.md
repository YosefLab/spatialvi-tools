# Downstream Analysis

## Latent space

Extract the latent representation and store it in `adata.obsm`:

```python
adata.obsm["X_spatialvi"] = model.get_latent_representation()
```

Use RAPIDS for GPU-accelerated downstream analysis:

```python
z = model.get_latent_representation(backend="rapids")  # returns cupy array
```

## Clustering

Apply standard scanpy clustering to the latent space:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_spatialvi")
sc.tl.leiden(adata)
sc.pl.embedding(adata, basis="spatial", color="leiden")
```

## Deconvolution (DestVI)

```python
proportions = model.get_proportions()   # DataFrame (n_spots × n_celltypes)
model.plot_cell_type_map(cell_type="Neuron")
```

## Niche prediction (scVIVA)

```python
ct_composition = model.predict_neighborhood()      # (n_cells, n_celltypes)
niche_activation = model.predict_niche_activation()  # (n_cells, n_niches)
```

## Differential expression

```python
de = model.differential_expression(groupby="cell_type", mode="change")
de[de["is_de_fdr_0.05"]].head(20)
```
