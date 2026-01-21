# spatialvi‑tools

**spatialvi‑tools** is a unified Python toolbox for modern spatial omics analysis.  
It provides a consistent API that wraps and extends several cutting‑edge algorithms for
spatial transcriptomics and proteomics, including niche detection, cell–cell
interaction inference, metabolic exchange analysis, variable selection, cell type
deconvolution, language‑model based cell annotation and prediction‑powered
inference.  

The aim of this project is to consolidate common functionality across the
following research packages into a single library with a scvi‑tools–style
structure:

* **NOLAN (NicheExplorer)** – self‑supervised identification of spatial niches and
  their relationships【688704965977238†L7-L13】.  
* **LAMBDA** – large language model driven, reference‑free cell type annotation for
  single‑cell data【245548828756402†L4-L32】.  
* **PPI** – prediction‑powered inference for scientifically rigorous estimation of
  statistical quantities【569057324233913†L11-L14】.  
* **VIVS** – variational inference for variable selection in omics data【666593781079131†L1-L8】.  
* **Harreman** – inference of metabolic exchanges from spatial transcriptomics
  via spatial correlation statistics【504187329151369†L11-L13】.  
* **AMICI** – attention‑based inference of cell–cell interactions from spatial
  transcriptomic data【767658563887085†L7-L10】.  
* **Starfysh** – generative modeling for reference‑free deconvolution and
  integration of spatial transcriptomics with histology【835840889957633†L6-L10】.  
* **SPARL** – spatial proteomics analysis with representation learning【218342748980084†L8-L10】.

The design follows the conventions established by `scvi‑tools` and the
scverse ecosystem.  Key characteristics include a clear separation between
data loading, model definition, neural network modules, training loops,
mixins for shared functionality, testing and documentation.  Users familiar
with `scvi-tools` will find the API familiar.

## Repository layout

```
spatialvi-tools/
├── docs/            # documentation source files
├── src/
│   └── spatialvi_tools/
│       ├── data/       # data loaders and preprocessing utilities
│       ├── mixin/      # reusable mixin classes
│       ├── models/     # high‑level model classes
│       │   └── modules/ # low‑level neural network modules
│       ├── nn/         # neural network building blocks
│       ├── train/      # training loops and callbacks
│       └── utils/      # helper functions
├── tests/          # test suite
│   └── data/         # small example datasets used in tests
├── pyproject.toml  # build and dependency configuration
└── README.md       # this file
```

### Installing

This project is still under active development.  To install the latest
development version, clone the repository and install it in editable mode:

```bash
git clone https://github.com/YosefLab/spatialvi-tools.git
cd spatialvi-tools
pip install -e .
```

Some models depend on external packages (e.g. `nolan`, `LAMBDA`, `vivs`,
`harreman`, `amici`, `starfysh`, `sparl` and `ppi_py`).  They are declared as
extras.  Install the full set of optional dependencies with:

```bash
pip install 'spatialvi-tools[all]'
```

## Quick start

The package exposes high‑level model classes under `spatialvi_tools.models`.  For
example, to detect spatial niches with the NOLAN algorithm:

```python
import anndata as ad
from spatialvi_tools.models import NolanModel

adata = ad.read_h5ad("sample_data.h5ad")
model = NolanModel(
    adata,
    emb_key="X_scVI",      # precomputed cell embeddings
    spatial_key="X_spatial", # coordinates of spots or cells
    num_niches=50,
)
model.train(num_epochs=10)
adata = model.predict(adata)

# the predicted niche assignments are stored in adata.obsm["X_nolan"]
```

Analogous interfaces exist for the other wrapped algorithms:

* `LambdaModel` – LLM‑based cell type annotation.
* `PPIInference` – statistical inference using prediction‑powered confidence
  intervals.
* `VIVSModel` – variable selection via variational inference.
* `HarremanModel` – analysis of metabolic exchanges.
* `AmiciModel` – attention‑based cell–cell interaction inference.
* `StarfyshModel` – reference‑free deconvolution and histology integration.
* `SparlModel` – representation learning for spatial proteomics.

Each model supports a `.train()` method to fit the underlying model and
`.predict()` or `.infer()` methods to produce results on new data.  See
the documentation in the `docs/` folder for detailed usage examples.

## License

This project is licensed under the BSD 3‑Clause license.  Note that some
wrapped algorithms have their own licensing terms; users must ensure that
they comply with the licenses of any external packages they install and use.