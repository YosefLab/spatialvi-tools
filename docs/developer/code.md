# Contributing Code

## Setting up a development environment

1. Fork the [repository](https://github.com/YosefLab/scviva-tools) on GitHub.

2. Clone your fork:

   ```bash
   git clone https://github.com/{your-username}/scviva-tools.git
   cd scviva-tools
   ```

3. Install the development dependencies in editable mode:

   ```bash
   pip install -e ".[dev,test]"
   ```

4. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Running tests

```bash
python -m pytest tests/ -v
```

Run a subset:

```bash
python -m pytest tests/model/test_scviva.py -v
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run it with:

```bash
ruff check src/
ruff format src/
```

Pre-commit hooks run these automatically on every commit.

## Adding a new model

1. Add the neural-network module under `src/scviva/module/`.
2. Add the model class under `src/scviva/model/`, inheriting from `SpatialBaseModel`.
3. Register it in `src/scviva/model/__init__.py` and `src/scviva/__init__.py`.
4. Write tests under `tests/model/`.

## Package layout

```
src/scviva/
├── model/           # High-level model classes
│   ├── base/        # Shared mixins and base class
│   └── utils/       # Model-level utilities and DE sub-packages
│       └── _scviva_de/  # scVIVA niche differential expression (math/data only; plots live in plotting/)
├── module/          # PyTorch neural-network modules
│   └── utils/       # Module-level component libraries
│       └── _nichevae_components.py
├── data/            # Custom AnnData fields
├── external/        # External/ported model adaptations
├── tools/           # Analysis tools (e.g. harreman/)
├── plotting/        # All plotting functions, exposed via `scviva.pl`
│   ├── harreman/    # Harreman plots
│   ├── scviva_de.py # scVIVA niche DE plots
│   └── _deconvolution.py  # DestVI cell-type-map plot
├── train/           # Training plan overrides
└── utils/           # Shared utilities (spatial coords, device resolution)
    ├── _spatial.py
    └── _device.py
```

Plotting functions live under `plotting/`, not inline in `model/`/`tools/` code. Model mixins and tool accessor classes expose thin `.plot_*()`/`.plot()` methods that delegate to a function in `plotting/` — see `SpatialDeconvolutionMixin.plot_cell_type_map` or `HarremanAnalysis.local_correlation_plot` for the pattern.
