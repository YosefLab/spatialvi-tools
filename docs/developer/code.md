# Contributing Code

## Setting up a development environment

1. Fork the [repository](https://github.com/YosefLab/spatialvi-tools) on GitHub.

2. Clone your fork:

   ```bash
   git clone https://github.com/{your-username}/spatialvi-tools.git
   cd spatialvi-tools
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

1. Add the neural-network module under `src/spatialvi/module/`.
2. Add the model class under `src/spatialvi/model/`, inheriting from `SpatialBaseModel`.
3. Register it in `src/spatialvi/model/__init__.py` and `src/spatialvi/__init__.py`.
4. Write tests under `tests/model/`.

## Package layout

```
src/spatialvi/
├── model/           # High-level model classes
│   ├── base/        # Shared mixins and base class
│   └── utils/       # Model-level utilities and DE sub-packages
│       └── _scviva_de/  # scVIVA niche differential expression
├── module/          # PyTorch neural-network modules
│   └── utils/       # Module-level component libraries
│       └── _nichevae_components.py
├── data/            # Custom AnnData fields
├── external/        # External/ported model adaptations
├── train/           # Training plan overrides
└── utils/           # Shared spatial utilities
```
