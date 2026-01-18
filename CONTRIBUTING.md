# Contributing to spatialvi-tools

Thank you for your interest in contributing to spatialvi-tools! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/spatialvi-tools.git
cd spatialvi-tools
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev,test,doc]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting. The configuration is in `pyproject.toml`.

- Line length: 100 characters
- Docstring style: NumPy
- Import sorting: isort-compatible

Run linting locally:
```bash
ruff check src/
ruff format src/
```

## Testing

We use pytest for testing. Run the test suite with:
```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=spatialvi --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Use fixtures from `conftest.py`
- Test both success and failure cases
- Use parametrize for testing multiple inputs

Example:
```python
import pytest
from spatialvi.module import SpatialVAEModule

class TestSpatialVAEModule:
    def test_initialization(self, n_genes, n_latent):
        module = SpatialVAEModule(
            n_input=n_genes,
            n_latent=n_latent,
        )
        assert module.n_input == n_genes

    @pytest.mark.parametrize("dispersion", ["gene", "gene-batch"])
    def test_dispersion_types(self, n_genes, dispersion):
        module = SpatialVAEModule(
            n_input=n_genes,
            dispersion=dispersion,
        )
        assert module.dispersion == dispersion
```

## Documentation

Documentation is built with Sphinx. Build locally:
```bash
cd docs
make html
```

View at `docs/_build/html/index.html`.

### Docstring Format

Use NumPy-style docstrings:
```python
def compute_spatial_neighbors(
    adata: AnnData,
    n_neighbors: int = 20,
) -> None:
    """Compute spatial nearest neighbors.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    n_neighbors
        Number of neighbors to compute.

    Returns
    -------
    None
        Modifies adata in place, adding neighbor information.

    Examples
    --------
    >>> compute_spatial_neighbors(adata, n_neighbors=10)
    """
```

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add feature: description"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request on GitHub

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)
- [ ] Pre-commit hooks pass
- [ ] CI passes

## Reporting Issues

When reporting issues, please include:
- spatialvi-tools version
- Python version
- Operating system
- Minimal reproducible example
- Full error traceback

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming environment for all contributors.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.
