# Contributing to scviva-tools

We welcome contributions! Please follow the guidelines below.

## Development Setup

```bash
git clone https://github.com/your-org/scviva-tools
cd scviva-tools
pip install -e ".[dev,test]"
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## References

When adding new models, update `docs/references.md` and `docs/references.bib`.
