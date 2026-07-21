# Installation

## Quick install

scVIVA-Tools requires Python ≥ 3.12. Install into a fresh virtual environment:

```bash
pip install scviva-tools
```

## Optional extras

| Extra | Command | Installs |
|-------|---------|---------|
| Spatial backends | `pip install "scviva-tools[spatial]"` | squidpy, spatialdata |
| GPU acceleration | `pip install "scviva-tools[rapids]"` | cuML, cuPy, cuGraph |
| Tutorials | `pip install "scviva-tools[tutorials]"` | jupyter, matplotlib, seaborn |
| All | `pip install "scviva-tools[all]"` | everything above |

## Development install

```bash
git clone https://github.com/YosefLab/scviva-tools.git
cd scviva-tools
pip install -e ".[dev,test]"
pre-commit install
```

## Verifying the installation

```python
import scviva
print(scviva.__version__)
```

## GPU support

RAPIDS-based acceleration (neighbor computation and latent representation) requires a CUDA-capable
GPU and the `rapids` extra. The base package runs on CPU without any GPU dependencies.
