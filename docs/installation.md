# Installation

## Quick install

spatialvi-tools requires Python ≥ 3.12. Install into a fresh virtual environment:

```bash
pip install spatialvi-tools
```

## Optional extras

| Extra | Command | Installs |
|-------|---------|---------|
| Spatial backends | `pip install "spatialvi-tools[spatial]"` | squidpy, spatialdata |
| GPU acceleration | `pip install "spatialvi-tools[rapids]"` | cuML, cuPy, cuGraph |
| Tutorials | `pip install "spatialvi-tools[tutorials]"` | jupyter, matplotlib, seaborn |
| All | `pip install "spatialvi-tools[all]"` | everything above |

## Development install

```bash
git clone https://github.com/YosefLab/spatialvi-tools.git
cd spatialvi-tools
pip install -e ".[dev,test]"
pre-commit install
```

## Verifying the installation

```python
import spatialvi
print(spatialvi.__version__)
```

## GPU support

RAPIDS-based acceleration (neighbor computation and latent representation) requires a CUDA-capable
GPU and the `rapids` extra. The base package runs on CPU without any GPU dependencies.
