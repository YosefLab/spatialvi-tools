# Installation

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- scvi-tools >= 1.1.0

## Installing spatialvi-tools

### From PyPI (recommended)

```bash
pip install spatialvi-tools
```

### From source

```bash
git clone https://github.com/yoseflab/spatialvi-tools.git
cd spatialvi-tools
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev,doc,test]"
```

## Optional Dependencies

### JAX support (for Tangram, VIVS)

```bash
pip install spatialvi-tools[jax]
```

### GPU support

spatialvi-tools automatically uses GPU if available. Make sure you have the correct PyTorch version installed for your CUDA version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Verifying Installation

```python
import spatialvi
print(spatialvi.__version__)
```
