# Installation

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- scvi-tools >= 1.0

## Installation Methods

### From PyPI (Recommended)
bash pip install spatialvi-tools``` 

### From Source
```

bash git clone https://github.com/your-org/spatialvi-tools.git cd spatialvi-tools pip install -e .``` 

### With Development Dependencies
```

bash pip install -e ".[dev]"``` 

### With Documentation Dependencies
```

bash pip install -e ".[docs]"``` 

## GPU Support

For GPU acceleration, ensure you have CUDA installed and install PyTorch with CUDA support:
```

bash pip install torch --index-url https://download.pytorch.org/whl/cu118``` 

## Verifying Installation
```

python import spatialvi print(spatialvi.version)