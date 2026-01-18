spatialvi-tools/
│
├── .github/
│   └── workflows/           # GitHub Actions workflows (empty)
│
├── docs/
│   ├── figs/                # Figures for documentation
│   └── source/
│       ├── api/             # API documentation
│       ├── tutorials/       # Tutorial guides
│       └── user_guide/      # User guide documentation
│
├── src/
│   └── spatialvi/           # Main package
│       ├── __init__.py
│       ├── _settings.py
│       ├── _constants.py
│       ├── data/            # Data handling modules
│       ├── external/        # External integrations
│       ├── model/           # Model definitions
│       ├── module/          # Neural network modules
│       ├── nn/              # Neural network components
│       ├── train/           # Training utilities
│       └── utils/           # Utility functions
│
├── tests/
│   ├── data/                # Test data
│   ├── external/            # External tests
│   └── model/               # Model tests
│
└── pyproject.toml           # Project configuration
