from __future__ import annotations

# Re-export scvi training configs — placeholder for future spatial-specific configs
from scvi.train._config import TrainerConfig, TrainingPlanConfig

__all__ = ["TrainingPlanConfig", "TrainerConfig"]
