"""Settings for spatialvi-tools."""

from __future__ import annotations

import logging
from typing import Literal


class SpatialVISettings:
    """Global settings for spatialvi-tools.

    Attributes
    ----------
    verbosity
        Logging verbosity level. One of: logging.DEBUG, logging.INFO,
        logging.WARNING, logging.ERROR, logging.CRITICAL.
    seed
        Random seed for reproducibility.
    progress_bar_style
        Progress bar style. One of: "rich", "tqdm", "auto".
    batch_size
        Default batch size for data loading.
    num_threads
        Number of threads to use for data loading.
    warnings_stacklevel
        Stack level for warnings.
    """

    def __init__(
        self,
        verbosity: int = logging.INFO,
        seed: int = 0,
        progress_bar_style: Literal["rich", "tqdm", "auto"] = "auto",
        batch_size: int = 128,
        num_threads: int = 0,
        warnings_stacklevel: int = 2,
    ):
        self._verbosity = verbosity
        self.seed = seed
        self.progress_bar_style = progress_bar_style
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.warnings_stacklevel = warnings_stacklevel

    @property
    def verbosity(self) -> int:
        """Logging verbosity level."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: int) -> None:
        self._verbosity = level
        logging.getLogger("spatialvi").setLevel(level)

    def reset(self) -> None:
        """Reset settings to default values."""
        self.__init__()


settings = SpatialVISettings()
