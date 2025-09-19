# config/settings.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class LomatceConfig:
        

    # Output
    VERBOSE: bool = True
    SAVE_EXPLANATIONS: bool = True
    OUTPUT_DIR: Optional[str] = "./outputs/"

    # Plotting
    PLOT_DPI: int = 300
    PLOT_FIGSIZE: Tuple[int, int] = (12, 6)

    # File paths
    BASE_DIR: str = "results"
    PLOTS_DIR: str = "results/plots"
    MODELS_DIR: str = "results/models"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Processing
    DEFAULT_N_JOBS: Optional[int] = None
    CPU_CORES_DIVISOR: int = 6
