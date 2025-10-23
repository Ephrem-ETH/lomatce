# config/settings.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class LomatceConfig:
        
    # LOMATCE settings
    FEATURE_IMPORTANCE_THRESHOLD = 0.0001
    
     # --- Clustering parameters ---
    MAX_STABLE_ITERATIONS = 4
    DEFAULT_K = 20
    KERNEL_WIDTH_MULTIPLIER = 2
     # Optional
    MINIBATCH_THRESHOLD = 20000
    SILHOUETTE_SAMPLE_CAP= None
    SILHOUETTE_FRACTION = 0.3
    LARGE_DATA_THRESHOLD = 50000
    
    # Output
    VERBOSE = True
    SAVE_EXPLANATIONS = True
    OUTPUT_DIR: Optional[str] = "./outputs/"

    # Plotting
    PLOT_DPI = 300
    PLOT_FIGSIZE: Tuple[int, int] = (12, 6)

    # File paths
    BASE_DIR: str = "results"
    PLOTS_DIR: str = "results/plots"
    MODELS_DIR: str = "results/models"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    
    
