# config/settings.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class LomatceConfig:
    # Clustering
    MAX_STABLE_ITERATIONS: int = 4
    DEFAULT_K: int = 20
    MIN_CLUSTER_SIZE: int = 5
    MAX_CLUSTER_SIZE: int = 50
    CLUSTERING_METHOD: str = 'kmeans'

    # SHAP settings (for comparison)
    TOP_N_FEATURES: int = 10
    FEATURE_IMPORTANCE_METHOD: str = 'shap'
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.0001
    
    # LIME settings (for compariosn)
    EXPLANATION_METHOD: str = 'lime'
    NUM_SAMPLES: int = 5000
    KERNEL_WIDTH: float = 0.75
    KERNEL_WIDTH_MULTIPLIER: float = 2.0

    # Perturbations
    NUM_PERTURBATIONS: int = 1000
    PERTURBATION_METHOD: str = 'gaussian'
    PERTURBATION_SCALE: float = 0.1

    # Models
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: ['ridge', 'lasso', 'elastic_net', 'random_forest'])
    DEFAULT_MODEL: str = 'ridge'

    

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
