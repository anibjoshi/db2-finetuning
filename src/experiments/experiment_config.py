from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    """Configuration for hyperparameter optimization experiments."""
    n_trials: int = 25
    study_name: str = "db2_finetuning"
    direction: str = "minimize"
    metric: str = "eval_loss"
    
    # Search space configuration
    lora_r_options: List[int] = field(default_factory=lambda: [16, 32, 64])
    lora_alpha_options: List[int] = field(default_factory=lambda: [8, 16, 32])
    lora_dropout_options: List[float] = field(default_factory=lambda: [0.1, 0.2])
    learning_rate_range: Tuple[float, float] = (1e-5, 5e-5)
    batch_size_options: List[int] = field(default_factory=lambda: [4, 8, 16]) 