from dataclasses import dataclass
from config import BASE_MODEL_DIR

@dataclass
class TrainingConfig:
    """Configuration for model training hyperparameters."""
    
    # Model configuration
    model_name: str = str(BASE_MODEL_DIR)
    load_in_8bit: bool = True
    use_gradient_checkpointing: bool = True
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 512
    num_epochs: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # Evaluation and saving
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Dataset configuration
    validation_split: float = 0.1
    seed: int = 42
    data_seed: int = 42
    group_by_length: bool = False