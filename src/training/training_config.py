from dataclasses import dataclass
from config import BASE_MODEL_DIR
from typing import Optional, Literal

@dataclass
class TrainingConfig:
    """Configuration for model training hyperparameters."""
    
    # Model configuration
    model_name: str = str(BASE_MODEL_DIR)
    load_in_8bit: bool = True
    use_fp16: bool = True  # Simple flag for mixed precision training
    
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
    
    # DataLoader configuration
    dataloader_num_workers: int = 8
    pin_memory: bool = True  # Enable pinned memory for faster GPU transfer
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Dataset configuration
    validation_split: float = 0.1
    seed: int = 42