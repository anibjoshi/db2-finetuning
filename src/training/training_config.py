from dataclasses import dataclass
from utils.config import BASE_MODEL_DIR
from typing import Optional, Literal

@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""
    
    # Model settings
    model_name: str = str(BASE_MODEL_DIR)
    max_length: int = 512
    
    # LoRA parameters
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    num_epochs: int = 2
    warmup_steps: int = 500
    
    # Evaluation settings
    validation_split: float = 0.1
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware settings
    use_bf16: bool = True
    dataloader_num_workers: int = 16
    pin_memory: bool = True
    seed: int = 42
    
    # Memory optimization
    prefetch_factor: int = 4
