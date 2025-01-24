from dataclasses import dataclass
from utils.config import BASE_MODEL_DIR
from typing import Optional, Literal

@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-v0.1"
    max_length: int = 512
    
    # LoRA parameters
    lora_r: int = 8  # LoRA attention dimension
    lora_alpha: int = 16  # Alpha scaling
    lora_dropout: float = 0.05
    
    # Training parameters
    learning_rate: float = 3e-4  # This seems high for LoRA
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    num_epochs: int = 2
    warmup_steps: int = 100
    
    # Evaluation settings
    validation_split: float = 0.1
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware settings
    load_in_8bit: bool = True
    use_bf16: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
