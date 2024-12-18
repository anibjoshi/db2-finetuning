"""Global configuration constants for DB2 Assistant."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Base directories
SRC_DIR = Path("src")
DATA_DIR = SRC_DIR / "data"
MODEL_DIR = SRC_DIR / "model"
LOGS_DIR = SRC_DIR / "logs"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
BASE_MODEL_DIR = MODEL_DIR / "base_model"
FINETUNED_MODEL_DIR = MODEL_DIR / "db2_llama_finetuned"
BEST_MODEL_DIR = FINETUNED_MODEL_DIR / "best_model"

# Data files
FIRST_TURN_CONV_FILE = PROCESSED_DATA_DIR / "SQL0000-0999_first_turn_conversations.jsonl"

# Log files
DEFAULT_LOG_FILE = "db2_assistant.log"

# Db2 versions
SUPPORTED_DB2_VERSIONS = ["11.1", "11.5", "12.1"]
DEFAULT_DB2_VERSION = "12.1" 

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

@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    model_path: Path = BEST_MODEL_DIR
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 4
    
# Default configurations
default_training_config = TrainingConfig()
default_inference_config = InferenceConfig() 