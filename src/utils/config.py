"""Global configuration constants for DB2 Assistant."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Base directories
SRC_DIR = Path("src")
DATA_DIR = SRC_DIR / "data"
MODEL_DIR = SRC_DIR / "model"
BASE_DIR = SRC_DIR
LOGS_DIR = BASE_DIR / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

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

# Add to existing config.py
EVALUATION_DATA_DIR = Path("src/data/evaluation")

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

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
default_inference_config = InferenceConfig() 