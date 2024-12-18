"""Global configuration constants for DB2 Assistant."""

from pathlib import Path

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