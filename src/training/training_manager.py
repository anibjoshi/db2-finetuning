import logging
from pathlib import Path
from typing import List, Dict, Any
from training.LoRA_training import Db2Trainer
from training.training_config import TrainingConfig

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages high-level training operations for Db2 model fine-tuning."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize training manager.
        
        Args:
            config: Training configuration parameters
        """
        self.config = config
        self.trainer = Db2Trainer(config)
    
    def _load_and_validate_files(self, data_path: Path) -> List[Path]:
        """Validate and return list of training data files."""
        if data_path.is_file():
            return [data_path]
            
        valid_files = list(data_path.glob("*.jsonl"))
        if not valid_files:
            raise ValueError(f"No .jsonl files found in {data_path}")
            
        logger.info(f"Found {len(valid_files)} training data files")
        return sorted(valid_files)

    def train(self, data_path: Path) -> Dict[str, Any]:
        """Run model training process on all files sequentially.
        
        Args:
            data_path: Path to training data file or directory
            
        Returns:
            Dict containing training metrics (eval_loss, etc.)
        """
        try:
            training_files = self._load_and_validate_files(data_path)
            metrics = {}
            
            for file_path in training_files:
                logger.info(f"Training on file: {file_path.name}")
                result = self.trainer.train(file_path)
                metrics.update(result)
            
            logger.info("Training completed successfully")
            return metrics
            
        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise RuntimeError(f"Training failed: {str(e)}")