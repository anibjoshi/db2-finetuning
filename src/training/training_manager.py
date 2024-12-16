import logging
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Optional, List, Dict
from training.LoRA_training import Db2FineTuningConfig, train_db2_model

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages model training operations.
    
    Handles sequential processing of multiple training data files and manages
    the training process with proper error handling and logging.
    
    Attributes:
        config: Configuration for DB2 fine-tuning
        batch_size: Number of examples to process at once
    """
    
    def __init__(self, config: Db2FineTuningConfig, batch_size: int = 1000):
        self.config = config
        self.batch_size = batch_size
    
    def _load_and_validate_files(self, data_path: Path) -> List[Path]:
        """Validate and return list of training data files.
        
        Args:
            data_path: Directory containing training data files
            
        Returns:
            List of valid training data file paths
            
        Raises:
            ValueError: If no valid training files found
        """
        if data_path.is_file():
            return [data_path]
            
        valid_files = list(data_path.glob("*.jsonl"))
        if not valid_files:
            raise ValueError(f"No .jsonl files found in {data_path}")
            
        logger.info(f"Found {len(valid_files)} training data files")
        return sorted(valid_files)
    
    def _process_file(self, file_path: Path) -> Dataset:
        """Process single training data file.
        
        Args:
            file_path: Path to training data file
            
        Returns:
            Processed dataset
            
        Raises:
            RuntimeError: If file processing fails
        """
        try:
            logger.info(f"Processing file: {file_path.name}")
            return load_dataset("json", data_files=str(file_path))["train"]
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}", exc_info=True)
            raise RuntimeError(f"File processing failed: {str(e)}")

    def train(self, data_path: Path) -> None:
        """Run model training process on all data files.
        
        Args:
            data_path: Path to directory containing training data or single file
            
        Raises:
            RuntimeError: If training process fails
            ValueError: If no valid training files found
        """
        try:
            training_files = self._load_and_validate_files(data_path)
            
            # Process files sequentially
            current_dataset = None
            for file_path in training_files:
                file_dataset = self._process_file(file_path)
                
                if current_dataset is None:
                    current_dataset = file_dataset
                else:
                    current_dataset = concatenate_datasets([current_dataset, file_dataset])
                
                # Check if batch size reached
                if len(current_dataset) >= self.batch_size:
                    logger.info(f"Training on batch of {len(current_dataset)} examples")
                    train_db2_model(self.config, current_dataset)
                    current_dataset = None
            
            # Train on remaining data if any
            if current_dataset is not None and len(current_dataset) > 0:
                logger.info(f"Training on final batch of {len(current_dataset)} examples")
                train_db2_model(self.config, current_dataset)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise RuntimeError(f"Training failed: {str(e)}") 