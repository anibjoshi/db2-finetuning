from typing import Dict, Any
import logging
from pathlib import Path
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from utils.config import LOGS_DIR

class MetricsLogger:
    """Handles logging of training and evaluation metrics using Tensorboard."""

    def __init__(
        self, 
        experiment_name: str,
        training_config: dict = None,
    ):
        """Initialize metrics logger with experiment tracking.
        
        Args:
            experiment_name: Name of the training/evaluation run
            training_config: Training configuration to log
        """
        self.logger = logging.getLogger("MetricsLogger")
        self.experiment_name = experiment_name
        
        # Initialize Tensorboard
        self.tensorboard_dir = LOGS_DIR / "tensorboard" / experiment_name
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # Initialize JSON logging
        self.metrics_dir = LOGS_DIR / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Log config parameters
        if training_config:
            self._save_config(training_config)
            for param_name, value in training_config.items():
                self.writer.add_text(f"config/{param_name}", str(value))
        
    def _save_config(self, config: dict):
        """Save training configuration to JSON."""
        config_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "config": config
        }
        self._append_to_json(config_entry)
        
    def _append_to_json(self, data: dict):
        """Append data to JSON file."""
        try:
            # Read existing data
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    file_data = json.load(f)
                if not isinstance(file_data, list):
                    file_data = [file_data]
            else:
                file_data = []
            
            # Append new data
            file_data.append(data)
            
            # Write back to file
            with open(self.metrics_file, 'w') as f:
                json.dump(file_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics to JSON: {e}")
        
    def log_metrics(self, metrics: Dict[str, Any], step: str = "evaluation", step_num: int = None) -> None:
        """Log metrics to Tensorboard and JSON file."""
        try:
            # Flatten nested metrics for logging
            flat_metrics = {}
            for category, values in metrics.items():
                if isinstance(values, dict):
                    for metric, value in values.items():
                        flat_metrics[f"{category}/{metric}"] = value
                else:
                    flat_metrics[category] = values
            
            # Log to Tensorboard
            for metric_name, value in flat_metrics.items():
                self.writer.add_scalar(f"{step}/{metric_name}", value, global_step=step_num)
            
            # Save to JSON
            metric_entry = {
                "timestamp": datetime.now().isoformat(),
                "experiment": self.experiment_name,
                "step": step,
                "step_num": step_num,
                "metrics": metrics
            }
            self._append_to_json(metric_entry)
            
            # Log to console
            if step_num is not None:
                self.logger.info(f"\n{'='*50}\nMetrics for {step} (Step {step_num}):")
            else:
                self.logger.info(f"\n{'='*50}\nMetrics for {step}:")
                
            for category, values in metrics.items():
                if isinstance(values, dict):
                    self.logger.info(f"\n{category}:")
                    for metric, value in values.items():
                        self.logger.info(f"  {metric}: {value:.4f}")
                else:
                    self.logger.info(f"{category}: {values:.4f}")
            self.logger.info(f"{'='*50}\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def close(self):
        """Clean up logging resources."""
        self.writer.close() 