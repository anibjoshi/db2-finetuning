from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import torch
import evaluate
from transformers import EvalPrediction
import numpy as np
import logging
from .base_metrics import BaseMetrics

@dataclass
class MetricsConfig:
    """Configuration for training metrics.
    
    Attributes:
        metrics_list: List of metrics to track during training
        log_to_file: Whether to log metrics to a file
        metrics_output_dir: Directory to save metrics logs
    """
    metrics_list: List[str] = None
    log_to_file: bool = True
    metrics_output_dir: str = "logs/metrics"

    def __post_init__(self):
        if self.metrics_list is None:
            self.metrics_list = ["accuracy", "f1", "precision", "recall"]

class TrainingMetrics(BaseMetrics):
    """Metrics computation during model training."""

    def __init__(self, tokenizer, config: MetricsConfig = None):
        """Initialize training metrics.
        
        Args:
            tokenizer: Tokenizer for text processing
            config: Metrics configuration
        """
        super().__init__(tokenizer)
        self.config = config or MetricsConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize metric computation objects."""
        try:
            self.metrics = {
                metric_name: evaluate.load(metric_name)
                for metric_name in self.config.metrics_list
            }
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics for predictions.
        
        Args:
            eval_pred: EvalPrediction object containing predictions and labels
            
        Returns:
            Dictionary of metric names and values
        """
        try:
            predictions = np.argmax(eval_pred.predictions, axis=-1)
            
            metrics_dict = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == "f1":
                    result = metric.compute(
                        predictions=predictions,
                        references=eval_pred.label_ids,
                        average="weighted"
                    )
                else:
                    result = metric.compute(
                        predictions=predictions,
                        references=eval_pred.label_ids
                    )
                metrics_dict.update(result)
            
            return metrics_dict
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log computed metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional training step number
        """
        try:
            step_info = f" at step {step}" if step is not None else ""
            self.logger.info(f"Metrics{step_info}:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise

    def compute_token_accuracy(self, eval_pred: EvalPrediction) -> float:
        """Compute token accuracy.
        
        Args:
            eval_pred: Contains predictions and labels
            
        Returns:
            Token accuracy
        """
        try:
            # Process in small batches to avoid OOM
            batch_size = self.config.eval_batch_size
            total_accuracy = 0
            total_samples = 0
            
            # Process predictions in batches
            for i in range(0, len(eval_pred.predictions), batch_size):
                # Get current batch
                batch_preds = eval_pred.predictions[i:i + batch_size]
                batch_labels = eval_pred.label_ids[i:i + batch_size]
                
                # Basic token-level metrics
                pred_ids = np.argmax(batch_preds, axis=-1)
                labels = np.where(
                    batch_labels != self.config.ignore_token_id,
                    batch_labels,
                    self.tokenizer.pad_token_id
                )
                
                # Calculate metrics for this batch
                if self.config.compute_token_accuracy:
                    batch_accuracy = np.mean((pred_ids == labels).astype(float))
                    
                    # Update totals
                    batch_size = len(batch_preds)
                    total_accuracy += batch_accuracy * batch_size
                    total_samples += batch_size
                
                # Clear GPU memory
                del batch_preds, batch_labels, pred_ids, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate final metrics
            token_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
            
            return token_accuracy
            
        except Exception as e:
            self.logger.error(f"Error computing token accuracy: {e}")
            return 0.0

def main():
    """Test metrics computation."""
    config = MetricsConfig()
    metrics = TrainingMetrics(tokenizer, config)
    
    # Example usage
    dummy_predictions = np.random.rand(100, 10)  # 100 examples, 10 classes
    dummy_labels = np.random.randint(0, 10, size=100)
    
    eval_pred = EvalPrediction(
        predictions=dummy_predictions,
        label_ids=dummy_labels
    )
    
    results = metrics.compute_metrics(eval_pred)
    metrics.log_metrics(results, step=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 