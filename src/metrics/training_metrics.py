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
        eval_batch_size: Batch size for evaluation
        compute_token_accuracy: Whether to compute token accuracy
        ignore_token_id: Token ID to ignore in token accuracy computation
        max_memory_usage: Maximum fraction of GPU memory to use
    """
    metrics_list: List[str] = None
    log_to_file: bool = True
    metrics_output_dir: str = "logs/metrics"
    eval_batch_size: int = 32
    compute_token_accuracy: bool = True
    ignore_token_id: int = -100
    max_memory_usage: float = 0.9

    def __post_init__(self):
        if self.metrics_list is None:
            self.metrics_list = ["accuracy", "f1", "precision", "recall"]

class TrainingMetrics(BaseMetrics):
    """Memory-efficient metrics computation during model training."""

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

    def _get_batch_size(self, total_samples: int) -> int:
        """Determine optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return min(self.config.eval_batch_size, 8)

        try:
            # Get available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = torch.cuda.memory_reserved(0)  # Changed from memory_allocated
            available_memory = (gpu_memory - free_memory) * 0.3  # Only use 30% of available memory
            
            # Very conservative memory estimate
            sample_memory = 1024 * 1024 * 16  # 16MB per sample estimate
            
            # Calculate maximum possible batch size with safety margin
            max_batch_size = max(1, int(available_memory / (sample_memory * 4)))  # 4x safety factor
            
            return min(max_batch_size, 8)  # Hard cap at 8
        except Exception:
            return 4  # Very conservative fallback

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Memory-efficient metric computation with O(n) complexity."""
        try:
            total_samples = len(eval_pred.predictions)
            
            # Limit maximum samples for evaluation
            max_eval_samples = 1000
            if total_samples > max_eval_samples:
                # Randomly sample indices
                indices = np.random.choice(total_samples, max_eval_samples, replace=False)
                eval_pred = EvalPrediction(
                    predictions=eval_pred.predictions[indices],
                    label_ids=eval_pred.label_ids[indices]
                )
                total_samples = max_eval_samples
            
            # Use very small batch size for prediction processing
            batch_size = min(self._get_batch_size(total_samples), 8)
            
            # Initialize accumulators as simple counters
            metric_sums = {}
            metric_counts = {}
            
            # Process in batches - O(n)
            for i in range(0, total_samples, batch_size):
                batch_end = min(i + batch_size, total_samples)
                
                # Process predictions in smaller chunks to avoid OOM
                with torch.no_grad():
                    # Move data to CPU immediately
                    batch_preds = torch.from_numpy(eval_pred.predictions[i:batch_end]).cpu()
                    batch_labels = torch.from_numpy(eval_pred.label_ids[i:batch_end]).cpu()
                    
                    # Convert to predictions
                    batch_pred_classes = torch.argmax(batch_preds, dim=-1).numpy()
                    batch_labels = batch_labels.numpy()
                
                # Clear memory immediately
                del batch_preds
                torch.cuda.empty_cache()
                
                # Process one metric at a time
                for metric_name, metric in self.metrics.items():
                    result = metric.compute(
                        predictions=batch_pred_classes,
                        references=batch_labels,
                        average="weighted" if metric_name == "f1" else None
                    )
                    
                    # Update running sums
                    for key, value in result.items():
                        if key not in metric_sums:
                            metric_sums[key] = 0.0
                            metric_counts[key] = 0
                        metric_sums[key] += value * len(batch_labels)
                        metric_counts[key] += len(batch_labels)
                
                # Clear batch data
                del batch_pred_classes, batch_labels
                torch.cuda.empty_cache()
            
            # Calculate final averages
            final_metrics = {
                key: metric_sums[key] / metric_counts[key]
                for key in metric_sums
                if metric_counts[key] > 0
            }
            
            return final_metrics
            
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
        """Memory-efficient token accuracy computation."""
        try:
            total_samples = len(eval_pred.predictions)
            batch_size = self._get_batch_size(total_samples)
            
            total_correct = 0
            total_tokens = 0
            
            for i in range(0, total_samples, batch_size):
                batch_end = min(i + batch_size, total_samples)
                batch_preds = eval_pred.predictions[i:batch_end]
                batch_labels = eval_pred.label_ids[i:batch_end]
                
                # Calculate token accuracy
                pred_ids = np.argmax(batch_preds, axis=-1)
                mask = batch_labels != self.config.ignore_token_id
                
                correct_tokens = np.sum((pred_ids == batch_labels) & mask)
                total_tokens += np.sum(mask)
                total_correct += correct_tokens
                
                # Clear memory
                del batch_preds, batch_labels, pred_ids, mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return total_correct / total_tokens if total_tokens > 0 else 0.0
            
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