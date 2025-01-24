from typing import Dict
import numpy as np
import torch
from transformers import EvalPrediction
from .base_metrics import BaseMetrics
from metrics.metrics_config import MetricsConfig

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

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute training evaluation metrics.
        
        Computes only essential metrics during training to avoid OOM:
        - Loss
        - Token accuracy
        - Basic ROUGE score
        
        Args:
            eval_pred: Contains predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Process in small batches to avoid OOM
            batch_size = self.config.eval_batch_size
            total_loss = 0
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
            metrics = {}
            if self.config.compute_token_accuracy:
                metrics["eval_accuracy"] = total_accuracy / total_samples if total_samples > 0 else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            # Return basic metrics if computation fails
            return {
                "eval_accuracy": 0.0,
            } 