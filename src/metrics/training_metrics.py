from typing import Dict, Any
import torch
import evaluate
from transformers import EvalPrediction
import numpy as np
from datasets import load_metric
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
        # Initialize metrics
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Computes:
        - Loss
        - Perplexity
        - BLEU score
        - ROUGE scores
        
        Args:
            eval_pred: Evaluation prediction object containing logits and labels
            
        Returns:
            Dictionary of computed metrics
        """
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Compute loss and perplexity
        loss = torch.nn.functional.cross_entropy(
            torch.tensor(logits.reshape(-1, logits.shape[-1])),
            torch.tensor(labels.reshape(-1)),
            ignore_index=-100
        )
        perplexity = torch.exp(loss)
        
        # Generate predictions
        predictions = np.argmax(logits, axis=-1)
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(
            [[l for l in label if l != -100] for label in labels],
            skip_special_tokens=True
        )
        
        # Compute ROUGE scores
        rouge_output = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Compute BLEU score
        bleu_output = self.bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels]
        )
        
        return {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
            "bleu": bleu_output["bleu"],
            "rouge1": rouge_output["rouge1"],
            "rouge2": rouge_output["rouge2"],
            "rougeL": rouge_output["rougeL"],
            "rougeLsum": rouge_output["rougeLsum"]
        }

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