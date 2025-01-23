from typing import Dict
import numpy as np
from transformers import EvalPrediction
from .base_metrics import BaseMetrics
import torch

class TrainingMetrics(BaseMetrics):
    """Metrics computation during model training."""

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute training evaluation metrics.
        
        Process predictions in chunks to avoid GPU OOM.
        
        Args:
            eval_pred: Contains predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        # Process in chunks to avoid OOM
        chunk_size = 32
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        total_predictions = []
        total_references = []
        
        # Process chunks
        for i in range(0, len(predictions), chunk_size):
            # Get chunk
            pred_chunk = predictions[i:i + chunk_size]
            label_chunk = labels[i:i + chunk_size]
            
            # Convert to token ids
            pred_ids = np.argmax(pred_chunk, axis=-1)
            label_ids = np.where(
                label_chunk != -100,
                label_chunk,
                self.tokenizer.pad_token_id
            )
            
            # Decode to text
            predicted_texts = [
                text.strip() 
                for text in self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            ]
            reference_texts = [
                text.strip() 
                for text in self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                if text.strip()
            ]
            
            total_predictions.extend(predicted_texts)
            total_references.extend(reference_texts)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Compute metrics on all processed texts
        metrics = self.compute_text_metrics(total_predictions, total_references)
        metrics['token_accuracy'] = np.mean(
            (np.argmax(predictions, axis=-1) == labels).astype(float)
        )
        
        return metrics 