from typing import Dict
import numpy as np
import evaluate
from transformers import EvalPrediction
import logging

class MetricsManager:
    """Manages computation of training evaluation metrics."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge_metric = evaluate.load('rouge')
        self.bleu_metric = evaluate.load('bleu')
        self.f1_metric = evaluate.load('f1')
        self.logger = logging.getLogger("MetricsManager")
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            eval_pred: Contains predictions and labels
            
        Returns:
            Dictionary of metrics (ROUGE, BLEU, F1, perplexity, token accuracy)
        """
        # Decode predictions and labels
        pred_ids = np.argmax(eval_pred.predictions, axis=-1)
        labels = np.where(eval_pred.label_ids != -100, eval_pred.label_ids, self.tokenizer.pad_token_id)
        
        predicted_texts = [
            text.strip() for text in self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ]
        reference_texts = [
            text.strip() for text in self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            if text.strip()
        ]
        
        # Check alignment of predictions and references
        if len(predicted_texts) != len(reference_texts):
            self.logger.warning("Mismatch between predicted and reference text counts")
        
        # Calculate metrics
        rouge = self.rouge_metric.compute(
            predictions=predicted_texts,
            references=reference_texts,
            use_stemmer=True
        )
        
        references = [[ref] for ref in reference_texts]
        bleu = self.bleu_metric.compute(
            predictions=predicted_texts,
            references=references
        )
        
        pred_words = [text.split() for text in predicted_texts]
        ref_words = [text.split() for text in reference_texts]
        f1 = self.f1_metric.compute(
            predictions=pred_words,
            references=ref_words,
            average='macro'
        )
        
        # Calculate token-level accuracy
        token_accuracy = np.mean((pred_ids == labels).astype(float))
        
        # Perplexity
        loss = eval_pred.metrics.get('loss', 0)
        if loss > 100:
            self.logger.warning(f"Loss value capped at 100 for perplexity calculation. Actual loss: {loss}")
        perplexity = np.exp(min(loss, 100))
        
        # Log key metrics
        self.logger.info(f"ROUGE: {rouge}, BLEU: {bleu}, F1: {f1}, Perplexity: {perplexity}, Token Accuracy: {token_accuracy}")
        
        return {
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'bleu': bleu['bleu'],
            'f1': f1['f1'],
            'perplexity': perplexity,
            'token_accuracy': token_accuracy
        }