from typing import Dict, Any
import numpy as np
import evaluate
from transformers import PreTrainedTokenizer, EvalPrediction
import logging

class BaseMetrics:
    """Base class for metrics computation."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize base metrics computation.
        
        Args:
            tokenizer: Tokenizer for text processing
        """
        self.tokenizer = tokenizer
        self.rouge_metric = evaluate.load('rouge')
        self.bleu_metric = evaluate.load('bleu')
        self.f1_metric = evaluate.load('f1')
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_text_metrics(
        self,
        predictions: list[str],
        references: list[str]
    ) -> Dict[str, float]:
        """Compute basic text metrics (ROUGE, BLEU, F1).
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            # Calculate ROUGE scores
            rouge = self.rouge_metric.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
            
            # Calculate BLEU score
            bleu = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            
            # Calculate F1 score
            pred_words = [text.split() for text in predictions]
            ref_words = [text.split() for text in references]
            f1 = self.f1_metric.compute(
                predictions=pred_words,
                references=ref_words,
                average='macro'
            )
            
            return {
                'rouge1': rouge['rouge1'],
                'rouge2': rouge['rouge2'],
                'rougeL': rouge['rougeL'],
                'bleu': bleu['bleu'],
                'f1': f1['f1']
            }
            
        except Exception as e:
            self.logger.error(f"Error computing text metrics: {e}")
            raise

    def prepare_eval_prediction(
        self,
        predictions: list[str],
        references: list[str]
    ) -> EvalPrediction:
        """Prepare predictions and references for evaluation.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            EvalPrediction object
        """
        pred_ids = self.tokenizer(predictions, padding=True, truncation=True)['input_ids']
        ref_ids = self.tokenizer(references, padding=True, truncation=True)['input_ids']
        
        return EvalPrediction(
            predictions=np.array(pred_ids),
            label_ids=np.array(ref_ids)
        ) 