from typing import Dict, List, Any
from pathlib import Path
import torch
import numpy as np
from transformers import PreTrainedModel
from datasets import Dataset
from .base_metrics import BaseMetrics
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class EvaluationMetrics(BaseMetrics):
    """Comprehensive model evaluation metrics."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_accuracy(
        self, 
        model: PreTrainedModel, 
        dataset: Dataset
    ) -> float:
        """Calculate response accuracy against ground truth.
        
        Args:
            model: The model to evaluate
            dataset: Dataset containing examples
            
        Returns:
            Token-level accuracy score
        """
        total_correct = 0
        total_tokens = 0
        
        for example in dataset:
            input_text = example.get('input', '')
            target_text = example.get('output', '')
            
            # Generate prediction
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs)
            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate token accuracy
            pred_tokens = self.tokenizer.encode(pred_text, add_special_tokens=False)
            target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)
            
            min_len = min(len(pred_tokens), len(target_tokens))
            correct_tokens = sum(p == t for p, t in zip(pred_tokens[:min_len], target_tokens[:min_len]))
            
            total_correct += correct_tokens
            total_tokens += min_len
        
        return total_correct / total_tokens if total_tokens > 0 else 0.0

    def evaluate_response_quality(
        self, 
        model: PreTrainedModel, 
        dataset: Dataset
    ) -> Dict[str, float]:
        """Evaluate quality metrics of model responses.
        
        Args:
            model: The model to evaluate
            dataset: Dataset containing examples
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            bleu_scores = []
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            response_lengths = []
            
            for example in dataset:
                input_text = example.get('input', '')
                target_text = example.get('output', '')
                
                # Generate prediction
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Calculate metrics
                reference = [target_text.split()]
                candidate = pred_text.split()
                bleu = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu)
                
                rouge_result = self.rouge_scorer.score(target_text, pred_text)
                for key in rouge_scores:
                    rouge_scores[key].append(rouge_result[key].fmeasure)
                
                response_lengths.append(len(candidate))
            
            metrics = {
                'bleu_score': np.mean(bleu_scores),
                'rouge1_f1': np.mean(rouge_scores['rouge1']),
                'rouge2_f1': np.mean(rouge_scores['rouge2']),
                'rougeL_f1': np.mean(rouge_scores['rougeL']),
                'avg_response_length': np.mean(response_lengths),
                'response_length_std': np.std(response_lengths)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing quality metrics: {e}")
            raise

    def evaluate_content_relevance(
        self, 
        model: PreTrainedModel, 
        dataset: Dataset
    ) -> Dict[str, float]:
        """Evaluate content relevance of responses.
        
        Args:
            model: The model to evaluate
            dataset: Dataset containing examples
            
        Returns:
            Dictionary of relevance metrics
        """
        try:
            keyword_matches = []
            context_scores = []
            
            for example in dataset:
                input_text = example.get('input', '')
                target_text = example.get('output', '')
                
                # Generate prediction
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Calculate metrics
                input_keywords = set(input_text.lower().split())
                pred_keywords = set(pred_text.lower().split())
                target_keywords = set(target_text.lower().split())
                
                keyword_match = len(pred_keywords & target_keywords) / len(target_keywords) if target_keywords else 0
                keyword_matches.append(keyword_match)
                
                context_relevance = len(pred_keywords & input_keywords) / len(input_keywords) if input_keywords else 0
                context_scores.append(context_relevance)
            
            return {
                'keyword_match_rate': np.mean(keyword_matches),
                'context_relevance': np.mean(context_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error computing relevance metrics: {e}")
            raise 