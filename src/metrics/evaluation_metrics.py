from typing import Dict, List, Any
import torch
import numpy as np
from transformers import PreTrainedModel
from datasets import Dataset
from .base_metrics import BaseMetrics
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class EvaluationMetrics(BaseMetrics):
    """Metrics for evaluating DB2 SQL code explanation model."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Match training format exactly
        self.prompt_template = """<|im_start|>user
{} What does this mean?<|im_end|>
<|im_start|>assistant
{} means """

        self.generation_config = {
            "max_length": 100,
            "min_length": 10,
            "temperature": 0.1,         # Even more focused
            "top_p": 0.3,              # Very conservative sampling
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "length_penalty": 0.3,      # Strongly favor shorter responses
            "early_stopping": True,
            "do_sample": False,        # Use greedy decoding
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def _generate_response(self, model: PreTrainedModel, input_text: str) -> str:
        """Generate model response for given input."""
        # Extract and format SQL code
        sql_code = input_text.split()[0] if input_text else ""
        if not sql_code.startswith("SQL"):
            sql_code = "SQL" + sql_code
        
        # Generate response
        prompt = self.prompt_template.format(sql_code, sql_code)  # Include code twice
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Extract assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        return response

    def _log_example(self, i: int, input_text: str, target_text: str, pred_text: str) -> None:
        """Log example inputs and outputs."""
        if i < 3:  # Log first 3 examples
            self.logger.info("\nExample evaluation:")
            self.logger.info(f"Input: {input_text}")
            self.logger.info(f"Target: {target_text}")
            self.logger.info(f"Prediction: {pred_text}")

    def calculate_accuracy(self, model: PreTrainedModel, dataset: Dataset) -> float:
        """Calculate token-level accuracy."""
        total_correct = 0
        total_tokens = 0
        
        self.logger.info(f"Starting accuracy evaluation on {len(dataset)} samples")
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                self.logger.info(f"Processing sample {i}/{len(dataset)}")
            
            input_text = example.get('input', '')
            target_text = example.get('output', '')
            pred_text = self._generate_response(model, input_text)
            
            self._log_example(i, input_text, target_text, pred_text)
            
            # Calculate token accuracy
            pred_tokens = self.tokenizer.encode(pred_text, add_special_tokens=False)
            target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)
            
            min_len = min(len(pred_tokens), len(target_tokens))
            correct_tokens = sum(p == t for p, t in zip(pred_tokens[:min_len], target_tokens[:min_len]))
            
            total_correct += correct_tokens
            total_tokens += min_len
        
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        self.logger.info(f"Completed accuracy evaluation. Final accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_response_quality(self, model: PreTrainedModel, dataset: Dataset) -> Dict[str, float]:
        """Evaluate BLEU, ROUGE scores and response lengths."""
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        response_lengths = []
        
        self.logger.info(f"Starting response quality evaluation on {len(dataset)} samples")
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                self.logger.info(f"Processing sample {i}/{len(dataset)}")
            
            input_text = example.get('input', '')
            target_text = example.get('output', '')
            pred_text = self._generate_response(model, input_text)
            
            self._log_example(i, input_text, target_text, pred_text)
            
            # Calculate metrics
            reference = [target_text.split()]
            candidate = pred_text.split()
            
            bleu_scores.append(sentence_bleu(reference, candidate))
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
        
        self.logger.info("Completed response quality evaluation:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics

    def evaluate_content_relevance(self, model: PreTrainedModel, dataset: Dataset) -> Dict[str, float]:
        """Evaluate keyword overlap between predictions and targets/inputs."""
        keyword_matches = []
        context_scores = []
        
        self.logger.info(f"Starting content relevance evaluation on {len(dataset)} samples")
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                self.logger.info(f"Processing sample {i}/{len(dataset)}")
            
            input_text = example.get('input', '')
            target_text = example.get('output', '')
            pred_text = self._generate_response(model, input_text)
            
            self._log_example(i, input_text, target_text, pred_text)
            
            # Calculate keyword overlap
            input_keywords = set(input_text.lower().split())
            pred_keywords = set(pred_text.lower().split())
            target_keywords = set(target_text.lower().split())
            
            keyword_match = len(pred_keywords & target_keywords) / len(target_keywords) if target_keywords else 0
            context_relevance = len(pred_keywords & input_keywords) / len(input_keywords) if input_keywords else 0
            
            keyword_matches.append(keyword_match)
            context_scores.append(context_relevance)
        
        results = {
            'keyword_match_rate': np.mean(keyword_matches),
            'context_relevance': np.mean(context_scores)
        }
        
        self.logger.info("Completed content relevance evaluation:")
        for metric, value in results.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return results 