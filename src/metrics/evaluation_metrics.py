from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
from transformers import PreTrainedModel
from datasets import load_dataset
from .base_metrics import BaseMetrics

class EvaluationMetrics(BaseMetrics):
    """Comprehensive model evaluation metrics."""

    def calculate_accuracy(
        self, 
        model: PreTrainedModel, 
        data_path: Path
    ) -> float:
        """Calculate response accuracy against ground truth."""
        dataset = load_dataset("json", data_files=str(data_path))["train"]
        predictions = []
        references = []
        
        for example in dataset:
            input_text = example.get('input', '')
            target_text = example.get('output', '')
            
            # Generate prediction
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model.generate(**inputs)
            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(pred_text)
            references.append(target_text)
            
        # Calculate token-level accuracy
        eval_pred = self.prepare_eval_prediction(predictions, references)
        token_accuracy = np.mean((eval_pred.predictions == eval_pred.label_ids).astype(float))
        
        return token_accuracy

    def evaluate_response_quality(
        self, 
        model: PreTrainedModel, 
        data_path: Path
    ) -> Dict[str, float]:
        """Evaluate quality metrics of model responses."""
        dataset = load_dataset("json", data_files=str(data_path))["train"]
        predictions = []
        references = []
        
        try:
            for example in dataset:
                input_text = example.get('input', '')
                target_text = example.get('output', '')
                
                # Generate prediction
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                predictions.append(pred_text)
                references.append(target_text)
            
            return self.compute_text_metrics(predictions, references)
            
        except Exception as e:
            self.logger.error(f"Error computing quality metrics: {e}")
            raise

    def evaluate_version_compatibility(
        self, 
        model: PreTrainedModel, 
        data_path: Path
    ) -> Dict[str, float]:
        """Evaluate DB2 version-specific performance."""
        dataset = load_dataset("json", data_files=str(data_path))["train"]
        version_correct = 0
        total = 0
        
        try:
            for example in dataset:
                if 'db2_version' in example:
                    total += 1
                    input_text = example.get('input', '')
                    target_text = example.get('output', '')
                    version = example.get('db2_version')
                    
                    # Generate prediction
                    inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        outputs = model.generate(**inputs)
                    pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Check if version-specific content is correct
                    if self._check_version_specific_content(pred_text, target_text, version):
                        version_correct += 1
            
            return {
                'version_accuracy': version_correct / total if total > 0 else 0.0,
                'version_consistency': self._calculate_version_consistency(model, dataset)
            }
            
        except Exception as e:
            self.logger.error(f"Error computing version compatibility metrics: {e}")
            raise

    def _check_version_specific_content(
        self,
        prediction: str,
        reference: str,
        version: str
    ) -> bool:
        """Check if prediction contains correct version-specific content."""
        version_mentioned = version in prediction
        content_match = any(
            ref_part in prediction 
            for ref_part in reference.split() 
            if version in ref_part
        )
        
        return version_mentioned and content_match

    def _calculate_version_consistency(
        self,
        model: PreTrainedModel,
        dataset: Any
    ) -> float:
        """Calculate consistency of responses across versions."""
        query_groups = {}
        for example in dataset:
            if 'query_id' in example and 'db2_version' in example:
                query_id = example['query_id']
                if query_id not in query_groups:
                    query_groups[query_id] = []
                query_groups[query_id].append(example)
        
        consistency_scores = []
        for query_group in query_groups.values():
            if len(query_group) > 1:
                responses = []
                for example in query_group:
                    inputs = self.tokenizer(example['input'], return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        outputs = model.generate(**inputs)
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    responses.append(response)
                
                consistency_scores.append(self._calculate_response_similarity(responses))
        
        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _calculate_response_similarity(self, responses: list) -> float:
        """Calculate similarity between responses."""
        if not responses:
            return 1.0
            
        word_sets = [set(response.lower().split()) for response in responses]
        similarities = []
        
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 1.0
                similarities.append(similarity)
                
        return np.mean(similarities) if similarities else 1.0 