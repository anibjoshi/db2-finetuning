from typing import Dict, Any, Optional
import logging
from pathlib import Path
import random
import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics.evaluation_metrics import EvaluationMetrics
from utils.config import (
    BEST_MODEL_DIR, 
    EVALUATION_DATA_DIR,
    PROCESSED_DATA_DIR
)
from datetime import datetime

class EvaluationManager:
    """Manages model evaluation processes."""

    def __init__(
        self, 
        model_path: Path = BEST_MODEL_DIR,
        eval_samples: int = 1000,
        seed: int = 42
    ):
        """Initialize evaluation manager.
        
        Args:
            model_path: Path to the model to evaluate
            eval_samples: Number of samples to use for evaluation
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger("EvaluationManager")
        self.model_path = model_path
        self.eval_samples = eval_samples
        self.seed = seed
        random.seed(seed)
        
        # Create directories
        self.eval_data_dir = EVALUATION_DATA_DIR
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Add results directory
        self.results_dir = Path("src/data/evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_evaluation_set(self, data_path: Path) -> Path:
        """Create and save evaluation dataset from training data."""
        try:
            # Generate fixed evaluation file name
            eval_file = self.eval_data_dir / f"eval_set_{data_path.stem}.jsonl"
            
            # If evaluation file exists, use it
            if eval_file.exists():
                self.logger.info(f"Using existing evaluation set: {eval_file}")
                return eval_file
            
            # Create new evaluation set
            self.logger.info(f"Creating new evaluation set: {eval_file}")
            
            # Read and process the JSONL file directly
            eval_examples = []
            with open(data_path, 'r') as f:
                lines = f.readlines()
                # Sample lines if we have more than we need
                if len(lines) > self.eval_samples:
                    random.seed(42)  # Fixed seed
                    lines = random.sample(lines, self.eval_samples)
                
                for line in lines:
                    dialogue = json.loads(line)['dialogue']
                    # Extract first turn Q&A
                    if len(dialogue) >= 2 and dialogue[0]['role'] == 'user' and dialogue[1]['role'] == 'assistant':
                        example = {
                            'input': dialogue[0]['content'],
                            'output': dialogue[1]['content']
                        }
                        eval_examples.append(example)
            
            # Save evaluation dataset
            with open(eval_file, 'w') as f:
                for example in eval_examples:
                    f.write(json.dumps(example) + '\n')
            
            self.logger.info(f"Created evaluation set with {len(eval_examples)} samples")
            # Log a few examples
            for i, example in enumerate(eval_examples[:3]):
                self.logger.info(f"\nExample {i+1}:")
                self.logger.info(f"Input: {example['input']}")
                self.logger.info(f"Output: {example['output']}")
            
            return eval_file
            
        except Exception as e:
            self.logger.error(f"Failed to create evaluation set: {e}")
            raise

    def prepare_eval_data(self, data_path: Optional[Path] = None) -> Dataset:
        """Prepare evaluation dataset."""
        try:
            # Use default processed data path if none provided
            if data_path is None:
                data_path = next(PROCESSED_DATA_DIR.glob("SQL*.jsonl"))
                self.logger.info(f"Using default training data: {data_path}")
            
            # Create evaluation set if using training data
            if PROCESSED_DATA_DIR in data_path.parents:
                data_path = self.create_evaluation_set(data_path)
            
            # Load and validate the dataset
            dataset = load_dataset("json", data_files=str(data_path))["train"]
            
            # Debug: Check dataset content
            self.logger.info(f"\nFirst 3 examples from evaluation dataset:")
            for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
                self.logger.info(f"\nExample {i+1}:")
                if isinstance(example, dict):
                    self.logger.info(f"Input: {example.get('input', 'NO INPUT')}")
                    self.logger.info(f"Output: {example.get('output', 'NO OUTPUT')}")
                else:
                    self.logger.info(f"Unexpected example format: {example}")
            
            if len(dataset) == 0:
                raise ValueError("Empty evaluation dataset")
            
            # Validate dataset format
            for example in dataset:
                if isinstance(example, dict):
                    if not example.get('input') or not example.get('output'):
                        self.logger.warning("Found example with missing input or output")
                        self.logger.warning(f"Example: {example}")
                else:
                    self.logger.warning(f"Found example with unexpected format: {example}")
            
            self.logger.info(f"Loaded {len(dataset)} samples for evaluation from {data_path}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to prepare evaluation data: {e}")
            raise

    def load_model(self) -> None:
        """Load model and tokenizer for evaluation."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True
            )
            self.logger.info("Model loaded successfully")
            self.logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            # Initialize metrics after tokenizer is loaded
            self.metrics = EvaluationMetrics(self.tokenizer)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save evaluation metrics to a file.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"eval_results_{timestamp}.json"
        
        # Add metadata to results
        results = {
            "timestamp": timestamp,
            "model_path": str(self.model_path),
            "num_samples": self.eval_samples,
            "metrics": metrics
        }
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved evaluation results to {results_file}")

    def evaluate(
        self, 
        training_data: Optional[Path] = None,
        evaluation_data: Optional[Path] = None
    ) -> Dict[str, float]:
        """Run evaluation suite on the model.
        
        Args:
            training_data: Optional path to training dataset. 
                         If None, uses default processed data.
            evaluation_data: Optional path to separate evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            self.load_model()
            
            # Use prepared evaluation data
            train_eval_data = self.prepare_eval_data(training_data)
            train_results = {
                "training_data_eval": {
                    "accuracy": self.metrics.calculate_accuracy(self.model, train_eval_data),
                    "response_quality": self.metrics.evaluate_response_quality(
                        self.model, train_eval_data
                    ),
                    "content_relevance": self.metrics.evaluate_content_relevance(
                        self.model, train_eval_data
                    )
                }
            }
            
            # If separate evaluation data is provided, evaluate on that too
            if evaluation_data is not None and evaluation_data.exists():
                eval_dataset = self.prepare_eval_data(evaluation_data)
                eval_results = {
                    "held_out_eval": {
                        "accuracy": self.metrics.calculate_accuracy(self.model, eval_dataset),
                        "response_quality": self.metrics.evaluate_response_quality(
                            self.model, eval_dataset
                        ),
                        "content_relevance": self.metrics.evaluate_content_relevance(
                            self.model, eval_dataset
                        )
                    }
                }
                train_results.update(eval_results)
            
            # Print results nicely
            print("\nEvaluation Results:")
            for dataset_name, metrics in train_results.items():
                print(f"\n{dataset_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"  {metric_name}:")
                        for submetric, subvalue in value.items():
                            print(f"    {submetric}: {subvalue:.4f}")
                    else:
                        print(f"  {metric_name}: {value:.4f}")
            
            # Save metrics to file
            self.save_metrics(train_results)
            
            return train_results
            
        except Exception as e:
            self.logger.error("Evaluation failed", exc_info=True)
            raise RuntimeError(f"Evaluation failed: {str(e)}") 