from typing import Dict, Any, Optional
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics.evaluation_metrics import EvaluationMetrics
from utils.config import BEST_MODEL_DIR, EVALUATION_DATA_DIR

class EvaluationManager:
    """Manages model evaluation processes."""

    def __init__(self, model_path: Path = BEST_MODEL_DIR):
        """Initialize evaluation manager.
        
        Args:
            model_path: Path to the model to evaluate
        """
        self.logger = logging.getLogger("EvaluationManager")
        self.model_path = model_path
        
    def load_model(self) -> None:
        """Load model and tokenizer for evaluation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True
            )
            # Initialize metrics after tokenizer is loaded
            self.metrics = EvaluationMetrics(self.tokenizer)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def evaluate(self, evaluation_data: Optional[Path] = None) -> Dict[str, float]:
        """Run evaluation suite on the model.
        
        Args:
            evaluation_data: Optional path to evaluation dataset
                           If None, uses default evaluation data
        
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            self.load_model()
            
            data_path = evaluation_data or EVALUATION_DATA_DIR
            if not data_path.exists():
                raise FileNotFoundError(f"Evaluation data not found at {data_path}")

            results = {
                "accuracy": self.metrics.calculate_accuracy(self.model, data_path),
                "response_quality": self.metrics.evaluate_response_quality(
                    self.model, data_path
                ),
                "version_compatibility": self.metrics.evaluate_version_compatibility(
                    self.model, data_path
                )
            }
            
            # Print results nicely
            print("\nEvaluation Results:")
            for metric, value in results.items():
                if isinstance(value, dict):
                    print(f"\n{metric}:")
                    for submetric, subvalue in value.items():
                        print(f"  {submetric}: {subvalue:.4f}")
                else:
                    print(f"{metric}: {value:.4f}")
            
            self.logger.info(f"Evaluation Results: {results}")
            return results
            
        except Exception as e:
            self.logger.error("Evaluation failed", exc_info=True)
            raise RuntimeError(f"Evaluation failed: {str(e)}") 