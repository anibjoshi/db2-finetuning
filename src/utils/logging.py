import logging
from typing import Dict

def log_training_metrics(metrics: Dict[str, float], step: int) -> None:
    """
    Log training and evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
    """
    logger = logging.getLogger(__name__)
    
    for metric_name, value in metrics.items():
        logger.info(f"Step {step} - {metric_name}: {value:.4f}") 