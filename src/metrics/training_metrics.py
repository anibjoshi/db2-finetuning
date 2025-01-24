from typing import Dict, Optional
import logging
from dataclasses import dataclass

@dataclass
class MetricsConfig:
    """Configuration for training metrics.
    
    Attributes:
        log_to_file: Whether to log metrics to a file
        metrics_output_dir: Directory to save metrics logs
    """
    log_to_file: bool = True
    metrics_output_dir: str = "logs/metrics"

class TrainingMetrics:
    """Simple metrics tracking during model training."""

    def __init__(self, config: MetricsConfig = None):
        """Initialize training metrics.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log computed metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional training step number
        """
        try:
            step_info = f" at step {step}" if step is not None else ""
            self.logger.info(f"Metrics{step_info}:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise 