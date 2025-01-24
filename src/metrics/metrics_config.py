from dataclasses import dataclass

@dataclass
class MetricsConfig:
    """Configuration for training and evaluation metrics."""
    
    # Batch processing settings
    eval_batch_size: int = 8  # Size of batches for evaluation
    
    # Token handling
    ignore_token_id: int = -100  # Token ID to ignore in loss calculation
    
    # Metric computation settings
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_token_accuracy: bool = True
    
    # Thresholds
    min_length_for_rouge: int = 10  # Minimum text length for ROUGE calculation
    max_sequence_length: int = 512  # Maximum sequence length for processing 