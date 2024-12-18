import logging
import argparse
from pathlib import Path
from typing import Optional

from data.processor import DataProcessor
from training.training_manager import TrainingManager
from inference.inference_manager import InferenceManager
from training.LoRA_training import Db2FineTuningConfig
from config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    BEST_MODEL_DIR,
    DEFAULT_LOG_FILE,
    SUPPORTED_DB2_VERSIONS,
    DEFAULT_DB2_VERSION
)

def setup_logging(log_file: Optional[str] = "training.log") -> None:
    """Configure application-wide logging.
    
    Sets up logging to both console and file with timestamp, logger name,
    and log level. Creates a new log file or appends to existing one.
    
    Args:
        log_file: Path to log file. If None, only logs to console
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments.
    
    Configures argument parser with three main actions:
    1. generate: Process raw DB2 documentation into training data
    2. train: Fine-tune the model on processed data
    3. infer: Run inference using the trained model
    
    Returns:
        Parsed command line arguments
        
    Example:
        python main.py generate --raw-dir data/raw --output-dir data/processed
        python main.py train --data-dir data/processed
        python main.py infer "How do I create a database?" --version 12.1
    """
    parser = argparse.ArgumentParser(
        description="DB2 Documentation Processing, Training and Inference"
    )
    
    # Create subparsers for different actions
    subparsers = parser.add_subparsers(dest='action', required=True)
    
    # Data generation parser
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument(
        '--raw-dir',
        type=Path,
        default=RAW_DATA_DIR,
        help='Directory containing raw DB2 documentation'
    )
    gen_parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROCESSED_DATA_DIR,
        help='Directory for processed training data'
    )
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-dir',
        type=Path,
        default=PROCESSED_DATA_DIR,
        help='Directory containing processed training data'
    )
    
    # Inference parser
    infer_parser = subparsers.add_parser('infer', help='Run model inference')
    infer_parser.add_argument(
        'question',
        type=str,
        help='DB2 question to answer'
    )
    infer_parser.add_argument(
        '--version',
        type=str,
        default=DEFAULT_DB2_VERSION,
        choices=SUPPORTED_DB2_VERSIONS,
        help='DB2 version'
    )
    
    # Global arguments
    parser.add_argument(
        '--log-file',
        type=str,
        default=DEFAULT_LOG_FILE,
        help='Path to log file'
    )
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the DB2 Assistant application.
    
    Orchestrates the complete workflow based on command line arguments:
    1. Data Generation: Process raw DB2 documentation into training format
    2. Training: Fine-tune the model using processed data
    3. Inference: Generate responses to DB2 questions
    
    The function handles all high-level error cases and ensures proper
    logging of any failures.
    
    Raises:
        SystemExit(1): If any stage of the process fails
    """
    args = parse_args()
    setup_logging(args.log_file)
    logger = logging.getLogger("main")
    
    try:
        if args.action == 'generate':
            # Process raw DB2 documentation into training data
            processor = DataProcessor(args.raw_dir, args.output_dir)
            processor.process_all()
        
        elif args.action == 'train':
            # Initialize training configuration and start training
            config = Db2FineTuningConfig()
            trainer = TrainingManager(config)
            
            # Process single file or directory of files
            trainer.train(args.data_dir)
            
        elif args.action == 'infer':
            # Load model and generate response
            inferencer = InferenceManager(BEST_MODEL_DIR)
            response = inferencer.generate_response(
                question=args.question,
                db2_version=args.version
            )
            
            # Display response
            if response:
                print("\nResponse:", response)
            else:
                print("\nNo response generated")
            
    except Exception as e:
        logger.error("Process failed", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
