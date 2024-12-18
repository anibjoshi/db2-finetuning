import logging
import argparse
from pathlib import Path
from typing import Optional

from preprocessing.data_processor import DataProcessor
from training.training_manager import TrainingManager
from inference.inference_manager import InferenceManager
from training.training_config import TrainingConfig
from experiments.experiments import ExperimentManager, ExperimentConfig

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
    
    Configures argument parser with four main actions:
    1. generate: Process raw DB2 documentation into training data
    2. train: Fine-tune the model on processed data
    3. run-experiment: Run hyperparameter optimization
    4. infer: Run inference using the trained model
    
    Returns:
        Parsed command line arguments
        
    Example:
        python main.py generate
        python main.py train
        python main.py run-experiment
        python main.py infer "How do I create a database?" 
    """
    parser = argparse.ArgumentParser(
        description="Db2 Documentation Processing, Training and Inference"
    )
    
    # Create subparsers for different actions
    subparsers = parser.add_subparsers(dest='action', required=True)
    
    # Data generation parser
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument(
        '--raw-dir',
        type=Path,
        default=RAW_DATA_DIR,
        help='Directory containing raw Db2 documentation'
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
    
    # Experiment parser - simple command without options
    subparsers.add_parser('run-experiment', help='Run hyperparameter optimization')
    
    # Inference parser
    infer_parser = subparsers.add_parser('infer', help='Run model inference')
    infer_parser.add_argument(
        'question',
        type=str,
        help='Db2 question to answer'
    )
    infer_parser.add_argument(
        '--version',
        type=str,
        default=DEFAULT_DB2_VERSION,
        choices=SUPPORTED_DB2_VERSIONS,
        help='Db2 version'
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
    """Main entry point for the Db2 Assistant application.
    
    Orchestrates the complete workflow based on command line arguments:
    1. Data Generation: Process raw Db2 documentation into training format
    2. Training: Fine-tune the model using processed data
    3. Experimentation: Run hyperparameter optimization
    4. Inference: Generate responses to Db2 questions
    
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
            # Process raw Db2 documentation into training data
            processor = DataProcessor(args.raw_dir, args.output_dir)
            processor.process_all()
        
        elif args.action == 'train':
            # Initialize training configuration and start training
            config = TrainingConfig()
            trainer = TrainingManager(config)
            print(args.data_dir)
            # Process single file or directory of files
            trainer.train(args.data_dir)
            
        elif args.action == 'run-experiment':
            # Run hyperparameter optimization
            exp_config = ExperimentConfig()  # Use default configuration
            experiment = ExperimentManager(exp_config)
            best_params = experiment.run_optimization()
            logger.info(f"Best hyperparameters found: {best_params}")
            
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
