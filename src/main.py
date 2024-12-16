import logging
import argparse
from pathlib import Path
from typing import Optional

from data.processor import DataProcessor
from training.training_manager import TrainingManager
from inference.inference_manager import InferenceManager
from training.LoRA_training import Db2FineTuningConfig

def setup_logging(log_file: Optional[str] = "training.log") -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
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
        default=Path("src/data/raw"),
        help='Directory containing raw DB2 documentation'
    )
    gen_parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("src/data/processed"),
        help='Directory for processed training data'
    )
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("src/data/processed"),
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
        default="12.1",
        choices=["11.1", "11.5", "12.1"],
        help='DB2 version'
    )
    
    # Global arguments
    parser.add_argument(
        '--log-file',
        type=str,
        default="db2_assistant.log",
        help='Path to log file'
    )
    
    return parser.parse_args()

def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_file)
    logger = logging.getLogger("main")
    
    try:
        if args.action == 'generate':
            processor = DataProcessor(args.raw_dir, args.output_dir)
            processor.process_all()
        
        elif args.action == 'train':
            trainer = TrainingManager(Db2FineTuningConfig())
            trainer.train(args.data_dir)
            
        elif args.action == 'infer':
            model_path = Path("src/model/db2_model")  # Use a sensible default
            inferencer = InferenceManager(model_path)
            response = inferencer.generate_response(
                question=args.question,
                db2_version=args.version
            )
            
            if response:
                print("\nResponse:", response)
            else:
                print("\nNo response generated")
            
    except Exception as e:
        logger.error("Process failed", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
