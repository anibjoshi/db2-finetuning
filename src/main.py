import logging
import argparse
from pathlib import Path
from typing import Optional
from transformers import LlamaTokenizer
from datasets import load_dataset

from preprocessing.first_turn_conv_generator import generate_first_turn_conversations
from finetuning.LoRA_training import (
    Db2FineTuningConfig,
    Db2Trainer,
    train_db2_model
)
from finetuning.inference import generate_db2_response

def setup_logging(log_file: Optional[str] = "training.log") -> None:
    """Configure global logging with both file and console output.
    
    Args:
        log_file: Path to log file. If None, only console logging is enabled.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def generate_data(input_path: Path, output_path: Path) -> None:
    """Generate training data from raw Db2 documentation.
    
    Args:
        input_path: Path to raw JSONL data file
        output_path: Path to save processed conversations
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If data generation fails
    """
    logger = logging.getLogger("data_generation")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Generating training data from {input_path}")
        generate_first_turn_conversations(
            input_path=input_path,
            output_path=output_path
        )
        logger.info(f"Training data saved to {output_path}")
    except Exception as e:
        logger.error("Data generation failed", exc_info=True)
        raise RuntimeError(f"Data generation failed: {str(e)}")

def train_model(config: Db2FineTuningConfig, data_path: Path) -> None:
    """Initialize and run model training using Hugging Face Trainer.
    
    Args:
        config: Training configuration
        data_path: Path to training data
        
    Raises:
        RuntimeError: If training fails
    """
    logger = logging.getLogger("model_training")
    
    try:
        logger.info("Loading dataset")
        dataset = load_dataset(
            "json", 
            data_files={"train": str(data_path)}
        )
        
        logger.info("Starting model training")
        train_db2_model(config)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error("Training failed", exc_info=True)
        raise RuntimeError(f"Training failed: {str(e)}")

def run_inference(
    question: str, 
    model_path: Path, 
    db2_version: str = "12.1",
    use_base_model: bool = False
) -> None:
    """Run inference using either base or fine-tuned model.
    
    Args:
        question: User's DB2 related question
        model_path: Path to model directory
        db2_version: DB2 version to consider (default: 12.1)
        use_base_model: If True, use base model instead of fine-tuned
        
    Raises:
        RuntimeError: If inference fails
    """
    logger = logging.getLogger("inference")
    
    try:
        logger.info(f"Generating response using {'base' if use_base_model else 'fine-tuned'} model")
        logger.info(f"Question: {question}")
        
        response = generate_db2_response(
            question=question,
            model_path=model_path,
            db2_version=db2_version,
            use_base_model=use_base_model
        )
        
        if not response:
            logger.warning("Empty response received")
            print("\nNo response generated")
        else:
            print("\nResponse:", response)
            logger.info("Response generated successfully")
        
    except Exception as e:
        logger.error("Inference failed", exc_info=True)
        raise RuntimeError(f"Inference failed: {str(e)}")

def main() -> None:
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Db2 Documentation Processing, Model Training and Inference"
    )
    parser.add_argument(
        'action',
        choices=['generate', 'train', 'infer', 'all'],
        help='Action to perform: generate data, train model, run inference, or generate+train'
    )
    parser.add_argument(
        '--input-path',
        type=Path,
        default=Path("src/data/raw/SQL0000-0999.jsonl"),
        help='Path to raw input data'
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path("src/data/processed/SQL0000-0999_first_turn_conversations.jsonl"),
        help='Path for processed output data'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default="training.log",
        help='Path to log file'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path("src/model/base_model"),
        help='Path to model directory'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='DB2 question for inference'
    )
    parser.add_argument(
        '--db2-version',
        type=str,
        default="12.1",
        choices=["11.1", "11.5", "12.1"],
        help='DB2 version for inference'
    )
    parser.add_argument(
        '--use-base-model',
        action='store_true',
        help='Use base model instead of fine-tuned model for inference'
    )
    
    args = parser.parse_args()
    setup_logging(args.log_file)
    logger = logging.getLogger("main")
    
    try:
        if args.action in ['generate', 'all']:
            logger.info("Starting data generation...")
            generate_data(args.input_path, args.output_path)
        
        if args.action in ['train', 'all']:
            logger.info("Starting model training...")
            config = Db2FineTuningConfig()
            train_model(config, args.output_path)
            
        if args.action == 'infer':
            if not args.question:
                raise ValueError("--question argument is required for inference")
            logger.info("Starting inference...")
            run_inference(
                question=args.question,
                model_path=args.model_path,
                db2_version=args.db2_version,
                use_base_model=args.use_base_model
            )
            
    except Exception as e:
        logger.error("Process failed", exc_info=True)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
