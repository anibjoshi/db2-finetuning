from pathlib import Path
import logging
from preprocessing.first_turn_conv_generator import generate_first_turn_conversations

def main():
    """Main entry point for DB2 training data generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_path = Path("data/raw/sql_error_codes.json")
    output_path = Path("data/processed/first_turn_conversations.jsonl")
    
    try:
        generate_first_turn_conversations(
            input_path=input_path,
            output_path=output_path
        )
        logging.info("Training data generation completed successfully")
    except Exception as e:
        logging.error(f"Error during training data generation: {e}")
        raise

if __name__ == "__main__":
    main() 