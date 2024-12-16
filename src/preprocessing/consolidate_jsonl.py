from pathlib import Path
import json
from typing import List, Dict, Any
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def consolidate_jsonl_files(
    input_dir: str | Path,
    output_file: str | Path,
    validate: bool = True
) -> None:
    """Consolidate multiple JSONL files into a single JSONL file.
    
    Args:
        input_dir: Directory containing input JSONL files
        output_file: Path to output consolidated JSONL file
        validate: Whether to validate JSON objects during consolidation
        
    Raises:
        ValueError: If input directory doesn't exist or no JSONL files found
        json.JSONDecodeError: If invalid JSON objects encountered
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
        
    # Get all JSONL files
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {input_dir}")
    
    logger.info(f"Found {len(jsonl_files)} JSONL files to consolidate")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if validate:
                            # Validate JSON object
                            try:
                                json_obj = json.loads(line.strip())
                                json.dump(json_obj, outfile, ensure_ascii=False)
                                outfile.write('\n')
                                total_records += 1
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON object in {jsonl_file}: {e}")
                                continue
                        else:
                            # Direct write without validation
                            outfile.write(line)
                            total_records += 1
                            
            except Exception as e:
                logger.error(f"Error processing file {jsonl_file}: {e}")
                continue
    
    logger.info(f"Consolidation complete. Total records: {total_records}")
    logger.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_directory = Path("src/data/processed")
    output_file = Path("src/data/processed/SQL_codes_first_turn_conversations.jsonl")
    
    try:
        consolidate_jsonl_files(
            input_dir=input_directory,
            output_file=output_file,
            validate=True
        )
    except Exception as e:
        logger.error(f"Consolidation failed: {e}") 