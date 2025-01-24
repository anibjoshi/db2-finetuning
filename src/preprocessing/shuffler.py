import json
from pathlib import Path
from typing import List, Dict, Optional
import random
from tqdm import tqdm
import logging
from dataclasses import dataclass
import sys

@dataclass
class ShuffleConfig:
    """Configuration for data shuffling.
    
    Attributes:
        chunk_size: Number of examples to process at once for memory efficiency
        random_seed: Seed for reproducible shuffling
        input_file: Path to input JSONL file
        output_file: Path to output shuffled JSONL file
    """
    chunk_size: int = 10000
    random_seed: Optional[int] = 42
    input_file: str = "src/data/processed/SQL_codes_first_turn_conversations.jsonl"
    output_file: str = "src/data/processed/SQL_codes_shuffled_conversations.jsonl"

class DataShuffler:
    """Handles efficient shuffling of large JSONL datasets."""
    
    def __init__(self, config: ShuffleConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the shuffling process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _count_lines(self, file_path: str) -> int:
        """Count number of lines in JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            Number of lines in file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _read_jsonl_chunks(self, file_path: str) -> List[Dict]:
        """Read JSONL file in chunks to manage memory.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of JSON objects from current chunk
        """
        chunk = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk.append(json.loads(line))
                    if len(chunk) >= self.config.chunk_size:
                        yield chunk
                        chunk = []
                if chunk:  # Yield remaining items
                    yield chunk
        except Exception as e:
            self.logger.error(f"Error reading JSONL file: {e}")
            raise
    
    def _write_jsonl(self, data: List[Dict], output_path: str, mode: str = 'w') -> None:
        """Write data to JSONL file.
        
        Args:
            data: List of JSON objects to write
            output_path: Path to output file
            mode: File opening mode ('w' for write, 'a' for append)
        """
        try:
            with open(output_path, mode, encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"Error writing JSONL file: {e}")
            raise

    def shuffle_data(self) -> None:
        """Shuffle the entire dataset while managing memory efficiently."""
        try:
            if self.config.random_seed is not None:
                random.seed(self.config.random_seed)
            
            total_lines = self._count_lines(self.config.input_file)
            self.logger.info(f"Processing {total_lines} examples...")
            
            # Read and shuffle in chunks
            all_chunks = []
            for chunk in tqdm(self._read_jsonl_chunks(self.config.input_file), 
                            desc="Reading chunks"):
                random.shuffle(chunk)
                all_chunks.append(chunk)
            
            # Shuffle the chunks themselves
            random.shuffle(all_chunks)
            
            # Write shuffled data
            self.logger.info("Writing shuffled data...")
            for i, chunk in enumerate(all_chunks):
                mode = 'w' if i == 0 else 'a'
                self._write_jsonl(chunk, self.config.output_file, mode)
            
            self.logger.info("Shuffling completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Shuffling failed: {e}")
            raise

def main():
    """Main entry point for data shuffling."""
    config = ShuffleConfig()
    shuffler = DataShuffler(config)
    shuffler.shuffle_data()

if __name__ == "__main__":
    main() 