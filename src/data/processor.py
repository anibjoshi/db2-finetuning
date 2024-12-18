import logging
from pathlib import Path
from typing import List
from preprocessing.first_turn_conv_generator import generate_first_turn_conversations

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data processing operations for Db2 documentation."""
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
    
    def get_raw_files(self) -> List[Path]:
        """Get all JSONL files from raw data directory."""
        return sorted(Path(self.input_path).glob("*.jsonl"))
    
    def get_output_path(self, input_file: Path) -> Path:
        """Generate output path for processed file."""
        return self.output_path / f"{input_file.stem}_first_turn_conversations.jsonl"
    
    def process_single_file(self, input_file: Path, output_file: Path) -> None:
        """Process a single documentation file."""
        try:
            logger.info(f"Processing {input_file.name}")
            generate_first_turn_conversations(
                input_path=input_file,
                output_path=output_file
            )
            logger.info(f"Saved processed data to {output_file}")
        except Exception as e:
            logger.error(f"Failed to process {input_file.name}", exc_info=True)
            raise
    
    def process_all(self) -> None:
        """Process all documentation files."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        
        if self.input_path.is_dir():
            self._process_directory()
        else:
            self._process_file()
    
    def _process_directory(self) -> None:
        """Process all files in the input directory."""
        input_files = self.get_raw_files()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        for input_file in input_files:
            output_file = self.get_output_path(input_file)
            
            if output_file.exists():
                logger.info(f"Skipping {input_file.name} - output exists")
                continue
                
            try:
                self.process_single_file(input_file, output_file)
            except Exception:
                continue
    
    def _process_file(self) -> None:
        """Process a single input file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_path.exists():
            logger.info(f"Skipping {self.input_path.name} - output exists")
            return
            
        self.process_single_file(self.input_path, self.output_path) 