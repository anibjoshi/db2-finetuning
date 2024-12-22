import logging
from pathlib import Path
from typing import Optional
from .inference import generate_db2_response
from config import (
    DEFAULT_DB2_VERSION,
    LOGS_DIR
)

logger = logging.getLogger(__name__)

class InferenceManager:
    """Manages high-level inference operations for Db2 question answering.
    
    This class provides a simplified interface for generating responses to
    Db2 questions, handling model selection, error handling, and logging.
    
    Attributes:
        model_path (Path): Path to the model directory
    """
    
    def __init__(self, model_path: Path):
        """Initialize inference manager.
        
        Args:
            model_path: Path to model directory containing weights and config
        """
        self.model_path = model_path
    
    def generate_response(
        self,
        question: str,
        db2_version: str = DEFAULT_DB2_VERSION,
        use_base_model: bool = False
    ) -> Optional[str]:
        """Generate response for Db2 question.
        
        Coordinates the response generation process, including:
        - Model selection (base vs. fine-tuned)
        - Response generation
        - Error handling and logging
        
        Args:
            question: The Db2-related question to answer
            db2_version: Target Db2 version for context
            use_base_model: Whether to use base model instead of fine-tuned
            
        Returns:
            Generated response text, or None if generation fails
            
        Raises:
            RuntimeError: If inference process fails
        """
        try:
            logger.info(f"Using {'base' if use_base_model else 'fine-tuned'} model")
            logger.info(f"Question: {question}")
            
            response = generate_db2_response(
                question=question,
                model_path=self.model_path,
                db2_version=db2_version,
                use_base_model=use_base_model
            )
            
            if not response:
                logger.warning("Empty response received")
                return None
                
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error("Inference failed", exc_info=True)
            raise RuntimeError(f"Inference failed: {str(e)}")