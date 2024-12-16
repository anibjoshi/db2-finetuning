import logging
from pathlib import Path
from typing import Optional
from .inference import generate_db2_response

logger = logging.getLogger(__name__)

class InferenceManager:
    """Manages model inference operations."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
    
    def generate_response(
        self,
        question: str,
        db2_version: str = "12.1",
        use_base_model: bool = False
    ) -> Optional[str]:
        """Generate response for DB2 question."""
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