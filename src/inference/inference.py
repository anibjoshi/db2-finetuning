from typing import Optional, Dict, Any
import torch
from pathlib import Path
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    LOGS_DIR,
    DEFAULT_DB2_VERSION,
    SUPPORTED_DB2_VERSIONS
)

logger = logging.getLogger(__name__)

class DB2ModelInference:
    """Handles inference for DB2 question answering using either base or fine-tuned model.
    
    This class manages the complete inference pipeline including:
    - Model and tokenizer loading with error handling
    - Input processing and prompt formatting
    - Response generation with configurable parameters
    - Response cleaning and formatting
    
    Attributes:
        model_path (Path): Path to the model directory
        max_length (int): Maximum sequence length for generation
        device (torch.device): Device to run inference on
        model (AutoModelForCausalLM): Loaded model instance
        tokenizer (AutoTokenizer): Model tokenizer instance
    """
    
    def __init__(
        self,
        model_path: Path,
        use_base_model: bool = False,
        max_length: int = 512,
    ):
        """Initialize the inference model.
        
        Args:
            model_path: Path to model directory
            use_base_model: Whether to use base model or fine-tuned model
            max_length: Maximum length for generated responses
            
        Raises:
            ValueError: If model path doesn't exist
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
            
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.load_model()
        
    def load_model(self) -> None:
        """Load the model and tokenizer.
        
        Loads the model and tokenizer from the specified path, handling
        tokenizer configuration and model loading with appropriate settings
        for inference.
        
        Raises:
            RuntimeError: If model or tokenizer loading fails
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer with padding token configuration
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with optimized settings for inference
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",  # Automatically handle device placement
                torch_dtype=torch.float16,  # Use half precision for efficiency
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def generate_response(
        self,
        question: str,
        db2_version: Optional[str] = DEFAULT_DB2_VERSION,
    ) -> str:
        """Generate response for a DB2 question.
        
        Processes the input question, formats it with DB2 version context,
        and generates a response using the loaded model.
        
        Args:
            question: The DB2-related question
            db2_version: Optional DB2 version context
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If unsupported DB2 version provided
            RuntimeError: If generation fails
        """
        try:
            # Validate DB2 version against supported versions
            if db2_version and db2_version not in SUPPORTED_DB2_VERSIONS:
                raise ValueError(f"Unsupported DB2 version: {db2_version}. Must be one of {SUPPORTED_DB2_VERSIONS}")
                
            # Format prompt with version context if provided
            if db2_version:
                prompt = f"For DB2 version {db2_version}, {question}"
            else:
                prompt = question
                
            # Add system context for better responses
            full_prompt = (
                "You are a DB2 database expert assistant. "
                "Provide clear and accurate answers to DB2 related questions. "
                "Give a single, complete response.\n\n"
                f"User: {prompt}\n"
                "Assistant:"
            )
            
            logger.info(f"Generated prompt: {full_prompt}")
            
            # Tokenize input with appropriate padding and truncation
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate response with carefully tuned parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    min_length=50,  # Ensure substantive responses
                    temperature=0.7,  # Balance between creativity and focus
                    top_p=0.9,  # Nucleus sampling for natural text
                    repetition_penalty=1.3,  # Discourage repetition
                    no_repeat_ngram_size=3,  # Prevent repeating phrases
                    length_penalty=1.0,  # Balanced length control
                    early_stopping=True,  # Efficient generation
                    do_sample=True,  # Enable sampling for natural text
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
            # Decode and clean up the generated response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract only the assistant's response
            parts = response.split("Assistant:", 1)
            if len(parts) > 1:
                response = parts[1].strip()
                response = response.split("User:")[0].split("Assistant:")[0].strip()
            else:
                response = response.strip()
                
            logger.info(f"Generated response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
            
    def save_conversation(
        self,
        question: str,
        response: str,
        db2_version: Optional[str],
        output_file: Path
    ) -> None:
        """Save the conversation to a file.
        
        Args:
            question: The original question
            response: The generated response
            db2_version: DB2 version if specified
            output_file: Path to output file
        """
        try:
            conversation = {
                "question": question,
                "response": response,
                "db2_version": db2_version
            }
            
            with open(output_file, 'a') as f:
                json.dump(conversation, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")

def generate_db2_response(
    question: str,
    model_path: Path,
    db2_version: Optional[str] = None,
    use_base_model: bool = False
) -> str:
    """Generate a response for a DB2 query.
    
    Args:
        question: The DB2-related question
        model_path: Path to model directory
        db2_version: Optional DB2 version
        use_base_model: Whether to use base model
        
    Returns:
        Generated response text
        
    Raises:
        RuntimeError: If generation fails
    """
    try:
        inference = DB2ModelInference(
            model_path=model_path,
            use_base_model=use_base_model
        )
        return inference.generate_response(question, db2_version)
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {str(e)}") 