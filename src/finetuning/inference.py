from typing import Optional, Dict, Any
import torch
from pathlib import Path
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
import subprocess

class Db2ModelInference:
    """Handles inference for fine-tuned Db2 model."""
    
    def __init__(
        self,
        model_path: str = "src/model/db2_llama_finetuned",
        base_model_path: str = "/Users/aniruddhajoshi/.llama/checkpoints/Llama3.1-8B-Instruct",
        max_length: int = 200,
    ):
        self.logger = logging.getLogger("Db2Inference")
        self.model_path = Path(model_path)
        self.base_model_path = Path(base_model_path)
        self.max_length = max_length
        self.setup_model()
        
    def setup_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Handle base model path conversion if needed
            base_model_path = Path(self.base_model_path)
            if (base_model_path / "consolidated.00.pth").exists():
                converted_path = base_model_path / "hf_converted"
                if not converted_path.exists():
                    # Use transformers-cli for conversion
                    print("Converting model to Hugging Face format...")
                    subprocess.run([
                        "transformers-cli",
                        "convert",
                        "--model_type", "llama",
                        "--input_dir", str(base_model_path),
                        "--output_dir", str(converted_path),
                        "--model_size", "8B"
                    ], check=True)
                base_model_path = converted_path
            
            # First try loading the fine-tuned model
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    local_files_only=True
                )
            except Exception as e:
                # If fine-tuned model not found, load base model
                self.logger.warning(f"Fine-tuned model not found, loading base model: {str(e)}")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    str(base_model_path),
                    local_files_only=True
                )
                self.model = LlamaForCausalLM.from_pretrained(
                    str(base_model_path),
                    device_map="auto",
                    local_files_only=True
                )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def format_prompt(self, query: str, db2_version: Optional[str] = None) -> str:
        """Format the input prompt for Db2 queries."""
        if db2_version:
            return f"user: For Db2 version {db2_version}, {query}\nassistant:"
        return f"user: {query}\nassistant:"
    
    def generate_response(
        self,
        query: str,
        db2_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response for Db2 query."""
        try:
            # Format prompt
            prompt = self.format_prompt(query, db2_version)
            
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "query": query,
                "db2_version": db2_version,
                "response": response,
                "full_prompt": prompt
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def save_response(self, response_data: Dict[str, Any], output_file: Path) -> None:
        """Save response to file."""
        try:
            with open(output_file, 'a') as f:
                json.dump(response_data, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save response: {str(e)}")

def main():
    """CLI interface for model inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Db2 Model Inference")
    parser.add_argument("query", help="Db2 query or question")
    parser.add_argument("--version", help="Db2 version")
    parser.add_argument(
        "--model-path", 
        default="src/model/db2_llama_finetuned",
        help="Path to fine-tuned model"
    )
    parser.add_argument("--output", type=Path, help="Save response to file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize inference
        inference = Db2ModelInference(model_path=args.model_path)
        
        # Generate response
        response = inference.generate_response(args.query, args.version)
        
        # Print response
        print("\nResponse:", response["response"])
        
        # Save if output specified
        if args.output:
            inference.save_response(response, args.output)
            
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 