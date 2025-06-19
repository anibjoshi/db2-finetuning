from pathlib import Path
import argparse
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_model_to_hub(
    model_path: Path,
    repo_name: str,
    hf_token: str,
    organization: str = None,
    private: bool = False,
    model_card: str = None
) -> None:
    """Upload fine-tuned model to HuggingFace Hub.
    
    Args:
        model_path: Path to the fine-tuned model directory
        repo_name: Name for the HuggingFace repository
        hf_token: HuggingFace API token
        organization: Optional organization to upload to
        private: Whether the repository should be private
        model_card: Optional path to model card markdown file
    """
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create full repo name
        full_repo_name = f"{organization}/{repo_name}" if organization else repo_name
        
        # Create repository
        logger.info(f"Creating repository: {full_repo_name}")
        create_repo(
            repo_id=full_repo_name,
            token=hf_token,
            private=private,
            exist_ok=True
        )
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Push to hub
        logger.info("Uploading model to HuggingFace Hub")
        model.push_to_hub(
            repo_id=full_repo_name,
            token=hf_token,
            commit_message="Upload fine-tuned DB2 assistant model"
        )
        
        tokenizer.push_to_hub(
            repo_id=full_repo_name,
            token=hf_token,
            commit_message="Upload tokenizer"
        )
        
        # Upload model card if provided
        if model_card:
            logger.info("Uploading model card")
            api.upload_file(
                path_or_fileobj=model_card,
                path_in_repo="README.md",
                repo_id=full_repo_name,
                token=hf_token
            )
            
        logger.info(f"Successfully uploaded model to: https://huggingface.co/{full_repo_name}")
        
    except Exception as e:
        logger.error(f"Failed to upload model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--repo-name", type=str, required=True, help="Name for HuggingFace repository")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--org", type=str, help="Optional organization name")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--model-card", type=Path, help="Path to model card markdown file")
    
    args = parser.parse_args()
    
    upload_model_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=args.token,
        organization=args.org,
        private=args.private,
        model_card=args.model_card
    )

if __name__ == "__main__":
    main() 