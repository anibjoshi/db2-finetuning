from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
import logging
import os
from pathlib import Path

class Db2FineTuningConfig:
    """Configuration for Db2-specific LoRA fine-tuning.
    
    Attributes:
        model_name (str): Base model identifier
        lora_r (int): Rank of LoRA adaptations
        lora_alpha (int): LoRA scaling factor
        lora_dropout (float): Dropout probability for LoRA layers
        learning_rate (float): Training learning rate
        batch_size (int): Per-device batch size
        max_length (int): Maximum sequence length
        num_epochs (int): Number of training epochs
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        warmup_steps (int): Number of steps to warm up learning rate
        load_in_8bit (bool): Whether to load model in 8-bit
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
    """
    def __init__(
        self,
        model_name: str = "src/model/base_model",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        max_length: int = 512,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        load_in_8bit: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.load_in_8bit = load_in_8bit
        self.use_gradient_checkpointing = use_gradient_checkpointing

def tokenize_function(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    """Tokenize and format Db2 dialogue data.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tokenized and formatted examples
    """
    # Print an example to debug the data structure
    print("Example data structure:", examples)
    
    # Handle both single strings and dialogue format
    texts = []
    for example in examples['text'] if 'text' in examples else examples['dialogue']:
        if isinstance(example, str):
            texts.append(example)
        elif isinstance(example, dict):
            # Handle dialogue format
            texts.append(f"User: {example.get('input', '')} Assistant: {example.get('output', '')}")
        elif isinstance(example, list):
            # Handle turn-based dialogue format
            dialogue = " ".join([f"{turn.get('role', 'Speaker')}: {turn.get('content', '')}" 
                               for turn in example])
            texts.append(dialogue)
            
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def train_db2_model(config: Db2FineTuningConfig, dataset: Optional[Dataset] = None) -> None:
    """Train Db2-specific model using LoRA fine-tuning.
    
    Args:
        config: Training configuration
        dataset: Optional pre-processed dataset. If None, loads from default path
        
    Raises:
        RuntimeError: If training fails
        MemoryError: If GPU memory is exceeded
    """
    accelerator = Accelerator()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=True,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try loading in 8-bit first, fall back to 16-bit if bitsandbytes is not available
        try:
            if config.load_in_8bit:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_8bit=True,
                    torch_dtype=torch.float16
                )
                model = prepare_model_for_kbit_training(model)
            else:
                raise ImportError("8-bit loading disabled by config")
        except (ImportError, RuntimeError) as e:
            print(f"Warning: 8-bit quantization failed ({str(e)}), falling back to 16-bit")
            config.load_in_8bit = False
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Print model architecture to identify correct target modules
    print("Model architecture:")
    for name, _ in model.named_modules():
        print(name)

    # Load and process dataset
    try:
        if dataset is None:
            # Use default dataset path if none provided
            dataset = load_dataset(
                "json", 
                data_files={"train": "src/data/processed/SQL0000-0999_first_turn_conversations.jsonl"}
            )
            dataset = dataset["train"]
        
        # Create a Dataset object with the correct format
        if not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a HuggingFace Dataset object")
        
        # Ensure dataset is in the correct format with a 'train' split
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
        
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, config.max_length),
            batched=True,
            remove_columns=dataset.column_names
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load or process dataset: {str(e)}")

    # Update LoRA config with correct target modules
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        # Update target modules based on model architecture
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        bias="none",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./db2_llama_finetuned",
        evaluation_strategy="no",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        gradient_checkpointing=config.use_gradient_checkpointing,
        report_to="tensorboard",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        group_by_length=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    try:
        trainer.train()
        model.save_pretrained("./db2_llama_finetuned")
        print("Fine-tuned model saved successfully")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

class Db2Trainer:
    """Manages Db2-specific LoRA fine-tuning process."""
    
    def __init__(self, config: Db2FineTuningConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure training-specific logging."""
        self.logger = logging.getLogger("Db2Trainer")
    
    def check_gpu(self) -> bool:
        """Check GPU availability and adjust config if needed."""
        if torch.cuda.is_available():
            self.logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 24:
                self.config.batch_size = min(self.config.batch_size, 2)
                self.logger.warning(f"Reduced batch size to {self.config.batch_size} due to GPU memory constraints")
            return True
        self.logger.warning("No GPU available, training will be slow on CPU")
        return False
    
    def prepare_training(self) -> None:
        """Prepare for training by setting up directories and checking data."""
        # Create output directory in src/model
        model_dir = Path("src/model/db2_llama_finetuned")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Log configuration
        self.logger.info("Training configuration:")
        for key, value in vars(self.config).items():
            self.logger.info(f"  {key}: {value}")
    
    def train(self, data_path: Path) -> None:
        """Execute the training process.
        
        Args:
            data_path: Path to training data file
            
        Raises:
            FileNotFoundError: If training data not found
            RuntimeError: If training fails
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
            
        try:
            self.check_gpu()
            self.prepare_training()
            
            # Start training process
            self.logger.info("Starting model training")
            train_db2_model(self.config)  # Your existing training function
            
            # Save model to src/model directory
            model_save_path = Path("src/model/db2_llama_finetuned")
            self.model.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(model_save_path)
            self.logger.info(f"Model saved to {model_save_path}")
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    config = Db2FineTuningConfig()
    trainer = Db2Trainer(config)
    trainer.train(Path("first_turn_conversations.jsonl"))
