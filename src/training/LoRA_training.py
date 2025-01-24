from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
import os
from pathlib import Path
from utils.config import FINETUNED_MODEL_DIR, BEST_MODEL_DIR, LOGS_DIR
from .training_config import TrainingConfig
from metrics.training_metrics import TrainingMetrics
from utils.metrics_logger import MetricsLogger

def tokenize_function(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    """Tokenize and format Db2 dialogue data.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tokenized and formatted examples
    """
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

class Db2Trainer:
    """Manages Db2-specific LoRA fine-tuning process.
    
    This class handles the complete training pipeline for fine-tuning a language model
    on Db2 documentation using LoRA (Low-Rank Adaptation). It manages:
    - GPU resource checking and optimization
    - Model and tokenizer loading with 8-bit quantization support
    - Dataset preparation and validation split
    - Training configuration and execution
    - Model checkpointing and saving
    
    Attributes:
        config (TrainingConfig): Configuration for model training
        logger (Logger): Training-specific logger instance
        model (PreTrainedModel): The model being fine-tuned
        tokenizer (AutoTokenizer): Tokenizer for the model
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.
        
        Args:
            config: Training configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger("Db2Trainer")
        self.model = None
        self.tokenizer = None
        self.metrics = None  # Will initialize after tokenizer is loaded
        
        # Set tokenizer parallelism based on number of workers
        if self.config.dataloader_num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable when using multiple workers
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"   # Enable for single process
    
    def check_gpu(self) -> bool:
        """Check GPU availability and adjust training parameters accordingly.
        
        Checks available GPU memory and adjusts batch size and gradient
        accumulation steps to prevent out-of-memory errors. For GPUs with
        less than 24GB memory, reduces batch size and increases gradient
        accumulation steps.
        
        Returns:
            bool: True if GPU is available, False if training on CPU
        """
        if not torch.cuda.is_available():
            self.logger.warning("No GPU available - training will be slow")
            return False
            
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 24:
            self.config.batch_size = min(self.config.batch_size, 2)
            self.config.gradient_accumulation_steps *= 2
            self.logger.warning(
                f"Limited GPU memory ({gpu_memory:.1f}GB) - adjusted batch size: {self.config.batch_size}"
            )
        return True
        
    def load_model_and_tokenizer(self) -> None:
        """Load and prepare the model and tokenizer for training.
        
        Attempts to load the model in 8-bit quantization first for memory efficiency.
        Falls back to 16-bit if 8-bit loading fails. Configures the tokenizer and
        applies LoRA adaptation to the model.
        
        Raises:
            RuntimeError: If model loading fails
        """
        # Load tokenizer and set padding token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=True,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        
        # Initialize metrics after tokenizer is loaded
        self.metrics = TrainingMetrics(self.tokenizer)
        
        # Try 8-bit loading first, fall back to 16-bit
        try:
            if self.config.load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_8bit=True,
                    torch_dtype=torch.bfloat16 if self.config.use_bf16 else None
                )
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                raise ImportError("8-bit loading disabled")
        except (ImportError, RuntimeError) as e:
            self.logger.warning(f"8-bit loading failed, using 16-bit: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16 if self.config.use_bf16 else None
            )
            
        # Configure and apply LoRA adaptation
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            inference_mode=False
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing by default for memory efficiency
        self.model.gradient_checkpointing_enable()
    
    def prepare_datasets(self, data_path: Path) -> Tuple[Dataset, Dataset]:
        """Load and prepare training and validation datasets.
        
        Loads data from the specified path, splits it into training and validation
        sets (90/10 split), applies shuffling, and tokenizes both sets.
        
        Args:
            data_path: Path to the training data file
            
        Returns:
            Tuple containing (training_dataset, validation_dataset)
            
        Raises:
            FileNotFoundError: If training data file not found
            RuntimeError: If dataset processing fails
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
            
        # Load dataset and shuffle before splitting
        dataset = load_dataset("json", data_files={"train": str(data_path)})["train"]
        dataset = dataset.shuffle(seed=self.config.seed)
        
        # Split into train/validation with shuffling
        dataset_dict = dataset.train_test_split(
            test_size=self.config.validation_split,
            seed=self.config.seed,
            shuffle=True
        )
        
        # Create tokenization function
        tokenize = lambda examples: tokenize_function(
            examples, 
            self.tokenizer, 
            self.config.max_length
        )
        
        # Process and shuffle both splits
        train_dataset = dataset_dict["train"].map(
            tokenize,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        ).shuffle(seed=self.config.seed)
        
        val_dataset = dataset_dict["test"].map(
            tokenize,
            batched=True,
            remove_columns=dataset_dict["test"].column_names
        ).shuffle(seed=self.config.seed)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics using metrics manager."""
        return self.metrics.compute_metrics(eval_pred)
    
    def train(self, data_path: Path) -> Dict[str, float]:
        """Execute the complete training process."""
        try:
            # Initialize metrics logger with config
            metrics_logger = MetricsLogger(
                experiment_name=f"training_{data_path.stem}",
                training_config=self.config.__dict__,  # Log all training hyperparameters
            )
            
            self.check_gpu()
            FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            self.load_model_and_tokenizer()
            train_dataset, val_dataset = self.prepare_datasets(data_path)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=str(FINETUNED_MODEL_DIR),
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=4,  # Small eval batch size
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_epochs,
                warmup_steps=self.config.warmup_steps,
                bf16=self.config.use_bf16,
                evaluation_strategy="steps",  # Light evaluation during training
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",  # Only track loss
                report_to=["tensorboard"],  # Only use tensorboard
                seed=self.config.seed,
                dataloader_num_workers=self.config.dataloader_num_workers,
                dataloader_pin_memory=self.config.pin_memory
            )

            # Initialize trainer with minimal metrics
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, 
                    mlm=False
                )
                # No compute_metrics for lighter evaluation
            )

            # Run training with basic evaluation
            self.logger.info("Starting training...")
            train_result = trainer.train()
            
            # Log final metrics
            metrics_logger.log_metrics(train_result.metrics, "training")
            metrics_logger.close()  # Clean up logging resources
            
            # Save the trained model
            self.logger.info("Saving trained model...")
            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(BEST_MODEL_DIR)
            self.tokenizer.save_pretrained(BEST_MODEL_DIR)
            
            return train_result.metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Training failed: {str(e)}")


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = Db2Trainer(config)
    trainer.train(Path("src/data/processed/SQL0000-0999_first_turn_conversations.jsonl"))
