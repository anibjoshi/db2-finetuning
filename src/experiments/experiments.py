from dataclasses import dataclass
from typing import Dict, Any, Optional
import optuna
from pathlib import Path
import logging
import json
from datetime import datetime
from config import PROCESSED_DATA_DIR

from training.training_manager import TrainingManager
from training.training_config import TrainingConfig
from experiments.experiment_config import ExperimentConfig

class ExperimentManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path("src/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ExperimentManager")
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Create config with trial suggestions
        config = TrainingConfig(
            lora_r=trial.suggest_categorical("lora_r", self.config.lora_r_options),
            lora_alpha=trial.suggest_categorical("lora_alpha", self.config.lora_alpha_options),
            lora_dropout=trial.suggest_categorical("lora_dropout", self.config.lora_dropout_options),
            learning_rate=trial.suggest_loguniform("learning_rate", *self.config.learning_rate_range),
            batch_size=trial.suggest_categorical("batch_size", self.config.batch_size_options)
        )
        
        # Log trial parameters
        self.logger.info(f"\nTrial {trial.number} parameters:")
        self.logger.info(f"  lora_r: {config.lora_r}")
        self.logger.info(f"  lora_alpha: {config.lora_alpha}")
        self.logger.info(f"  lora_dropout: {config.lora_dropout}")
        self.logger.info(f"  learning_rate: {config.learning_rate}")
        self.logger.info(f"  batch_size: {config.batch_size}\n")
        
        try:
            trainer = TrainingManager(config)
            result = trainer.train(PROCESSED_DATA_DIR)
            return result[self.config.metric]
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run the hyperparameter optimization study."""
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
        )
        
        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            callbacks=[self._optimization_callback]
        )
        
        # Save final results
        self._save_study_results(study)
        
        return study.best_params
    
    def _log_trial(self, trial_num: int, params: Dict[str, Any], result: Dict[str, Any]):
        """Log individual trial results."""
        trial_data = {
            "trial_number": trial_num,
            "parameters": params,
            "metrics": result,
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = self.results_dir / f"trial_{trial_num}.json"
        with open(output_file, "w") as f:
            json.dump(trial_data, f, indent=2)
    
    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback to track optimization progress."""
        logging.info(f"Trial {trial.number} finished with value: {trial.value}")
        logging.info(f"Current best value: {study.best_value}")
    
    def _save_study_results(self, study: optuna.Study):
        """Save final study results and best parameters."""
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "timestamp": datetime.now().isoformat()
        }
        
        output_file = self.results_dir / "final_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2) 