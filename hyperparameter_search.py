"""
Hyperparameter Search System for Token Field Networks

A comprehensive hyperparameter optimization system that can sweep through
different models (TFN, Enhanced TFN) and parameters with early stopping
and detailed result logging.

Usage:
    python hyperparameter_search.py \
        --models tfn_classifier enhanced_tfn_classifier \
        --param_sweep embed_dim:128,256,512 num_layers:2,4,6 kernel_type:rbf,compact \
        --epochs 20 --patience 5 \
        --output_dir ./search_results
"""

import argparse
import json
import os
import time
import torch
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import itertools
import random

from data_pipeline import get_dataloader
from model import registry
from src.trainer import Trainer
from src import metrics


@dataclass
class TrialResult:
    """Results from a single hyperparameter trial."""
    trial_id: str
    model_name: str
    parameters: Dict[str, Any]
    start_time: str
    end_time: str
    duration_seconds: float
    epochs_completed: int
    early_stopped: bool
    early_stop_reason: Optional[str]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float
    best_val_mse: float
    best_val_mae: float
    final_train_loss: float
    final_train_accuracy: float
    final_train_mse: float
    final_train_mae: float
    final_val_loss: float
    final_val_accuracy: float
    final_val_mse: float
    final_val_mae: float
    training_history: List[Dict[str, Any]]
    flops_stats: Optional[Dict[str, Any]] = None


class SearchSpace:
    """Defines the hyperparameter search space."""
    
    def __init__(self, param_sweep: Dict[str, List[Any]]):
        """
        Initialize search space.
        
        Args:
            param_sweep: Dictionary mapping parameter names to lists of values
        """
        self.param_sweep = param_sweep
        self.param_combinations = list(itertools.product(*param_sweep.values()))
        self.param_names = list(param_sweep.keys())
    
    def __len__(self):
        return len(self.param_combinations)
    
    def get_trial_params(self, trial_idx: int) -> Dict[str, Any]:
        """Get parameters for a specific trial."""
        if trial_idx >= len(self.param_combinations):
            raise IndexError(f"Trial index {trial_idx} out of range")
        
        param_values = self.param_combinations[trial_idx]
        return dict(zip(self.param_names, param_values))


class ResultsLogger:
    """Handles logging of trial results to JSON files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.trials_dir = os.path.join(output_dir, "trials")
        
        # Handle Kaggle environment - ensure we can write to the directory
        try:
            os.makedirs(self.trials_dir, exist_ok=True)
        except OSError as e:
            if "Read-only file system" in str(e):
                # If we can't write to the specified directory, try Kaggle working directory
                if os.path.exists('/kaggle/working'):
                    self.output_dir = '/kaggle/working/search_results'
                    self.trials_dir = os.path.join(self.output_dir, "trials")
                    os.makedirs(self.trials_dir, exist_ok=True)
                    print(f"Warning: Using Kaggle working directory: {self.output_dir}")
                else:
                    raise e
            else:
                raise e
        
        # Initialize summary
        self.summary = {
            "search_start_time": datetime.now().isoformat(),
            "total_trials": 0,
            "completed_trials": 0,
            "best_trial": None,
            "trial_summaries": []
        }
    
    def log_trial(self, result: TrialResult) -> None:
        """Log a single trial result."""
        # Save individual trial result
        trial_file = os.path.join(self.trials_dir, f"{result.trial_id}.json")
        with open(trial_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Update summary
        self.summary["total_trials"] += 1
        self.summary["completed_trials"] += 1
        
        trial_summary = {
            "trial_id": result.trial_id,
            "model_name": result.model_name,
            "best_val_loss": result.best_val_loss,
            "best_val_accuracy": result.best_val_accuracy,
            "epochs_completed": result.epochs_completed,
            "early_stopped": result.early_stopped,
            "duration_seconds": result.duration_seconds
        }
        self.summary["trial_summaries"].append(trial_summary)
        
        # Update best trial
        if (self.summary["best_trial"] is None or 
            result.best_val_loss < self.summary["best_trial"]["best_val_loss"]):
            self.summary["best_trial"] = trial_summary
    
    def save_summary(self) -> None:
        """Save the search summary."""
        self.summary["search_end_time"] = datetime.now().isoformat()
        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2, default=str)
    
    def save_search_config(self, config: Dict[str, Any]) -> None:
        """Save the search configuration."""
        config_file = os.path.join(self.output_dir, "search_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)


class Trial:
    """Individual hyperparameter trial with early stopping."""
    
    def __init__(self, 
                 trial_id: str,
                 model_name: str,
                 parameters: Dict[str, Any],
                 config: Dict[str, Any],
                 patience: int = 5,
                 min_epochs: int = 3):
        self.trial_id = trial_id
        self.model_name = model_name
        self.parameters = parameters
        self.config = config
        self.patience = patience
        self.min_epochs = min_epochs
        self.start_time = datetime.now()
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopped = False
        self.early_stop_reason = None
        
        # Training history
        self.training_history = []
        self.flops_stats = None
    
    def should_stop(self, epoch: int, val_loss: float) -> bool:
        """Check if training should stop early."""
        # Always track the best loss, regardless of min_epochs
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Only allow early stopping after min_epochs
        if epoch < self.min_epochs:
            return False
        
        if self.patience_counter >= self.patience:
            self.early_stopped = True
            self.early_stop_reason = f"Validation loss stagnated for {self.patience} epochs"
            return True
        
        return False
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  train_mse: float, train_mae: float, val_loss: float, val_acc: float,
                  val_mse: float, val_mae: float, learning_rate: float = None) -> None:
        """Log metrics for current epoch."""
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "learning_rate": learning_rate
        }
        self.training_history.append(epoch_data)
    
    def get_result(self) -> TrialResult:
        """Get the final trial result."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Get final metrics from last epoch
        final_metrics = self.training_history[-1] if self.training_history else {}
        
        # Get best epoch metrics (best_epoch is 1-indexed, but list is 0-indexed)
        best_metrics = {}
        if self.best_epoch > 0 and self.best_epoch <= len(self.training_history):
            best_metrics = self.training_history[self.best_epoch - 1]
        elif self.training_history:
            # Fallback to first epoch if best_epoch is invalid
            best_metrics = self.training_history[0]
        
        return TrialResult(
            trial_id=self.trial_id,
            model_name=self.model_name,
            parameters=self.parameters,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            epochs_completed=len(self.training_history),
            early_stopped=self.early_stopped,
            early_stop_reason=self.early_stop_reason,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            best_val_accuracy=best_metrics.get("val_accuracy", 0.0),
            best_val_mse=best_metrics.get("val_mse", 0.0),
            best_val_mae=best_metrics.get("val_mae", 0.0),
            final_train_loss=final_metrics.get("train_loss", 0.0),
            final_train_accuracy=final_metrics.get("train_accuracy", 0.0),
            final_train_mse=final_metrics.get("train_mse", 0.0),
            final_train_mae=final_metrics.get("train_mae", 0.0),
            final_val_loss=final_metrics.get("val_loss", 0.0),
            final_val_accuracy=final_metrics.get("val_accuracy", 0.0),
            final_val_mse=final_metrics.get("val_mse", 0.0),
            final_val_mae=final_metrics.get("val_mae", 0.0),
            training_history=self.training_history,
            flops_stats=self.flops_stats
        )


class HyperparameterSearch:
    """Main hyperparameter search orchestrator."""
    
    def __init__(self, 
                 models: List[str],
                 param_sweep: Dict[str, List[Any]],
                 config: Dict[str, Any],
                 output_dir: str,
                 patience: int = 5,
                 min_epochs: int = 3,
                 seed: int = 42):
        """
        Initialize hyperparameter search.
        
        Args:
            models: List of model names to search
            param_sweep: Dictionary mapping parameter names to lists of values
            config: Base configuration for training
            output_dir: Directory to save results
            patience: Early stopping patience
            min_epochs: Minimum epochs before early stopping
            seed: Random seed for reproducibility
        """
        self.models = models
        self.param_sweep = param_sweep
        self.config = config
        self.output_dir = output_dir
        self.patience = patience
        self.min_epochs = min_epochs
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize components
        self.search_space = SearchSpace(param_sweep)
        self.logger = ResultsLogger(output_dir)
        
        # Save search configuration
        search_config = {
            "models": models,
            "param_sweep": param_sweep,
            "patience": patience,
            "min_epochs": min_epochs,
            "seed": seed,
            "total_trials": len(models) * len(self.search_space)
        }
        self.logger.save_search_config(search_config)
    
    def run_search(self) -> None:
        """Run the complete hyperparameter search."""
        print(f"Starting hyperparameter search with {len(self.models)} models and {len(self.search_space)} parameter combinations")
        print(f"Total trials: {len(self.models) * len(self.search_space)}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)
        
        trial_counter = 1
        
        for model_name in self.models:
            print(f"\nSearching model: {model_name}")
            
            for trial_idx in range(len(self.search_space)):
                parameters = self.search_space.get_trial_params(trial_idx)
                trial_id = f"trial_{trial_counter:03d}"
                
                print(f"\nTrial {trial_counter}: {model_name} with parameters:")
                for param, value in parameters.items():
                    print(f"  {param}: {value}")
                print("  Training progress:")
                
                try:
                    result = self._run_trial(trial_id, model_name, parameters)
                    self.logger.log_trial(result)
                    
                    print(f"  ✓ Completed in {result.duration_seconds:.1f}s")
                    print(f"  ✓ Best val_loss: {result.best_val_loss:.4f} (epoch {result.best_epoch})")
                    print(f"  ✓ Best val_accuracy: {result.best_val_accuracy:.4f}")
                    if result.early_stopped:
                        print(f"  ⚠ Early stopped: {result.early_stop_reason}")
                    else:
                        print(f"  ✓ Completed all {result.epochs_completed} epochs")
                
                except Exception as e:
                    print(f"  Trial failed: {str(e)}")
                    # Log failed trial
                    failed_result = TrialResult(
                        trial_id=trial_id,
                        model_name=model_name,
                        parameters=parameters,
                        start_time=datetime.now().isoformat(),
                        end_time=datetime.now().isoformat(),
                        duration_seconds=0.0,
                        epochs_completed=0,
                        early_stopped=True,
                        early_stop_reason=f"Trial failed: {str(e)}",
                        best_epoch=0,
                        best_val_loss=float('inf'),
                        best_val_accuracy=0.0,
                        final_train_loss=0.0,
                        final_train_accuracy=0.0,
                        final_val_loss=0.0,
                        final_val_accuracy=0.0,
                        training_history=[]
                    )
                    self.logger.log_trial(failed_result)
                
                trial_counter += 1
        
        self.logger.save_summary()
        print(f"\nSearch completed! Results saved to {self.output_dir}")
    
    def _run_trial(self, trial_id: str, model_name: str, parameters: Dict[str, Any]) -> TrialResult:
        """Run a single trial."""
        # Create trial instance
        trial = Trial(trial_id, model_name, parameters, self.config, self.patience, self.min_epochs)
        
        # Update config with trial parameters
        trial_config = self.config.copy()
        
        # Update model parameters in the model section
        if 'model' not in trial_config:
            trial_config['model'] = {}
        trial_config['model'].update(parameters)
        trial_config['model_name'] = model_name
        
        # Set random seed for this trial
        trial_seed = self.seed + hash(trial_id) % 10000
        torch.manual_seed(trial_seed)
        
        # Get data loaders
        train_loader = get_dataloader(trial_config, split='train')
        val_loader = get_dataloader(trial_config, split='val')
        test_loader = get_dataloader(trial_config, split='test')
        
        # Build model
        model = self._build_model(model_name, trial_config)
        
        # Print model parameter count and debug info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Model: {model_name} | Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
        print(f"    Model class: {type(model).__name__}")
        print(f"    Model module: {model.__class__.__module__}")
        
        # Print first few layers for debugging
        print(f"    Model layers:")
        for i, (name, module) in enumerate(model.named_modules()):
            if i < 5:  # Show first 5 layers
                print(f"      {name}: {type(module).__name__}")
            else:
                break
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            task=trial_config.get('task', 'classification'),
            device=trial_config.get('device', 'cpu'),
            lr=trial_config.get('learning_rate', 1e-3),
            weight_decay=trial_config.get('weight_decay', 0.0),
            epochs=trial_config.get('epochs', 10),
            grad_clip=trial_config.get('grad_clip', 1.0),
            log_interval=trial_config.get('log_interval', 100),
            warmup_epochs=trial_config.get('warmup_epochs', 2),
            track_flops=True  # Enable FLOPs tracking for hyperparameter search
        )
        
        # Custom training loop with early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        print(f"    Starting training for {trainer.epochs} epochs...")
        
        for epoch in range(1, trainer.epochs + 1):
            # Train epoch
            train_metrics = trainer._run_epoch(trainer.train_loader, train=True)
            
            # Validate epoch
            val_metrics = trainer._run_epoch(trainer.val_loader, train=False)
            
            # Extract metrics
            train_loss = train_metrics[0]  # avg_loss
            train_acc = train_metrics[1] if train_metrics[1] is not None else 0.0  # avg_acc
            train_mse = train_metrics[2] if train_metrics[2] is not None else 0.0  # avg_mse
            train_mae = train_metrics[3] if train_metrics[3] is not None else 0.0  # avg_mae
            val_loss = val_metrics[0]  # avg_loss
            val_acc = val_metrics[1] if val_metrics[1] is not None else 0.0  # avg_acc
            val_mse = val_metrics[2] if val_metrics[2] is not None else 0.0  # avg_mse
            val_mae = val_metrics[3] if val_metrics[3] is not None else 0.0  # avg_mae
            
            # Determine task type for appropriate metric printing
            task = trial_config.get('task', 'classification')
            
            if task in ('regression', 'time_series'):
                # Print regression metrics (MSE, MAE)
                print(f"    Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_mse={train_mse:.4f}, train_mae={train_mae:.4f}, "
                      f"val_loss={val_loss:.4f}, val_mse={val_mse:.4f}, val_mae={val_mae:.4f}", flush=True)
            else:
                # Print classification metrics (accuracy)
                print(f"    Epoch {epoch:2d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}", flush=True)
            
            # Log epoch
            trial.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                train_mse=train_mse,
                train_mae=train_mae,
                val_loss=val_loss,
                val_acc=val_acc,
                val_mse=val_mse,
                val_mae=val_mae,
                learning_rate=trainer.optimizer.param_groups[0]['lr']
            )
            
            # Check early stopping
            if trial.should_stop(epoch, val_loss):
                print(f"    Early stopping at epoch {epoch}")
                break
        
        # Get FLOPS stats if tracking was enabled
        flops_stats = None
        if hasattr(trainer, 'flops_tracker') and trainer.flops_tracker:
            flops_stats = trainer.flops_tracker.get_flops_stats()
            print(f"    FLOPS Stats:")
            print(f"      Total FLOPS: {flops_stats['total_flops']:,}")
            print(f"      Avg FLOPS per pass: {flops_stats['avg_flops_per_pass']:,.0f}")
            print(f"      FLOPS per second: {flops_stats['flops_per_second']:,.0f}")
        
        # Update trial with FLOPS stats
        trial.flops_stats = flops_stats
        
        return trial.get_result()
    
    def _build_model(self, model_name: str, config: Dict[str, Any]) -> torch.nn.Module:
        """Build model from registry using the same logic as train.py."""
        model_info = registry.get_model_config(model_name)
        model_cls = model_info['class']
        task_type = model_info['task_type']
        model_args = dict(model_info.get('defaults', {}))
        
        # Get model config from the config dictionary
        model_cfg = config.get('model', {})
        model_args.update(model_cfg)
        
        # Add data config parameters that might be needed by the model
        data_cfg = config.get('data')
        if data_cfg is not None:
            # Pass output_len from data config to model if needed
            if 'output_len' in data_cfg and 'output_len' in model_info.get('required_params', []):
                model_args['output_len'] = data_cfg['output_len']
        
        if 'task' in model_cls.__init__.__code__.co_varnames:
            model_args['task'] = task_type
        allowed = set(model_info.get('required_params', []) + model_info.get('optional_params', []))
        filtered_args = {k: v for k, v in model_args.items() if k in allowed or k == 'task'}
        
        # Debug: Print what we're passing to the model
        print(f"    Debug - Model class: {model_cls.__name__}")
        print(f"    Debug - Model params: {filtered_args}")
        
        return model_cls(**filtered_args)


def parse_param_sweep(param_sweep_str: str) -> Dict[str, List[Any]]:
    """Parse parameter sweep string into dictionary."""
    param_sweep = {}
    
    for param_str in param_sweep_str.split():
        if ':' not in param_str:
            continue
        
        # Split on the last colon to handle nested parameter names
        parts = param_str.split(':')
        if len(parts) < 2:
            continue
            
        # Handle nested parameter names (e.g., "evolution:type" -> "evolution_type")
        if len(parts) > 2:
            param_name = '_'.join(parts[:-1])  # Join all parts except the last as parameter name
            values_str = parts[-1]  # Last part contains the values
        else:
            param_name, values_str = parts
        
        values = []
        
        for value_str in values_str.split(','):
            value_str = value_str.strip()
            
            # Try to convert to appropriate type
            try:
                # Handle scientific notation (e.g., 1e-3)
                if 'e' in value_str.lower():
                    values.append(float(value_str))
                elif '.' in value_str:
                    values.append(float(value_str))
                else:
                    values.append(int(value_str))
            except ValueError:
                # Keep as string
                values.append(value_str)
        
        param_sweep[param_name] = values
    
    return param_sweep


def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update config with command line arguments using the same logic as train.py."""
    # Special mappings for CLI arguments that don't follow dot notation
    special_mappings = {
        "learning_rate": "training.lr",
        "batch_size": "training.batch_size", 
        "epochs": "training.epochs",
        "weight_decay": "training.weight_decay",
        "optimizer": "training.optimizer",
        "warmup_epochs": "training.warmup_epochs",
    }
    
    for arg_key, arg_val in vars(args).items():
        if arg_key == "config" or arg_val is None:
            continue
            
        # Handle special mappings
        if arg_key in special_mappings:
            config_key = special_mappings[arg_key]
        else:
            config_key = arg_key
            
        keys = config_key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = arg_val
    return config


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for TFN models")
    parser.add_argument("--config", type=str, default="configs/tests/synthetic_copy_test.yaml", 
                       help="Path to YAML config file")
    parser.add_argument("--models", nargs='+', 
                       default=["tfn_classifier", "enhanced_tfn_classifier"],
                       help="Models to search")
    parser.add_argument("--param_sweep", type=str, 
                       default="embed_dim:128,256 num_layers:2,4 kernel_type:rbf,compact learning_rate:1e-3,1e-4",
                       help="Parameter sweep specification")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    parser.add_argument("--min_epochs", type=int, default=3,
                       help="Minimum epochs before early stopping")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Maximum training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                       help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--track_flops", action="store_true",
                       help="Enable FLOPS tracking during training")
    
    args = parser.parse_args()
    
    # Handle Kaggle environment - use writable directory
    if args.output_dir is None:
        # Check if we're in Kaggle environment
        if os.path.exists('/kaggle/working'):
            args.output_dir = '/kaggle/working/search_results'
        else:
            args.output_dir = './search_results'
    
    # Load base config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with CLI args
    config = update_config_with_args(config, args)
    
    # Parse parameter sweep
    param_sweep = parse_param_sweep(args.param_sweep)
    
    # Create and run search
    search = HyperparameterSearch(
        models=args.models,
        param_sweep=param_sweep,
        config=config,
        output_dir=args.output_dir,
        patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed
    )
    
    search.run_search()


if __name__ == "__main__":
    main() 