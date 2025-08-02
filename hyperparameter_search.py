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
from model.utils import build_model  # centralised model builder


@dataclass
class TrialResult:
    """Results from a single hyperparameter trial."""
    trial_id: str
    model_name: str
    parameters: Dict[str, Any]
    status: str  # 'completed', 'failed'
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
    error_message: Optional[str] = None  # For failed trials


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
        """Save a summary of all trials."""
        summary = {
            "total_trials": len(self.trials),
            "successful_trials": len([t for t in self.trials if t.get('status') == 'completed']),
            "failed_trials": len([t for t in self.trials if t.get('status') == 'failed']),
            "trials": self.trials
        }
        
        # Add failed trial details for debugging
        failed_trials = [t for t in self.trials if t.get('status') == 'failed']
        if failed_trials:
            summary["failed_trial_details"] = [
                {
                    "trial_id": t.get('trial_id'),
                    "model_name": t.get('model_name'),
                    "parameters": t.get('parameters'),
                    "error_message": t.get('error_message', 'Unknown error')
                }
                for t in failed_trials
            ]
        
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print(f"\nSearch Summary:")
        print(f"  Total trials: {summary['total_trials']}")
        print(f"  Successful: {summary['successful_trials']}")
        print(f"  Failed: {summary['failed_trials']}")
        if summary['failed_trials'] > 0:
            print(f"  Failed trials:")
            for failed in summary.get('failed_trial_details', []):
                print(f"    - {failed['trial_id']} ({failed['model_name']}): {failed['error_message']}")
    
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
            status='completed',
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
                 config: Dict[str, Any],
                 output_dir: str,
                 patience: int = 5,
                 min_epochs: int = 3,
                 seed: int = 42):
        """
        Initialize hyperparameter search from YAML config.
        
        Args:
            config: Complete configuration including search_space section
            output_dir: Directory to save results
            patience: Early stopping patience
            min_epochs: Minimum epochs before early stopping
            seed: Random seed for reproducibility
        """
        # Extract search space from config
        search_space_config = config.get("search_space")
        if not search_space_config:
            raise ValueError(
                "Config file must contain a 'search_space' section for hyperparameter search.\n"
                "See example configs/searches/ for reference."
            )
        
        # Extract models and parameters
        self.models = search_space_config.get("models", [config.get("model_name")])
        if not self.models or self.models == [None]:
            raise ValueError("No models specified in search_space.models or model_name")
        
        # Parse parameter sweep from config
        param_config = search_space_config.get("params", {})
        self.param_sweep = self._parse_param_config(param_config)
        
        # Store configuration
        self.config = config
        self.output_dir = output_dir
        self.patience = patience
        self.min_epochs = min_epochs
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize components
        self.search_space = SearchSpace(self.param_sweep)
        self.logger = ResultsLogger(output_dir)
        
        # Save search configuration
        search_config = {
            "models": self.models,
            "param_sweep": self.param_sweep,
            "search_space_config": search_space_config,
            "patience": patience,
            "min_epochs": min_epochs,
            "seed": seed,
            "total_trials": len(self.models) * len(self.search_space)
        }
        self.logger.save_search_config(search_config)
    
    def _parse_param_config(self, param_config: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Parse parameter configuration from YAML into sweep format.
        
        Supports both old flat format and new scoped format:
        
        Old format:
        params:
          model.embed_dim: [128, 256, 512]
          training.lr: {logspace: [0.0001, 0.01], steps: 5}
        
        New scoped format:
        params:
          model:
            embed_dim: [128, 256, 512]
            num_layers: [2, 4, 6]
          training:
            lr: {logspace: [0.0001, 0.01], steps: 5}
            batch_size: [16, 32, 64]
        """
        param_sweep = {}
        
        # Check if this is the new scoped format
        if any(isinstance(v, dict) and not any(key in v for key in ['values', 'range', 'logspace']) 
               for v in param_config.values()):
            # New scoped format
            for scope, scope_params in param_config.items():
                for param_name, param_def in scope_params.items():
                    full_param_name = f"{scope}.{param_name}"
                    param_sweep[full_param_name] = self._parse_param_definition(param_def)
        else:
            # Old flat format - maintain backward compatibility
            for param_name, param_def in param_config.items():
                param_sweep[param_name] = self._parse_param_definition(param_def)
        
        return param_sweep
    
    def _parse_param_definition(self, param_def: Any) -> List[Any]:
        """Parse a single parameter definition into a list of values."""
        if isinstance(param_def, dict):
            if "values" in param_def:
                # Direct value list: {values: [128, 256, 512]}
                return param_def["values"]
            elif "range" in param_def:
                # Range specification: {range: [0.0001, 0.01], steps: 5}
                start, end = param_def["range"]
                steps = param_def.get("steps", 3)
                if param_def.get("log_scale", False):
                    import numpy as np
                    return np.logspace(np.log10(start), np.log10(end), steps).tolist()
                else:
                    import numpy as np
                    return np.linspace(start, end, steps).tolist()
            elif "logspace" in param_def:
                # Logarithmic range: {logspace: [0.0001, 0.01], steps: 5}
                start, end = param_def["logspace"]
                steps = param_def.get("steps", 3)
                import numpy as np
                return np.logspace(np.log10(start), np.log10(end), steps).tolist()
            else:
                raise ValueError(f"Unknown parameter definition: {param_def}")
        elif isinstance(param_def, list):
            # Direct list: [128, 256, 512]
            return param_def
        else:
            raise ValueError(f"Invalid parameter definition: {param_def}")
    
    def run_search(self) -> None:
        """Run the complete hyperparameter search."""
        print(f"Starting hyperparameter search with {len(self.models)} models and {len(self.search_space)} parameter combinations")
        print(f"Total trials: {len(self.models) * len(self.search_space)}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)
        
        trial_counter = 1
        successful_trials = 0
        failed_trials = 0
        
        for model_name in self.models:
            print(f"\nSearching model: {model_name}")
            
            for trial_idx in range(len(self.search_space)):
                parameters = self.search_space.get_trial_params(trial_idx)
                trial_id = f"trial_{trial_counter:03d}"
                
                print(f"\nTrial {trial_counter}: {model_name} with parameters:")
                for param, value in parameters.items():
                    print(f"  {param}: {value}")
                print("  Training progress:")
                
                # Step 4: Implement resilient trial execution
                try:
                    result = self._run_trial(trial_id, model_name, parameters)
                    self.logger.log_trial(result)
                    successful_trials += 1
                    
                    print(f"  ✓ Completed in {result.duration_seconds:.1f}s")
                    print(f"  ✓ Best val_loss: {result.best_val_loss:.4f} (epoch {result.best_epoch})")
                    print(f"  ✓ Best val_accuracy: {result.best_val_accuracy:.4f}")
                    if result.early_stopped:
                        print(f"  ⚠ Early stopped: {result.early_stop_reason}")
                    else:
                        print(f"  ✓ Completed all {result.epochs_completed} epochs")
                
                except Exception as e:
                    failed_trials += 1
                    print(f"  ❌ TRIAL FAILED: {trial_id} for model {model_name}")
                    print(f"  ERROR: {e}")
                    
                    # Log the full traceback for debugging
                    import traceback
                    traceback.print_exc()
                    
                    # Create failed trial result
                    failed_result = TrialResult(
                        trial_id=trial_id,
                        model_name=model_name,
                        parameters=parameters,
                        status='failed',
                        start_time=datetime.now().isoformat(),
                        end_time=datetime.now().isoformat(),
                        duration_seconds=0.0,
                        epochs_completed=0,
                        early_stopped=True,
                        early_stop_reason=f"Trial failed: {str(e)}",
                        best_epoch=0,
                        best_val_loss=float('inf'),
                        best_val_accuracy=0.0,
                        best_val_mse=float('inf'),
                        best_val_mae=float('inf'),
                        final_train_loss=0.0,
                        final_train_accuracy=0.0,
                        final_train_mse=0.0,
                        final_train_mae=0.0,
                        final_val_loss=0.0,
                        final_val_accuracy=0.0,
                        final_val_mse=0.0,
                        final_val_mae=0.0,
                        training_history=[],
                        error_message=str(e)
                    )
                    
                    # Log the failed trial and continue to the next trial
                    self.logger.log_trial(failed_result)
                    continue  # This is the crucial part - continue to next trial
                
                trial_counter += 1
        
        # Print final summary
        total_trials = successful_trials + failed_trials
        print(f"\n" + "="*80)
        print(f"HYPERPARAMETER SEARCH COMPLETED")
        print(f"Total trials: {total_trials}")
        print(f"Successful: {successful_trials}")
        print(f"Failed: {failed_trials}")
        print(f"Success rate: {successful_trials/total_trials*100:.1f}%" if total_trials > 0 else "No trials completed")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
    
    def _run_trial(self, trial_id: str, model_name: str, parameters: Dict[str, Any]) -> TrialResult:
        """Run a single trial."""
        # Create trial instance
        trial = Trial(trial_id, model_name, parameters, self.config, self.patience, self.min_epochs)
        
        # Start with a deep copy of the original config to preserve all static parameters
        import copy
        trial_config = copy.deepcopy(self.config)
        
        # Organize swept parameters by scope (model, training, data, etc.)
        scoped_params = {}
        for param_name, param_value in parameters.items():
            if '.' in param_name:
                scope, param = param_name.split('.', 1)
                if scope not in scoped_params:
                    scoped_params[scope] = {}
                scoped_params[scope][param] = param_value
            else:
                # Handle non-scoped parameters (fallback)
                if 'model' not in scoped_params:
                    scoped_params['model'] = {}
                scoped_params['model'][param_name] = param_value
        
        # Update config sections with scoped parameters while preserving static params
        for param_scope, param_values in scoped_params.items():
            if param_scope not in trial_config:
                trial_config[param_scope] = {}
            trial_config[param_scope].update(param_values)
        
        # Ensure model_name is set for this trial
        trial_config['model_name'] = model_name
        
        # Handle learning rate alias for backward compatibility
        if 'training' in trial_config and 'learning_rate' in trial_config['training']:
            trial_config['training']['lr'] = trial_config['training'].pop('learning_rate')
        
        # Set random seed for this trial
        trial_seed = self.seed + hash(trial_id) % 10000
        torch.manual_seed(trial_seed)
        
        # Get data loaders
        train_loader = get_dataloader(trial_config, split='train')
        val_loader = get_dataloader(trial_config, split='val')
        test_loader = get_dataloader(trial_config, split='test')
        
        # Debug: Print trial configuration for debugging
        print(f"    Trial config keys: {list(trial_config.keys())}")
        print(f"    Model config: {trial_config.get('model', {})}")
        print(f"    Data config: {trial_config.get('data', {})}")
        
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
        training_config = trial_config.get('training', {})
        
        # Handle learning rate: if learning_rate is swept, it should override lr
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
        else:
            lr = training_config.get('lr', 1e-3)
        
        # Ensure lr is float (handle string values from YAML)
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                raise ValueError(f"Invalid learning rate value: {lr!r}")
        
        weight_decay = training_config.get('weight_decay', 0.0)
        
        # Create strategy for this trial
        from train import create_task_strategy
        model_info = registry.get_model_config(model_name)
        task_type = model_info['task_type']
        strategy = create_task_strategy(task_type, trial_config)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            strategy=strategy,  # Use strategy instead of task string
            device=training_config.get('device', 'cpu'),
            lr=float(lr),  # Ensure lr is float
            weight_decay=float(weight_decay),  # Ensure weight_decay is float
            epochs=int(training_config.get('epochs', 10)),  # Ensure epochs is int
            grad_clip=float(training_config.get('grad_clip', 1.0)),  # Ensure grad_clip is float
            log_interval=int(training_config.get('log_interval', 100)),  # Ensure log_interval is int
            warmup_epochs=int(training_config.get('warmup_epochs', 2)),  # Ensure warmup_epochs is int
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
        """Build model with validation of required parameters."""
        # Get model info from registry
        model_info = registry.get_model_config(model_name)
        required_params = model_info.get('required_params', [])
        optional_params = model_info.get('optional_params', [])
        defaults = model_info.get('defaults', {})
        
        # Validate that all required parameters are present
        model_config = config.get('model', {})
        missing_params = []
        for param in required_params:
            if param not in model_config:
                missing_params.append(param)
        
        if missing_params:
            error_msg = f"Missing required parameters for model '{model_name}': {missing_params}\n"
            error_msg += f"Required parameters: {required_params}\n"
            error_msg += f"Optional parameters: {optional_params}\n"
            error_msg += f"Available model config keys: {list(model_config.keys())}\n"
            error_msg += f"Full config keys: {list(config.keys())}\n"
            error_msg += f"Model config contents: {model_config}\n"
            error_msg += f"Data config contents: {config.get('data', {})}"
            raise ValueError(error_msg)
        
        # Add defaults for missing optional parameters
        for param, default_value in defaults.items():
            if param not in model_config:
                model_config[param] = default_value
        
        # Build the model using the centralized builder
        # Pass model_cfg and data_cfg separately as expected by build_model
        return build_model(model_name, model_config, config.get('data', {}))



def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update config with command line arguments using generic override system."""
    
    # Handle direct convenience arguments
    convenience_mappings = {
        'output_dir': 'output_dir',
        'device': 'training.device',
        'seed': 'seed'
    }
    
    for arg_name, config_path in convenience_mappings.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            _set_nested_config_value(config, config_path, arg_value)
    
    # Handle generic --set overrides
    for override in getattr(args, 'set', []):
        try:
            key, value = override.split('=', 1)
        except ValueError:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'")
        
        # Smart type casting
        typed_value = _cast_config_value(value)
        _set_nested_config_value(config, key, typed_value)
    
    return config


def _set_nested_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key_path.split('.')
    d = config
    
    # Navigate to nested dictionary, creating as needed
    for k in keys[:-1]:
        if k not in d or not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    
    # Set the final value
    d[keys[-1]] = value


def _cast_config_value(value: str) -> Any:
    """Intelligently cast string values to appropriate types."""
    # Strip whitespace
    value = value.strip()
    
    # Handle boolean strings
    if value.lower() in ('true', 'yes', 'on', '1'):
        return True
    elif value.lower() in ('false', 'no', 'off', '0'):
        return False
    
    # Handle None/null
    if value.lower() in ('none', 'null', ''):
        return None
    
    # Handle numbers
    try:
        # Scientific notation or decimal
        if 'e' in value.lower() or '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Handle lists (comma-separated)
    if ',' in value:
        items = [_cast_config_value(item.strip()) for item in value.split(',')]
        return items
    
    # Keep as string
    return value


def run_search(config: Dict[str, Any], output_dir: str = None, device: str = "auto", seed: int = None) -> Dict[str, Any]:
    """
    Run hyperparameter search with the given configuration.
    
    Args:
        config: Configuration dictionary with 'search_space' section
        output_dir: Output directory for results (overrides config)
        device: Device to use ("auto", "cuda", "cpu")
        seed: Random seed (overrides config)
    
    Returns:
        Dictionary containing search results and summary
    """
    # Handle special arguments
    if output_dir:
        config['output_dir'] = output_dir
    if seed:
        config['seed'] = seed
    
    # Set defaults
    output_dir = config.get('output_dir', './search_results')
    
    # Handle Kaggle environment
    if os.path.exists('/kaggle/working') and not os.access(output_dir, os.W_OK):
        output_dir = '/kaggle/working/search_results'
        print(f"Warning: Using Kaggle working directory: {output_dir}")
    
    print(f"🔧 Hyperparameter Search Configuration:")
    print(f"   Output directory: {output_dir}")
    
    # Device handling
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if 'training' not in config:
        config['training'] = {}
    config['training']['device'] = device
    
    print(f"   Device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        try:
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception as e:
            print(f"   GPU info unavailable: {e}")
    
    # Create and run search
    search = HyperparameterSearch(
        config=config,
        output_dir=output_dir,
        patience=config.get('patience', 5),
        min_epochs=config.get('min_epochs', 3),
        seed=config.get('seed', 42)
    )
    
    # Run search and return results
    search.run_search()
    
    # Return summary information
    return {
        'output_dir': output_dir,
        'total_trials': len(search.models) * len(search.search_space),
        'models': search.models,
        'param_sweep': search.param_sweep
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for TFN models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file with 'search_space' section")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (overrides YAML)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: cuda | cpu | auto (overrides YAML)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (overrides YAML)")
    
    # Generic override system (same as train.py)
    parser.add_argument("--set", nargs='+', default=[],
                       help="Override config parameters with key=value pairs")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides using same logic as train.py
    config = update_config_with_args(config, args)
    
    # Run search
    run_search(
        config=config,
        output_dir=args.output_dir,
        device=args.device or "auto",
        seed=args.seed
    )


if __name__ == "__main__":
    main() 