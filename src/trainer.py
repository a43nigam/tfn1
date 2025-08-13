from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from src import metrics
from src.flops_tracker import create_flops_tracker, log_flops_stats
from src.task_strategies import TaskStrategy
import math
import os
import json
from datetime import datetime

# Optional wandb import for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

class Trainer:
    """
    Universal Trainer using the Strategy design pattern for task-specific logic.
    Handles device, optimizer, scheduler, loss, metrics, and robust batch formats.
    Enhanced with comprehensive experiment tracking.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        strategy: TaskStrategy = None,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 10,
        grad_clip: Optional[float] = 1.0,
        log_interval: int = 100,
        warmup_epochs: int = 1,
        track_flops: bool = False,
        # Experiment tracking parameters
        experiment_name: Optional[str] = None,
        project_name: str = "tfn-experiments",
        use_wandb: bool = False,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_hyperparams: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.strategy = strategy
        self.device = torch.device(device)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.track_flops = track_flops
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.project_name = project_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.log_hyperparams = log_hyperparams
        
        # Create checkpoint directory with error handling
        if self.save_checkpoints:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                print(f"✅ Checkpoint directory created/verified: {self.checkpoint_dir}")
            except PermissionError:
                print(f"❌ Permission denied creating checkpoint directory: {self.checkpoint_dir}")
                print("   Consider setting checkpoint_dir to a writable location in your config")
                raise
            except Exception as e:
                print(f"❌ Error creating checkpoint directory {self.checkpoint_dir}: {e}")
                raise
        
        self.history = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], 
            "train_mse": [], "val_mse": [], "train_mae": [], "val_mae": [], 
            "train_relative_l2": [], "val_relative_l2": [],  # PDE-specific metrics
            "learning_rates": [], "epochs": []
        }

        # Create FLOPS tracker if enabled
        if self.track_flops:
            self.flops_tracker = create_flops_tracker(self.model)
            self.model = self.flops_tracker
        else:
            self.flops_tracker = None

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = self._build_scheduler(lr, warmup_epochs)
        
        # The strategy provides the loss function
        self.criterion = self.strategy.get_criterion() if self.strategy else torch.nn.MSELoss()
        
        # Store hyperparameters for logging (MUST be before wandb initialization)
        self.hyperparams = {
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'grad_clip': grad_clip,
            'warmup_epochs': warmup_epochs,
            'track_flops': track_flops,
            'device': str(device),
            'model_name': model.__class__.__name__,
            'num_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        # Add model-specific hyperparameters
        if hasattr(model, 'embed_dim'):
            self.hyperparams['embed_dim'] = model.embed_dim
        if hasattr(model, 'num_layers'):
            self.hyperparams['num_layers'] = model.num_layers
        if hasattr(model, 'kernel_type'):
            self.hyperparams['kernel_type'] = model.kernel_type
        if hasattr(model, 'evolution_type'):
            self.hyperparams['evolution_type'] = model.evolution_type
        
        # Initialize wandb AFTER hyperparameters are ready
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases experiment tracking."""
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available. Skipping wandb initialization.")
            return
        
        # Defensive check to ensure hyperparams are available
        if not hasattr(self, 'hyperparams'):
            print("Warning: hyperparams not initialized. Creating default hyperparams for wandb.")
            self.hyperparams = {
                'lr': getattr(self, 'lr', 1e-3),
                'weight_decay': getattr(self, 'weight_decay', 0.0),
                'epochs': getattr(self, 'epochs', 10),
                'device': str(getattr(self, 'device', 'cpu')),
                'model_name': getattr(self.model, '__class__.__name__', 'Unknown'),
            }
            
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.hyperparams,
            tags=["tfn", "deep-learning", "field-networks"]
        )
        
        # Define custom x-axis for epoch-level metrics
        wandb.define_metric("epoch/epoch")
        wandb.define_metric("epoch/*", step_metric="epoch/epoch")
        
        # Log model architecture
        if hasattr(self.model, 'named_modules'):
            model_summary = []
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    model_summary.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'parameters': sum(p.numel() for p in module.parameters())
                    })
            wandb.log({"model_architecture": wandb.Table(
                columns=["name", "type", "parameters"],
                data=[[m['name'], m['type'], m['parameters']] for m in model_summary]
            )})

    def _log_metrics(self, epoch: int, metrics: Dict[str, Optional[float]], step: str = "epoch"):
        """Log metrics to wandb and console."""
        # Add epoch to metrics
        metrics['epoch'] = epoch
        
        # Log to wandb - filter out None values
        if self.use_wandb:
            # Prefix metrics with "epoch/" to link them to the custom step
            wandb_metrics = {f"epoch/{k}": v for k, v in metrics.items() if v is not None}
            # Add the epoch number itself to the log
            wandb_metrics["epoch/epoch"] = epoch
            wandb.log(wandb_metrics)
        
        # Log to console - handle None values properly
        metric_parts = []
        for k, v in metrics.items():
            if k != 'epoch':
                if v is None:
                    metric_parts.append(f"{k}: None")
                else:
                    metric_parts.append(f"{k}: {v:.2e}")
        metric_str = " | ".join(metric_parts)
        print(f"Epoch {epoch:02d} | {step.capitalize()} | {metric_str}")

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Optional[float]], is_best: bool = False):
        """Save model checkpoint."""
        if not self.save_checkpoints:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'hyperparams': self.hyperparams,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_latest.pt")
        torch.save(checkpoint, latest_path)

    def _build_scheduler(self, lr: float, warmup_epochs: int):
        total_steps = self.epochs * len(self.train_loader)
        warmup_steps = warmup_epochs * len(self.train_loader)
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _unpack_batch(self, batch: Dict[str, Any]):
        """
        Unpack batch data using the standardized data pipeline format.
        
        All data loaders now return a standardized format:
        - For regression/copy tasks: {"inputs": ..., "targets": ...}
        - For classification tasks: {"inputs": ..., "labels": ...}
        - For Transformer language modeling: {"inputs": ..., "labels": ..., "attention_mask": ...}
        - Optional "attention_mask" for NLP tasks
        - Optional "positions" for PDE datasets
        - Optional "calendar_features" for time-based positional encoding
        
        This method assumes the standardized format and provides clear error messages.
        """
        # --- START FIX ---
        # Look for calendar features and move them to the correct device
        calendar_features = batch.get('calendar_features')
        if calendar_features is not None:
            calendar_features = {
                key: val.to(self.device) for key, val in calendar_features.items()
            }
        # --- END FIX ---

        # Standard format from the unified data pipeline
        if "inputs" in batch and "targets" in batch:
            # Regression/copy tasks
            inputs = batch['inputs'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Handle models that need an attention mask (e.g., Transformers)
            if 'attention_mask' in batch:
                model_input = (inputs, batch['attention_mask'].to(self.device))
            else:
                model_input = inputs
            
            # NEW: Check for explicit positions and pass them to the model
            positions = batch.get('positions')
            if positions is not None:
                positions = positions.to(self.device)
            
            return model_input, targets, positions, calendar_features
        
        elif "inputs" in batch and "labels" in batch:
            # Classification tasks or Transformer language modeling
            inputs = batch['inputs'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Handle models that need an attention mask (e.g., Transformers)
            if 'attention_mask' in batch:
                model_input = (inputs, batch['attention_mask'].to(self.device))
            else:
                model_input = inputs
            
            # NEW: Check for explicit positions and pass them to the model
            positions = batch.get('positions')
            if positions is not None:
                positions = positions.to(self.device)
            
            return model_input, labels, positions, calendar_features
        
        else:
            # Enhanced error message with debugging information
            available_keys = list(batch.keys())
            error_msg = f"Invalid batch format: keys {available_keys}\n"
            error_msg += "Expected standardized format:\n"
            error_msg += "  - 'inputs' and 'targets' (regression/copy tasks)\n"
            error_msg += "  - 'inputs' and 'labels' (classification/transformer tasks)\n"
            error_msg += "  - Optional 'attention_mask' for NLP tasks\n"
            error_msg += "  - Optional 'positions' for PDE datasets\n"
            error_msg += "  - Optional 'calendar_features' for time-based positional encoding\n"
            error_msg += f"Available keys: {available_keys}"
            raise ValueError(error_msg)

    def _fmt(self, val):
        return f"{val:.4f}" if val is not None else "N/A"
    
    def _print_detailed_config(self, config: Dict[str, Any]) -> None:
        """Print detailed configuration information after W&B has started."""
        
        # Model info
        model_info = config.get("model_info", {})
        print(f"\n🤖 MODEL DETAILS:")
        print(f"   Type: {config.get('model_name', 'unknown')}")
        print(f"   Task: {model_info.get('task_type', 'unknown')}")
        print(f"   Components: {', '.join(model_info.get('components', []))}")
        print(f"   Evolution Types: {', '.join(model_info.get('evolution_types', []))}")
        
        # Model hyperparameters
        model_cfg = config.get("model", {})
        print(f"\n⚙️  MODEL HYPERPARAMETERS:")
        for key, value in model_cfg.items():
            print(f"   {key}: {value}")
        
        # Training hyperparameters
        train_cfg = config.get("training", {})
        print(f"\n🎯 TRAINING HYPERPARAMETERS:")
        for key, value in train_cfg.items():
            print(f"   {key}: {value}")
        
        # Model architecture details
        print(f"\n🏗️  MODEL ARCHITECTURE:")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Device: {self.device}")
        
        # Print model structure (first few layers)
        print(f"\n📋 MODEL STRUCTURE (first 3 layers):")
        for i, (name, module) in enumerate(self.model.named_modules()):
            if i >= 3:  # Only show first 3 layers
                break
            if len(list(module.children())) == 0:  # Leaf modules only
                print(f"   {name}: {module}")
        
        print()

    def fit(self, on_epoch_end: Optional[callable] = None, full_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the model with optional epoch-end callback and full configuration logging.
        
        Args:
            on_epoch_end: Optional callback function called at the end of each epoch.
                         Signature: on_epoch_end(epoch, metrics, trainer) -> bool
                         Return True to continue training, False to stop early.
            full_config: Full configuration dictionary to print detailed information
        """
        # Print model name and parameter count at the start
        model_name = getattr(self.model, 'name', self.model.__class__.__name__)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {model_name} | Total parameters: {num_params:,}")
        
        # Log detailed configuration after W&B has started
        if full_config and self.log_hyperparams:
            self._print_detailed_config(full_config)
        
        # Log basic hyperparameters
        if self.log_hyperparams:
            print("\nHyperparameters:")
            for key, value in self.hyperparams.items():
                print(f"  {key}: {value}")
            print()
        
        # Track best validation metrics
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_mse = float('inf')
        best_val_mae = float('inf')
        best_val_relative_l2 = float('inf')  # PDE-specific metric
        
        # Track learning rates for each epoch
        learning_rates = []
        
        for epoch in range(1, self.epochs + 1):
            # Training phase
            train_loss, train_acc, train_mse, train_mae, train_relative_l2 = self._run_epoch(self.train_loader, train=True)
            
            # Validation phase
            val_loss, val_acc, val_mse, val_mae, val_relative_l2 = (None, None, None, None, None)
            if self.val_loader is not None:
                val_loss, val_acc, val_mse, val_mae, val_relative_l2 = self._run_epoch(self.val_loader, train=False)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_mse"].append(train_mse)
            self.history["val_mse"].append(val_mse)
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            self.history["train_relative_l2"].append(train_relative_l2)
            self.history["val_relative_l2"].append(val_relative_l2)
            self.history["learning_rates"].append(current_lr)
            self.history["epochs"].append(epoch)
            
            # Prepare metrics for logging
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_relative_l2': train_relative_l2,
                'val_relative_l2': val_relative_l2,
                'learning_rate': current_lr
            }
            
            # Log metrics
            self._log_metrics(epoch, metrics)
            
            # Check for best model
            is_best = False
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True
            if val_mse is not None and val_mse < best_val_mse:
                best_val_mse = val_mse
                is_best = True
            if val_mae is not None and val_mae < best_val_mae:
                best_val_mae = val_mae
                is_best = True
            if val_relative_l2 is not None and val_relative_l2 < best_val_relative_l2:
                best_val_relative_l2 = val_relative_l2
                is_best = True
            
            # Save checkpoint
            self._save_checkpoint(epoch, metrics, is_best)
            
            # Call epoch-end callback if provided
            if on_epoch_end is not None:
                should_continue = on_epoch_end(epoch, metrics, self)
                if not should_continue:
                    print(f"Training stopped early at epoch {epoch} by callback")
                    break
            
            # Step the scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
        
        # Log FLOPS stats at the end if tracking is enabled
        flops_stats = None
        if self.track_flops and self.flops_tracker:
            flops_stats = self.flops_tracker.get_flops_stats()
            print("\n" + "="*50)
            print("FLOPS STATISTICS")
            print("="*50)
            log_flops_stats(flops_stats, prefix="  ")
            self.history["flops_stats"] = flops_stats
            
            # Log FLOPS to wandb
            if self.use_wandb:
                # Prefix keys to avoid conflicts with other metrics
                wandb_flops_stats = {f"flops/{k}": v for k, v in flops_stats.items()}
                wandb.log(wandb_flops_stats)
        
        # Final logging
        if self.use_wandb:
            wandb.finish()
        
        # Save final history
        history_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history

    def _run_epoch(self, loader: DataLoader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0
        
        for batch_idx, batch in enumerate(loader, start=1):
            # DEBUG: Print batch keys to verify calendar_features survives DataLoader
            # print(f"🔍 Trainer._run_epoch: batch.keys() = {list(batch.keys())}")
            
            # --- START FIX ---
            # Unpack four items now (x, y, positions, calendar_features)
            unpacked = self._unpack_batch(batch)
            x, y, positions, calendar_features = unpacked
            # --- END FIX ---
            
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                
            with torch.set_grad_enabled(train):
                # Delegate forward pass and loss calculation to strategy
                logits, loss = self.strategy.process_forward_pass(self.model, x, y, positions=positions, calendar_features=calendar_features)
                
                if train:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
            
            # Delegate metric calculation to strategy
            batch_metrics = self.strategy.calculate_metrics(logits, y)
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            n_batches += 1
            
            # Log batch metrics to wandb if enabled
            if self.use_wandb and batch_idx % self.log_interval == 0:
                batch_metrics_log = {f"batch_{k}": v for k, v in batch_metrics.items()}
                batch_metrics_log['batch_loss'] = loss.item()
                wandb.log(batch_metrics_log)
        
        # Calculate averages
        avg_loss = total_loss / n_batches if n_batches > 0 else None
        avg_metrics = {}
        for key, total_value in total_metrics.items():
            avg_metrics[key] = total_value / n_batches if n_batches > 0 else None
        
        # Return in the expected format (loss, acc, mse, mae) for backward compatibility
        avg_acc = avg_metrics.get("acc", None)
        avg_mse = avg_metrics.get("mse", None)
        avg_mae = avg_metrics.get("mae", None)
        avg_relative_l2 = avg_metrics.get("relative_l2", None)  # PDE-specific metric
        
        return avg_loss, avg_acc, avg_mse, avg_mae, avg_relative_l2

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['metrics']

if __name__ == "__main__":
    print("Trainer class ready for use. Run via train.py.") 