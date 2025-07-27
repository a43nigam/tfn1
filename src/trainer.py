from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from src import metrics
from src.flops_tracker import create_flops_tracker, log_flops_stats
import math

class Trainer:
    """
    Universal Trainer for classification, regression, time series, NER, and language modeling.
    Handles device, optimizer, scheduler, loss, metrics, and robust batch formats.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        task: str = "classification",
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 10,
        grad_clip: Optional[float] = 1.0,
        log_interval: int = 100,
        warmup_epochs: int = 1,
        track_flops: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task = task
        self.device = torch.device(device)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.track_flops = track_flops
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_mse": [], "val_mse": [], "train_mae": [], "val_mae": [], "learning_rates": []}

        # Create FLOPS tracker if enabled
        if self.track_flops:
            self.flops_tracker = create_flops_tracker(self.model)
            self.model = self.flops_tracker
        else:
            self.flops_tracker = None

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = self._build_scheduler(lr, warmup_epochs)
        self.criterion = self._select_criterion(task)

    def _select_criterion(self, task: str):
        if task in ("classification", "ner", "language_modeling"):
            return torch.nn.CrossEntropyLoss()
        else:
            return torch.nn.MSELoss()

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
        # Handles all batch formats
        if "input_ids" in batch and "labels" in batch:
            x = batch["input_ids"].to(self.device)
            y = batch["labels"].to(self.device)
            # Optionally pass attention_mask if present
            if "attention_mask" in batch:
                return (x, batch["attention_mask"].to(self.device)), y
            return x, y
        elif "labels" in batch and "source" in batch:
            # Synthetic classification: use 'source' as input, 'labels' as target
            x = batch["source"].to(self.device)
            y = batch["labels"].to(self.device)
            return x, y
        elif "source" in batch:
            x = batch["source"].to(self.device)
            y = batch["target"].to(self.device)
            return x, y
        elif "input" in batch:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)
            return x, y
        else:
            raise ValueError(f"Unknown batch format: keys {list(batch.keys())}")

    def _accuracy(self, logits, targets):
        # For classification/NER
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == targets).float().sum().item()
        total = targets.numel()
        return correct / total

    def _fmt(self, val):
        return f"{val:.4f}" if val is not None else "N/A"

    def fit(self) -> Dict[str, Any]:
        # Print model name and parameter count at the start
        model_name = getattr(self.model, 'name', self.model.__class__.__name__)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {model_name} | Total parameters: {num_params:,}")
        
        # Track learning rates for each epoch
        learning_rates = []
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, train_mse, train_mae = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc, val_mse, val_mae = (None, None, None, None)
            if self.val_loader is not None:
                val_loss, val_acc, val_mse, val_mae = self._run_epoch(self.val_loader, train=False)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_mse"].append(train_mse)
            self.history["val_mse"].append(val_mse)
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            self.history["learning_rates"].append(current_lr)
            
            print(f"Epoch {epoch:02d} | Train Loss: {self._fmt(train_loss)} | Val Loss: {self._fmt(val_loss)} | "
                  f"Train Acc: {self._fmt(train_acc)} | Val Acc: {self._fmt(val_acc)} | "
                  f"Train MSE: {self._fmt(train_mse)} | Val MSE: {self._fmt(val_mse)} | "
                  f"Train MAE: {self._fmt(train_mae)} | Val MAE: {self._fmt(val_mae)} | "
                  f"LR: {current_lr:.6f}")
            
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
        
        return self.history

    def _run_epoch(self, loader: DataLoader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss, total_acc, total_mse, total_mae, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for batch_idx, batch in enumerate(loader, start=1):
            x, y = self._unpack_batch(batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                if self.task in ("classification", "ner"):
                    if isinstance(x, tuple):
                        logits = self.model(x[0])
                    else:
                        logits = self.model(x)
                    if logits.dim() == 3 and logits.size(1) > 1:
                        logits = logits.mean(dim=1)
                    loss = self.criterion(logits, y)
                    acc = self._accuracy(logits, y)
                    mse_val = mae_val = None
                elif self.task in ("regression", "time_series"):
                    preds = self.model(x)
                    # Handle multi-dimensional targets for time series
                    if y.dim() > 1:
                        # Flatten targets for loss calculation
                        y_flat = y.view(y.size(0), -1)
                        preds_flat = preds.view(preds.size(0), -1)
                        loss = self.criterion(preds_flat, y_flat)
                    else:
                        loss = self.criterion(preds, y)
                    acc = None
                    mse_val = metrics.mse(preds, y)
                    mae_val = metrics.mae(preds, y)
                elif self.task == "language_modeling":
                    # For LM, x may be a tuple (input_ids, attention_mask)
                    if isinstance(x, tuple):
                        logits = self.model(x[0])
                    else:
                        logits = self.model(x)
                    # logits: [B, L, vocab_size], y: [B, L]
                    logits = logits.view(-1, logits.size(-1))
                    y = y.view(-1)
                    loss = self.criterion(logits, y)
                    acc = None
                    mse_val = mae_val = None
                else:
                    raise ValueError(f"Unknown task: {self.task}")
                if train:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
            total_loss += loss.item()
            if acc is not None:
                total_acc += acc
            if mse_val is not None:
                total_mse += mse_val
            if mae_val is not None:
                total_mae += mae_val
            n_batches += 1
        avg_loss = total_loss / n_batches if n_batches > 0 else None
        avg_acc = total_acc / n_batches if n_batches > 0 else None
        avg_mse = total_mse / n_batches if n_batches > 0 else None
        avg_mae = total_mae / n_batches if n_batches > 0 else None
        return avg_loss, avg_acc, avg_mse, avg_mae

if __name__ == "__main__":
    print("Trainer class ready for use. Run via train.py.") 