__all__ = ["SyntheticCopyDataset", "pad_collate", "language_modeling_collate", "synthetic_seq_collate", "get_dataloader"]

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional

def language_modeling_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for language modeling datasets (WikiText, PG19).
    
    Args:
        batch: List of dictionaries with standardized format:
               'inputs', 'attention_mask', 'labels'
    
    Returns:
        Dictionary with standardized format:
        {'inputs': ..., 'attention_mask': ..., 'labels': ...}
    """
    # Stack all tensors
    inputs = torch.stack([item['inputs'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'inputs': inputs,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def synthetic_seq_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for synthetic sequence datasets (e.g., Delayed Copy)
    that do not have an attention mask.
    """
    inputs = torch.stack([item['inputs'] for item in batch])
    # The synthetic loader provides 'targets', which the trainer maps to the 'y' variable (labels)
    labels = torch.stack([item['targets'] for item in batch])
    
    return {
        'inputs': inputs,
        'labels': labels,
    }


def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0, task: str = "copy") -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of dictionaries with standardized format:
               - 'inputs' and 'targets' for regression/copy tasks
               - 'inputs' and 'labels' for classification tasks
        pad_idx: Padding token index
        task: Task type ("copy", "classification", "regression")
    
    Returns:
        Dictionary with standardized format:
        - {"inputs": ..., "targets": ...} for regression/copy
        - {"inputs": ..., "labels": ...} for classification
    """
    if not batch:
        raise ValueError("Empty batch provided to pad_collate")
    
    # All datasets now use standardized format
    first_item = batch[0]
    
    # Verify we have the expected standardized format
    if task == "classification":
        if "inputs" not in first_item or "labels" not in first_item:
            raise ValueError(f"Classification batch must have 'inputs' and 'labels' keys. Got: {list(first_item.keys())}")
    else:
        if "inputs" not in first_item or "targets" not in first_item:
            raise ValueError(f"Regression/copy batch must have 'inputs' and 'targets' keys. Got: {list(first_item.keys())}")
    
    batch_size = len(batch)
    lengths = [item['inputs'].size(0) for item in batch]
    max_len = max(lengths)
    
    # Determine data type and shape based on task and first item
    first_item_data = first_item['inputs']
    if task == "regression":
        # For regression, we have feature vectors [seq_len, input_dim]
        input_dim = first_item_data.size(1) if first_item_data.dim() > 1 else 1
        dtype = torch.float
        src = torch.full((batch_size, max_len, input_dim), pad_idx, dtype=dtype)
        
        # For regression, targets may have different shape than inputs
        first_target = first_item['targets']
        target_dim = first_target.size(1) if first_target.dim() > 1 else 1
        target_len = first_target.size(0)
        tgt = torch.full((batch_size, target_len, target_dim), pad_idx, dtype=dtype)
    else:
        # For classification/copy, we have token sequences [seq_len]
        dtype = torch.long
        src = torch.full((batch_size, max_len), pad_idx, dtype=dtype)
        if task != "classification":
            tgt = torch.full((batch_size, max_len), pad_idx, dtype=dtype)
    
    for i, item in enumerate(batch):
        seq_len = item['inputs'].size(0)
        source_data = item['inputs'].to(dtype)
        
        if task == "regression":
            # Handle 3D tensors for regression
            src[i, :seq_len, :] = source_data
            target_data = item['targets'].to(dtype)
            target_len = target_data.size(0)
            tgt[i, :target_len, :] = target_data
        else:
            # For classification/copy, pad token sequences
            src[i, :seq_len] = source_data
            if task != "classification":
                target_data = item['targets'].to(dtype)
                # For copy tasks, input and target have same shape
                tgt[i, :seq_len] = target_data
    
    # Return standardized format
    if task == "classification":
        # For classification, we have inputs and labels
        batch_dict = {"inputs": src}
        # Stack labels
        labels = torch.stack([item["labels"] for item in batch])
        batch_dict["labels"] = labels
    else:
        # For copy/regression, we have inputs and targets
        batch_dict = {"inputs": src, "targets": tgt}
    
    return batch_dict

class SyntheticCopyDataset(Dataset):
    """Synthetic dataset that generates integer sequences on-the-fly.

    Each item is a dict with keys:
        ``source`` – the input sequence (LongTensor[seq_len])
        ``target`` – identical to ``source`` for the copy task
        ``label`` – (optional) integer class label for classification
    """

    def __init__(
        self,
        dataset_size: int,
        seq_len: int,
        vocab_size: int,
        pad_idx: int = 0,
        task: str = "copy",  # "copy" or "classification"
        num_classes: int = 2, # for classification
        rng: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= pad_idx:
            raise ValueError("vocab_size must be > pad_idx (padding reserved)")
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.task = task
        self.num_classes = num_classes
        self.rng = rng if rng is not None else torch.Generator()
        self.rng.manual_seed(42)  # deterministic across workers unless reset

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Randomly pick a length *independent* of the index for more variety
        length = torch.randint(1, self.seq_len + 1, (), generator=self.rng).item()
        
        if self.task == "regression":
            # For regression, generate feature vectors instead of token sequences
            # Generate random features with input_dim dimensions
            input_dim = 8  # Default input dimension for regression
            seq = torch.randn(length, input_dim, generator=self.rng)
            item = {"inputs": seq, "targets": seq.clone()}
        elif self.task == "classification":
            # For classification, generate token sequences with single labels
            # Sample integers uniformly from the vocabulary *excluding* the pad token.
            if self.vocab_size <= 1:
                raise ValueError("vocab_size must be at least 2 to reserve a pad token.")

            # Strategy: sample from [0, vocab_size-2] then shift by +1 for indices >= pad_idx
            raw = torch.randint(0, self.vocab_size - 1, (length,), dtype=torch.long, generator=self.rng)
            # Shift values greater than or equal to pad_idx so that pad_idx is skipped
            seq = raw + (raw >= self.pad_idx).long()
            
            # For classification, use sum of sequence mod num_classes as label
            label = int(seq.sum().item()) % self.num_classes
            item = {"inputs": seq, "labels": torch.tensor(label, dtype=torch.long)}
        else:
            # For copy tasks, generate token sequences
            # Sample integers uniformly from the vocabulary *excluding* the pad token.
            if self.vocab_size <= 1:
                raise ValueError("vocab_size must be at least 2 to reserve a pad token.")

            # Strategy: sample from [0, vocab_size-2] then shift by +1 for indices >= pad_idx
            raw = torch.randint(0, self.vocab_size - 1, (length,), dtype=torch.long, generator=self.rng)
            # Shift values greater than or equal to pad_idx so that pad_idx is skipped
            seq = raw + (raw >= self.pad_idx).long()
            item = {"inputs": seq, "targets": seq.clone()}
            
        return item

def dataloader_factory(config: Dict[str, Any], split: str = 'train') -> Dataset:
    """
    Factory to select the correct dataset based on config['data']['dataset_name'].
    Uses the data registry pattern for extensibility.
    
    Args:
        config: full config dict
        split: 'train', 'val', or 'test'
    Returns:
        torch.utils.data.Dataset
    """
    from data.registry import create_dataset
    
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset_name", "synthetic")
    
    try:
        return create_dataset(dataset_name, config, split)
    except Exception as e:
        raise ValueError(f"Failed to create dataset '{dataset_name}': {e}")

def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """
    Get DataLoader with proper collation based on dataset and task.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        DataLoader with appropriate collation function
    """
    # Get dataset
    dataset = dataloader_factory(config, split)
    
    # Determine task type for proper collation
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "synthetic")
    
    # Auto-detect task based on dataset and model
    task = data_cfg.get("task", "regression")  # Default to regression for time series
    
    # Override based on dataset type
    if dataset_name in ["ett", "jena", "stock"]:
        task = "regression"
    elif dataset_name in ["glue", "arxiv", "imdb"]:
        task = "classification"
    elif dataset_name in ["pg19", "wikitext", "delayed_copy"]:
        task = "language_modeling"
    elif dataset_name == "synthetic":
        task = data_cfg.get("task", "copy")
    
    # Get batch size from config
    batch_size = config.get("training", {}).get("batch_size", 32)
    
    # Use appropriate collation function
    if dataset_name == "delayed_copy":  # Special handling for this synthetic task
        collate_fn = synthetic_seq_collate
    elif task == "classification":
        # For classification, we need special handling
        collate_fn = lambda batch: pad_collate(batch, task="classification")
    elif task == "language_modeling":
        # For other LM tasks like PG19 that use tokenizers
        collate_fn = language_modeling_collate
    else:
        # For regression/copy tasks
        collate_fn = lambda batch: pad_collate(batch, task=task)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for debugging
        drop_last=(split == 'train')
    )

if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser(description="Test DataLoader creation from config YAML.")
    parser.add_argument("--config", type=str, default="configs/synthetic_copy.yaml")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    loader = get_dataloader(cfg, split=args.split)
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    
    # Handle both old and new standardized keys
    if "inputs" in batch:
        print("inputs shape:", batch["inputs"].shape)
        print("targets shape:", batch["targets"].shape)
    elif "source" in batch:
        print("source shape:", batch["source"].shape)
        print("target shape:", batch["target"].shape)
    elif "input_ids" in batch:
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        print("labels shape:", batch["labels"].shape)
    else:
        print("Unknown batch format:", batch.keys()) 