__all__ = ["SyntheticCopyDataset", "pad_collate", "get_dataloader"]

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional
from data.timeseries_loader import ETTDataset
from data.nlp_loader import NLPDataset
from data.jena_loader import JenaClimateDataset
from data.stock_loader import StockMarketDataset
from data.glue_loader import GLUEDataset
from data.arxiv_loader import ArxivDataset
from data.pg19_loader import PG19Dataset
from data.wikitext_loader import WikiTextDataset

def language_modeling_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for language modeling datasets (WikiText, PG19).
    
    Args:
        batch: List of dictionaries with 'input_ids', 'attention_mask', 'labels'
    
    Returns:
        Dictionary with batched tensors
    """
    # Stack all tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0, task: str = "copy") -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of dictionaries with 'source' and 'target'/'label' keys
        pad_idx: Padding token index
        task: Task type ("copy", "classification", "regression")
    
    Returns:
        Dictionary with padded tensors
    """
    if not batch:
        raise ValueError("Empty batch provided to pad_collate")
    
    # Step 1: Eliminate Key Ambiguity - inspect first item to determine keys
    first_item = batch[0]
    input_key = 'input' if 'input' in first_item else 'source'
    
    # For classification, we have 'label' instead of 'target'
    if task == "classification":
        target_key = 'label'
    else:
        target_key = 'target' if 'target' in first_item else 'target'
    
    batch_size = len(batch)
    lengths = [item[input_key].size(0) for item in batch]
    max_len = max(lengths)
    
    # Determine data type and shape based on task and first item
    first_item_data = first_item[input_key]
    if task == "regression":
        # For regression, we have feature vectors [seq_len, input_dim]
        input_dim = first_item_data.size(1) if first_item_data.dim() > 1 else 1
        dtype = torch.float
        src = torch.full((batch_size, max_len, input_dim), pad_idx, dtype=dtype)
        if task != "classification":
            tgt = torch.full((batch_size, max_len, input_dim), pad_idx, dtype=dtype)
    else:
        # For classification/copy, we have token sequences [seq_len]
        dtype = torch.long
        src = torch.full((batch_size, max_len), pad_idx, dtype=dtype)
        if task != "classification":
            tgt = torch.full((batch_size, max_len), pad_idx, dtype=dtype)
    
    for i, item in enumerate(batch):
        seq_len = item[input_key].size(0)
        source_data = item[input_key].to(dtype)
        
        if task == "regression":
            # Step 2: Correctly handle 3D tensors for regression
            src[i, :seq_len, :] = source_data
            if task != "classification":
                target_data = item[target_key].to(dtype)
                tgt[i, :seq_len, :] = target_data
        else:
            # For classification/copy, pad token sequences
            src[i, :seq_len] = source_data
            if task != "classification":
                target_data = item[target_key].to(dtype)
                tgt[i, :seq_len] = target_data
    
    # Step 3: Standardize the output dictionary
    if task == "classification":
        # For classification, we have inputs and labels
        batch_dict = {"inputs": src}
        # Stack labels
        labels = torch.stack([item["label"] for item in batch])
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
            item = {"source": seq, "target": seq.clone()}
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
            item = {"source": seq, "label": torch.tensor(label, dtype=torch.long)}
        else:
            # For copy tasks, generate token sequences
            # Sample integers uniformly from the vocabulary *excluding* the pad token.
            if self.vocab_size <= 1:
                raise ValueError("vocab_size must be at least 2 to reserve a pad token.")

            # Strategy: sample from [0, vocab_size-2] then shift by +1 for indices >= pad_idx
            raw = torch.randint(0, self.vocab_size - 1, (length,), dtype=torch.long, generator=self.rng)
            # Shift values greater than or equal to pad_idx so that pad_idx is skipped
            seq = raw + (raw >= self.pad_idx).long()
            item = {"source": seq, "target": seq.clone()}
            
        return item

def dataloader_factory(config: Dict[str, Any], split: str = 'train') -> Dataset:
    # Map split names for datasets that use different naming conventions
    split_mapping = {
        'val': 'validation',  # WikiText uses 'validation' instead of 'val'
        'validation': 'validation',
        'train': 'train',
        'test': 'test'
    }
    mapped_split = split_mapping.get(split, split)
    
    """
    Factory to select the correct dataset based on config['data']['dataset_name'].
    Args:
        config: full config dict
        split: 'train', 'val', or 'test'
    Returns:
        torch.utils.data.Dataset
    """
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset_name", "synthetic")
    
    if dataset_name == "synthetic":
        return SyntheticCopyDataset(
            dataset_size=data_cfg.get("dataset_size", 1000),
            seq_len=data_cfg.get("seq_len", 50),
            vocab_size=data_cfg.get("vocab_size", 20),
            pad_idx=data_cfg.get("pad_idx", 0),
            task=data_cfg.get("task", "copy"),
            num_classes=data_cfg.get("num_classes", 2),
        )
    
    # Step 1: Unified pattern for datasets with get_splits method
    elif dataset_name == "ett":
        # Handle ETT dataset with new normalization parameters
        csv_path = data_cfg.get("csv_path", "data/ETTh1.csv")
        input_len = data_cfg.get("input_len", 96)
        output_len = data_cfg.get("output_len", 24)
        normalization_strategy = data_cfg.get("normalization_strategy", "global")
        instance_normalize = data_cfg.get("instance_normalize", False)
        
        # Get splits with new parameters
        train_ds, val_ds, test_ds = ETTDataset.get_splits(
            csv_path=csv_path,
            input_len=input_len,
            output_len=output_len,
            normalization_strategy=normalization_strategy,
            instance_normalize=instance_normalize
        )
        
        if split == 'train':
            return train_ds
        elif split == 'val':
            return val_ds
        elif split == 'test':
            return test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
    
    elif dataset_name == "jena":
        # Step 2: Use get_splits pattern for Jena dataset
        csv_path = data_cfg.get("csv_path", "data/jena_climate_2009_2016.csv")
        input_len = data_cfg.get("input_len", 96)
        output_len = data_cfg.get("output_len", 24)
        
        train_ds, val_ds, test_ds = JenaClimateDataset.get_splits(
            csv_path=csv_path,
            input_len=input_len,
            output_len=output_len,
        )
        
        if split == 'train':
            return train_ds
        elif split == 'val':
            return val_ds
        elif split == 'test':
            return test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
    
    elif dataset_name == "stock":
        # Step 2: Use get_splits pattern for Stock dataset
        csv_path = data_cfg.get("csv_path", "data/all_stocks_5yr.csv")
        input_len = data_cfg.get("input_len", 60)
        output_len = data_cfg.get("output_len", 5)
        ticker = data_cfg.get("ticker", None)
        
        train_ds, val_ds, test_ds = StockMarketDataset.get_splits(
            csv_path=csv_path,
            input_len=input_len,
            output_len=output_len,
            ticker=ticker,
        )
        
        if split == 'train':
            return train_ds
        elif split == 'val':
            return val_ds
        elif split == 'test':
            return test_ds
        else:
            raise ValueError(f"Unknown split: {split}")
    
    elif dataset_name == "glue":
        return GLUEDataset(
            task=data_cfg.get("task", "sst2"),
            split=mapped_split,
            max_length=data_cfg.get("max_length", 512),
            tokenizer_name=data_cfg.get("tokenizer_name", "bert-base-uncased"),
        )
    
    elif dataset_name == "arxiv":
        return ArxivDataset(
            csv_path=data_cfg.get("csv_path", "data/arxiv_sample.csv"),
            max_length=data_cfg.get("max_length", 512),
            tokenizer_name=data_cfg.get("tokenizer_name", "bert-base-uncased"),
        )
    
    elif dataset_name == "pg19":
        return PG19Dataset(
            data_dir=data_cfg.get("data_dir", "data/pg19"),
            split=mapped_split,
            max_length=data_cfg.get("max_length", 512),
            tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
        )
    
    elif dataset_name == "wikitext":
        return WikiTextDataset(
            dataset_name=data_cfg.get("wikitext_dataset_name", "wikitext-2-raw-v1"),
            tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
            max_length=data_cfg.get("max_length", 512),
            split=mapped_split,
            text_col=data_cfg.get("text_col", "text"),
            use_streaming=data_cfg.get("use_streaming", False),
            max_samples=data_cfg.get("max_samples", None),
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """
    Create a DataLoader for the specified dataset and split.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        DataLoader with appropriate collate function
    """
    dataset = dataloader_factory(config, split)
    
    # Step 1: Use the model registry to determine task type
    model_name = config.get("model_name", "tfn_classifier")
    try:
        from model.registry import get_model_config
        model_info = get_model_config(model_name)
        task_type = model_info['task_type']
    except (ImportError, KeyError, ValueError):
        # Fallback to config-based task determination
        task_type = config.get("model", {}).get("task", "classification")
    
    # Step 2: Generalize collation selection based on task type
    if task_type == "language_modeling":
        collate_fn = lambda batch: language_modeling_collate(batch)
    elif task_type in ("regression", "time_series"):
        # This now works because pad_collate is fixed
        collate_fn = lambda batch: pad_collate(batch, task="regression")
    else:  # classification and other tasks
        collate_fn = lambda batch: pad_collate(batch, task="classification")
    
    # Get batch size
    batch_size = config.get("training", {}).get("batch_size", 32)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for debugging, increase for production
        drop_last=(split == 'train'),
    )
    
    return dataloader

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