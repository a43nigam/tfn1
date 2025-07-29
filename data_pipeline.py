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

def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0, task: str = "copy") -> Dict[str, torch.Tensor]:
    """Custom ``collate_fn`` that pads variable-length sequences.

    The result tensors have shape ``[B, max_len]`` where ``max_len`` is the
    longest sequence in the batch.
    """
    batch_size = len(batch)
    lengths = [item["source"].size(0) for item in batch]
    max_len = max(lengths)

    # Determine data type and shape based on task and first item
    first_item = batch[0]["source"]
    if task == "regression":
        # For regression, we have feature vectors [seq_len, input_dim]
        input_dim = first_item.size(1) if first_item.dim() > 1 else 1
        dtype = torch.float
        src = torch.full((batch_size, max_len, input_dim), pad_idx, dtype=dtype)
        tgt = torch.full((batch_size, max_len, input_dim), pad_idx, dtype=dtype)
    else:
        # For classification/copy, we have token sequences [seq_len]
        dtype = torch.long
        src = torch.full((batch_size, max_len), pad_idx, dtype=dtype)
        tgt = torch.full((batch_size, max_len), pad_idx, dtype=dtype)

    for i, item in enumerate(batch):
        seq_len = item["source"].size(0)
        source_data = item["source"].to(dtype)
        target_data = item["target"].to(dtype)
        
        if task == "regression":
            # For regression, pad feature vectors
            src[i, :seq_len, :] = source_data
            tgt[i, :seq_len, :] = target_data
        else:
            # For classification/copy, pad token sequences
            src[i, :seq_len] = source_data
            tgt[i, :seq_len] = target_data

    batch_dict = {"source": src, "target": tgt}
    if task == "classification":
        labels = torch.stack([item["label"] for item in batch])
        batch_dict["labels"] = labels
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
        else:
            # For classification/copy tasks, generate token sequences
            # Sample integers uniformly from the vocabulary *excluding* the pad token.
            if self.vocab_size <= 1:
                raise ValueError("vocab_size must be at least 2 to reserve a pad token.")

            # Strategy: sample from [0, vocab_size-2] then shift by +1 for indices >= pad_idx
            raw = torch.randint(0, self.vocab_size - 1, (length,), dtype=torch.long, generator=self.rng)
            # Shift values greater than or equal to pad_idx so that pad_idx is skipped
            seq = raw + (raw >= self.pad_idx).long()

        item = {"source": seq, "target": seq.clone()}
        if self.task == "classification":
            # Example: use sum of sequence mod num_classes as label
            label = int(seq.sum().item()) % self.num_classes
            item["label"] = torch.tensor(label, dtype=torch.long)
        return item

def dataloader_factory(config: Dict[str, Any], split: str = 'train') -> Dataset:
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
        return JenaClimateDataset(
            csv_path=data_cfg.get("csv_path", "data/jena_climate_2009_2016.csv"),
            input_len=data_cfg.get("input_len", 96),
            output_len=data_cfg.get("output_len", 24),
        )
    
    elif dataset_name == "stock":
        return StockMarketDataset(
            csv_path=data_cfg.get("csv_path", "data/all_stocks_5yr.csv"),
            input_len=data_cfg.get("input_len", 60),
            output_len=data_cfg.get("output_len", 5),
        )
    
    elif dataset_name == "glue":
        return GLUEDataset(
            task=data_cfg.get("task", "sst2"),
            split=split,
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
            split=split,
            max_length=data_cfg.get("max_length", 512),
            tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
        )
    
    elif dataset_name == "wikitext":
        return WikiTextDataset(
            data_dir=data_cfg.get("data_dir", "data/wikitext"),
            split=split,
            max_length=data_cfg.get("max_length", 512),
            tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
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
    
    # Determine task type for collate function
    task = config.get("model", {}).get("task", "classification")
    if task == "regression":
        collate_fn = lambda batch: pad_collate(batch, task="regression")
    else:
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
    if "source" in batch:
        print("source shape:", batch["source"].shape)
        print("target shape:", batch["target"].shape)
    else:
        print("input shape:", batch["input"].shape)
        print("target shape:", batch["target"].shape) 