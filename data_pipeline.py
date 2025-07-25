__all__ = ["SyntheticCopyDataset", "pad_collate", "get_dataloader"]

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
from data.timeseries_loader import ETTDataset
from data.nlp_loader import NLPDataset
from data.jena_loader import JenaClimateDataset
from data.stock_loader import StockMarketDataset
from data.glue_loader import GLUEDataset
from data.arxiv_loader import ArxivDataset
from data.pg19_loader import PG19Dataset

def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_idx: int = 0, task: str = "copy") -> Dict[str, torch.Tensor]:
    """Custom ``collate_fn`` that pads variable-length sequences.

    The result tensors have shape ``[B, max_len]`` where ``max_len`` is the
    longest sequence in the batch.
    """
    batch_size = len(batch)
    lengths = [item["source"].size(0) for item in batch]
    max_len = max(lengths)

    src = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    tgt = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["source"].size(0)
        src[i, :seq_len] = item["source"]
        tgt[i, :seq_len] = item["target"]

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
        model_cfg = config["model"]
        # Explicit task field in YAML takes precedence; fall back to model_cfg for legacy.
        task = data_cfg.get("task", model_cfg.get("task", "copy"))
        num_classes = model_cfg.get("num_classes", 2)
        return SyntheticCopyDataset(
            dataset_size=data_cfg["dataset_size"],
            seq_len=data_cfg["seq_len"],
            vocab_size=model_cfg["vocab_size"],
            pad_idx=data_cfg.get("pad_idx", 0),
            task=task,
            num_classes=num_classes,
        )
    elif dataset_name == "ett":
        input_len = data_cfg.get("input_len", 96)
        output_len = data_cfg.get("output_len", 24)
        csv_path = data_cfg.get("csv_path", "data/dummy/ETTh1.csv")
        train, val, test = ETTDataset.get_splits(csv_path, input_len, output_len)
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")
    elif dataset_name == "jena":
        input_len = data_cfg.get("input_len", 96)
        output_len = data_cfg.get("output_len", 24)
        csv_path = data_cfg.get("csv_path", "data/dummy/jena_climate_2009_2016.csv")
        target_col = data_cfg.get("target_col", "T (degC)")
        train, val, test = JenaClimateDataset.get_splits(csv_path, input_len, output_len, target_col=target_col)
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")
    elif dataset_name == "stock":
        input_len = data_cfg.get("input_len", 60)
        output_len = data_cfg.get("output_len", 5)
        csv_path = data_cfg.get("csv_path", "data/dummy/all_stocks_5yr.csv")
        target_col = data_cfg.get("target_col", "close")
        ticker = data_cfg.get("ticker", None)
        train, val, test = StockMarketDataset.get_splits(csv_path, input_len, output_len, target_col=target_col, ticker=ticker)
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")
    elif dataset_name == "glue":
        file_path = data_cfg.get("file_path")
        tokenizer_name = data_cfg.get("tokenizer_name", "bert-base-uncased")
        max_length = data_cfg.get("max_length", 128)
        text_col = data_cfg.get("text_col", "sentence")
        label_col = data_cfg.get("label_col", "label")
        split_frac = data_cfg.get("split_frac", {"train": 0.8, "val": 0.1, "test": 0.1})
        train, val, test = GLUEDataset.get_splits(file_path, tokenizer_name, max_length, split_frac, text_col, label_col)
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")
    elif dataset_name == "arxiv":
        file_path = data_cfg.get("file_path")
        tokenizer_name = data_cfg.get("tokenizer_name", "bert-base-uncased")
        max_length = data_cfg.get("max_length", 256)
        text_col = data_cfg.get("text_col", "abstract")
        label_col = data_cfg.get("label_col", "category")
        split_frac = data_cfg.get("split_frac", {"train": 0.8, "val": 0.1, "test": 0.1})
        train, val, test = ArxivDataset.get_splits(file_path, tokenizer_name, max_length, split_frac, text_col, label_col)
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")
    elif dataset_name == "pg19":
        import os
        from data import resolve_data_path
        # If a direct file_path is given, use it for all splits (assume pre-split tiny CSV)
        direct_path = data_cfg.get("file_path")
        if direct_path is not None:
            file_path = resolve_data_path(direct_path)
        else:
            base_path = data_cfg.get("base_path", "/kaggle/input/the-pg19-language-modeling-benchmark-dataset")
            split_files = {
                "train": os.path.join(base_path, "train.csv"),
                "val": os.path.join(base_path, "validation.csv"),
                "test": os.path.join(base_path, "test.csv"),
            }
            file_path = resolve_data_path(split_files[split])
        tokenizer_name = data_cfg.get("tokenizer_name", "gpt2")
        max_length = data_cfg.get("max_length", 512)
        text_col = data_cfg.get("text_col", "text")
        return PG19Dataset(file_path, tokenizer_name, max_length, text_col=text_col)
    elif dataset_name == "nlp":
        file_path = data_cfg.get("file_path")
        tokenizer_name = data_cfg.get("tokenizer_name", "bert-base-uncased")
        max_length = data_cfg.get("max_length", 128)
        split_frac = data_cfg.get("split_frac", {"train": 0.8, "val": 0.1, "test": 0.1})
        train, val, test = NLPDataset.get_splits(file_path, tokenizer_name, max_length, split_frac)
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")
    elif dataset_name == "imdb":
        from data.imdb_loader import IMDBDataset
        file_path = data_cfg.get("file_path")
        tokenizer_name = data_cfg.get("tokenizer_name", "bert-base-uncased")
        max_length = data_cfg.get("max_length", 256)
        text_col = data_cfg.get("text_col", "Overview")
        label_col = data_cfg.get("label_col", "Genre")
        regression_col = data_cfg.get("regression_col", "IMDB_Rating")
        # Determine task from model_name or config
        model_name = config.get("model_name", "")
        if "regressor" in model_name or data_cfg.get("task") == "regression":
            task = "regression"
        else:
            task = "classification"
        return IMDBDataset(file_path, tokenizer_name, max_length, task, text_col, label_col, regression_col)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

def get_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """
    Given a config dictionary (from YAML), return a PyTorch DataLoader for the specified dataset.
    Args:
        config: full config dict
        split: 'train', 'val', or 'test'
    Returns:
        DataLoader
    """
    dataset = dataloader_factory(config, split)
    train_cfg = config["training"]
    dataset_name = config["data"].get("dataset_name", "synthetic")
    # For synthetic dataset, use its internal task attribute (set by YAML)
    task = getattr(dataset, "task", "copy")
    # Use drop_last only for the training split so we don't silently discard
    # small validation / test sets (which would lead to 0 batches and hence
    # the "N/A" metrics you observed).
    drop_last_flag = (split == 'train')

    if dataset_name == "synthetic":
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=(split == 'train'),
            collate_fn=lambda batch: pad_collate(batch, pad_idx=config["data"].get("pad_idx", 0), task=task),
            drop_last=drop_last_flag,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=(split == 'train'),
            drop_last=drop_last_flag,
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