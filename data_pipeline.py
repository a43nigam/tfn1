__all__ = ["SyntheticCopyDataset", "pad_collate", "language_modeling_collate", "synthetic_seq_collate", "get_dataloader"]

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional

def heat_equation_transformer_collate(batch: List[Dict[str, torch.Tensor]], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Special collate function for heat equation data when used with Transformer models.
    
    This function converts continuous floating-point values to discrete token indices
    that the Transformer's embedding layer can handle.
    
    Args:
        batch: List of dictionaries with heat equation data
        config: Configuration dictionary containing model and data parameters
    
    Returns:
        Dictionary with tokenized format for Transformer models
    """
    try:
        # Get quantization parameters from config with sensible defaults
        vocab_size = config.get("model", {}).get("vocab_size", 1000)
        min_val = config.get("data", {}).get("min_val", -1.0)
        max_val = config.get("data", {}).get("max_val", 1.0)
        
        print(f"🔧 Quantization parameters: vocab_size={vocab_size}, range=[{min_val}, {max_val}]")
        
        # Stack inputs and targets
        inputs = torch.stack([item['inputs'] for item in batch])  # [B, seq_len, features]
        targets = torch.stack([item['targets'] for item in batch])  # [B, seq_len, features]
        
        # Get positions if available
        positions = None
        if 'positions' in batch[0]:
            positions = torch.stack([item['positions'] for item in batch])
        
        # Quantize continuous values to discrete tokens
        # Scale values from [min_val, max_val] to [0, vocab_size-1]
        def quantize_to_tokens(values: torch.Tensor) -> torch.Tensor:
            # Clamp values to valid range
            values_clamped = torch.clamp(values, min_val, max_val)
            
            # Normalize to [0, 1]
            values_normalized = (values_clamped - min_val) / (max_val - min_val)
            
            # Scale to [0, vocab_size-1] and convert to long
            tokens = (values_normalized * (vocab_size - 1)).long()
            
            return tokens
        
        # Convert inputs and targets to discrete tokens
        input_tokens = quantize_to_tokens(inputs.squeeze(-1))  # Remove feature dim, [B, seq_len]
        target_tokens = quantize_to_tokens(targets.squeeze(-1))  # Remove feature dim, [B, seq_len]
        
        # Create attention mask (all positions are valid for heat equation)
        attention_mask = torch.ones_like(input_tokens, dtype=torch.long)
        
        # Return in language modeling format for Transformer
        result = {
            'inputs': input_tokens,
            'attention_mask': attention_mask,
            'labels': target_tokens
        }
        
        # Add positions if available
        if positions is not None:
            result['positions'] = positions
        
        # --- START FIX: Add calendar features handling ---
        # Check if calendar features are present in the first item
        if 'calendar_features' in batch[0]:
            # If they exist, collate them into a single dictionary of tensors
            calendar_features_collated = {}
            # Get the keys from the first sample's calendar_features
            feature_keys = batch[0]['calendar_features'].keys()
            
            for key in feature_keys:
                # Stack the tensors for this feature from all items in the batch
                calendar_features_collated[key] = torch.stack(
                    [item['calendar_features'][key] for item in batch]
                )
            
                    # Add the collated dictionary to the final result
        result['calendar_features'] = calendar_features_collated
        # print(f"🔧 heat_equation_transformer_collate: Added calendar_features with keys: {list(calendar_features_collated.keys())}")
        # --- END FIX ---
        
        print(f"✅ Successfully quantized batch: {input_tokens.shape} -> {input_tokens.dtype}")
        return result
        
    except Exception as e:
        print(f"❌ Error in heat_equation_transformer_collate: {e}")
        print(f"   Batch keys: {[item.keys() for item in batch]}")
        print(f"   First item shapes: {[(k, v.shape, v.dtype) for k, v in batch[0].items()]}")
        raise


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
    
    batch_dict = {
        'inputs': inputs,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    
    # --- START FIX: Add calendar features handling ---
    # Check if calendar features are present in the first item
    if 'calendar_features' in batch[0]:
        # If they exist, collate them into a single dictionary of tensors
        calendar_features_collated = {}
        # Get the keys from the first sample's calendar_features
        feature_keys = batch[0]['calendar_features'].keys()
        
        for key in feature_keys:
            # Stack the tensors for this feature from all items in the batch
            calendar_features_collated[key] = torch.stack(
                [item['calendar_features'][key] for item in batch]
            )
        
        # Add the collated dictionary to the final batch dictionary
        batch_dict['calendar_features'] = calendar_features_collated
        # print(f"🔧 language_modeling_collate: Added calendar_features with keys: {list(calendar_features_collated.keys())}")
    # --- END FIX ---
    
    return batch_dict


def synthetic_seq_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for synthetic sequence datasets (e.g., Delayed Copy)
    that do not have an attention mask.
    """
    inputs = torch.stack([item['inputs'] for item in batch])
    # The synthetic loader provides 'targets', which the trainer maps to the 'y' variable (labels)
    labels = torch.stack([item['targets'] for item in batch])
    
    batch_dict = {
        'inputs': inputs,
        'labels': labels,
    }
    
    # --- START FIX: Add calendar features handling ---
    # Check if calendar features are present in the first item
    if 'calendar_features' in batch[0]:
        # If they exist, collate them into a single dictionary of tensors
        calendar_features_collated = {}
        # Get the keys from the first sample's calendar_features
        feature_keys = batch[0]['calendar_features'].keys()
        
        for key in feature_keys:
            # Stack the tensors for this feature from all items in the batch
            calendar_features_collated[key] = torch.stack(
                [item['calendar_features'][key] for item in batch]
            )
        
        # Add the collated dictionary to the final batch dictionary
        batch_dict['calendar_features'] = calendar_features_collated
        # print(f"🔧 synthetic_seq_collate: Added calendar_features with keys: {list(calendar_features_collated.keys())}")
    # --- END FIX ---
    
    return batch_dict


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
    
    # --- START FIX: Add calendar features handling ---
    # Initialize the final batch dictionary
    batch_dict = {}
    
    # Return standardized format
    if task == "classification":
        # For classification, we have inputs and labels
        batch_dict["inputs"] = src
        # Stack labels
        labels = torch.stack([item["labels"] for item in batch])
        batch_dict["labels"] = labels
    else:
        # For copy/regression, we have inputs and targets
        batch_dict["inputs"] = src
        batch_dict["targets"] = tgt
    
    # Now, check if calendar features are present in the first item
    if 'calendar_features' in batch[0]:
        # If they exist, collate them into a single dictionary of tensors
        calendar_features_collated = {}
        # Get the keys from the first sample's calendar_features
        feature_keys = batch[0]['calendar_features'].keys()
        
        for key in feature_keys:
            # Stack the tensors for this feature from all items in the batch
            calendar_features_collated[key] = torch.stack(
                [item['calendar_features'][key] for item in batch]
            )
        
        # Add the collated dictionary to the final batch dictionary
        batch_dict['calendar_features'] = calendar_features_collated
        # print(f"🔧 pad_collate: Added calendar_features with keys: {list(calendar_features_collated.keys())}")
    else:
        # print(f"ℹ️  pad_collate: No calendar_features found in batch")
        pass
    # --- END FIX ---
    
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
    Get DataLoader with proper collation based on dataset and model.
    
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
    model_name = config.get("model", {}).get("model_name", "unknown")
    
    # Debug information
    print(f"🔍 DataLoader Debug:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Model: {model_name}")
    print(f"   Split: {split}")
    
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
    
    # Special handling for heat equation with Transformer models
    # Heat equation returns continuous values but Transformers need discrete tokens
    # Check both config and actual dataset instance
    is_heat_equation = (
        dataset_name == "heat_equation" or 
        (hasattr(dataset, 'dataset_type') and dataset.dataset_type == 'heat_equation') or
        (hasattr(dataset, 'file_path') and 'heat_equation' in str(dataset.file_path))
    )
    is_transformer_model = (
        "transformer" in model_name.lower() or 
        "baseline" in model_name.lower()
    )
    
    # Fallback detection: check if dataset has continuous float data
    has_continuous_data = False
    if hasattr(dataset, 'inputs') and hasattr(dataset, 'targets'):
        try:
            # Check if the data is continuous (float) and has the right shape for heat equation
            sample_inputs = dataset.inputs[0] if len(dataset.inputs) > 0 else None
            if sample_inputs is not None:
                has_continuous_data = (
                    sample_inputs.dtype == torch.float32 and 
                    sample_inputs.dim() >= 2 and  # [seq_len, features] or similar
                    sample_inputs.shape[-1] == 1  # Single feature dimension
                )
        except:
            pass
    
    if (is_heat_equation or has_continuous_data) and is_transformer_model:
        print(f"⚠️  Detected heat equation/continuous data with Transformer/Baseline model: {model_name}")
        print(f"   Dataset: {dataset_name} (type: {getattr(dataset, 'dataset_type', 'unknown')})")
        print(f"   File path: {getattr(dataset, 'file_path', 'unknown')}")
        print(f"   Has continuous data: {has_continuous_data}")
        print(f"   Task: {task}")
        print(f"   Using special quantization collate function")
        collate_fn = lambda batch: heat_equation_transformer_collate(batch, config)
    # Use appropriate collation function
    elif dataset_name == "delayed_copy":  # Special handling for this synthetic task
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
    
    # Get batch size from config
    batch_size = config.get("training", {}).get("batch_size", 32)
    
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