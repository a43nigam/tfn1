import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple
from datasets import load_dataset
import gc

# Dynamic data path resolution (e.g., Kaggle input datasets)
from . import resolve_data_path

class WikiTextDataset(Dataset):
    """
    WikiTextDataset for language modeling on WikiText-2 and WikiText-103.
    Loads from HuggingFace datasets, tokenizes, and returns dicts for DataLoader.
    Memory-optimized for Colab/Kaggle environments.
    """
    def __init__(
        self,
        dataset_name: str = 'wikitext-2-raw-v1',
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        split: Optional[str] = None,
        text_col: str = 'text',
        use_streaming: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset from HuggingFace with memory optimizations
        if use_streaming:
            # Streaming mode for very large datasets
            self.dataset = load_dataset('wikitext', dataset_name, split=split, streaming=True)
            self.texts = None  # Will load on-demand
            self.use_streaming = True
        else:
            # Regular mode with optional sample limiting
            self.dataset = load_dataset('wikitext', dataset_name, split=split)
            if max_samples and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))
            self.texts = self.dataset[text_col]
            self.use_streaming = False
        
        self.max_length = max_length
        self.text_col = text_col
        
    def __len__(self) -> int:
        if self.use_streaming:
            # For streaming, we need to estimate or set a fixed length
            return 10000  # Conservative estimate
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_streaming:
            # Get item from streaming dataset
            item = next(iter(self.dataset.skip(idx)))
            text = item[self.text_col]
        else:
            text = self.texts[idx]
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        # For LM: labels are input_ids shifted by 1 (next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # ignore index for last token
        
        # Clear memory after tokenization
        del enc
        gc.collect()
        
        return {
            'inputs': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    @staticmethod
    def get_splits(dataset_name: str = 'wikitext-2-raw-v1', tokenizer_name: str = 'gpt2', max_length: int = 512, text_col: str = 'text', use_streaming: bool = False, max_samples: Optional[int] = None):
        return (
            WikiTextDataset(dataset_name, tokenizer_name, max_length, split='train', text_col=text_col, use_streaming=use_streaming, max_samples=max_samples),
            WikiTextDataset(dataset_name, tokenizer_name, max_length, split='validation', text_col=text_col, use_streaming=use_streaming, max_samples=max_samples),
            WikiTextDataset(dataset_name, tokenizer_name, max_length, split='test', text_col=text_col, use_streaming=use_streaming, max_samples=max_samples),
        )

class MemoryEfficientWikiTextDataset(Dataset):
    """
    Memory-efficient WikiText dataset for very limited environments.
    Loads data on-demand and uses aggressive memory management.
    """
    def __init__(
        self,
        dataset_name: str = 'wikitext-2-raw-v1',
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load only a subset for memory efficiency
        self.dataset = load_dataset('wikitext', dataset_name, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        self.max_length = max_length
        self._cached_samples = {}  # Simple LRU cache
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Simple caching to avoid re-tokenization
        if idx in self._cached_samples:
            return self._cached_samples[idx]
        
        text = self.dataset[idx]['text']
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        result = {
            'input_ids': input_ids,
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': labels,
        }
        
        # Cache with size limit
        if len(self._cached_samples) < 100:  # Keep only 100 samples in cache
            self._cached_samples[idx] = result
        
        return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test WikiTextDataset loader.")
    parser.add_argument('--dataset_name', type=str, default='wikitext-2-raw-v1', help='WikiText dataset name (wikitext-2-raw-v1, wikitext-103-raw-v1)')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples for memory efficiency')
    args = parser.parse_args()
    ds = WikiTextDataset(args.dataset_name, args.tokenizer, args.max_length, args.split, max_samples=args.max_samples)
    sample = ds[0]
    print(f"input_ids shape: {sample['input_ids'].shape}, attention_mask shape: {sample['attention_mask'].shape}, labels shape: {sample['labels'].shape}")
    print(f"Dataset size: {len(ds)} samples") 