import os
import json
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class Stage1Dataset(Dataset):

    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        subset_size: Optional[int] = None,
    ):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if subset_size is not None and subset_size < len(self.data):
            self.data = self.data[:subset_size]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        text = item['text']
        label = float(item.get('label', 0.0))
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


def get_stage1_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    train_subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:

    train_dataset = Stage1Dataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length,
        subset_size=train_subset_size,
    )
    
    val_dataset = Stage1Dataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        subset_size=val_subset_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader

