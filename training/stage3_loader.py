import os
import json
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class Stage3Dataset(Dataset):

    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        n_value_tokens: int = 1,
        max_length: int = 512,
        subset_size: Optional[int] = None,
    ):

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if subset_size is not None and subset_size < len(self.data):
            self.data = self.data[:subset_size]
        
        self.tokenizer = tokenizer
        self.n_value_tokens = n_value_tokens
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item.get('prompt', '')
        
        safe_response = ''
        harmful_response = ''
        
        if 'harmful_response' in item:
            harmful_response = item.get('harmful_response', '')
        
        if 'safe_response' in item:
            safe_response = item.get('safe_response', '')
        elif 'response' in item:
            response = item.get('response', '')
            is_harmful = item.get('response_is_harmful', False)
            
            if is_harmful:
                if not harmful_response:
                    harmful_response = response
            else:
                safe_response = response
        
        response = safe_response
        
        full_text = prompt + response

        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        response_enc = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        if harmful_response:
            harmful_enc = self.tokenizer(
                harmful_response,
                truncation=True,
                max_length=self.max_length // 2,
                return_tensors="pt",
                add_special_tokens=False,
            )
            harmful_ids = harmful_enc['input_ids'].squeeze(0)
            harmful_mask = harmful_enc['attention_mask'].squeeze(0)
        else:
            harmful_ids = torch.tensor([], dtype=torch.long)
            harmful_mask = torch.tensor([], dtype=torch.long)
        
        prompt_ids = prompt_enc['input_ids'].squeeze(0)  # [prompt_len]
        response_ids = response_enc['input_ids'].squeeze(0)  # [response_len]
        prompt_mask = prompt_enc['attention_mask'].squeeze(0)
        response_mask = response_enc['attention_mask'].squeeze(0)
    
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        
        available_length = self.max_length - 1  # -1 for EOS
        
        prompt_len = prompt_ids.size(0)
        response_len = response_ids.size(0)
        
        if prompt_len + response_len > available_length:
            if response_len > available_length:
                response_ids = response_ids[:available_length]
                response_mask = response_mask[:available_length]
                prompt_len = 0
            else:
                max_prompt_len = available_length - response_len
                prompt_ids = prompt_ids[:max_prompt_len]
                prompt_mask = prompt_mask[:max_prompt_len]
                prompt_len = max_prompt_len
        
        K = self.n_value_tokens
        input_ids = torch.cat([
            prompt_ids,
            response_ids,
            torch.tensor([eos_token_id], dtype=torch.long)
        ])
        

        prompt_labels = torch.full((prompt_ids.size(0),), -100, dtype=torch.long)
        response_labels = response_ids.clone()  
        eos_label = torch.tensor([eos_token_id], dtype=torch.long)
        
        labels = torch.cat([
            prompt_labels,
            response_labels,
            eos_label
        ])
        
        attention_mask = torch.cat([
            prompt_mask,
            response_mask,
            torch.tensor([1], dtype=torch.long)  # EOS
        ])
        

        M = prompt_ids.size(0)
        L = response_ids.size(0)
        
        position_ids_prompt = torch.arange(0, M, dtype=torch.long)  # [0..M-1]
        position_ids_response = torch.arange(M + K, M + K + L + 1, dtype=torch.long)  # [M+K..M+K+L] (+1 for EOS)
        position_ids = torch.cat([position_ids_prompt, position_ids_response])
        
        seq_len = input_ids.size(0)
        if seq_len < self.max_length:
            # Padding
            pad_length = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
            if pad_length > 0:
                last_pos = position_ids[-1].item()
                position_ids = torch.cat([position_ids, torch.full((pad_length,), last_pos, dtype=torch.long)])
        elif seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            position_ids = position_ids[:self.max_length]
        
        prompt_len_final = prompt_ids.size(0)
        prompt_end_idx = prompt_len_final - 1 if prompt_len_final > 0 else 0
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_len": torch.tensor(prompt_len_final, dtype=torch.long),
            "prompt_end_idx": torch.tensor(prompt_end_idx, dtype=torch.long),
        }


def get_stage3_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: PreTrainedTokenizer,
    n_value_tokens: int = 1,
    max_length: int = 512,
    batch_size: int = 8,
    train_subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = Stage3Dataset(
        data_path=train_path,
        tokenizer=tokenizer,
        n_value_tokens=n_value_tokens,
        max_length=max_length,
        subset_size=train_subset_size,
    )
    
    val_dataset = Stage3Dataset(
        data_path=val_path,
        tokenizer=tokenizer,
        n_value_tokens=n_value_tokens,
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

