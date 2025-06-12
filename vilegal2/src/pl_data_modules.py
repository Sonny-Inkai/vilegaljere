from typing import Any, Optional, Union, List
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, default_data_collator

class VietnameseLegalDataset(Dataset):
    """Dataset for Vietnamese legal documents, reading from a single JSON file."""
    
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, conf: DictConfig):
        self.tokenizer = tokenizer
        self.conf = conf
        self.prefix = conf.source_prefix
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.examples = []
        for key, value in raw_data.items():
            self.examples.append({
                'input_text': value['formatted_context_sent'],
                'target_text': value['extracted_relations_text']
            })

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = self.prefix + example['input_text']
        target_text = example['target_text']

        # Tokenize using the modern API
        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.conf.max_source_length,
            padding="max_length" if self.conf.pad_to_max_length else False,
            truncation=True,
            return_tensors="pt"
        )
        
        tokenized_target = self.tokenizer(
            text_target=target_text,
            max_length=self.conf.max_target_length,
            padding="max_length" if self.conf.pad_to_max_length else False,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = tokenized_target['input_ids']
        if self.conf.pad_to_max_length and self.conf.ignore_pad_token_for_loss:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokenized_input['input_ids'].squeeze(0),
            'attention_mask': tokenized_input['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


class VietnameseLegalPLDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Vietnamese Legal Dataset."""
    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=-100 if conf.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = VietnameseLegalDataset(self.conf.train_file, self.tokenizer, self.conf)
            self.val_dataset = VietnameseLegalDataset(self.conf.validation_file, self.tokenizer, self.conf)
        if stage == "test" or stage is None:
            self.test_dataset = VietnameseLegalDataset(self.conf.test_file, self.tokenizer, self.conf)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.conf.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        ) 