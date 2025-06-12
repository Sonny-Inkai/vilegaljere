from typing import Any, Union, List, Optional
import json
import os
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)


class VietnameseLegalDataset(Dataset):
    """Dataset for Vietnamese legal documents"""
    
    def __init__(self, data_file: str, tokenizer: AutoTokenizer, max_source_length: int, 
                 max_target_length: int, prefix: str = "", padding: str = "max_length"):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        self.padding = padding
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert to list of examples
        self.examples = []
        for key, value in self.data.items():
            self.examples.append({
                'input_text': value['formatted_context_sent'],
                'target_text': value['extracted_relations_text']
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Prepare input
        input_text = self.prefix + example['input_text']
        target_text = example['target_text']
        
        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt"
            )
        
        # Handle padding for loss computation
        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels['input_ids']).squeeze()
        }


class VietnameseLegalPLDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Vietnamese Legal Documents
    """

    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.model = model
        
        self.prefix = conf.source_prefix if conf.source_prefix is not None else ""
        self.max_target_length = conf.max_target_length
        self.padding = "max_length" if conf.pad_to_max_length else False

        # Data paths
        self.train_file = conf.train_file
        self.validation_file = conf.validation_file
        self.test_file = conf.test_file if hasattr(conf, 'test_file') else None

        # Data collator
        label_pad_token_id = -100 if conf.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if conf.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorForSeq2Seq(
                self.tokenizer, 
                self.model, 
                label_pad_token_id=label_pad_token_id
            )

    def prepare_data(self, *args, **kwargs):
        # Download or prepare data if needed
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = VietnameseLegalDataset(
                data_file=self.train_file,
                tokenizer=self.tokenizer,
                max_source_length=self.conf.max_source_length,
                max_target_length=self.max_target_length,
                prefix=self.prefix,
                padding=self.padding
            )
            
            if self.conf.max_train_samples is not None:
                self.train_dataset.examples = self.train_dataset.examples[:self.conf.max_train_samples]

        if stage == "fit" or stage == "validate" or stage is None:
            if os.path.exists(self.validation_file):
                self.val_dataset = VietnameseLegalDataset(
                    data_file=self.validation_file,
                    tokenizer=self.tokenizer,
                    max_source_length=self.conf.max_source_length,
                    max_target_length=self.conf.val_max_target_length,
                    prefix=self.prefix,
                    padding=self.padding
                )
                
                if self.conf.max_val_samples is not None:
                    self.val_dataset.examples = self.val_dataset.examples[:self.conf.max_val_samples]

        if stage == "test" or stage is None:
            if self.test_file and os.path.exists(self.test_file):
                self.test_dataset = VietnameseLegalDataset(
                    data_file=self.test_file,
                    tokenizer=self.tokenizer,
                    max_source_length=self.conf.max_source_length,
                    max_target_length=self.conf.val_max_target_length,
                    prefix=self.prefix,
                    padding=self.padding
                )
                
                if self.conf.max_test_samples is not None:
                    self.test_dataset.examples = self.test_dataset.examples[:self.conf.max_test_samples]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.conf.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        ) 