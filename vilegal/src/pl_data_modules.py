import torch
import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class ViLegalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, max_target_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert dict to list if needed
        if isinstance(self.data, dict):
            self.data = list(self.data.values())
        
        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract input and target texts
        input_text = item['formatted_context_sent']
        target_text = item['extracted_relations_text']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (shift target for decoder)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'target_text': target_text,
            'input_text': input_text
        }

class ViLegalDataModule(pl.LightningDataModule):
    def __init__(self, conf, tokenizer):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Training data
            train_path = f"{self.conf.data_path}/{self.conf.finetune_file_name}"
            self.train_dataset = ViLegalDataset(
                train_path,
                self.tokenizer,
                max_length=self.conf.max_length,
                max_target_length=self.conf.max_target_length
            )
            
            # Validation data (use same as train for now, or split)
            self.val_dataset = self.train_dataset

        if stage == "test" or stage is None:
            # Test data
            test_path = self.conf.test_data_path
            self.test_dataset = ViLegalDataset(
                test_path,
                self.tokenizer,
                max_length=self.conf.max_length,
                max_target_length=self.conf.max_target_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=True
        ) 