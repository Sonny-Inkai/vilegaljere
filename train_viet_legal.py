"""
Fine-tuning script for Vietnamese Legal Joint Entity and Relation Extraction
Adapted from REBEL project for VietAI/vit5-base model
Dataset: Vietnamese Legal Dataset
"""

import os
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
import re

class VietnameseLegalDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert dict to list if needed
        if isinstance(self.data, dict):
            self.data = list(self.data.values())
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract input text and target triplets
        input_text = item['formatted_context_sent']
        target_text = item['extracted_relations_text']
        
        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target - using text_target to avoid deprecation warning
        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
            
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

class VietLegalModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        train_dataset_path: str = None,
        val_dataset_path: str = None,
        batch_size: int = 4,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens for Vietnamese legal NER
        special_tokens = [
            '<LEGAL_PROVISION>', '<ORGANIZATION>', '<PERSON>', '<LOCATION>', 
            '<DATE/TIME>', '<RIGHT/DUTY>', '<RELATION>', '<Relates_To>',
            '<Effective_From>', '<Applicable_In>'
        ]
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
        
        # Resize token embeddings to accommodate new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        # Calculate total training steps
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def train_dataloader(self):
        dataset = VietnameseLegalDataset(
            self.hparams.train_dataset_path, 
            self.tokenizer, 
            self.hparams.max_length
        )
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=4
        )
    
    def val_dataloader(self):
        if self.hparams.val_dataset_path:
            dataset = VietnameseLegalDataset(
                self.hparams.val_dataset_path, 
                self.tokenizer, 
                self.hparams.max_length
            )
            return DataLoader(
                dataset, 
                batch_size=self.hparams.batch_size, 
                shuffle=False, 
                num_workers=4
            )
        return None

def extract_vietnamese_triplets(text: str) -> List[tuple]:
    """
    Extract triplets from Vietnamese legal text output
    Format: <ENTITY_TYPE> entity_text <ENTITY_TYPE> entity_text <RELATION_TYPE>
    """
    triplets = []
    
    # Pattern to match the Vietnamese legal format
    pattern = r'<([^>]+)>\s*([^<]+?)\s*<([^>]+)>\s*([^<]+?)\s*<([^>]+)>'
    matches = re.findall(pattern, text)
    
    for match in matches:
        if len(match) == 5:
            head_type, head_text, tail_type, tail_text, relation = match
            triplets.append((
                f"{head_type.strip()}: {head_text.strip()}", 
                f"{tail_type.strip()}: {tail_text.strip()}", 
                relation.strip()
            ))
    
    return triplets

def main():
    # Configuration - exactly as requested
    data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning"
    test_data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning-test"
    finetune_file_name = "finetune.json"  # Your main training file  
    test_file_name = "test.json"  # For validation
    model_name = "VietAI/vit5-base"
    out_dir = '/kaggle/working/vit5-base'  # Using model name as requested
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model
    model = VietLegalModel(
        model_name=model_name,
        learning_rate=5e-5,
        warmup_steps=1000,
        train_dataset_path=os.path.join(data_path, finetune_file_name),
        val_dataset_path=os.path.join(test_data_path, test_file_name),
        batch_size=4,
        max_length=512
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename='vit5-vietnamese-legal-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=out_dir,
        name='vit5-vietnamese-legal'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,  # Effective batch size = 4 * 4 = 16
        precision=16 if torch.cuda.is_available() else 32,
        val_check_interval=0.5,  # Validate twice per epoch
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model)
    
    # Save final model
    model.model.save_pretrained(os.path.join(out_dir, 'final_model'))
    model.tokenizer.save_pretrained(os.path.join(out_dir, 'final_model'))
    
    print(f"Training completed! Model saved to {out_dir}")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main() 
