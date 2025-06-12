#!/usr/bin/env python3
"""
Vietnamese Legal Joint Entity and Relation Extraction Training Script
Based on REBEL approach but adapted for VietAI/vit5-base model
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Domain-specific tokens for Vietnamese legal domain
DOMAIN_SPECIAL_TOKENS = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
    "<Relates_To>", "<Amended_By>"
]

class ViLegalDataset(Dataset):
    """Vietnamese Legal Dataset for Joint Entity and Relation Extraction"""
    
    def __init__(self, data_path: str, tokenizer, max_source_length: int = 512, max_target_length: int = 256):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.samples = []
        for item_id, item_data in self.data.items():
            self.samples.append({
                'id': item_id,
                'input_text': item_data['formatted_context_sent'],
                'target_text': item_data['extracted_relations_text']
            })
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input
        source_encoding = self.tokenizer(
            sample['input_text'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            sample['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids but with -100 for padding tokens)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze()
        }

class ViLegalT5Model(pl.LightningModule):
    """Vietnamese Legal T5 Model for Joint Entity and Relation Extraction"""
    
    def __init__(self, model_name: str = "VietAI/vit5-base", learning_rate: float = 1e-4, 
                 warmup_steps: int = 1000, max_steps: int = 10000):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Add domain-specific tokens
        self.tokenizer.add_tokens(DOMAIN_SPECIAL_TOKENS)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"✅ Added {len(DOMAIN_SPECIAL_TOKENS)} domain-specific tokens")
        logger.info(f"✅ Model vocabulary size: {len(self.tokenizer)}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Linear warmup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def generate_relations(self, input_text: str, max_length: int = 256, num_beams: int = 4):
        """Generate relations for given input text"""
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

def create_data_loaders(train_path: str, val_path: str, tokenizer, batch_size: int = 8):
    """Create train and validation data loaders"""
    
    train_dataset = ViLegalDataset(train_path, tokenizer)
    val_dataset = ViLegalDataset(val_path, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese Legal Joint Entity and Relation Extraction Model')
    
    # Data arguments
    parser.add_argument('--train_path', type=str, default='/kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, default='/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json',
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/vilegal-t5',
                        help='Output directory for model checkpoints')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='VietAI/vit5-base',
                        help='Pre-trained model name')
    parser.add_argument('--max_source_length', type=int, default=512,
                        help='Maximum source sequence length')
    parser.add_argument('--max_target_length', type=int, default=256,
                        help='Maximum target sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum number of training steps')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16,
                        help='Training precision (16 or 32)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = ViLegalT5Model(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.train_path,
        args.val_path,
        model.tokenizer,
        args.batch_size
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='vilegal-t5-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.patience,
        verbose=True
    )
    
    # Logger
    logger_tb = TensorBoardLogger(
        save_dir=args.output_dir,
        name='vilegal_logs'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gpus=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger_tb,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        val_check_interval=0.5,
        log_every_n_steps=50
    )
    
    # Train
    logger.info("🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model')
    model.model.save_pretrained(final_model_path)
    model.tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"✅ Training completed! Model saved to {final_model_path}")

if __name__ == '__main__':
    main() 