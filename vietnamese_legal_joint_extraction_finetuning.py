#!/usr/bin/env python3
"""
Vietnamese Legal Joint Extraction & Relation Extraction Fine-tuning
Using VietAI/vit5-base model

Inspired by Tony Stark's engineering approach - modular, efficient, and precise!

Task: Extract entities and relations from Vietnamese legal text
Format: <Entity_Type> Entity_Text <Entity_Type> Entity_Text <Relation_Type>
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score
import rouge_score
from rouge_score import rouge_scorer
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration - Tony Stark style modularity"""
    model_name: str = "VietAI/vit5-base"
    max_input_length: int = 512
    max_output_length: int = 1024
    learning_rate: float = 3e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 100
    
class VietnameseLegalDataset(Dataset):
    """Custom dataset for Vietnamese legal text joint extraction"""
    
    def __init__(self, data_file: str, tokenizer: T5Tokenizer, config: ModelConfig, is_train: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        
        # Load and process data
        self.data = self._load_data(data_file)
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
    def _load_data(self, data_file: str) -> List[Dict]:
        """Load data with Tony Stark's attention to detail"""
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        for key, item in raw_data.items():
            # Extract input and target
            input_text = item['formatted_context_sent']
            target_text = item['extracted_relations_text']
            
            # Add task prefix for T5
            input_text = f"Trích xuất thực thể và quan hệ từ văn bản luật: {input_text}"
            
            processed_data.append({
                'input_text': input_text,
                'target_text': target_text,
                'original_id': key
            })
            
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            item['input_text'],
            max_length=self.config.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            item['target_text'],
            max_length=self.config.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'original_id': item['original_id']
        }

class VietnameseLegalJointExtractionTrainer:
    """Main trainer class - the Arc Reactor of our system"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        self.model.to(self.device)
        
        # Add special tokens if needed
        special_tokens = [
            '<LEGAL_PROVISION>', '<ORGANIZATION>', '<PERSON>', '<LOCATION>', 
            '<DATE/TIME>', '<RIGHT/DUTY>', '<RELATION>', 
            '<Relates_To>', '<Applicable_In>', '<Effective_From>'
        ]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model loaded: {config.model_name}")
        logger.info(f"Vocabulary size: {len(self.tokenizer)}")
    
    def create_datasets(self, train_file: str, eval_file: str = None):
        """Create training and evaluation datasets"""
        train_dataset = VietnameseLegalDataset(train_file, self.tokenizer, self.config, is_train=True)
        
        eval_dataset = None
        if eval_file and os.path.exists(eval_file):
            eval_dataset = VietnameseLegalDataset(eval_file, self.tokenizer, self.config, is_train=False)
            
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics - precision like Tony's engineering"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE scores for text generation quality
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        # Entity extraction metrics
        exact_matches = 0
        total_predictions = len(decoded_preds)
        
        for pred, label in zip(decoded_preds, decoded_labels):
            # ROUGE scores
            scores = scorer.score(label, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
            
            # Exact match
            if pred.strip() == label.strip():
                exact_matches += 1
        
        # Calculate average scores
        metrics = {
            'exact_match': exact_matches / total_predictions,
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        return metrics
    
    def train(self, train_dataset, eval_dataset, output_dir: str):
        """Train the model with Tony Stark's methodical approach"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='exact_match',
            greater_is_better=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            fp16=True if torch.cuda.is_available() else False,
            push_to_hub=False,
            report_to="none"  # Disable default wandb integration
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Starting training - Let's build something amazing!")
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        return trainer
    
    def evaluate_model(self, model_path: str, eval_file: str):
        """Evaluate the fine-tuned model"""
        logger.info("Loading model for evaluation...")
        
        # Load fine-tuned model
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        # Load evaluation dataset
        eval_dataset = VietnameseLegalDataset(eval_file, tokenizer, self.config, is_train=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.config.batch_size)
        
        predictions = []
        references = []
        
        logger.info("Running evaluation...")
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Generate predictions
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode predictions
                batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # Calculate metrics
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        exact_match_score = exact_matches / len(predictions)
        
        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        # Print results
        results = {
            'exact_match': exact_match_score,
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL']),
            'total_samples': len(predictions)
        }
        
        logger.info("Evaluation Results:")
        for metric, score in results.items():
            if metric != 'total_samples':
                logger.info(f"{metric}: {score:.4f}")
            else:
                logger.info(f"{metric}: {score}")
        
        return results, predictions, references

def main():
    """Main execution function - Tony Stark's workshop in action"""
    
    # Configuration
    config = ModelConfig()
    
    # File paths - Kaggle environment
    data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning"
    finetune_file_name = "finetune.json"
    train_file = os.path.join(data_path, finetune_file_name)
    eval_file = "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
    
    # Output directory
    model_name_safe = config.model_name.replace("/", "_")
    out_dir = f'/kaggle/working/{model_name_safe}_vietnamese_legal_joint_extraction'
    
    logger.info("🚀 Starting Vietnamese Legal Joint Extraction Fine-tuning")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Train file: {train_file}")
    logger.info(f"Eval file: {eval_file}")
    logger.info(f"Output directory: {out_dir}")
    
    # Initialize trainer
    trainer = VietnameseLegalJointExtractionTrainer(config)
    
    # Create datasets
    train_dataset, eval_dataset = trainer.create_datasets(train_file, eval_file)
    
    # Train model
    trained_model = trainer.train(train_dataset, eval_dataset, out_dir)
    
    # Evaluate model
    results, predictions, references = trainer.evaluate_model(out_dir, eval_file)
    
    # Save evaluation results
    eval_output = {
        'metrics': results,
        'sample_predictions': [
            {'input': train_dataset.data[i]['input_text'][:200] + "...", 
             'predicted': predictions[i], 
             'reference': references[i]}
            for i in range(min(10, len(predictions)))
        ]
    }
    
    with open(os.path.join(out_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)
    
    logger.info("🎯 Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {out_dir}")
    logger.info("Ready for deployment - Tony Stark style! 🔥")

if __name__ == "__main__":
    main() 