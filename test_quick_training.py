#!/usr/bin/env python3
"""
Quick test script to verify model can learn the Vietnamese legal format
Uses only first 10 samples for fast iteration
"""

import json
import torch
from train_viet_legal import VietLegalModel, VietnameseLegalDataset
from evaluate_viet_legal import VietLegalEvaluator
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os

def create_small_dataset(input_path, output_path, num_samples=10):
    """Create small dataset for quick testing"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = list(data.values())[:num_samples]
        small_data = {f"item_{i}": item for i, item in enumerate(items)}
    else:
        small_data = data[:num_samples]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(small_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created small dataset with {len(small_data)} samples at {output_path}")

def quick_test():
    print("🚀 Starting Quick Test Training...")
    
    # Paths
    data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning"
    test_data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning-test"
    out_dir = '/kaggle/working/vit5-base-quick-test'
    
    # Create small datasets
    os.makedirs(out_dir, exist_ok=True)
    small_train_path = os.path.join(out_dir, 'small_train.json')
    small_test_path = os.path.join(out_dir, 'small_test.json')
    
    create_small_dataset(
        os.path.join(data_path, 'finetune.json'), 
        small_train_path, 
        num_samples=20
    )
    create_small_dataset(
        os.path.join(test_data_path, 'test.json'), 
        small_test_path, 
        num_samples=5
    )
    
    # Initialize model
    model = VietLegalModel(
        model_name="VietAI/vit5-base",
        learning_rate=5e-4,  # Higher LR for quick learning
        warmup_steps=50,
        train_dataset_path=small_train_path,
        val_dataset_path=small_test_path,
        batch_size=2,  # Smaller batch for quick training
        max_length=256  # Shorter sequences
    )
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename='quick-test-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    
    # Trainer for quick test
    trainer = pl.Trainer(
        max_epochs=5,  # Very few epochs for quick test
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        val_check_interval=1.0,  # Validate once per epoch
        enable_progress_bar=True
    )
    
    print("Training on small dataset...")
    trainer.fit(model)
    
    # Save model
    final_model_path = os.path.join(out_dir, 'final_model')
    model.model.save_pretrained(final_model_path)
    model.tokenizer.save_pretrained(final_model_path)
    
    print(f"✅ Quick training completed! Model saved to {final_model_path}")
    
    # Quick evaluation
    print("\n🧪 Running quick evaluation...")
    evaluator = VietLegalEvaluator(final_model_path)
    
    # Test on one sample
    with open(small_test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if isinstance(test_data, dict):
        sample = list(test_data.values())[0]
    else:
        sample = test_data[0]
    
    input_text = sample['formatted_context_sent']
    gold_text = sample['extracted_relations_text']
    
    prediction = evaluator.predict(input_text, num_beams=3)
    
    print("="*80)
    print("QUICK TEST RESULT:")
    print("="*80)
    print(f"Input: {input_text[:200]}...")
    print(f"\nGold: {gold_text}")
    print(f"\nPrediction: {prediction}")
    print("="*80)
    
    # Check if prediction contains special tokens
    special_tokens = ['<LEGAL_PROVISION>', '<ORGANIZATION>', '<LOCATION>', '<DATE/TIME>', '<RIGHT/DUTY>']
    found_tokens = [token for token in special_tokens if token in prediction]
    
    if found_tokens:
        print(f"✅ Model learned to use special tokens: {found_tokens}")
    else:
        print("❌ Model did not learn special tokens yet")
    
    return final_model_path

if __name__ == "__main__":
    quick_test() 