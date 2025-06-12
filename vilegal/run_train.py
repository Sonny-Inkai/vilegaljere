#!/usr/bin/env python3
"""
Simple training script for Vietnamese Legal Joint Entity-Relation Extraction
Usage: python run_train.py
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import main

if __name__ == "__main__":
    # Set default arguments for Kaggle environment
    import sys
    
    default_args = [
        "--model_name_or_path", "VietAI/vit5-base",
        "--data_path", "/kaggle/input/vietnamese-legal-dataset-finetuning", 
        "--finetune_file_name", "finetune.json",
        "--test_data_path", "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json",
        "--output_dir", "/kaggle/working/vilegal-vit5",
        "--batch_size", "64",
        "--gradient_accumulation_steps", "4", 
        "--learning_rate", "3e-5",
        "--num_epochs", "10",
        "--eval_beams", "3",
        "--max_length", "512",
        "--max_target_length", "512",
        "--precision", "16-mixed",
        "--gpus", "1",
        "--seed", "42"
    ]
    
    # Use default args if no command line args provided
    if len(sys.argv) == 1:
        sys.argv.extend(default_args)
        print("🚀 Using default training configuration for Kaggle environment")
        print("📝 Override with command line arguments if needed")
    
    main() 