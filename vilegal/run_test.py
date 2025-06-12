#!/usr/bin/env python3
"""
Simple test script for Vietnamese Legal Joint Entity-Relation Extraction
Usage: python run_test.py --model_path /path/to/trained/model
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.test import main

if __name__ == "__main__":
    # Set default arguments for Kaggle environment
    import sys
    import argparse
    
    # Parse only the model_path first
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model")
    args, remaining = parser.parse_known_args()
    
    default_args = [
        "--model_path", args.model_path,
        "--test_data_path", "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json",
        "--output_dir", "/kaggle/working/test_results",
        "--batch_size", "4",
        "--eval_beams", "3",
        "--max_length", "512",
        "--max_target_length", "512",
        "--gpus", "1",
        "--seed", "42",
        "--show_examples", "5"
    ]
    
    # Use default args combined with any additional provided args
    sys.argv = [sys.argv[0]] + default_args + remaining
    
    print("🧪 Using default test configuration for Kaggle environment")
    print(f"🤖 Testing model: {args.model_path}")
    print("📝 Override with command line arguments if needed")
    
    main() 