#!/usr/bin/env python3
"""
Simple script to run Vietnamese Legal Relation Extraction training
For Kaggle environment
"""

import os
import sys
import subprocess

# Add src to Python path
sys.path.append('src')

def main():
    """Run the training"""
    print("Starting Vietnamese Legal Relation Extraction Training...")
    print("Model: VietAI/vit5-base")
    print("Dataset: Vietnamese Legal Documents")
    
    # Change to src directory
    os.chdir('src')
    
    try:
        # Run training
        subprocess.run([
            'python', 'train.py'
        ], check=True)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("Model saved to: /kaggle/working/vietnamese_legal_vit5")
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 