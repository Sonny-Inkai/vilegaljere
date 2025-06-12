#!/usr/bin/env python3
"""
Simple script to run evaluation for Vietnamese Legal Relation Extraction
For Kaggle environment
"""

import os
import sys
import subprocess

# Add src to Python path
sys.path.append('src')

def main():
    """Run the evaluation"""
    print("Starting Vietnamese Legal Relation Extraction Evaluation...")
    
    # Change to src directory
    os.chdir('src')
    
    try:
        # Run evaluation
        subprocess.run([
            'python', 'test.py', 
            'checkpoint_path=/kaggle/working/vietnamese_legal_vit5/last.ckpt'
        ], check=True)
        
        print("\n" + "="*50)
        print("Evaluation completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 