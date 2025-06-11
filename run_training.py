#!/usr/bin/env python3
"""
Simple script to run Vietnamese Legal NER/RE training in Kaggle environment
"""

import subprocess
import sys
import os


def run_training():
    """Run the training script"""
    print("Starting Vietnamese Legal NER/RE training...")
    print("="*50)
    
    # Import and run training
    from train_viet_legal import main
    main()

def run_evaluation(model_path=None):
    """Run evaluation after training"""
    if model_path is None:
        model_path = "/kaggle/working/vit5-base/final_model"
    
    test_data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
    
    if os.path.exists(model_path) and os.path.exists(test_data_path):
        print("\nRunning evaluation...")
        print("="*50)
        
        from evaluate_viet_legal import VietLegalEvaluator
        
        # Initialize evaluator
        evaluator = VietLegalEvaluator(model_path)
        
        # Run evaluation
        metrics, _, _ = evaluator.evaluate_dataset(test_data_path, verbose=True)
        
        # Print results
        print("\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        print(f"Triplet F1 Score: {metrics['f1']:.4f}")
        print(f"Entity F1 Score: {metrics['entity_f1']:.4f}")
        print(f"Relation F1 Score: {metrics['relation_f1']:.4f}")
        print("="*50)
    else:
        print(f"Model path {model_path} or test data {test_data_path} not found!")

if __name__ == "__main__":
    
    
    # Run training
    run_training()
    
    # Run evaluation
    run_evaluation()
    
    print("\nAll tasks completed!") 