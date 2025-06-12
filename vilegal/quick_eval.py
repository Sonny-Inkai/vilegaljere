#!/usr/bin/env python3
"""
Quick evaluation script for Vietnamese Legal Relation Extraction
Works with partially trained model
"""

import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add src to Python path
sys.path.append('src')
from src.utils import extract_vietnamese_legal_triplets
from src.score import score

def load_test_data():
    """Load test data"""
    test_file = "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
    if not os.path.exists(test_file):
        test_file = "test.json"  # Fallback to local file
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for key, value in data.items():
        examples.append({
            'input_text': value['formatted_context_sent'],
            'target_text': value['extracted_relations_text']
        })
    
    return examples[:5]  # Only test on first 5 examples for speed

def quick_evaluate():
    """Quick evaluation on a few examples"""
    print("Quick Evaluation: Vietnamese Legal Relation Extraction")
    print("=" * 60)
    
    # Load test data
    test_examples = load_test_data()
    print(f"Loaded {len(test_examples)} test examples")
    
    # Load model and tokenizer
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(
        "VietAI/vit5-base",
        additional_special_tokens=domain_special_tokens
    )
    
    # Try to load trained model, fallback to base model
    model_path = "/kaggle/working/vietnamese_legal_vit5"
    
    try:
        if os.path.exists(f"{model_path}/pytorch_model.bin"):
            print("Loading trained model...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            print("Loading base model (VietAI/vit5-base)...")
            model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    except:
        print("Loading base model (VietAI/vit5-base)...")
        model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded on: {device}")
    print("-" * 60)
    
    # Evaluate examples
    all_predicted = []
    all_actual = []
    
    for i, example in enumerate(test_examples):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input_text'][:100]}...")
        
        # Tokenize input
        inputs = tokenizer(
            example['input_text'],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=3,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode and extract triplets
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        predicted_triplets = extract_vietnamese_legal_triplets(generated_text)
        actual_triplets = extract_vietnamese_legal_triplets(example['target_text'])
        
        all_predicted.append(predicted_triplets)
        all_actual.append(actual_triplets)
        
        print(f"Generated: {generated_text[:150]}...")
        print(f"Predicted triplets: {len(predicted_triplets)}")
        print(f"Actual triplets: {len(actual_triplets)}")
        
        # Score this example
        example_score = score(predicted_triplets, actual_triplets)
        print(f"Example F1: {example_score['micro']['f1']:.3f}")
    
    # Overall evaluation
    print("\n" + "=" * 60)
    print("OVERALL RESULTS:")
    
    total_predicted = sum(len(p) for p in all_predicted)
    total_actual = sum(len(a) for a in all_actual)
    
    # Aggregate scoring
    all_scores = []
    for pred, actual in zip(all_predicted, all_actual):
        example_score = score(pred, actual)
        all_scores.append(example_score['micro'])
    
    if all_scores:
        avg_precision = sum(s['p'] for s in all_scores) / len(all_scores)
        avg_recall = sum(s['r'] for s in all_scores) / len(all_scores)
        avg_f1 = sum(s['f1'] for s in all_scores) / len(all_scores)
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
    
    print(f"Total Predicted Triplets: {total_predicted}")
    print(f"Total Actual Triplets: {total_actual}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'total_predicted': total_predicted,
        'total_actual': total_actual
    }

if __name__ == "__main__":
    results = quick_evaluate() 