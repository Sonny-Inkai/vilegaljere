"""
Evaluation script for Vietnamese Legal Joint Entity and Relation Extraction
Adapted from REBEL project evaluation methodology
"""

import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np

def extract_vietnamese_triplets(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract triplets from Vietnamese legal text output
    Format: <ENTITY_TYPE> entity_text <ENTITY_TYPE> entity_text <RELATION_TYPE>
    Expected format: <LEGAL_PROVISION> 01/1999/NĐ-CP <DATE/TIME> ngày 13 tháng 01 năm 1999 <Effective_From>
    """
    triplets = []
    
    # Updated pattern to match the exact format in your data
    pattern = r'<([^>]+)>\s*([^<]+?)\s*<([^>]+)>\s*([^<]+?)\s*<([^>]+)>'
    matches = re.findall(pattern, text)
    
    for match in matches:
        if len(match) == 5:
            head_type, head_text, tail_type, tail_text, relation = match
            triplets.append((
                f"{head_text.strip()}", 
                f"{tail_text.strip()}", 
                relation.strip()
            ))
    
    return triplets

def normalize_triplet(triplet: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Normalize triplet for comparison"""
    head, tail, relation = triplet
    return (
        head.strip().lower(),
        tail.strip().lower(), 
        relation.strip().lower()
    )

def compute_f1_score(predicted_triplets: List[List[Tuple]], 
                     gold_triplets: List[List[Tuple]], 
                     verbose: bool = False) -> Dict[str, float]:
    """
    Compute F1 score for triplet extraction
    """
    total_predicted = 0
    total_gold = 0
    total_correct = 0
    
    for pred_list, gold_list in zip(predicted_triplets, gold_triplets):
        # Normalize triplets for comparison
        pred_set = set(normalize_triplet(t) for t in pred_list)
        gold_set = set(normalize_triplet(t) for t in gold_list)
        
        total_predicted += len(pred_set)
        total_gold += len(gold_set)
        total_correct += len(pred_set.intersection(gold_set))
        
        if verbose:
            print(f"Predicted: {pred_list}")
            print(f"Gold: {gold_list}")
            print(f"Intersection: {pred_set.intersection(gold_set)}")
            print("---")
    
    # Compute metrics
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_predicted': total_predicted,
        'total_gold': total_gold,
        'total_correct': total_correct
    }

def evaluate_entity_extraction(predicted_triplets: List[List[Tuple]], 
                             gold_triplets: List[List[Tuple]]) -> Dict[str, float]:
    """
    Evaluate entity extraction performance (heads and tails)
    """
    total_predicted_entities = 0
    total_gold_entities = 0
    total_correct_entities = 0
    
    for pred_list, gold_list in zip(predicted_triplets, gold_triplets):
        # Extract entities (heads and tails)
        pred_entities = set()
        gold_entities = set()
        
        for head, tail, _ in pred_list:
            pred_entities.add(head.strip().lower())
            pred_entities.add(tail.strip().lower())
            
        for head, tail, _ in gold_list:
            gold_entities.add(head.strip().lower())
            gold_entities.add(tail.strip().lower())
        
        total_predicted_entities += len(pred_entities)
        total_gold_entities += len(gold_entities)
        total_correct_entities += len(pred_entities.intersection(gold_entities))
    
    precision = total_correct_entities / total_predicted_entities if total_predicted_entities > 0 else 0
    recall = total_correct_entities / total_gold_entities if total_gold_entities > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'entity_precision': precision,
        'entity_recall': recall,
        'entity_f1': f1
    }

def evaluate_relation_classification(predicted_triplets: List[List[Tuple]], 
                                   gold_triplets: List[List[Tuple]]) -> Dict[str, float]:
    """
    Evaluate relation classification (assuming entities are correctly identified)
    """
    total_predicted_relations = 0
    total_gold_relations = 0
    total_correct_relations = 0
    
    for pred_list, gold_list in zip(predicted_triplets, gold_triplets):
        # Create entity pair to relation mapping
        pred_relations = {}
        gold_relations = {}
        
        for head, tail, relation in pred_list:
            key = (head.strip().lower(), tail.strip().lower())
            pred_relations[key] = relation.strip().lower()
            
        for head, tail, relation in gold_list:
            key = (head.strip().lower(), tail.strip().lower())
            gold_relations[key] = relation.strip().lower()
        
        total_predicted_relations += len(pred_relations)
        total_gold_relations += len(gold_relations)
        
        # Count correct relations (same entity pair, same relation)
        for key, relation in pred_relations.items():
            if key in gold_relations and gold_relations[key] == relation:
                total_correct_relations += 1
    
    precision = total_correct_relations / total_predicted_relations if total_predicted_relations > 0 else 0
    recall = total_correct_relations / total_gold_relations if total_gold_relations > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'relation_precision': precision,
        'relation_recall': recall,
        'relation_f1': f1
    }

class VietLegalEvaluator:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, input_text: str, max_length: int = 512, num_beams: int = 5) -> str:
        """Generate prediction for input text"""
        # Add same instruction prefix as training
        input_with_instruction = f"Trích xuất entities và relations từ văn bản luật sau: {input_text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_with_instruction,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate with better parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=20,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.2,  # Reduce repetition
                no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                length_penalty=1.0
            )
        
        # Decode
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
    
    def evaluate_dataset(self, test_data_path: str, verbose: bool = False) -> Dict[str, float]:
        """Evaluate model on test dataset"""
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if isinstance(test_data, dict):
            test_data = list(test_data.values())
        
        predicted_triplets = []
        gold_triplets = []
        
        print(f"Evaluating on {len(test_data)} samples...")
        
        for item in tqdm(test_data):
            input_text = item['formatted_context_sent']
            gold_text = item['extracted_relations_text']
            
            # Get prediction
            prediction = self.predict(input_text)
            
            # Extract triplets
            pred_triplets = extract_vietnamese_triplets(prediction)
            gold_triplets_item = extract_vietnamese_triplets(gold_text)
            
            predicted_triplets.append(pred_triplets)
            gold_triplets.append(gold_triplets_item)
            
            if verbose and len(predicted_triplets) <= 5:  # Show first 5 examples
                print(f"\nInput: {input_text[:200]}...")
                print(f"Gold: {gold_text}")
                print(f"Prediction: {prediction}")
                print(f"Gold triplets: {gold_triplets_item}")
                print(f"Pred triplets: {pred_triplets}")
                print("-" * 100)
        
        # Compute metrics
        metrics = {}
        
        # Overall triplet F1
        triplet_metrics = compute_f1_score(predicted_triplets, gold_triplets, verbose=False)
        metrics.update(triplet_metrics)
        
        # Entity extraction F1
        entity_metrics = evaluate_entity_extraction(predicted_triplets, gold_triplets)
        metrics.update(entity_metrics)
        
        # Relation classification F1
        relation_metrics = evaluate_relation_classification(predicted_triplets, gold_triplets)
        metrics.update(relation_metrics)
        
        return metrics, predicted_triplets, gold_triplets

def main():
    parser = argparse.ArgumentParser(description='Evaluate Vietnamese Legal NER/RE Model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the fine-tuned model directory')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data JSON file')
    parser.add_argument('--output_path', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed evaluation information')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = VietLegalEvaluator(args.model_path)
    
    # Run evaluation
    metrics, predictions, gold = evaluator.evaluate_dataset(
        args.test_data_path, 
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Triplet Extraction:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"\nEntity Extraction:")
    print(f"  Precision: {metrics['entity_precision']:.4f}")
    print(f"  Recall: {metrics['entity_recall']:.4f}")
    print(f"  F1 Score: {metrics['entity_f1']:.4f}")
    print(f"\nRelation Classification:")
    print(f"  Precision: {metrics['relation_precision']:.4f}")
    print(f"  Recall: {metrics['relation_recall']:.4f}")
    print(f"  F1 Score: {metrics['relation_f1']:.4f}")
    print(f"\nDataset Statistics:")
    print(f"  Total Predicted Triplets: {metrics['total_predicted']}")
    print(f"  Total Gold Triplets: {metrics['total_gold']}")
    print(f"  Total Correct Triplets: {metrics['total_correct']}")
    
    # Save results
    results = {
        'metrics': metrics,
        'model_path': args.model_path,
        'test_data_path': args.test_data_path,
        'num_samples': len(predictions)
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main() 
