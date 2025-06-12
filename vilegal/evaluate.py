#!/usr/bin/env python3
"""
Vietnamese Legal Joint Entity and Relation Extraction Evaluation Script
Based on REBEL evaluation approach but adapted for Vietnamese legal domain
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Set
import re
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViLegalEvaluator:
    """Evaluator for Vietnamese Legal Joint Entity and Relation Extraction"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Loaded model from {model_path}")
        logger.info(f"✅ Using device: {self.device}")
    
    def generate_relations(self, input_text: str, max_length: int = 256, num_beams: int = 4) -> str:
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
    
    def parse_relations(self, relation_text: str) -> Set[Tuple[str, str, str]]:
        """
        Parse relation text into triplets
        Format: <Entity_Type> Entity_Text <Entity_Type> Entity_Text <Relation_Type>
        """
        triplets = set()
        
        # Enhanced regex pattern to handle Vietnamese legal domain
        # Pattern: <TYPE> text <TYPE> text <RELATION>
        pattern = r'<([^>]+)>\s*([^<]+?)\s*<([^>]+)>\s*([^<]+?)\s*<([^>]+)>'
        
        matches = re.findall(pattern, relation_text)
        
        for match in matches:
            entity1_type, entity1_text, entity2_type, entity2_text, relation_type = match
            
            # Clean text
            entity1_text = entity1_text.strip()
            entity2_text = entity2_text.strip()
            
            if entity1_text and entity2_text and relation_type:
                triplet = (
                    f"{entity1_type}:{entity1_text}",
                    relation_type,
                    f"{entity2_type}:{entity2_text}"
                )
                triplets.add(triplet)
        
        return triplets
    
    def evaluate_sample(self, input_text: str, gold_relations: str) -> Dict:
        """Evaluate a single sample"""
        # Generate predictions
        pred_relations = self.generate_relations(input_text)
        
        # Parse triplets
        gold_triplets = self.parse_relations(gold_relations)
        pred_triplets = self.parse_relations(pred_relations)
        
        # Calculate metrics
        true_positives = len(gold_triplets & pred_triplets)
        false_positives = len(pred_triplets - gold_triplets)
        false_negatives = len(gold_triplets - pred_triplets)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'gold_triplets': gold_triplets,
            'pred_triplets': pred_triplets,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_text': pred_relations
        }
    
    def evaluate_dataset(self, data_path: str) -> Dict:
        """Evaluate entire dataset"""
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        total_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        logger.info(f"Evaluating {len(data)} samples...")
        
        for item_id, item_data in tqdm(data.items(), desc="Evaluating"):
            input_text = item_data['formatted_context_sent']
            gold_relations = item_data['extracted_relations_text']
            
            sample_result = self.evaluate_sample(input_text, gold_relations)
            sample_result['id'] = item_id
            sample_result['input_text'] = input_text
            sample_result['gold_relations'] = gold_relations
            
            results.append(sample_result)
            
            # Accumulate metrics
            total_metrics['true_positives'] += sample_result['true_positives']
            total_metrics['false_positives'] += sample_result['false_positives']
            total_metrics['false_negatives'] += sample_result['false_negatives']
        
        # Calculate overall metrics
        overall_precision = total_metrics['true_positives'] / (
            total_metrics['true_positives'] + total_metrics['false_positives']
        ) if (total_metrics['true_positives'] + total_metrics['false_positives']) > 0 else 0
        
        overall_recall = total_metrics['true_positives'] / (
            total_metrics['true_positives'] + total_metrics['false_negatives']
        ) if (total_metrics['true_positives'] + total_metrics['false_negatives']) > 0 else 0
        
        overall_f1 = 2 * overall_precision * overall_recall / (
            overall_precision + overall_recall
        ) if (overall_precision + overall_recall) > 0 else 0
        
        # Calculate per-sample averages
        sample_precisions = [r['precision'] for r in results]
        sample_recalls = [r['recall'] for r in results]
        sample_f1s = [r['f1'] for r in results]
        
        return {
            'results': results,
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'true_positives': total_metrics['true_positives'],
                'false_positives': total_metrics['false_positives'],
                'false_negatives': total_metrics['false_negatives']
            },
            'average_metrics': {
                'precision': np.mean(sample_precisions),
                'recall': np.mean(sample_recalls),
                'f1': np.mean(sample_f1s)
            },
            'num_samples': len(results)
        }
    
    def analyze_errors(self, results: List[Dict]) -> Dict:
        """Analyze common error patterns"""
        error_analysis = {
            'missing_relations': [],
            'extra_relations': [],
            'entity_type_errors': [],
            'relation_type_errors': []
        }
        
        for result in results:
            gold_triplets = result['gold_triplets']
            pred_triplets = result['pred_triplets']
            
            # Missing relations (False Negatives)
            missing = gold_triplets - pred_triplets
            for triplet in missing:
                error_analysis['missing_relations'].append({
                    'id': result['id'],
                    'triplet': triplet
                })
            
            # Extra relations (False Positives)
            extra = pred_triplets - gold_triplets
            for triplet in extra:
                error_analysis['extra_relations'].append({
                    'id': result['id'],
                    'triplet': triplet
                })
        
        return error_analysis

def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON"""
    # Convert sets to lists for JSON serialization
    serializable_results = results.copy()
    for result in serializable_results['results']:
        result['gold_triplets'] = list(result['gold_triplets'])
        result['pred_triplets'] = list(result['pred_triplets'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Results saved to {output_path}")

def print_metrics(results: Dict):
    """Print evaluation metrics"""
    overall = results['overall_metrics']
    average = results['average_metrics']
    
    print("\n" + "="*50)
    print("VIETNAMESE LEGAL RELATION EXTRACTION EVALUATION")
    print("="*50)
    
    print(f"\n📊 OVERALL METRICS (Micro-averaged):")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1-Score:  {overall['f1']:.4f}")
    
    print(f"\n📊 AVERAGE METRICS (Macro-averaged):")
    print(f"  Precision: {average['precision']:.4f}")
    print(f"  Recall:    {average['recall']:.4f}")
    print(f"  F1-Score:  {average['f1']:.4f}")
    
    print(f"\n📈 STATISTICS:")
    print(f"  Total Samples:     {results['num_samples']}")
    print(f"  True Positives:    {overall['true_positives']}")
    print(f"  False Positives:   {overall['false_positives']}")
    print(f"  False Negatives:   {overall['false_negatives']}")
    
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Vietnamese Legal Joint Entity and Relation Extraction Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--test_data', type=str, default='/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json',
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/evaluation_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams for generation')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum generation length')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ViLegalEvaluator(args.model_path, device)
    
    # Run evaluation
    logger.info("🚀 Starting evaluation...")
    results = evaluator.evaluate_dataset(args.test_data)
    
    # Print metrics
    print_metrics(results)
    
    # Analyze errors
    logger.info("🔍 Analyzing errors...")
    error_analysis = evaluator.analyze_errors(results['results'])
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    save_results(results, results_path)
    
    # Save error analysis
    error_path = os.path.join(args.output_dir, 'error_analysis.json')
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Error analysis saved to {error_path}")
    
    # Save sample predictions
    sample_predictions = []
    for i, result in enumerate(results['results'][:10]):  # Save first 10 samples
        sample_predictions.append({
            'id': result['id'],
            'input': result['input_text'][:200] + "..." if len(result['input_text']) > 200 else result['input_text'],
            'gold': result['gold_relations'],
            'predicted': result['predicted_text'],
            'f1': result['f1']
        })
    
    samples_path = os.path.join(args.output_dir, 'sample_predictions.json')
    with open(samples_path, 'w', encoding='utf-8') as f:
        json.dump(sample_predictions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Sample predictions saved to {samples_path}")
    logger.info("✅ Evaluation completed!")

if __name__ == '__main__':
    main() 