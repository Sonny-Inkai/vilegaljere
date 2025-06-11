#!/usr/bin/env python3
"""
Evaluation Script for Vietnamese Legal Joint Extraction Model
Inspired by Tony Stark's precision testing approach
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityRelationEvaluator:
    """Advanced evaluator for entity-relation extraction"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def extract_triplets(self, text: str) -> List[tuple]:
        """Extract entity-relation triplets from text"""
        triplets = []
        
        # Pattern to match: <TYPE> entity <TYPE> entity <RELATION>
        pattern = r'<([^>]+)>\s+([^<]+?)\s+<([^>]+)>\s+([^<]+?)\s+<([^>]+)>'
        matches = re.findall(pattern, text)
        
        for match in matches:
            head_type, head_text, tail_type, tail_text, relation = match
            triplets.append((
                head_type.strip(),
                head_text.strip(), 
                tail_type.strip(),
                tail_text.strip(),
                relation.strip()
            ))
        
        return triplets
    
    def evaluate_triplets(self, predicted_triplets: List[tuple], 
                         reference_triplets: List[tuple]) -> Dict[str, float]:
        """Evaluate triplet extraction performance"""
        
        # Convert to sets for easier comparison
        pred_set = set(predicted_triplets)
        ref_set = set(reference_triplets)
        
        # Calculate metrics
        true_positives = len(pred_set & ref_set)
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(ref_set) if ref_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'predicted_count': len(pred_set),
            'reference_count': len(ref_set)
        }
    
    def evaluate_entities(self, predicted_triplets: List[tuple], 
                         reference_triplets: List[tuple]) -> Dict[str, float]:
        """Evaluate entity extraction separately"""
        
        # Extract entities (head and tail)
        pred_entities = set()
        ref_entities = set()
        
        for triplet in predicted_triplets:
            pred_entities.add((triplet[0], triplet[1]))  # head_type, head_text
            pred_entities.add((triplet[2], triplet[3]))  # tail_type, tail_text
            
        for triplet in reference_triplets:
            ref_entities.add((triplet[0], triplet[1]))
            ref_entities.add((triplet[2], triplet[3]))
        
        # Calculate metrics
        true_positives = len(pred_entities & ref_entities)
        precision = true_positives / len(pred_entities) if pred_entities else 0
        recall = true_positives / len(ref_entities) if ref_entities else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1,
            'entity_true_positives': true_positives,
            'entity_predicted_count': len(pred_entities),
            'entity_reference_count': len(ref_entities)
        }
    
    def evaluate_relations(self, predicted_triplets: List[tuple], 
                          reference_triplets: List[tuple]) -> Dict[str, float]:
        """Evaluate relation extraction separately"""
        
        # Extract relations
        pred_relations = set(triplet[4] for triplet in predicted_triplets)
        ref_relations = set(triplet[4] for triplet in reference_triplets)
        
        # Calculate metrics
        true_positives = len(pred_relations & ref_relations)
        precision = true_positives / len(pred_relations) if pred_relations else 0
        recall = true_positives / len(ref_relations) if ref_relations else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'relation_precision': precision,
            'relation_recall': recall,
            'relation_f1': f1,
            'relation_true_positives': true_positives,
            'relation_predicted_count': len(pred_relations),
            'relation_reference_count': len(ref_relations)
        }
    
    def predict(self, input_text: str) -> str:
        """Generate prediction for input text"""
        input_text = f"Trích xuất thực thể và quan hệ từ văn bản luật: {input_text}"
        
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=1024,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                temperature=1.0
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
    
    def evaluate_dataset(self, test_file: str) -> Dict:
        """Evaluate on full test dataset"""
        logger.info(f"Loading test data from {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        all_metrics = []
        detailed_results = []
        
        # ROUGE scorer for text similarity
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        logger.info("Starting evaluation...")
        
        for i, (key, item) in enumerate(test_data.items()):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(test_data)} samples")
            
            input_text = item['formatted_context_sent']
            reference_text = item['extracted_relations_text']
            
            # Generate prediction
            prediction_text = self.predict(input_text)
            
            # Extract triplets
            predicted_triplets = self.extract_triplets(prediction_text)
            reference_triplets = self.extract_triplets(reference_text)
            
            # Calculate metrics
            triplet_metrics = self.evaluate_triplets(predicted_triplets, reference_triplets)
            entity_metrics = self.evaluate_entities(predicted_triplets, reference_triplets)
            relation_metrics = self.evaluate_relations(predicted_triplets, reference_triplets)
            
            # ROUGE scores
            rouge_result = rouge_scorer_obj.score(reference_text, prediction_text)
            for metric in rouge_scores:
                rouge_scores[metric].append(rouge_result[metric].fmeasure)
            
            # Combine all metrics
            sample_metrics = {**triplet_metrics, **entity_metrics, **relation_metrics}
            all_metrics.append(sample_metrics)
            
            # Store detailed result
            detailed_results.append({
                'id': key,
                'input': input_text[:200] + "..." if len(input_text) > 200 else input_text,
                'predicted': prediction_text,
                'reference': reference_text,
                'predicted_triplets': predicted_triplets,
                'reference_triplets': reference_triplets,
                'metrics': sample_metrics
            })
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        # Add ROUGE scores
        for metric in rouge_scores:
            avg_metrics[metric] = np.mean(rouge_scores[metric])
        
        # Calculate exact match
        exact_matches = sum(1 for r in detailed_results 
                           if r['predicted'].strip() == r['reference'].strip())
        avg_metrics['exact_match'] = exact_matches / len(detailed_results)
        
        logger.info("Evaluation completed!")
        
        return {
            'average_metrics': avg_metrics,
            'detailed_results': detailed_results[:20],  # Save only first 20 for space
            'total_samples': len(detailed_results)
        }

def main():
    """Main evaluation function"""
    
    # Paths
    model_path = "/kaggle/working/VietAI_vit5-base_vietnamese_legal_joint_extraction"
    test_file = "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
    output_file = "/kaggle/working/detailed_evaluation_results.json"
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Available models in /kaggle/working:")
        for item in os.listdir("/kaggle/working"):
            if os.path.isdir(f"/kaggle/working/{item}"):
                logger.info(f"  - {item}")
        return
    
    # Initialize evaluator
    evaluator = EntityRelationEvaluator(model_path)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_file)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*50)
    
    metrics = results['average_metrics']
    
    logger.info(f"Total Samples: {results['total_samples']}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info("")
    logger.info("Triplet Extraction:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")
    logger.info("")
    logger.info("Entity Extraction:")
    logger.info(f"  Precision: {metrics['entity_precision']:.4f}")
    logger.info(f"  Recall: {metrics['entity_recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['entity_f1']:.4f}")
    logger.info("")
    logger.info("Relation Extraction:")
    logger.info(f"  Precision: {metrics['relation_precision']:.4f}")
    logger.info(f"  Recall: {metrics['relation_recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['relation_f1']:.4f}")
    logger.info("")
    logger.info("ROUGE Scores:")
    logger.info(f"  ROUGE-1: {metrics['rouge1']:.4f}")
    logger.info(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    logger.info(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    
    # Save detailed results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    logger.info("🎯 Evaluation completed successfully! Tony Stark would be proud! 🔥")

if __name__ == "__main__":
    main() 