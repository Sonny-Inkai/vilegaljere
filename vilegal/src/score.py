import numpy as np
from collections import defaultdict


def score(predicted_triplets, actual_triplets, mode='strict'):
    """
    Score predicted triplets against actual triplets
    
    Args:
        predicted_triplets: List of predicted triplets
        actual_triplets: List of actual triplets
        mode: 'strict' or 'boundaries'
    
    Returns:
        Dictionary with precision, recall, F1 scores
    """
    if not predicted_triplets and not actual_triplets:
        return {'micro': {'p': 1.0, 'r': 1.0, 'f1': 1.0}}
    
    if not predicted_triplets:
        return {'micro': {'p': 0.0, 'r': 0.0, 'f1': 0.0}}
    
    if not actual_triplets:
        return {'micro': {'p': 0.0, 'r': 0.0, 'f1': 0.0}}
    
    # Convert triplets to comparable format
    predicted_set = set()
    actual_set = set()
    
    for triplet in predicted_triplets:
        if isinstance(triplet, dict):
            key = (triplet.get('head', '').strip(), 
                   triplet.get('type', '').strip(), 
                   triplet.get('tail', '').strip())
            predicted_set.add(key)
    
    for triplet in actual_triplets:
        if isinstance(triplet, dict):
            key = (triplet.get('head', '').strip(), 
                   triplet.get('type', '').strip(), 
                   triplet.get('tail', '').strip())
            actual_set.add(key)
    
    # Calculate intersection
    intersection = predicted_set.intersection(actual_set)
    
    # Calculate metrics
    if len(predicted_set) == 0:
        precision = 0.0
    else:
        precision = len(intersection) / len(predicted_set)
    
    if len(actual_set) == 0:
        recall = 0.0
    else:
        recall = len(intersection) / len(actual_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'micro': {
            'p': precision,
            'r': recall,
            'f1': f1
        }
    }


def re_score(predicted_triplets, actual_triplets, mode='strict'):
    """
    Alternative scoring function for relation extraction
    """
    return score(predicted_triplets, actual_triplets, mode) 