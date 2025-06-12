import numpy as np

def score(predicted_triplets, actual_triplets):
    """
    Scores predicted triplets against actual triplets.
    The triplets are dictionaries with keys: 'head_type', 'head', 'tail_type', 'tail', 'type'.
    """
    
    # Helper to create a comparable representation of a triplet
    def triplet_to_tuple(triplet):
        return (
            triplet.get('head_type', '').strip(),
            triplet.get('head', '').strip(),
            triplet.get('tail_type', '').strip(),
            triplet.get('tail', '').strip(),
            triplet.get('type', '').strip(),
        )

    # Use sets for efficient comparison
    predicted_set = {triplet_to_tuple(t) for t in predicted_triplets}
    actual_set = {triplet_to_tuple(t) for t in actual_triplets}
    
    # Calculate True Positives, Precision, and Recall
    tp = len(predicted_set.intersection(actual_set))
    
    precision = tp / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = tp / len(actual_set) if len(actual_set) > 0 else 0.0
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'micro': {
            'p': precision,
            'r': recall,
            'f1': f1,
            'tp': tp,
            'pred_count': len(predicted_set),
            'actual_count': len(actual_set),
        }
    }

def re_score(predicted_triplets, actual_triplets):
    """Alias for score function."""
    return score(predicted_triplets, actual_triplets) 