import re
import torch
import torch.nn.functional as F
from typing import List, Tuple

# Vietnamese Legal domain special tokens
DOMAIN_SPECIAL_TOKENS = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
    "<Relates_To>", "<Amended_By>"
]

def shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids to the left."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id
    return shifted_input_ids

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """Compute label smoothed cross entropy loss."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    
    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def extract_vilegal_triplets(text: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Extract triplets from Vietnamese Legal text format:
    <Loại_Head> Text_Head <Loại_Tail> Text_Tail <Loại_Quan_Hệ>
    
    Returns: List of (head_type, head_text, tail_type, tail_text, relation)
    """
    triplets = []
    
    # Clean text
    text = text.replace('</s>', '').replace('<s>', '').strip()
    
    # Pattern to match: <TYPE> text <TYPE> text <RELATION>
    pattern = r'<([^>]+)>\s*([^<]*?)\s*<([^>]+)>\s*([^<]*?)\s*<([^>]+)>'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        head_type, head_text, tail_type, tail_text, relation = match
        
        # Clean extracted parts
        head_type = head_type.strip()
        head_text = head_text.strip()
        tail_type = tail_type.strip()
        tail_text = tail_text.strip()
        relation = relation.strip()
        
        # Skip if any part is empty
        if all([head_type, head_text, tail_type, tail_text, relation]):
            triplets.append((head_type, head_text, tail_type, tail_text, relation))
    
    return triplets

def setup_tokenizer_for_vilegal(tokenizer):
    """Add domain-specific special tokens to tokenizer."""
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': DOMAIN_SPECIAL_TOKENS}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"✅ Added {num_added_toks} domain-specific tokens")
    print(f"📝 Special tokens: {DOMAIN_SPECIAL_TOKENS}")
    
    return tokenizer, num_added_toks

def print_model_info(model, tokenizer):
    """Print model and tokenizer information."""
    print(f"🤖 Model: {model.config.name_or_path if hasattr(model.config, 'name_or_path') else 'Unknown'}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔤 Vocabulary size: {len(tokenizer)}")
    print(f"🎯 Model type: {model.config.model_type}")

def format_example_for_display(input_text, target_text, predicted_text=None):
    """Format training/test examples for display."""
    print("\n" + "="*80)
    print("📄 INPUT:")
    print(input_text)
    print("\n🎯 TARGET:")
    print(target_text)
    
    if predicted_text:
        print("\n🔮 PREDICTED:")
        print(predicted_text)
        
        # Extract and compare triplets
        target_triplets = extract_vilegal_triplets(target_text)
        pred_triplets = extract_vilegal_triplets(predicted_text)
        
        print(f"\n📊 TRIPLETS COMPARISON:")
        print(f"Target ({len(target_triplets)}): {target_triplets}")
        print(f"Predicted ({len(pred_triplets)}): {pred_triplets}")
        
        # Calculate match
        target_set = set(target_triplets)
        pred_set = set(pred_triplets)
        matches = target_set.intersection(pred_set)
        print(f"✅ Matches ({len(matches)}): {matches}")
    
    print("="*80)

def compute_rouge_scores(predictions, references):
    """Compute ROUGE scores for generated text."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        
        # Average scores
        avg_scores = {}
        for key in scores:
            avg_scores[key] = sum(scores[key]) / len(scores[key])
        
        return avg_scores
    except ImportError:
        print("⚠️ rouge_score not installed. Skipping ROUGE computation.")
        return {}

def validate_data_format(data):
    """Validate Vietnamese Legal dataset format."""
    required_fields = ['formatted_context_sent', 'extracted_relations_text']
    
    if isinstance(data, dict):
        # Check if it's a nested dict structure
        sample_key = list(data.keys())[0]
        sample = data[sample_key]
    else:
        sample = data[0]
    
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing required field: {field}")
    
    print("✅ Data format validation passed")
    return True 