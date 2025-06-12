import torch
from torch import nn

def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """Shift input ids one token to the left, used for T5."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id
    return shifted_input_ids

def extract_vietnamese_legal_triplets(text: str, special_tokens: list):
    """
    Extracts triplets from a generated string based on the custom format:
    <HEAD_TYPE> Head_Text <TAIL_TYPE> Tail_Text <RELATION_TYPE>
    """
    triplets = []
    
    # Create a regex pattern to find all special tokens
    # This also helps split the string by these tokens
    token_pattern = f"({'|'.join(special_tokens)})"
    
    # Split the text by the special tokens, keeping the tokens
    parts = [p.strip() for p in re.split(token_pattern, text) if p.strip()]
    if not parts:
        return []

    # The text is expected to be a sequence of:
    # [HEAD_TYPE, Head_Text, TAIL_TYPE, Tail_Text, RELATION_TYPE]
    # We iterate through the parts list in chunks of 5.
    i = 0
    while i < len(parts):
        chunk = parts[i : i + 5]
        if len(chunk) == 5:
            head_type, head_text, tail_type, tail_text, rel_type = chunk
            
            # Basic validation: ensure the types are actual special tokens
            if (head_type in special_tokens and
                tail_type in special_tokens and
                rel_type in special_tokens and
                head_type.startswith('<') and
                tail_type.startswith('<') and
                rel_type.startswith('<')):
                
                triplets.append({
                    'head_type': head_type,
                    'head': head_text,
                    'tail_type': tail_type,
                    'tail': tail_text,
                    'type': rel_type,
                })
        i += 5 # Move to the next potential triplet

    return triplets

import re

def extract_vietnamese_legal_triplets(text: str, special_tokens: list):
    """
    Extracts triplets from a generated string based on the custom format:
    <HEAD_TYPE> Head_Text <TAIL_TYPE> Tail_Text <RELATION_TYPE>
    This is a stream of triplets.
    """
    triplets = []
    
    # Create a regex pattern that matches any of the special tokens
    # This is used to split the string while keeping the delimiters (the tokens)
    pattern = f"({'|'.join(map(re.escape, special_tokens))})"
    
    # Split the string by the tokens. Filter out empty strings that can result from the split.
    parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
    
    # We process the list of parts in chunks. Each valid triplet consists of 5 parts:
    # [HEAD_TYPE, Head_Text, TAIL_TYPE, Tail_Text, RELATION_TYPE]
    i = 0
    while i <= len(parts) - 5:
        chunk = parts[i : i + 5]
        
        head_type = chunk[0]
        head_text = chunk[1]
        tail_type = chunk[2]
        tail_text = chunk[3]
        relation  = chunk[4]
        
        # Validate that the extracted types are indeed special tokens
        # and that the text parts are not special tokens
        is_chunk_valid = (
            head_type in special_tokens and
            tail_type in special_tokens and
            relation in special_tokens and
            head_text not in special_tokens and
            tail_text not in special_tokens
        )
        
        if is_chunk_valid:
            triplets.append({
                'head_type': head_type,
                'head': head_text,
                'tail_type': tail_type,
                'tail': tail_text,
                'type': relation,
            })
            # If valid, advance by 5 to look for the next triplet
            i += 5
        else:
            # If the chunk is not a valid triplet, advance by 1 to find the start of a new potential triplet
            i += 1
            
    return triplets 