#!/usr/bin/env python3
"""
Simple test for T5 span corruption function
"""

import numpy as np
from utils import load_custom_tokenizer

def create_noise_mask(length, noise_density, mean_noise_span_length):
    """Create random spans noise mask like Google T5"""
    if noise_density == 0.0:
        return [False] * length
    
    length = max(length, 2)
    
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = max(1, int(round(num_noise_tokens / mean_noise_span_length)))
    num_nonnoise_tokens = length - num_noise_tokens
    
    def random_segmentation(num_items, num_segments):
        if num_segments >= num_items:
            return [1] * num_items
        
        breaks = sorted(np.random.choice(num_items - 1, num_segments - 1, replace=False))
        breaks = [0] + [b + 1 for b in breaks] + [num_items]
        lengths = [breaks[i+1] - breaks[i] for i in range(len(breaks) - 1)]
        return lengths
    
    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)
    
    interleaved_span_lengths = []
    for i in range(num_noise_spans):
        interleaved_span_lengths.append(nonnoise_span_lengths[i])
        interleaved_span_lengths.append(noise_span_lengths[i])
    
    mask = []
    is_noise = False
    for span_length in interleaved_span_lengths:
        mask.extend([is_noise] * span_length)
        is_noise = not is_noise
    
    return mask[:length]

def create_t5_spans(tokens, tokenizer, noise_density=0.15, mean_noise_span_length=3.0):
    """Create T5 span corruption that returns STRING"""
    num_tokens = len(tokens)
    if num_tokens <= 1:
        return "", ""

    noise_mask = create_noise_mask(num_tokens, noise_density, mean_noise_span_length)
    
    sentinel_base_id = tokenizer.convert_tokens_to_ids('<extra_id_0>') 
    
    input_ids = []
    labels = []
    
    prev_token_is_noise = False
    sentinel_idx = 0
    
    for i, token in enumerate(tokens):
        is_noise = noise_mask[i] if i < len(noise_mask) else False
        
        if is_noise:
            if not prev_token_is_noise:
                sentinel_id = sentinel_base_id - sentinel_idx
                input_ids.append(sentinel_id)
                labels.append(sentinel_id)
                sentinel_idx += 1
            labels.append(token)
        else:
            input_ids.append(token)
            
        prev_token_is_noise = is_noise
    
    labels.append(tokenizer.eos_token_id)
    
    input_string = tokenizer.decode(input_ids, skip_special_tokens=False)
    labels_string = tokenizer.decode(labels, skip_special_tokens=False)
    
    return input_string, labels_string

def main():
    print("🧪 Simple T5 Span Test")
    
    # Load tokenizer
    tokenizer = load_custom_tokenizer(master_process=True)
    
    # Test text
    text = "Điều 2 của Luật này quy định về việc thực hiện các biện pháp bảo vệ môi trường"
    print(f"Original: {text}")
    
    # Encode
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"Tokens: {tokens}")
    
    print(f"Decoded: {tokenizer.decode(tokens)}")
    atokens = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Các token tương ứng: {atokens}")
    
    # Create spans
    input_str, label_str = create_t5_spans(tokens, tokenizer, noise_density=0.15)
    print(f"\nCorrupted input: {input_str}")
    print(f"Target output: {label_str}")
    
    # Test encode/decode cycle
    input_tokens = tokenizer.encode(input_str, add_special_tokens=False)
    label_tokens = tokenizer.encode(label_str, add_special_tokens=False)
    
    print(f"\nRe-encoded input: {input_tokens}")
    tokens = tokenizer.convert_ids_to_tokens(input_tokens)
    print(f"Các token tương ứng: {tokens}")
    print(f"Re-encoded label: {label_tokens}")
    tokens = tokenizer.convert_ids_to_tokens(label_tokens)
    print(f"Các token tương ứng: {tokens}")
    
    input_redecoded = tokenizer.decode(input_tokens, skip_special_tokens=False)
    label_redecoded = tokenizer.decode(label_tokens, skip_special_tokens=False)
    
    print(f"\nRe-decoded input: {input_redecoded}")
    print(f"Re-decoded label: {label_redecoded}")
    
    print(f"\n✅ Input consistent: {input_str == input_redecoded}")
    print(f"✅ Label consistent: {label_str == label_redecoded}")

if __name__ == "__main__":
    main() 