#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script để kiểm tra token IDs có nằm trong range không
"""

import torch
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

def debug_token_ids():
    """Debug token ID ranges"""
    print("🔍 DEBUGGING TOKEN ID RANGES")
    print("=" * 60)
    
    # Setup tokenizer  
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    print(f"Original tokenizer vocab size: {len(tokenizer)}")
    
    # Add special tokens
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    num_added = tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    print(f"Added {num_added} new tokens")
    print(f"New tokenizer vocab size: {len(tokenizer)}")
    
    # Test input
    test_text = "Điều 51: Tham gia của nhà đầu tư nước ngoài"
    
    # Tokenize
    tokens = tokenizer(test_text, return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    
    print(f"\n🧪 TEST TEXT: {test_text}")
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Max token ID: {input_ids.max().item()}")
    print(f"Min token ID: {input_ids.min().item()}")
    
    # Check if token IDs are within range
    max_id = input_ids.max().item()
    vocab_size = len(tokenizer)
    
    if max_id >= vocab_size:
        print(f"❌ TOKEN ID OUT OF RANGE!")
        print(f"   Max token ID: {max_id}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Token IDs >= vocab_size: {(input_ids >= vocab_size).sum().item()}")
        
        # Show which tokens are problematic
        for i, token_id in enumerate(input_ids.tolist()):
            if token_id >= vocab_size:
                print(f"   Position {i}: token_id={token_id} >= vocab_size={vocab_size}")
    else:
        print(f"✅ All token IDs are within range [0, {vocab_size-1}]")
    
    # ✅ FIX: Test with correct vocab size
    print(f"\n🔧 CREATING MODEL WITH EXACT VOCAB SIZE...")
    config = ViLegalConfig(vocab_size=len(tokenizer))
    model = ViLegalJERE(config)
    
    print(f"Model embedding vocab size: {model.shared.num_embeddings}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Test embedding lookup
    try:
        with torch.no_grad():
            embeddings = model.shared(input_ids)
        print(f"✅ Embedding lookup successful!")
        print(f"Embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Embedding lookup failed: {e}")
    
    # Test special tokens
    print(f"\n🧪 TESTING SPECIAL TOKENS:")
    for token in domain_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"   {token}: {token_id}")
        
        if token_id >= len(tokenizer):
            print(f"   ❌ OUT OF RANGE!")
        else:
            print(f"   ✅ OK")

if __name__ == "__main__":
    debug_token_ids() 