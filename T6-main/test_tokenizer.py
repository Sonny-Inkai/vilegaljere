#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra tokenizer có xử lý đúng special tokens không
"""

from transformers import AutoTokenizer

def test_tokenizer():
    """Test special tokens in tokenizer"""
    print("🧪 TESTING TOKENIZER SPECIAL TOKENS")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    print(f"Original vocab size: {len(tokenizer)}")
    
    # Add special tokens
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    print(f"New vocab size: {len(tokenizer)}")
    print(f"Added {len(domain_special_tokens)} special tokens")
    
    # Test tokenization
    print("\n📝 TESTING TOKENIZATION:")
    print("-" * 40)
    
    # Sample text with special tokens
    test_text = "<ORGANIZATION> Chính phủ <LEGAL_PROVISION> Điều 51 <Relates_To>"
    
    # Tokenize
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    
    print(f"Input text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Test each special token
    print("\n🏷️ TESTING INDIVIDUAL SPECIAL TOKENS:")
    print("-" * 50)
    
    for token in domain_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded = tokenizer.decode([token_id])
        
        if token_id == tokenizer.unk_token_id:
            status = "❌ NOT RECOGNIZED (UNK)"
        else:
            status = "✅ OK"
        
        print(f"{token:20} → ID: {token_id:5} → Decoded: '{decoded}' {status}")
    
    # Test sample relations text
    print("\n🎯 TESTING SAMPLE RELATIONS:")
    print("-" * 40)
    
    sample_relations = "<ORGANIZATION> tổ chức kinh tế có vốn đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To>"
    
    encoded = tokenizer.encode(sample_relations)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {sample_relations}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    
    # Check if decode matches original
    if sample_relations.strip() == decoded.strip():
        print("✅ Perfect round-trip encoding/decoding!")
    else:
        print("⚠️ Round-trip encoding/decoding has differences")
    
    print("\n" + "=" * 60)
    return tokenizer

if __name__ == "__main__":
    tokenizer = test_tokenizer() 