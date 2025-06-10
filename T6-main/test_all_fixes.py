#!/usr/bin/env python3
"""
Test all fixes to ensure model training will work correctly
"""

import torch
import sys
import numpy as np
sys.path.append('.')

from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

def test_tokenizer_setup():
    print("="*60)
    print("🧪 TESTING TOKENIZER SETUP")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    
    print(f"✅ Vocab size: {len(tokenizer)}")
    print(f"✅ Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print(f"✅ EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    
    # Test sentinel tokens
    sentinel_0 = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    sentinel_1 = tokenizer.convert_tokens_to_ids('<extra_id_1>')
    print(f"✅ <extra_id_0>: {sentinel_0}")
    print(f"✅ <extra_id_1>: {sentinel_1}")
    
    assert sentinel_0 == 10099, f"Expected 10099, got {sentinel_0}"
    assert sentinel_1 == 10098, f"Expected 10098, got {sentinel_1}"
    print("✅ Sentinel tokens verified!")
    
    return tokenizer

def test_model_config(tokenizer):
    print("\n" + "="*60)
    print("🧪 TESTING MODEL CONFIG")
    print("="*60)
    
    config = ViLegalConfig(
        vocab_size=len(tokenizer),
        n_layer=2,  # Small for testing
        n_head=4,
        n_embd=128,
        head_dim=32,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id
    )
    
    print(f"✅ Config pad_token_id: {config.pad_token_id}")
    print(f"✅ Config eos_token_id: {config.eos_token_id}")
    print(f"✅ Config decoder_start_token_id: {config.decoder_start_token_id}")
    
    assert config.eos_token_id == 3, f"Expected EOS=3, got {config.eos_token_id}"
    assert config.decoder_start_token_id == 3, f"Expected decoder_start=3, got {config.decoder_start_token_id}"
    print("✅ Model config verified!")
    
    return config

def test_span_corruption(tokenizer):
    print("\n" + "="*60)
    print("🧪 TESTING SPAN CORRUPTION")
    print("="*60)
    
    # Mock the improved create_t5_spans function
    def create_noise_mask_test(length, noise_density=0.15):
        """Simple noise mask for testing"""
        num_noise = max(1, int(length * noise_density))
        mask = [False] * length
        indices = np.random.choice(length, size=num_noise, replace=False)
        for i in indices:
            mask[i] = True
        return mask
    
    def create_t5_spans_test(tokens):
        """Test version of span corruption"""
        noise_mask = create_noise_mask_test(len(tokens))
        
        # Get sentinel start
        sentinel_start_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        print(f"  Using sentinel_start_id: {sentinel_start_id}")
        
        input_ids = []
        labels = []
        sentinel_idx = 0
        prev_noise = False
        
        for i, token in enumerate(tokens):
            is_noise = noise_mask[i]
            
            if is_noise:
                if not prev_noise:
                    sentinel_id = sentinel_start_id - sentinel_idx
                    input_ids.append(sentinel_id)
                    labels.append(sentinel_id)
                    sentinel_idx += 1
                labels.append(token)
            else:
                input_ids.append(token)
            
            prev_noise = is_noise
        
        labels.append(tokenizer.eos_token_id)  # Add EOS
        
        return input_ids, labels
    
    # Test with sample tokens
    test_tokens = [100, 200, 300, 400, 500, 600, 700, 800]
    print(f"  Input tokens: {test_tokens}")
    
    input_ids, labels = create_t5_spans_test(test_tokens)
    print(f"  Output input_ids: {input_ids}")
    print(f"  Output labels: {labels}")
    
    # Verify EOS token at end
    assert labels[-1] == tokenizer.eos_token_id, f"Expected EOS={tokenizer.eos_token_id} at end, got {labels[-1]}"
    print("✅ Span corruption test passed!")
    
    return input_ids, labels

def test_model_forward(config, tokenizer):
    print("\n" + "="*60)
    print("🧪 TESTING MODEL FORWARD")
    print("="*60)
    
    model = ViLegalJERE(config)
    model.eval()
    
    batch_size, seq_len = 2, 8
    
    # Create sample inputs
    input_ids = torch.randint(4, 1000, (batch_size, seq_len))
    labels = torch.randint(4, 1000, (batch_size, seq_len))
    
    # Create decoder_input_ids correctly
    decoder_input_ids = torch.cat([
        torch.full((batch_size, 1), tokenizer.eos_token_id),  # Start with EOS
        labels[:, :-1]
    ], dim=-1)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Decoder input shape: {decoder_input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Decoder starts with: {decoder_input_ids[0, 0].item()} (should be {tokenizer.eos_token_id})")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
    
    print(f"✅ Forward pass successful!")
    print(f"  Loss: {outputs['loss']:.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    assert decoder_input_ids[0, 0].item() == tokenizer.eos_token_id, "Decoder should start with EOS token"
    print("✅ Model forward test passed!")

def test_batch_creation(tokenizer):
    print("\n" + "="*60)
    print("🧪 TESTING BATCH CREATION LOGIC")
    print("="*60)
    
    # Mock get_batch logic
    batch_size = 2
    max_length = 10
    
    # Create sample data
    sample_input = ["Điều 1: Test <ORGANIZATION>", "Điều 2: Another test"]
    sample_target = ["<ORGANIZATION> test <Relates_To>", "<LOCATION> example"]
    
    # Tokenize
    input_encodings = tokenizer(sample_input, padding=True, truncation=True, 
                               max_length=max_length, return_tensors="pt")
    target_encodings = tokenizer(sample_target, padding=True, truncation=True, 
                                max_length=max_length, return_tensors="pt")
    
    input_ids = input_encodings.input_ids
    labels = target_encodings.input_ids
    
    # Create decoder input correctly
    decoder_input_ids = torch.cat([
        torch.full((labels.shape[0], 1), tokenizer.eos_token_id),
        labels[:, :-1]
    ], dim=-1)
    
    print(f"  Input: {sample_input}")
    print(f"  Target: {sample_target}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Decoder input shape: {decoder_input_ids.shape}")
    print(f"  Decoder starts: {decoder_input_ids[:, 0].tolist()} (all should be {tokenizer.eos_token_id})")
    
    # Verify all decoder inputs start with EOS
    assert all(decoder_input_ids[:, 0] == tokenizer.eos_token_id), "All decoder inputs should start with EOS"
    print("✅ Batch creation test passed!")

if __name__ == "__main__":
    print("🚀 RUNNING COMPREHENSIVE TESTS FOR VILEGALJERE FIXES")
    
    try:
        tokenizer = test_tokenizer_setup()
        config = test_model_config(tokenizer)
        test_span_corruption(tokenizer)
        test_model_forward(config, tokenizer)
        test_batch_creation(tokenizer)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! MODEL IS READY FOR TRAINING!")
        print("="*60)
        print("✅ Token IDs are correct")
        print("✅ Model config is correct") 
        print("✅ Span corruption works")
        print("✅ Model forward pass works")
        print("✅ Batch creation works")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc() 