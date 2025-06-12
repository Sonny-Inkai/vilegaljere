#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra training có hoạt động đúng không với minimal setup
"""

import torch
import json
import os
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

def test_minimal_training():
    """Test minimal training setup"""
    print("🧪 TESTING MINIMAL TRAINING SETUP")
    print("=" * 60)
    
    # 1. Setup tokenizer
    print("1️⃣ Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # 2. Setup model
    print("2️⃣ Setting up model...")
    model_args = dict(
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=512,
        bias=False,
        head_dim=64,
        rank=4,
        q_rank=8,
        using_groupnorm=True,
        vocab_size=len(tokenizer),
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id
    )
    
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)
    print(f"   Model vocab size: {model.config.vocab_size}")
    print(f"   Model parameters: {model.get_num_params() / 1e6:.1f}M")
    
    # 3. Test sample data
    print("3️⃣ Testing sample data...")
    
    sample_input = "Điều 51: Tham gia của nhà đầu tư nước ngoài trên thị trường chứng khoán Việt Nam."
    sample_target = "<ORGANIZATION> nhà đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To>"
    
    # Tokenize
    input_ids = tokenizer.encode(sample_input, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    target_ids = tokenizer.encode(sample_target, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    
    # Create decoder input
    decoder_input_ids = torch.cat([torch.full((target_ids.shape[0], 1), tokenizer.eos_token_id), target_ids[:, :-1]], dim=-1)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Target shape: {target_ids.shape}")
    print(f"   Decoder input shape: {decoder_input_ids.shape}")
    
    # 4. Test forward pass
    print("4️⃣ Testing forward pass...")
    
    try:
        with torch.no_grad():
            attention_mask = (input_ids != tokenizer.pad_token_id)
            decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=target_ids
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            print(f"   ✅ Forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Logits shape: {logits.shape}")
            
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # 5. Test generation
    print("5️⃣ Testing generation...")
    
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,
                do_sample=False,
                num_beams=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(generated[0, 1:], skip_special_tokens=False)
            print(f"   ✅ Generation successful!")
            print(f"   Generated: {generated_text}")
            
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED! Setup is ready for training.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_minimal_training() 