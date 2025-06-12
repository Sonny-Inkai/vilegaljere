#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script để kiểm tra vocab size mismatch giữa model và tokenizer
"""

import torch
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE
import os

def debug_vocab_size():
    """Debug vocab size issues"""
    print("🔍 DEBUGGING VOCAB SIZE MISMATCH")
    print("=" * 60)
    
    # 1. Check tokenizer vocab size
    print("1️⃣ Checking tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    print(f"Original tokenizer vocab size: {len(tokenizer)}")
    
    # Add special tokens
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    print(f"Tokenizer vocab size after adding special tokens: {len(tokenizer)}")
    
    # 2. Check model path và config
    model_path = "/kaggle/working/out_vilegal_t5small"
    if not os.path.exists(model_path):
        print(f"❌ Model path không tồn tại: {model_path}")
        
        # Try local path
        model_path = "out_vilegal_t5small"
        if not os.path.exists(model_path):
            print(f"❌ Local model path cũng không tồn tại: {model_path}")
            print("🔧 Tạo model config mới với vocab size đúng...")
            
            # Create new model with correct vocab size
            config = ViLegalConfig(vocab_size=len(tokenizer))
            model = ViLegalJERE(config)
            print(f"✅ Created new model with vocab size: {config.vocab_size}")
            return model, tokenizer
        else:
            print(f"✅ Found local model path: {model_path}")
    else:
        print(f"✅ Found model path: {model_path}")
    
    # 3. Try to load model config
    try:
        print("\n2️⃣ Checking saved model config...")
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            print(f"Saved model vocab size: {config_dict.get('vocab_size', 'NOT FOUND')}")
        else:
            print(f"❌ Config file không tồn tại: {config_path}")
    except Exception as e:
        print(f"❌ Error reading config: {e}")
    
    # 4. Try to load model
    try:
        print("\n3️⃣ Attempting to load model...")
        model = ViLegalJERE.from_pretrained(model_path)
        print(f"✅ Loaded model successfully")
        print(f"Model embedding size: {model.shared.num_embeddings}")
        
        # Check if we need to resize
        if model.shared.num_embeddings != len(tokenizer):
            print(f"🔧 VOCAB SIZE MISMATCH DETECTED!")
            print(f"   Model: {model.shared.num_embeddings}")
            print(f"   Tokenizer: {len(tokenizer)}")
            print(f"   Resizing model embeddings...")
            
            model.resize_token_embeddings(len(tokenizer))
            print(f"✅ Resized to {len(tokenizer)}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"🔧 Creating new model with correct vocab size...")
        
        # Create new model with correct vocab size
        config = ViLegalConfig(vocab_size=len(tokenizer))
        model = ViLegalJERE(config)
        print(f"✅ Created new model with vocab size: {config.vocab_size}")
        return model, tokenizer

def test_model_generation(model, tokenizer):
    """Test generation with corrected model"""
    print("\n4️⃣ Testing generation...")
    
    # Simple test
    test_text = "Điều 1: Phạm vi điều chỉnh Bộ luật lao động"
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids'][0][:10].tolist()}")  # First 10 tokens
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Simple forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]])
            )
        print(f"✅ Forward pass successful!")
        print(f"Logits shape: {outputs['logits'].shape}")
        
        # Test generation
        try:
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50,
                do_sample=False  # Use greedy first
            )
            print(f"✅ Generation successful!")
            print(f"Generated shape: {generated.shape}")
            
            # Decode
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Generated text: {decoded}")
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
    
    except Exception as e:
        print(f"❌ Forward pass error: {e}")

if __name__ == "__main__":
    try:
        model, tokenizer = debug_vocab_size()
        test_model_generation(model, tokenizer)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc() 