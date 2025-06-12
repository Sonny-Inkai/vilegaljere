#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINIMAL TEST SCRIPT - Debug generation step by step
"""

import torch
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

def minimal_test():
    """Test generation step by step"""
    print("🧪 MINIMAL GENERATION TEST")
    print("=" * 50)
    
    # Setup tokenizer
    print("1️⃣ Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
    
    # Create model
    print("\n2️⃣ Creating model...")
    config = ViLegalConfig(vocab_size=len(tokenizer))
    model = ViLegalJERE(config)
    model.eval()
    print(f"✅ Model vocab size: {model.config.vocab_size}")
    
    # Test input
    print("\n3️⃣ Testing tokenization...")
    test_text = "Điều 1: Phạm vi điều chỉnh"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    print(f"Input text: {test_text}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Token range: [{inputs['input_ids'].min()}, {inputs['input_ids'].max()}]")
    
    # Test forward pass
    print("\n4️⃣ Testing forward pass...")
    try:
        with torch.no_grad():
            # Test encoder
            encoder_outputs = model.encode(inputs['input_ids'], inputs['attention_mask'])
            print(f"✅ Encoder OK: {encoder_outputs.last_hidden_state.shape}")
            
            # Test decoder with proper start token
            decoder_start_token = torch.tensor([[model.config.decoder_start_token_id]])
            print(f"Decoder start token: {decoder_start_token}")
            
            decoder_outputs = model.decode(
                decoder_start_token,
                encoder_outputs.last_hidden_state,
                encoder_attention_mask=inputs['attention_mask']
            )
            print(f"✅ Decoder OK: {decoder_outputs.logits.shape}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test simple generation (step by step)
    print("\n5️⃣ Testing simple generation...")
    try:
        with torch.no_grad():
            # Encode
            encoder_outputs = model.encode(inputs['input_ids'], inputs['attention_mask'])
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # Initialize decoder input
            decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
            print(f"Initial decoder input: {decoder_input_ids}")
            
            generated_tokens = [model.config.decoder_start_token_id]
            
            # Generate one token at a time
            for step in range(5):  # Only 5 steps for testing
                print(f"\n   Step {step + 1}:")
                print(f"   Decoder input: {decoder_input_ids}")
                
                # Decode
                decoder_outputs = model.decode(
                    decoder_input_ids,
                    encoder_hidden_states,
                    encoder_attention_mask=inputs['attention_mask']
                )
                
                # Get next token
                logits = decoder_outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                print(f"   Next token: {next_token.item()}")
                
                # Check token range
                if next_token.item() >= len(tokenizer):
                    print(f"   ❌ Token {next_token.item()} >= vocab_size {len(tokenizer)}")
                    break
                
                # Add to sequence
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    print(f"   ✅ EOS token reached")
                    break
                    
            print(f"✅ Generation successful!")
            print(f"Generated tokens: {generated_tokens}")
            
            # Decode
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            print(f"Generated text: {generated_text}")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test using model.generate()
    print("\n6️⃣ Testing model.generate()...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=10,  # Very short for testing
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            print(f"✅ model.generate() successful!")
            print(f"Output shape: {outputs.shape}")
            print(f"Output tokens: {outputs[0].tolist()}")
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(f"Decoded: {decoded}")
            
    except Exception as e:
        print(f"❌ model.generate() failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check beam search specifically
        print(f"\n🔧 Checking if issue is with beam search...")
        try:
            # Test with very simple generation
            simple_outputs = model._greedy_sample_generate(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=inputs['attention_mask'],
                max_length=10,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id,
                no_repeat_ngram_size=0
            )
            print(f"✅ Greedy generation works!")
            print(f"Simple output: {simple_outputs[0].tolist()}")
            
        except Exception as e2:
            print(f"❌ Even greedy generation fails: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    minimal_test() 