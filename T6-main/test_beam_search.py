#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test beam search specifically
"""

import torch
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

def test_beam_search():
    """Test beam search generation"""
    print("🧪 TESTING BEAM SEARCH")
    print("=" * 50)
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    
    config = ViLegalConfig(vocab_size=len(tokenizer))
    model = ViLegalJERE(config)
    model.eval()
    
    # Test input
    test_text = "Điều 1: Phạm vi điều chỉnh"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    
    print(f"Testing with: {test_text}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # Test different beam sizes
    for num_beams in [1, 2, 3]:
        print(f"\n🔧 Testing num_beams={num_beams}")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=15,
                    do_sample=False,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                print(f"✅ num_beams={num_beams} successful!")
                print(f"Output shape: {outputs.shape}")
                print(f"Output tokens: {outputs[0].tolist()}")
                
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
                print(f"Decoded: {decoded}")
                
        except Exception as e:
            print(f"❌ num_beams={num_beams} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_beam_search() 