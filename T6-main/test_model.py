#!/usr/bin/env python3
"""
✅ STANDALONE TEST SCRIPT for ViLegalJERE Model
Test the fine-tuned model for Vietnamese Legal Joint Entity and Relation Extraction

Usage:
    python test_model.py --model_path /path/to/model --test_file /path/to/test.json
"""

import torch
import json
import argparse
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalJERE

def load_custom_tokenizer():
    """Load tokenizer with domain-specific tokens"""
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    
    # Add domain-specific tokens
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    special_tokens_dict = {'additional_special_tokens': domain_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"✅ Added {num_added_toks} domain-specific tokens")
    
    return tokenizer

def test_single_input(model, tokenizer, text, device):
    """Test model with a single input"""
    model.eval()
    
    # Add task prefix
    input_text = f"extract relations: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True, 
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        if hasattr(model, 'generate_relations'):
            outputs = model.generate_relations(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=3,
                early_stopping=True,
                length_penalty=1.0
            )
        else:
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256
            )
    
    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def evaluate_model(model_path, test_file_path=None):
    """Evaluate model performance"""
    print(f"🚀 Loading model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = load_custom_tokenizer()
    
    try:
        model = ViLegalJERE.from_pretrained(model_path)
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"🔧 Using device: {device}")
    print(f"📊 Model vocab size: {model.config.vocab_size}")
    print(f"📊 Tokenizer vocab size: {len(tokenizer)}")
    
    # Test with sample inputs
    test_cases = [
        "Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam 1. Nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài khi tham gia đầu tư, hoạt động trên thị trường chứng khoán Việt Nam tuân thủ quy định về tỷ lệ sở hữu nước ngoài.",
        
        "Điều 173: Trách nhiệm của Kiểm soát viên 1. Tuân thủ đúng pháp luật, Điều lệ công ty, nghị quyết Đại hội đồng cổ đông và đạo đức nghề nghiệp trong thực hiện quyền và nghĩa vụ được giao.",
        
        "Điều 63: Bù trừ và thanh toán giao dịch chứng khoán 1. Hoạt động bù trừ, xác định nghĩa vụ thanh toán tiền và chứng khoán được thực hiện thông qua Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam."
    ]
    
    print(f"\n{'='*80}")
    print("🧪 TESTING MODEL WITH SAMPLE CASES")
    print(f"{'='*80}")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}:")
        print(f"📝 INPUT: {test_case[:100]}...")
        
        try:
            result = test_single_input(model, tokenizer, test_case, device)
            print(f"🤖 OUTPUT: {result}")
            
            # Check for domain tokens
            domain_tokens = ["<ORGANIZATION>", "<LOCATION>", "<LEGAL_PROVISION>", 
                           "<RIGHT/DUTY>", "<PERSON>", "<Relates_To>"]
            found_tokens = [token for token in domain_tokens if token in result]
            
            if found_tokens:
                print(f"✅ Domain tokens found: {found_tokens}")
            else:
                print(f"⚠️ No domain tokens found in output")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    # Load and test from file if provided
    if test_file_path:
        print(f"\n{'='*80}")
        print(f"📁 TESTING WITH FILE: {test_file_path}")
        print(f"{'='*80}")
        
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            correct_predictions = 0
            total_predictions = 0
            
            for key, value in list(test_data.items())[:5]:  # Test first 5 cases
                input_text = value.get("formatted_context_sent", "")
                expected_output = value.get("extracted_relations_text", "")
                
                if input_text and expected_output:
                    print(f"\n🧪 TEST: {key}")
                    print(f"📝 INPUT: {input_text[:100]}...")
                    print(f"🎯 EXPECTED: {expected_output[:100]}...")
                    
                    try:
                        result = test_single_input(model, tokenizer, input_text, device)
                        print(f"🤖 GENERATED: {result[:100]}...")
                        
                        # Simple quality check
                        expected_tokens = ["<ORGANIZATION>", "<LOCATION>", "<LEGAL_PROVISION>", 
                                         "<RIGHT/DUTY>", "<PERSON>", "<Relates_To>"]
                        has_expected_format = any(token in result for token in expected_tokens)
                        
                        if has_expected_format:
                            correct_predictions += 1
                            print("✅ Output has expected format")
                        else:
                            print("❌ Output missing expected format")
                            
                        total_predictions += 1
                        
                    except Exception as e:
                        print(f"❌ Generation failed: {e}")
                        total_predictions += 1
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions * 100
                print(f"\n📊 FORMAT ACCURACY: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            
        except Exception as e:
            print(f"❌ Failed to load test file: {e}")
    
    print(f"\n{'='*80}")
    print("🎯 TESTING COMPLETED")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Test ViLegalJERE Model')
    parser.add_argument('--model_path', type=str, default='/kaggle/working/out_vilegal_t5small',
                      help='Path to the trained model')
    parser.add_argument('--test_file', type=str, default=None,
                      help='Path to test JSON file')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_file)

if __name__ == "__main__":
    main() 