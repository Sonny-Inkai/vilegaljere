#!/usr/bin/env python3
"""
Demo script for Vietnamese Legal Joint Entity-Relation Extraction
Usage: python demo.py --model_path /path/to/trained/model
"""

import os
import sys
import argparse
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import extract_vilegal_triplets, format_example_for_display

def load_model(model_path):
    """Load trained model and tokenizer"""
    print(f"🤖 Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)
    
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"🔤 Vocabulary size: {len(tokenizer)}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer

def predict(text, model, tokenizer, max_length=512, num_beams=3):
    """Generate prediction for input text"""
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Generate prediction
    with torch.no_grad():
        generated_tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=False,
            length_penalty=0,
            use_cache=True
        )
    
    # Decode prediction
    prediction = tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
    prediction = prediction.replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip()
    
    return prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model")
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_beams", default=3, type=int)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    print("\n🚀 Vietnamese Legal Joint Entity-Relation Extraction Demo")
    print("="*80)
    print("📝 Enter Vietnamese legal text to extract entities and relations")
    print("💡 Type 'quit' or 'exit' to stop")
    print("="*80)
    
    # Interactive demo
    while True:
        print("\n📄 Enter legal text:")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not text:
            print("⚠️ Please enter some text")
            continue
        
        print("\n🔮 Generating prediction...")
        
        try:
            prediction = predict(text, model, tokenizer, args.max_length, args.num_beams)
            
            print("\n" + "="*80)
            print("📄 INPUT:")
            print(text)
            print("\n🎯 PREDICTED OUTPUT:")
            print(prediction)
            
            # Extract and display triplets
            triplets = extract_vilegal_triplets(prediction)
            print(f"\n📊 EXTRACTED TRIPLETS ({len(triplets)}):")
            if triplets:
                for i, (head_type, head_text, tail_type, tail_text, relation) in enumerate(triplets, 1):
                    print(f"  {i}. {head_type}: '{head_text}' --[{relation}]--> {tail_type}: '{tail_text}'")
            else:
                print("  No triplets extracted")
            
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")

# Sample examples for testing
SAMPLE_EXAMPLES = [
    "Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam 1. Nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài khi tham gia đầu tư, hoạt động trên thị trường chứng khoán Việt Nam tuân thủ quy định về tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư theo quy định của pháp luật về chứng khoán và thị trường chứng khoán.",
    
    "Chính phủ quy định chi tiết tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư, việc tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam.",
    
    "Theo Luật Doanh nghiệp 2020, doanh nghiệp có trách nhiệm tuân thủ các quy định về bảo vệ môi trường và phát triển bền vững."
]

def demo_with_samples():
    """Run demo with sample texts"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model")
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_beams", default=3, type=int)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    print("\n🎯 Running demo with sample Vietnamese legal texts")
    print("="*80)
    
    for i, text in enumerate(SAMPLE_EXAMPLES, 1):
        print(f"\n--- SAMPLE {i} ---")
        
        try:
            prediction = predict(text, model, tokenizer, args.max_length, args.num_beams)
            format_example_for_display(text, "", prediction)
            
            input("\nPress Enter to continue to next sample...")
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and "--samples" in sys.argv:
        sys.argv.remove("--samples")
        demo_with_samples()
    else:
        main() 