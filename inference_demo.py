#!/usr/bin/env python3
"""
Inference Demo for Vietnamese Legal Joint Extraction Model
Test your trained model with custom input!
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VietnameseLegalInference:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, text: str, max_length: int = 1024, num_beams: int = 4) -> str:
        """Generate entity-relation extraction prediction"""
        
        # Add task prefix
        input_text = f"Trích xuất thực thể và quan hệ từ văn bản luật: {text}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction

def main():
    parser = argparse.ArgumentParser(description='Vietnamese Legal Joint Extraction Inference')
    parser.add_argument('--model_path', type=str, 
                       default='/kaggle/working/VietAI_vit5-base_vietnamese_legal_joint_extraction',
                       help='Path to trained model')
    parser.add_argument('--text', type=str, help='Input text to process')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    inferencer = VietnameseLegalInference(args.model_path)
    
    if args.interactive:
        print("🚀 Vietnamese Legal Joint Extraction - Interactive Mode")
        print("Enter Vietnamese legal text (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            text = input("\nInput: ").strip()
            
            if text.lower() == 'quit':
                break
                
            if not text:
                continue
            
            try:
                prediction = inferencer.predict(text)
                print(f"\nOutput: {prediction}")
                print("-" * 50)
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.text:
        print(f"Input: {args.text}")
        prediction = inferencer.predict(args.text)
        print(f"Output: {prediction}")
    
    else:
        # Demo with example text
        demo_text = """Điều 1: 01/1999/NĐ-CP của chính phủ số 01/1999/nđ-cp ngày 13 tháng 01 năm 1999 về việc thành lập thị trấn, huyện lỵ huyện dương minh châu, tỉnh tây ninh"""
        
        print("🔥 Demo Mode - Testing with example:")
        print(f"Input: {demo_text}")
        
        prediction = inferencer.predict(demo_text)
        print(f"Output: {prediction}")

if __name__ == "__main__":
    main() 