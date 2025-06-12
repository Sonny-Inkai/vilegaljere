#!/usr/bin/env python3
"""
Vietnamese Legal Joint Entity and Relation Extraction Demo Script
Interactive demo for testing the trained model
"""

import os
import argparse
import logging
from typing import Dict, List, Tuple, Set
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViLegalDemo:
    """Demo class for Vietnamese Legal Joint Entity and Relation Extraction"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Model loaded successfully on {self.device}")
    
    def generate_relations(self, input_text: str, max_length: int = 256, num_beams: int = 4) -> str:
        """Generate relations for given input text"""
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded
    
    def parse_relations(self, relation_text: str) -> List[Dict]:
        """
        Parse relation text into structured triplets
        Format: <Entity_Type> Entity_Text <Entity_Type> Entity_Text <Relation_Type>
        """
        triplets = []
        
        # Enhanced regex pattern to handle Vietnamese legal domain
        pattern = r'<([^>]+)>\s*([^<]+?)\s*<([^>]+)>\s*([^<]+?)\s*<([^>]+)>'
        
        matches = re.findall(pattern, relation_text)
        
        for match in matches:
            entity1_type, entity1_text, entity2_type, entity2_text, relation_type = match
            
            # Clean text
            entity1_text = entity1_text.strip()
            entity2_text = entity2_text.strip()
            
            if entity1_text and entity2_text and relation_type:
                triplet = {
                    'subject': {
                        'type': entity1_type,
                        'text': entity1_text
                    },
                    'relation': relation_type,
                    'object': {
                        'type': entity2_type,
                        'text': entity2_text
                    }
                }
                triplets.append(triplet)
        
        return triplets
    
    def extract_relations(self, input_text: str) -> Dict:
        """Extract relations from input text and return structured output"""
        # Generate relations
        raw_output = self.generate_relations(input_text)
        
        # Parse relations
        triplets = self.parse_relations(raw_output)
        
        return {
            'input_text': input_text,
            'raw_output': raw_output,
            'extracted_triplets': triplets,
            'num_triplets': len(triplets)
        }
    
    def print_results(self, results: Dict):
        """Pretty print extraction results"""
        print("\n" + "="*80)
        print("🏛️  VIETNAMESE LEGAL RELATION EXTRACTION RESULTS")
        print("="*80)
        
        print(f"\n📖 INPUT TEXT:")
        input_text = results['input_text']
        if len(input_text) > 200:
            print(f"  {input_text[:200]}...")
        else:
            print(f"  {input_text}")
        
        print(f"\n🤖 RAW MODEL OUTPUT:")
        print(f"  {results['raw_output']}")
        
        print(f"\n🔗 EXTRACTED RELATIONS ({results['num_triplets']} triplets):")
        
        if results['extracted_triplets']:
            for i, triplet in enumerate(results['extracted_triplets'], 1):
                print(f"\n  {i}. Subject: [{triplet['subject']['type']}] {triplet['subject']['text']}")
                print(f"     Relation: {triplet['relation']}")
                print(f"     Object: [{triplet['object']['type']}] {triplet['object']['text']}")
        else:
            print("  No relations extracted.")
        
        print("\n" + "="*80)

def interactive_demo(demo: ViLegalDemo):
    """Run interactive demo"""
    print("\n🏛️  Vietnamese Legal Joint Entity and Relation Extraction Demo")
    print("Enter Vietnamese legal text to extract entities and relations.")
    print("Type 'quit' or 'exit' to end the demo.\n")
    
    while True:
        try:
            # Get input from user
            user_input = input("📖 Enter Vietnamese legal text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                print("❌ Please enter some text.")
                continue
            
            # Extract relations
            print("\n🔄 Processing...")
            results = demo.extract_relations(user_input)
            
            # Print results
            demo.print_results(results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_demo(demo: ViLegalDemo, input_file: str, output_file: str):
    """Run batch demo on file"""
    import json
    
    logger.info(f"Processing file: {input_file}")
    
    results_list = []
    
    if input_file.endswith('.json'):
        # Process JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item_id, item_data in data.items():
            input_text = item_data.get('formatted_context_sent', '')
            if input_text:
                logger.info(f"Processing item: {item_id}")
                results = demo.extract_relations(input_text)
                results['id'] = item_id
                results_list.append(results)
    
    else:
        # Process plain text file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                logger.info(f"Processing line {i+1}")
                results = demo.extract_relations(line)
                results['id'] = f"line_{i+1}"
                results_list.append(results)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Results saved to {output_file}")
    logger.info(f"Processed {len(results_list)} items")

def main():
    parser = argparse.ArgumentParser(description='Vietnamese Legal Joint Entity and Relation Extraction Demo')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive',
                        help='Demo mode: interactive or batch')
    parser.add_argument('--input_file', type=str,
                        help='Input file for batch mode')
    parser.add_argument('--output_file', type=str,
                        help='Output file for batch mode')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams for generation')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum generation length')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize demo
    demo = ViLegalDemo(args.model_path, device)
    
    if args.mode == 'interactive':
        interactive_demo(demo)
    
    elif args.mode == 'batch':
        if not args.input_file or not args.output_file:
            print("❌ Batch mode requires --input_file and --output_file arguments")
            return
        
        if not os.path.exists(args.input_file):
            print(f"❌ Input file not found: {args.input_file}")
            return
        
        batch_demo(demo, args.input_file, args.output_file)

# Example usage and sample texts
SAMPLE_TEXTS = [
    """Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam 1. Nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài khi tham gia đầu tư, hoạt động trên thị trường chứng khoán Việt Nam tuân thủ quy định về tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư theo quy định của pháp luật về chứng khoán và thị trường chứng khoán.""",
    
    """Điều 173: Trách nhiệm của Kiểm soát viên 1. Tuân thủ đúng pháp luật, Điều lệ công ty, nghị quyết Đại hội đồng cổ đông và đạo đức nghề nghiệp trong thực hiện quyền và nghĩa vụ được giao.""",
    
    """Điều 109: Báo cáo thực trạng quản trị công ty bao gồm các thông tin sau đây: a) Thông tin về cơ quan đại diện chủ sở hữu, người đứng đầu và cấp phó của người đứng đầu cơ quan đại diện chủ sở hữu; b) Thông tin về người quản lý công ty."""
]

if __name__ == '__main__':
    if len(os.sys.argv) == 1:
        # Show help and examples if no arguments provided
        print("🏛️  Vietnamese Legal Joint Entity and Relation Extraction Demo")
        print("\nUsage:")
        print("  python demo.py --model_path /path/to/model --mode interactive")
        print("  python demo.py --model_path /path/to/model --mode batch --input_file data.json --output_file results.json")
        print("\nSample Vietnamese legal texts:")
        for i, text in enumerate(SAMPLE_TEXTS, 1):
            print(f"\n{i}. {text[:100]}...")
    else:
        main() 