#!/usr/bin/env python3
"""
Demo script for Vietnamese Legal Relation Extraction
"""

import sys
sys.path.append('src')

from src.test import predict_single

def demo():
    """Demo the model with sample legal text"""
    
    # Sample Vietnamese legal text
    sample_text = """
    Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam
    1. Nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài khi tham gia đầu tư, hoạt động trên thị trường chứng khoán Việt Nam tuân thủ quy định về tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư theo quy định của pháp luật về chứng khoán và thị trường chứng khoán.
    2. Chính phủ quy định chi tiết tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư, việc tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam.
    """
    
    print("Demo: Vietnamese Legal Relation Extraction")
    print("=" * 50)
    print(f"Input text: {sample_text[:200]}...")
    print("\nProcessing...")
    
    try:
        # Use trained model for prediction
        model_path = "/kaggle/working/vietnamese_legal_vit5"
        triplets = predict_single(sample_text, model_path)
        
        print(f"\nExtracted Relations ({len(triplets)} found):")
        print("-" * 30)
        
        for i, triplet in enumerate(triplets, 1):
            print(f"{i}. Head: {triplet['head']}")
            print(f"   Type: {triplet['head_type']}")
            print(f"   Relation: {triplet['type']}")
            print(f"   Tail: {triplet['tail']}")
            print(f"   Tail Type: {triplet['tail_type']}")
            print()
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure the model is trained first!")

if __name__ == "__main__":
    demo() 