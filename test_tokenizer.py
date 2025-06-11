from transformers import AutoTokenizer
import torch

# Test texts
text1 = "<LEGAL_PROVISION> Điều 90 <RIGHT/DUTY> phải chịu trách nhiệm hình sự <Relates_To> <LEGAL_PROVISION> Điều 90 <LEGAL_PROVISION> Bộ luật hình sự <Relates_To>"
text2 = "<LEGAL_PROVISION> Điều 96 <ORGANIZATION> Tòa án <Relates_To> <RIGHT/DUTY> phải chấp hành đầy đủ những nghĩa vụ về học tập, học nghề, lao động, sinh hoạt <ORGANIZATION> nhà trường <Relates_To>"

# Tokenizer models to test
models = [
    "sonny36/vilegaljere",
    "VietAI/vit5-base", 
    "google/mt5-base"
]

def test_tokenizer(model_name, text, text_label):
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"Text: {text_label}")
    print(f"{'='*80}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        print(f"Original text: {text}")
        print(f"\nNumber of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print(f"\nToken IDs: {token_ids}")
        
        # Decode back to check
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"Decoded back: {decoded}")
        
  
    except Exception as e:
        print(f"Error with {model_name}: {e}")

def main():
    print("Testing Vietnamese Legal Text Tokenization")
    print("=" * 80)
    
    texts = [
        (text1, "Text 1 - Điều 90"),
        (text2, "Text 2 - Điều 96")
    ]
    
    for model_name in models:
        for text, label in texts:
            test_tokenizer(model_name, text, label)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    for text, label in texts:
        print(f"\n{label}: {text}")
        for model_name in models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokens = tokenizer.tokenize(text)
                print(f"  {model_name}: {len(tokens)} tokens")
            except:
                print(f"  {model_name}: ERROR")

if __name__ == "__main__":
    main()
