from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import os

def batch_iterator(file_path, batch_size=100000):
    """Generator to read text from local file in batches"""
    print(f"Reading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Processing lines"):
            line = line.strip()
            if line:  # Skip empty lines
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:  # Yield remaining lines
            yield batch

def main():
    # Configuration parameters
    DATA_FILE = "dataset.txt"  # Local file in same directory
    EXISTING_TOKENIZER_TEMPLATE = "t5-small"  # Use T5 tokenizer template
    OUTPUT_TOKENIZER_NAME = "sonny36/vilegaljere"  # HuggingFace repo name
    VOCAB_SIZE = 15000  # Standard T5 vocab size
    PUSH_TO_HUB = True
    
    # Domain-specific tokens for Vietnamese legal documents
    DOMAIN_SPECIAL_TOKENS = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    print("=== VIETNAMESE LEGAL T5 TOKENIZER TRAINING ===")
    print(f"Data file: {DATA_FILE}")
    print(f"Base tokenizer: {EXISTING_TOKENIZER_TEMPLATE}")
    print(f"Target vocab size: {VOCAB_SIZE}")
    print(f"Output name: {OUTPUT_TOKENIZER_NAME}")
    print(f"Domain tokens: {len(DOMAIN_SPECIAL_TOKENS)}")
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found!")
        return
    
    # Load base tokenizer (T5-small for special tokens compatibility)
    print(f"\nLoading base tokenizer: {EXISTING_TOKENIZER_TEMPLATE}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(EXISTING_TOKENIZER_TEMPLATE)
        print(f"Base tokenizer loaded successfully. Original vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading base tokenizer: {e}")
        return
    
    # Add domain-specific special tokens
    print(f"\nAdding {len(DOMAIN_SPECIAL_TOKENS)} domain-specific tokens...")
    num_added_tokens = tokenizer.add_special_tokens({
        'additional_special_tokens': DOMAIN_SPECIAL_TOKENS
    })
    print(f"Added {num_added_tokens} new tokens to vocabulary")
    
    # Train new tokenizer from iterator
    print(f"\nTraining new tokenizer with vocab size: {VOCAB_SIZE}")
    print("This may take several minutes...")
    
    try:
        # Train new tokenizer from the text data
        new_tokenizer = tokenizer.train_new_from_iterator(
            text_iterator=batch_iterator(DATA_FILE), 
            vocab_size=VOCAB_SIZE
        )
        print("Tokenizer training completed successfully!")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        return
    
    # Re-add the domain-specific tokens to the new tokenizer
    print("\nRe-adding domain-specific tokens to new tokenizer...")
    new_tokenizer.add_special_tokens({
        'additional_special_tokens': DOMAIN_SPECIAL_TOKENS
    })
    
    # Save tokenizer locally
    local_save_path = "vietnamese_legal_t5_tokenizer"
    print(f"\nSaving tokenizer locally to: {local_save_path}")
    new_tokenizer.save_pretrained(local_save_path)
    
    # Test the tokenizer
    print("\n=== TESTING TOKENIZER ===")
    test_texts = [
        "Điều 1: <LEGAL_PROVISION> về quyền và nghĩa vụ của <PERSON>.",
        "<ORGANIZATION> ban hành quy định <Effective_From> ngày 01/01/2024.",
        "Văn bản này <Relates_To> Luật số 123/2023 và <Amended_By> Nghị định 456."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        tokens = new_tokenizer.tokenize(text)
        ids = new_tokenizer.encode(text)
        decoded = new_tokenizer.decode(ids)
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  IDs: {ids[:10]}{'...' if len(ids) > 10 else ''}")
        print(f"  Decoded: {decoded}")
    
    # Check special tokens
    print(f"\n=== SPECIAL TOKENS CHECK ===")
    print(f"Vocab size: {new_tokenizer.vocab_size}")
    print(f"PAD token: '{new_tokenizer.pad_token}' (ID: {new_tokenizer.pad_token_id})")
    print(f"EOS token: '{new_tokenizer.eos_token}' (ID: {new_tokenizer.eos_token_id})")
    
    # Check domain tokens
    print("\nDomain-specific tokens:")
    for token in DOMAIN_SPECIAL_TOKENS[:5]:  # Show first 5
        token_id = new_tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")
    
    # Check extra_id tokens (T5 specific)
    print("\nT5 extra_id tokens:")
    for i in [0, 50, 99]:
        token = f"<extra_id_{i}>"
        token_id = new_tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")
    
    # Push to HuggingFace Hub
    if PUSH_TO_HUB:
        print(f"\n=== PUSHING TO HUGGINGFACE HUB ===")
        print(f"Uploading to: {OUTPUT_TOKENIZER_NAME}")
        try:
            new_tokenizer.push_to_hub(OUTPUT_TOKENIZER_NAME)
            print(f"✅ Successfully pushed tokenizer to: https://huggingface.co/{OUTPUT_TOKENIZER_NAME}")
        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Make sure you're logged in: huggingface-cli login")
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Local tokenizer saved to: {local_save_path}")
    if PUSH_TO_HUB:
        print(f"Online tokenizer: https://huggingface.co/{OUTPUT_TOKENIZER_NAME}")

if __name__ == "__main__":
    main() 