from transformers import AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')

print(f"Vocab size: {len(tokenizer)}")
print(f"Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
print(f"EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")

# Check sentinel
try:
    extra_id_0 = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    print(f"<extra_id_0>: {extra_id_0}")
except Exception as e:
    print(f"Error: {e}")

print("Test complete!") 