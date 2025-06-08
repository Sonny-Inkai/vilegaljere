from transformers import AutoTokenizer

# Tải tokenizer từ Hugging Face Hub
# Thư viện sẽ tự động đọc tất cả các file cấu hình liên quan
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')

# # Print all special tokens with their IDs
# print("=== Special Tokens and their IDs ===")
# print("Special tokens:")
# for token in tokenizer.all_special_tokens:
#     token_id = tokenizer.convert_tokens_to_ids(token)
#     print(f"Token: {token:<20} ID: {token_id}")

# print("\nAdditional special tokens:")
# for token in tokenizer.additional_special_tokens:
#     token_id = tokenizer.convert_tokens_to_ids(token)
#     print(f"Token: {token:<20} ID: {token_id}")

# # Print total vocabulary size
# print(f"\nTotal vocabulary size: {len(tokenizer)}")

# # Phương thức len() trên đối tượng tokenizer sẽ trả về tổng số token
# # bao gồm cả từ điển gốc và các token đặc biệt đã được thêm vào.
# total_tokens = len(tokenizer)

# # In ra kết quả
# print(f"Tổng số token trong tokenizer 'sonny36/vilegaljere' là: {total_tokens}")

# all_special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.additional_special_tokens]
# max_token_id = max([tokenizer.vocab_size - 1] + all_special_token_ids)
# actual_vocab_size = max_token_id + 1

# print(actual_vocab_size)

# a = tokenizer.additional_special_tokens_ids
# print(a)
# print('--------------------------------')
# for i in range(len(a)):
#     print(tokenizer.decode(a[i]))

# # Các token đặc biệt cho lĩnh vực pháp luật
# domain_special_tokens = [
#     "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
#     "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
#     "<Relates_To>", "<Amended_By>"
# ]

# print("=== Legal Domain Tokens and their IDs ===")
# for token in domain_special_tokens:
#     token_id = tokenizer.convert_tokens_to_ids(token)
#     print(f"Token: {token:<20} ID: {token_id}")

print("eos token id:")
print(tokenizer.eos_token_id)
print("pad token id:")
print(tokenizer.pad_token_id)
print("unk token id:")
print(tokenizer.unk_token_id)
print("bos token id:")
print(tokenizer.bos_token_id)
print("eos token:")
print(tokenizer.eos_token)
print("pad token:")
print(tokenizer.pad_token)