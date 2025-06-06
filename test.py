from transformers import AutoTokenizer

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')

# Lấy danh sách các special token (dạng chữ)
special_tokens = tokenizer.all_special_tokens

# Chuyển danh sách đó thành các ID tương ứng
special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

print("Các special token và ID tương ứng:")
# In ra từng cặp token và ID
for token, token_id in zip(special_tokens, special_token_ids):
    print(f"{token}: {token_id}")