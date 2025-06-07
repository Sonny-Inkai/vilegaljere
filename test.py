from transformers import AutoTokenizer

# Tải tokenizer từ Hugging Face Hub
# Thư viện sẽ tự động đọc tất cả các file cấu hình liên quan
tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')

# Phương thức len() trên đối tượng tokenizer sẽ trả về tổng số token
# bao gồm cả từ điển gốc và các token đặc biệt đã được thêm vào.
total_tokens = len(tokenizer)

# In ra kết quả
print(f"Tổng số token trong tokenizer 'sonny36/vilegaljere' là: {total_tokens}")

all_special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.additional_special_tokens]
max_token_id = max([tokenizer.vocab_size - 1] + all_special_token_ids)
actual_vocab_size = max_token_id + 1

print(actual_vocab_size)

a = tokenizer.additional_special_tokens_ids
print(a)
print('--------------------------------')
for i in range(len(a)):
    print(tokenizer.decode(a[i]))

print('--------------------------------')
print(a[0])
print(tokenizer.decode(32099))