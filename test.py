from transformers import AutoTokenizer

# Tải tokenizer từ Hugging Face Hub
# Thư viện sẽ tự động đọc tất cả các file cấu hình liên quan
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')

# Phương thức len() trên đối tượng tokenizer sẽ trả về tổng số token
# bao gồm cả từ điển gốc và các token đặc biệt đã được thêm vào.
total_tokens = len(tokenizer)

# In ra kết quả
print(f"Tổng số token trong tokenizer 'sonny36/vilegaljere' là: {total_tokens}")

# Để kiểm tra token có ID lớn nhất
# ID lớn nhất sẽ là total_tokens - 1
highest_id = total_tokens - 1
token_with_highest_id = tokenizer.convert_ids_to_tokens(highest_id)

print(f"Token có ID lớn nhất là '{token_with_highest_id}' với ID là: {highest_id}")