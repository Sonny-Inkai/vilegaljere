from transformers import T5Tokenizer

# Khởi tạo tokenizer T5 từ một mô hình đã được huấn luyện trước (ví dụ: 't5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Chuỗi văn bản đầu vào chứa các token đặc biệt
text = "The <extra_id_0> walks in <extra_id_1> park"

# Mã hóa (Encode) chuỗi văn bản
# Việc này sẽ chuyển đổi chuỗi ký tự thành một chuỗi các ID số nguyên
input_ids = tokenizer.encode(text)

# In ra các ID đã được mã hóa
print(f"Chuỗi văn bản gốc: '{text}'")
print(f"Các ID sau khi mã hóa: {input_ids}")

# Giải mã (Decode) các ID trở lại thành chuỗi văn bản
# skip_special_tokens=False để đảm bảo các token đặc biệt được giữ lại trong kết quả
decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)

# In ra chuỗi đã được giải mã
print(f"Chuỗi văn bản sau khi giải mã: '{decoded_text}'")

# Để xem các token cụ thể mà tokenizer đã tách ra
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"Các token tương ứng: {tokens}")