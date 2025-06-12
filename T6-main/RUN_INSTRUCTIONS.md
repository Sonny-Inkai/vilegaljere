# 🚀 HƯỚNG DẪN CHẠY ViLegalJERE

## 🔧 Các vấn đề đã được sửa:

### ❌ **Vấn đề trước đây:**
1. **Tokenizer thiếu special tokens** → Model không hiểu `<ORGANIZATION>`, `<LEGAL_PROVISION>`, etc.
2. **Vocab size mismatch** → Model và tokenizer không khớp
3. **Sai format parsing** → Extraction function không parse đúng format

### ✅ **Các fix đã áp dụng:**
1. **Thêm đúng special tokens vào tokenizer**
2. **Resize model embeddings để match tokenizer**
3. **Parse đúng format: `<HEAD_TYPE> head_text <TAIL_TYPE> tail_text <RELATION>`**
4. **Optimize generation parameters**

---

## 📋 TRÌNH TỰ CHẠY:

### **Bước 1: Test Tokenizer**
```bash
cd T6-main
python test_tokenizer.py
```
**Kỳ vọng:** Tất cả special tokens hiển thị ✅ OK

### **Bước 2: Test Minimal Training Setup**
```bash
python test_training.py
```
**Kỳ vọng:** ALL TESTS PASSED!

### **Bước 3: Chạy Training**
```bash
# Pre-training (nếu muốn)
python train_vilegal_jere.py

# Hoặc chỉ Fine-tuning
# Đảm bảo finetune = True trong script
python train_vilegal_jere.py
```

### **Bước 4: Test Generation**
```bash
python generate.py
```

---

## 🎯 Format dữ liệu của cháu:

**Input (Context):**
```
Điều 51: Tham gia của nhà đầu tư nước ngoài trên thị trường chứng khoán Việt Nam...
```

**Output (Relations):**
```
<ORGANIZATION> nhà đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To> <LEGAL_PROVISION> pháp luật về chứng khoán <LOCATION> thị trường chứng khoán Việt Nam <Relates_To>
```

**Cấu trúc:** `<HEAD_TYPE> head_text <TAIL_TYPE> tail_text <RELATION_TYPE>`

---

## 🚨 Troubleshooting:

### Nếu vẫn generate vô nghĩa:
1. **Check tokenizer:** Run `test_tokenizer.py` xem special tokens có ✅ không
2. **Check model vocab:** Đảm bảo `model.config.vocab_size == len(tokenizer)`
3. **Check data format:** Xem sample trong training log có đúng format không

### Nếu training bị lỗi:
1. **Memory error:** Giảm `batch_size` từ 32 → 16
2. **CUDA error:** Set `device = 'cpu'` để test
3. **Data loading error:** Check `data_path` và file `finetune.json`

---

## 📊 Expected Results:

**Trước khi sửa:**
```
🤖 MODEL GENERATED:
,,, nghiệp hợp doanh cơ thông đất từ bù hợp hộ Tòa có được quy trong theo quy
```

**Sau khi sửa (kỳ vọng):**
```
🤖 MODEL GENERATED:
<ORGANIZATION> nhà đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To>
🎯 Extracted 1 relations:
   1. nhà đầu tư nước ngoài (ORGANIZATION) --Relates_To--> thị trường chứng khoán Việt Nam (LOCATION)
```

---

## 🎯 Key Changes Made:

1. **train_vilegal_jere.py:**
   - ✅ Added domain special tokens to tokenizer
   - ✅ Removed unnecessary triplet tokens
   - ✅ Fixed data loading format

2. **generate.py:**
   - ✅ Fixed tokenizer setup
   - ✅ Improved relation extraction parsing
   - ✅ Better generation parameters

3. **model/ViLegalJERE.py:**
   - ✅ Proper token embedding resizing
   - ✅ Correct attention mask handling

**Chạy ngay để test nhé cháu! 🚀** 