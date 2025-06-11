# Vietnamese Legal Joint Entity and Relation Extraction

Fine-tuning VietAI/vit5-base model cho task joint extraction và relation extraction với văn bản luật Việt Nam.

## 📁 Files

- `train_viet_legal.py`: Script fine-tuning model VietAI/vit5-base
- `evaluate_viet_legal.py`: Script đánh giá model 
- `run_training.py`: Script chạy tự động training + evaluation
- `requirements.txt`: Dependencies cần thiết

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị data

Đảm bảo data ở đúng vị trí:
- Dataset path: `/kaggle/input/vietnamese-legal-dataset-finetuning/`
- Training file: `finetune.json`
- Test file: `test.json`

### 3. Chạy training

#### Option 1: Chạy tự động (recommended)
```bash
python run_training.py
```

#### Option 2: Chạy từng bước
```bash
# Training
python train_viet_legal.py

# Evaluation  
python evaluate_viet_legal.py --model_path /kaggle/working/vit5-base/final_model --test_data_path /kaggle/input/vietnamese-legal-dataset-finetuning/test.json
```

## 📊 Model Output Format

**Input:** Văn bản luật Việt Nam

**Output:** Triplets dưới dạng
```
<LEGAL_PROVISION> 01/1999/NĐ-CP <DATE/TIME> ngày 13 tháng 01 năm 1999 <Effective_From>
<LEGAL_PROVISION> 01/1999/NĐ-CP <LOCATION> tỉnh Tây Ninh <Applicable_In>
```

## 🎯 Evaluation Metrics

- **Triplet F1**: F1 score cho toàn bộ triplet (entity + relation)
- **Entity F1**: F1 score cho entity extraction
- **Relation F1**: F1 score cho relation classification

## ⚙️ Configuration

Trong `train_viet_legal.py`, các tham số chính:

```python
model_name = "VietAI/vit5-base"
data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning"
finetune_file_name = "finetune.json"
out_dir = '/kaggle/working/vit5-base'
learning_rate = 5e-5
batch_size = 4
max_epochs = 10
```

## 🔍 Tham khảo từ REBEL

Code này được adapt từ REBEL project:
- **Training architecture**: Sử dụng PyTorch Lightning như `src/train.py` và `src/pl_modules.py`
- **Evaluation methodology**: Adapt từ `src/score.py` 
- **Triplet extraction**: Tham khảo `src/utils.py` extract_triplets function
- **Dataset structure**: Tương tự format trong REBEL

## 📈 Expected Results

Model sẽ được lưu tại:
- Checkpoints: `/kaggle/working/vit5-base/`
- Final model: `/kaggle/working/vit5-base/final_model/`
- Evaluation results: `evaluation_results.json`

## 🐛 Troubleshooting

1. **CUDA out of memory**: Giảm `batch_size` từ 4 xuống 2 hoặc 1
2. **Dataset not found**: Kiểm tra đường dẫn dataset
3. **Model loading error**: Đảm bảo có internet để download VietAI/vit5-base

Chúc bạn thành công! 🎉 