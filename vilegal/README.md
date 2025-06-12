# Vietnamese Legal Joint Entity and Relation Extraction

## 🏛️ Overview

This project implements a Vietnamese Legal Joint Entity and Relation Extraction system based on the REBEL approach, but adapted for Vietnamese legal documents using VietAI/vit5-base model.

### Key Features

- **Vietnamese Legal Domain**: Specialized for Vietnamese legal text processing
- **Joint Extraction**: Simultaneously extracts entities and relations in one pass
- **T5-based Architecture**: Uses VietAI/vit5-base as the backbone model
- **Domain-Specific Tokens**: Custom tokens for legal entities and relations
- **Text-to-Text Generation**: Converts relation extraction to text generation task

## 🎯 Domain-Specific Tokens

The model uses the following special tokens for Vietnamese legal domain:

```
<ORGANIZATION>    - Legal organizations (Chính phủ, Tòa án, etc.)
<LOCATION>        - Geographic locations and jurisdictions
<DATE/TIME>       - Temporal expressions and effective dates
<LEGAL_PROVISION> - Legal provisions, articles, and clauses
<RIGHT/DUTY>      - Rights, duties, and obligations
<PERSON>          - Individual persons and roles
<Effective_From>  - Effective date relations
<Applicable_In>   - Jurisdiction relations
<Relates_To>      - General relation type
<Amended_By>      - Amendment relations
```

## 📊 Data Format

### Input Format
Vietnamese legal text (article content)

### Output Format
The model generates structured relations in the format:
```
<Entity_Type> Entity_Text <Entity_Type> Entity_Text <Relation_Type>
```

Example:
```
<ORGANIZATION> Chính phủ <LEGAL_PROVISION> Điều 51 <Relates_To>
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd vilegal

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train.py \
    --train_path /kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json \
    --val_path /kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json \
    --output_dir /kaggle/working/vilegal-t5 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 10
```

### Evaluation

```bash
python evaluate.py \
    --model_path /kaggle/working/vilegal-t5/final_model \
    --test_data /kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json \
    --output_dir /kaggle/working/evaluation_results
```

### Demo

```bash
# Interactive demo
python demo.py --model_path /kaggle/working/vilegal-t5/final_model --mode interactive

# Batch processing
python demo.py \
    --model_path /kaggle/working/vilegal-t5/final_model \
    --mode batch \
    --input_file input.json \
    --output_file results.json
```

## 📁 Project Structure

```
vilegal/
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── demo.py               # Demo script
├── requirements.txt      # Dependencies
├── README.md            # This file
├── finetune.json        # Training data
└── test.json           # Test data
```

## 🔧 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `VietAI/vit5-base` | Pretrained model |
| `--batch_size` | `8` | Training batch size |
| `--learning_rate` | `1e-4` | Learning rate |
| `--num_epochs` | `10` | Number of epochs |
| `--max_source_length` | `512` | Max input length |
| `--max_target_length` | `256` | Max output length |

### Model Architecture

- **Base Model**: VietAI/vit5-base
- **Task**: Text-to-text generation
- **Objective**: Joint entity and relation extraction
- **Language**: Vietnamese
- **Domain**: Legal documents

## 📈 Evaluation Metrics

The evaluation script provides:

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1-Score**: Harmonic mean of precision and recall
- **Micro-averaged**: Overall metrics across all relations
- **Macro-averaged**: Average metrics per sample

## 🎯 Use Cases

1. **Legal Document Analysis**: Extract key entities and their relationships
2. **Compliance Checking**: Identify regulatory requirements and obligations
3. **Legal Research**: Find connections between legal provisions
4. **Document Summarization**: Extract structured information from legal texts

## 📝 Sample Usage

```python
from demo import ViLegalDemo

# Initialize model
demo = ViLegalDemo('/path/to/model')

# Extract relations
text = """Điều 51: Tham gia của nhà đầu tư nước ngoài trên thị trường chứng khoán Việt Nam"""
results = demo.extract_relations(text)

# Print results
demo.print_results(results)
```

## 🔍 Example Outputs

### Input
```
Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam tuân thủ quy định của pháp luật về chứng khoán.
```

### Output
```
<ORGANIZATION> tổ chức kinh tế có vốn đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To> <LEGAL_PROVISION> pháp luật về chứng khoán và thị trường chứng khoán <LOCATION> thị trường chứng khoán Việt Nam <Relates_To>
```

### Parsed Relations
1. **Subject**: [ORGANIZATION] tổ chức kinh tế có vốn đầu tư nước ngoài
   **Relation**: Relates_To
   **Object**: [LOCATION] thị trường chứng khoán Việt Nam

2. **Subject**: [LEGAL_PROVISION] pháp luật về chứng khoán và thị trường chứng khoán
   **Relation**: Relates_To  
   **Object**: [LOCATION] thị trường chứng khoán Việt Nam

## 🛠️ Hardware Requirements

### Minimum
- **GPU**: 4GB VRAM (e.g., GTX 1060)
- **RAM**: 8GB
- **Storage**: 5GB free space

### Recommended
- **GPU**: 8GB+ VRAM (e.g., RTX 3070, V100)
- **RAM**: 16GB+
- **Storage**: 10GB+ free space

## 📚 References

1. [REBEL: Relation Extraction By End-to-end Language generation](https://arxiv.org/abs/2104.07650)
2. [VietAI T5 Models](https://huggingface.co/VietAI)
3. [Text-to-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- REBEL authors for the original approach
- VietAI team for the Vietnamese T5 models
- Hugging Face for the transformers library
- PyTorch Lightning for the training framework 