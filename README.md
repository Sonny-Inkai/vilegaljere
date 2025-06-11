# Vietnamese Legal Joint Extraction & Relation Extraction

🚀 **Fine-tuning VietAI/vit5-base for Vietnamese Legal Text Analysis**

Inspired by Tony Stark's engineering precision - this project implements joint entity and relation extraction for Vietnamese legal documents using the VietAI/vit5-base model.

## 📋 Task Description

Extract entities and their relationships from Vietnamese legal text in the format:
```
<Entity_Type> Entity_Text <Entity_Type> Entity_Text <Relation_Type>
```

**Example:**
- Input: "Điều 1: 01/1999/NĐ-CP của chính phủ ngày 13 tháng 01 năm 1999..."
- Output: `<LEGAL_PROVISION> 01/1999/NĐ-CP <DATE/TIME> ngày 13 tháng 01 năm 1999 <Effective_From>`

## 🏗️ Project Structure

```
├── vietnamese_legal_joint_extraction_finetuning.py  # Main training script
├── evaluate_model.py                                # Comprehensive evaluation
├── inference_demo.py                               # Interactive testing
├── requirements.txt                                # Dependencies
└── README.md                                      # This file
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch transformers datasets scikit-learn rouge-score numpy pandas tqdm accelerate
```

## 🚀 Quick Start

### 1. Training

```bash
python vietnamese_legal_joint_extraction_finetuning.py
```

**Configuration:**
- Model: `VietAI/vit5-base`
- Training data: `/kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json`
- Eval data: `/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json`
- Output: `/kaggle/working/VietAI_vit5-base_vietnamese_legal_joint_extraction`

### 2. Evaluation

```bash
python evaluate_model.py
```

**Metrics Calculated:**
- **Triplet Extraction**: Precision, Recall, F1-Score
- **Entity Extraction**: Entity-level metrics
- **Relation Extraction**: Relation-level metrics  
- **ROUGE Scores**: Text generation quality
- **Exact Match**: Perfect prediction accuracy

### 3. Inference Demo

```bash
# Interactive mode
python inference_demo.py --interactive

# Single prediction
python inference_demo.py --text "Your Vietnamese legal text here"

# Demo mode
python inference_demo.py
```

## 📊 Model Configuration

```python
@dataclass
class ModelConfig:
    model_name: str = "VietAI/vit5-base"
    max_input_length: int = 512
    max_output_length: int = 1024
    learning_rate: float = 3e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
```

## 🎯 Entity Types Supported

- `<LEGAL_PROVISION>`: Legal provisions, articles, laws
- `<ORGANIZATION>`: Government bodies, institutions
- `<PERSON>`: Individuals, officials
- `<LOCATION>`: Geographic locations
- `<DATE/TIME>`: Temporal expressions
- `<RIGHT/DUTY>`: Rights and obligations

## 🔗 Relation Types Supported

- `<Relates_To>`: General relationship
- `<Applicable_In>`: Geographic applicability
- `<Effective_From>`: Temporal effectiveness

## 📈 Training Features

✅ **Advanced Data Processing**
- Task-specific prefixing for T5
- Special token integration
- Efficient tokenization

✅ **Robust Training**
- Early stopping
- Learning rate scheduling
- Gradient accumulation
- Mixed precision (FP16)

✅ **Comprehensive Evaluation**
- Multi-metric assessment
- ROUGE scores
- Entity/Relation separation
- Detailed error analysis

## 🎪 Evaluation Metrics

The evaluation script provides comprehensive metrics:

### Triplet-Level Metrics
- **Precision**: Correct triplets / Predicted triplets
- **Recall**: Correct triplets / Reference triplets  
- **F1-Score**: Harmonic mean of precision and recall

### Entity-Level Metrics
- Separate evaluation for entity extraction
- Type-aware entity matching

### Relation-Level Metrics
- Relation type extraction accuracy
- Relation coverage analysis

### Text Generation Metrics
- **ROUGE-1/2/L**: N-gram overlap scores
- **Exact Match**: Perfect prediction rate

## 🔬 Advanced Features

### Data Augmentation
The training script includes:
- Task-specific prompting
- Special token vocabulary expansion
- Context-aware processing

### Evaluation Analysis
- Triplet extraction parsing
- Detailed error analysis
- Sample prediction inspection
- Performance breakdown by metric

### Inference Capabilities
- Beam search generation
- Temperature control
- Interactive testing
- Batch processing support

## 📝 Dataset Format

**Input Format:**
```json
{
  "sample_id": {
    "so_hieu_original": "54/2019/QH14",
    "dieu_original": "Điều 51", 
    "formatted_context_sent": "Legal text content...",
    "extracted_relations_text": "<ENTITY_TYPE> entity <ENTITY_TYPE> entity <RELATION>"
  }
}
```

## 🚨 Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Enable gradient accumulation
- Use FP16 training

### Poor Performance
- Increase training epochs
- Adjust learning rate
- Check data quality
- Verify special tokens

### CUDA Errors
- Verify PyTorch CUDA installation
- Check GPU memory usage
- Reduce model/batch size

## 🎖️ Performance Tips

1. **Memory Optimization**
   - Use gradient checkpointing
   - Enable FP16 training
   - Optimize batch sizes

2. **Training Efficiency**
   - Use learning rate scheduling
   - Enable early stopping
   - Monitor validation metrics

3. **Inference Speed**
   - Use beam search efficiently
   - Batch multiple predictions
   - Consider model distillation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VietAI Team** for the excellent vit5-base model
- **Transformers Library** for the robust framework
- **Tony Stark** for the engineering inspiration ⚡

## 📞 Contact

For questions or support, please open an issue or contact the maintainers.

---

**Built with ❤️ for Vietnamese Legal NLP**

*"Sometimes you gotta run before you can walk" - Tony Stark* 