#!/bin/bash

# Quick fix for AdamW import error and run training
echo "🔧 Quick Fix: AdamW Import Error"
echo "=================================="

# The main fix is already applied to train.py (AdamW from torch.optim instead of transformers)
echo "✅ AdamW import fixed in train.py"

# Set paths
TRAIN_PATH="/kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json"
VAL_PATH="/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
OUTPUT_DIR="/kaggle/working/vilegal-t5"
MODEL_NAME="VietAI/vit5-base"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✅ GPU detected: $GPU_COUNT GPUs"
else
    GPU_COUNT=0
    echo "⚠️ No GPU detected"
fi

# Set environment variables for stability
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

echo "🚀 Starting training with fixed imports..."

# Run training directly (imports are already fixed)
python train.py \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --warmup_steps 1000 \
    --max_steps 10000 \
    --patience 3 \
    --gpus $GPU_COUNT \
    --precision 16 \
    --max_source_length 512 \
    --max_target_length 256

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Run evaluation
    echo "🔍 Starting evaluation..."
    python evaluate.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --test_data "$VAL_PATH" \
        --output_dir "$OUTPUT_DIR/evaluation_results" \
        --device auto \
        --num_beams 4 \
        --max_length 256
    
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation completed!"
        
        # Show results if available
        if [ -f "$OUTPUT_DIR/evaluation_results/evaluation_results.json" ]; then
            echo "📈 Quick Results:"
            python -c "
import json
try:
    with open('$OUTPUT_DIR/evaluation_results/evaluation_results.json', 'r') as f:
        results = json.load(f)
        overall = results['overall_metrics']
        print(f'  Precision: {overall[\"precision\"]:.4f}')
        print(f'  Recall:    {overall[\"recall\"]:.4f}')
        print(f'  F1-Score:  {overall[\"f1\"]:.4f}')
        print(f'  Samples:   {results[\"num_samples\"]}')
except Exception as e:
    print(f'  Could not load results: {e}')
"
        fi
        
        # Create quick demo
        echo "🎯 Creating demo..."
        cat > "$OUTPUT_DIR/quick_demo.py" << 'EOL'
#!/usr/bin/env python3
import sys
sys.path.append('/kaggle/working/vilegaljere/vilegal')
from demo import ViLegalDemo

# Quick demo
model_path = '/kaggle/working/vilegal-t5/final_model'
demo = ViLegalDemo(model_path)

# Test with sample
sample_text = """Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam tuân thủ quy định của pháp luật về chứng khoán."""

print("🧪 Testing model...")
results = demo.extract_relations(sample_text)
demo.print_results(results)
EOL
        
        echo "✅ Quick demo created: $OUTPUT_DIR/quick_demo.py"
        echo ""
        echo "🎉 All done! To test the model:"
        echo "   python $OUTPUT_DIR/quick_demo.py"
    fi
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
fi 