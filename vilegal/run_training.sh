#!/bin/bash

# Vietnamese Legal Joint Entity and Relation Extraction Training Script
# Optimized for Kaggle GPU environment

echo "🏛️ Vietnamese Legal Joint Entity and Relation Extraction Training"
echo "=================================================================="

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Check GPU availability
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Number of GPUs: $GPU_COUNT"
else
    echo "⚠️  No GPU detected, using CPU"
    GPU_COUNT=0
fi

# Training parameters
TRAIN_PATH="/kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json"
VAL_PATH="/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
OUTPUT_DIR="/kaggle/working/vilegal-t5"
MODEL_NAME="VietAI/vit5-base"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if data files exist
if [ ! -f "$TRAIN_PATH" ]; then
    echo "❌ Training data not found at $TRAIN_PATH"
    echo "Please ensure the Vietnamese legal dataset is uploaded to Kaggle"
    exit 1
fi

if [ ! -f "$VAL_PATH" ]; then
    echo "❌ Validation data not found at $VAL_PATH"
    echo "Please ensure the Vietnamese legal test dataset is uploaded to Kaggle"
    exit 1
fi

echo "✅ Data files found"
echo "📂 Training data: $TRAIN_PATH"
echo "📂 Validation data: $VAL_PATH"
echo "📂 Output directory: $OUTPUT_DIR"

# Install dependencies if not already installed
echo "📦 Installing dependencies..."
pip install -q torch transformers>=4.30.0 pytorch-lightning>=1.8.0 datasets tokenizers "numpy>=1.21.0,<2.0.0" tqdm tensorboard scikit-learn pandas regex sentencepiece accelerate>=0.15.0

# Start training
echo "🚀 Starting training..."
echo "Model: $MODEL_NAME"
echo "Training parameters:"
echo "  - Batch size: 8"
echo "  - Learning rate: 1e-4"
echo "  - Max epochs: 10"
echo "  - Precision: 16-bit"

python train.py \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size 8 \
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
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo "✅ Evaluation completed successfully!"
        echo "📊 Results saved to $OUTPUT_DIR/evaluation_results/"
        
        # Show evaluation summary
        if [ -f "$OUTPUT_DIR/evaluation_results/evaluation_results.json" ]; then
            echo "📈 Evaluation Summary:"
            python -c "
import json
with open('$OUTPUT_DIR/evaluation_results/evaluation_results.json', 'r') as f:
    results = json.load(f)
    overall = results['overall_metrics']
    print(f'  Precision: {overall[\"precision\"]:.4f}')
    print(f'  Recall:    {overall[\"recall\"]:.4f}')
    print(f'  F1-Score:  {overall[\"f1\"]:.4f}')
    print(f'  Samples:   {results[\"num_samples\"]}')
"
        fi
    else
        echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
    fi
    
    # Create demo script
    echo "🎯 Creating demo script..."
    cat > "$OUTPUT_DIR/run_demo.sh" << EOL
#!/bin/bash
# Demo script for Vietnamese Legal Relation Extraction

echo "🏛️ Vietnamese Legal Relation Extraction Demo"
echo "============================================="

# Interactive demo
python demo.py \\
    --model_path "$OUTPUT_DIR/final_model" \\
    --mode interactive \\
    --device auto

EOL
    chmod +x "$OUTPUT_DIR/run_demo.sh"
    
    echo "✅ Demo script created: $OUTPUT_DIR/run_demo.sh"
    
    # Show final summary
    echo ""
    echo "🎉 Training pipeline completed successfully!"
    echo "📁 Model files:"
    ls -la "$OUTPUT_DIR/final_model/" 2>/dev/null || echo "  Model files not found"
    echo ""
    echo "📊 Evaluation files:"
    ls -la "$OUTPUT_DIR/evaluation_results/" 2>/dev/null || echo "  Evaluation files not found"
    echo ""
    echo "🚀 To run the demo:"
    echo "  bash $OUTPUT_DIR/run_demo.sh"
    echo ""
    echo "📖 For batch processing:"
    echo "  python demo.py --model_path $OUTPUT_DIR/final_model --mode batch --input_file input.json --output_file results.json"
    
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check the logs above for error details"
    exit $TRAINING_EXIT_CODE
fi 