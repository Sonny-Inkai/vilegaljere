#!/usr/bin/env python3
"""
Kaggle Environment Setup Script
Prepares the environment for Vietnamese Legal Joint Extraction training
"""

import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    
    packages = [
        "torch>=1.12.0",
        "transformers>=4.21.0", 
        "datasets>=2.4.0",
        "scikit-learn>=1.1.0",
        "rouge-score>=0.1.2",
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "tqdm>=4.64.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "sentencepiece>=0.1.97",
        "protobuf>=3.20.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")

def verify_environment():
    """Verify the environment is set up correctly"""
    logger.info("Verifying environment...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA device: {torch.cuda.get_device_name()}")
        
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
        
        from rouge_score import rouge_scorer
        logger.info("✅ ROUGE scorer available")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    
    return True

def check_data_paths():
    """Check if data paths exist"""
    logger.info("Checking data paths...")
    
    train_path = "/kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json"
    eval_path = "/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json"
    
    if os.path.exists(train_path):
        logger.info(f"✅ Training data found: {train_path}")
    else:
        logger.warning(f"⚠️ Training data not found: {train_path}")
    
    if os.path.exists(eval_path):
        logger.info(f"✅ Evaluation data found: {eval_path}")
    else:
        logger.warning(f"⚠️ Evaluation data not found: {eval_path}")
    
    # Create output directory
    output_dir = "/kaggle/working"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"✅ Created output directory: {output_dir}")
    else:
        logger.info(f"✅ Output directory exists: {output_dir}")

def print_system_info():
    """Print system information"""
    logger.info("System Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Check GPU info
    if os.path.exists("/proc/driver/nvidia/version"):
        try:
            with open("/proc/driver/nvidia/version", "r") as f:
                nvidia_version = f.readline().strip()
                logger.info(f"NVIDIA Driver: {nvidia_version}")
        except:
            pass
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.info("psutil not available for memory info")

def create_training_script():
    """Create a simple training launcher"""
    script_content = '''#!/bin/bash

echo "🚀 Starting Vietnamese Legal Joint Extraction Training"
echo "======================================================="

# Check if training data exists
if [ ! -f "/kaggle/input/vietnamese-legal-dataset-finetuning/finetune.json" ]; then
    echo "❌ Training data not found!"
    echo "Please make sure the dataset is properly uploaded to Kaggle"
    exit 1
fi

# Run training
python vietnamese_legal_joint_extraction_finetuning.py

# Run evaluation if training succeeds
if [ $? -eq 0 ]; then
    echo "🎯 Training completed! Running evaluation..."
    python evaluate_model.py
else
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ All tasks completed successfully!"
'''
    
    with open("/kaggle/working/run_training.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("/kaggle/working/run_training.sh", 0o755)
    logger.info("✅ Created training script: /kaggle/working/run_training.sh")

def main():
    """Main setup function"""
    logger.info("🔧 Setting up Vietnamese Legal Joint Extraction environment")
    logger.info("=" * 60)
    
    # Print system info
    print_system_info()
    
    # Install requirements
    install_requirements()
    
    # Verify environment
    if not verify_environment():
        logger.error("❌ Environment verification failed!")
        return False
    
    # Check data paths
    check_data_paths()
    
    # Create training script
    create_training_script()
    
    logger.info("=" * 60)
    logger.info("🎉 Setup completed successfully!")
    logger.info("")
    logger.info("📋 Next steps:")
    logger.info("1. Upload your datasets to Kaggle")
    logger.info("2. Run: python vietnamese_legal_joint_extraction_finetuning.py")
    logger.info("3. Or run: bash /kaggle/working/run_training.sh")
    logger.info("")
    logger.info("🔥 Ready to fine-tune like Tony Stark!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 