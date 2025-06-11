#!/usr/bin/env python3
"""
Run All Script - One-click training and evaluation
"""

import os
import sys
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Vietnamese Legal Joint Extraction - Complete Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Training
    logger.info("Step 1: Starting model training...")
    try:
        result = subprocess.run([sys.executable, "vietnamese_legal_joint_extraction_finetuning.py"], 
                               check=True, capture_output=True, text=True)
        logger.info("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    
    # Step 2: Evaluation
    logger.info("Step 2: Starting model evaluation...")
    try:
        result = subprocess.run([sys.executable, "evaluate_model.py"], 
                               check=True, capture_output=True, text=True)
        logger.info("✅ Evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Evaluation failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    
    # Step 3: Demo
    logger.info("Step 3: Running inference demo...")
    try:
        result = subprocess.run([sys.executable, "inference_demo.py"], 
                               check=True, capture_output=True, text=True)
        logger.info("✅ Demo completed successfully!")
        logger.info(f"Demo output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
    
    logger.info("=" * 60)
    logger.info("🎉 All tasks completed! Tony Stark would be proud!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 