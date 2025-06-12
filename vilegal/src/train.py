import os
import warnings
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pl_modules import ViLegalJEREModule  
from pl_data_modules import ViLegalDataModule
from utils import setup_tokenizer_for_vilegal, print_model_info, validate_data_format
import json

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name_or_path", default="VietAI/vit5-base", type=str)
    parser.add_argument("--config_name", default="VietAI/vit5-base", type=str)
    parser.add_argument("--tokenizer_name", default="VietAI/vit5-base", type=str)
    
    # Data arguments
    parser.add_argument("--data_path", default="/kaggle/input/vietnamese-legal-dataset-finetuning", type=str)
    parser.add_argument("--finetune_file_name", default="finetune.json", type=str)
    parser.add_argument("--test_data_path", default="/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--max_target_length", default=512, type=int)
    
    # Training arguments
    parser.add_argument("--output_dir", default="/kaggle/working/vilegal-vit5", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    
    # Generation arguments
    parser.add_argument("--eval_beams", default=3, type=int)
    parser.add_argument("--val_max_target_length", default=512, type=int)
    
    # Optimization arguments
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--lr_scheduler", default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--ignore_pad_token_for_loss", default=True, type=bool)
    
    # Other arguments
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--save_top_k", default=3, type=int)
    parser.add_argument("--patience", default=3, type=int)
    
    args = parser.parse_args()
    
    # Set seeds
    pl.seed_everything(args.seed)
    
    print("🚀 Starting Vietnamese Legal Joint Entity-Relation Extraction Training")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and setup tokenizer
    print(f"🔤 Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Add domain-specific tokens
    tokenizer, num_added_tokens = setup_tokenizer_for_vilegal(tokenizer)
    
    # Load model config and model
    print(f"⚙️ Loading model config: {args.config_name}")
    config = AutoConfig.from_pretrained(args.config_name)
    
    print(f"🤖 Loading model: {args.model_name_or_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    
    # Resize token embeddings to account for new tokens
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"🔧 Resized token embeddings to {len(tokenizer)}")
    
    # Print model info
    print_model_info(model, tokenizer)
    
    # Validate data format
    print("🔍 Validating data format...")
    train_data_path = f"{args.data_path}/{args.finetune_file_name}"
    with open(train_data_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    validate_data_format(sample_data)
    
    # Setup data module
    print("📊 Setting up data module...")
    data_module = ViLegalDataModule(args, tokenizer)
    
    # Setup model module
    print("🧠 Setting up model module...")
    model_module = ViLegalJEREModule(args, config, tokenizer, model)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="vilegal-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.patience,
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="vilegal_logs"
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        devices=args.gpus if args.gpus > 0 else "auto",
        accelerator="gpu" if args.gpus > 0 else "cpu",
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        val_check_interval=0.5,  # Check validation every half epoch
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Start training
    print("🏃‍♂️ Starting training...")
    trainer.fit(model_module, data_module)
    
    # Save final model and tokenizer
    print("💾 Saving final model and tokenizer...")
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("✅ Training completed!")
    print(f"📁 Model saved to: {final_model_path}")
    print(f"🏆 Best checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main() 