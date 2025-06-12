import os
import json
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pl_modules import ViLegalJEREModule
from pl_data_modules import ViLegalDataModule  
from utils import setup_tokenizer_for_vilegal, extract_vilegal_triplets, format_example_for_display

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model")
    parser.add_argument("--model_name_or_path", default="VietAI/vit5-base", type=str)
    parser.add_argument("--config_name", default="VietAI/vit5-base", type=str)
    parser.add_argument("--tokenizer_name", default="VietAI/vit5-base", type=str)
    
    # Data arguments  
    parser.add_argument("--test_data_path", default="/kaggle/input/vietnamese-legal-dataset-finetuning-test/test.json", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--max_target_length", default=512, type=int)
    
    # Test arguments
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--output_dir", default="/kaggle/working/test_results", type=str)
    
    # Generation arguments
    parser.add_argument("--eval_beams", default=3, type=int)
    parser.add_argument("--val_max_target_length", default=512, type=int)
    
    # Other arguments
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--show_examples", default=5, type=int, help="Number of examples to display")
    
    args = parser.parse_args()
    
    # Set seeds
    pl.seed_everything(args.seed)
    
    print("🧪 Starting Vietnamese Legal JERE Model Testing")
    print(f"🤖 Model path: {args.model_path}")
    print(f"📊 Test data: {args.test_data_path}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and setup special tokens
    print(f"🔤 Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model config and model
    print(f"⚙️ Loading model config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    
    print(f"🤖 Loading trained model from: {args.model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, config=config)
    
    print(f"🔤 Vocabulary size: {len(tokenizer)}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup data module for testing
    print("📊 Setting up test data module...")
    data_module = ViLegalDataModule(args, tokenizer)
    
    # Setup model module
    print("🧠 Setting up model module...")
    model_module = ViLegalJEREModule(args, config, tokenizer, model)
    
    # Setup trainer for testing
    trainer = pl.Trainer(
        devices=args.gpus if args.gpus > 0 else "auto",
        accelerator="gpu" if args.gpus > 0 else "cpu",
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False
    )
    
    # Run test
    print("🏃‍♂️ Running test...")
    test_results = trainer.test(model_module, data_module)
    
    # Save test results
    results_file = os.path.join(args.output_dir, "test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results[0], f, ensure_ascii=False, indent=2)
    
    print(f"💾 Test results saved to: {results_file}")
    
    # Generate predictions on test set for detailed analysis
    print("🔮 Generating detailed predictions...")
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()
    
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= 20:  # Limit to first 20 batches for detailed analysis
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate predictions
            gen_kwargs = {
                "max_length": args.val_max_target_length,
                "early_stopping": False,
                "length_penalty": 0,
                "num_beams": args.eval_beams,
            }
            
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **gen_kwargs,
            )
            
            # Decode predictions and targets
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            decoded_inputs = batch['input_text']
            decoded_targets = batch['target_text']
            
            all_predictions.extend(decoded_preds)
            all_targets.extend(decoded_targets)
            all_inputs.extend(decoded_inputs)
    
    # Show examples
    print(f"\n🎯 Showing {min(args.show_examples, len(all_predictions))} test examples:")
    for i in range(min(args.show_examples, len(all_predictions))):
        print(f"\n--- Example {i+1} ---")
        format_example_for_display(
            all_inputs[i], 
            all_targets[i], 
            all_predictions[i]
        )
    
    # Compute detailed metrics
    print("\n📊 Computing detailed metrics...")
    all_pred_triplets = [extract_vilegal_triplets(pred) for pred in all_predictions]
    all_gold_triplets = [extract_vilegal_triplets(target) for target in all_targets]
    
    # Overall metrics
    total_pred = sum(len(triplets) for triplets in all_pred_triplets)
    total_gold = sum(len(triplets) for triplets in all_gold_triplets)
    total_correct = 0
    
    for pred_list, gold_list in zip(all_pred_triplets, all_gold_triplets):
        pred_set = set(pred_list)
        gold_set = set(gold_list)
        total_correct += len(pred_set.intersection(gold_set))
    
    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    detailed_results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_predicted_triplets': total_pred,
        'total_gold_triplets': total_gold,
        'total_correct_triplets': total_correct,
        'num_test_examples': len(all_predictions)
    }
    
    # Save detailed results
    detailed_file = os.path.join(args.output_dir, "detailed_results.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # Save predictions
    predictions_file = os.path.join(args.output_dir, "predictions.json")
    predictions_data = []
    for i in range(len(all_predictions)):
        predictions_data.append({
            'input': all_inputs[i],
            'target': all_targets[i],
            'prediction': all_predictions[i],
            'target_triplets': all_gold_triplets[i],
            'predicted_triplets': all_pred_triplets[i]
        })
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== FINAL TEST RESULTS ===")
    print(f"📊 Precision: {precision:.4f}")
    print(f"📊 Recall: {recall:.4f}")  
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"🎯 Total Predicted Triplets: {total_pred}")
    print(f"🎯 Total Gold Triplets: {total_gold}")
    print(f"✅ Total Correct Triplets: {total_correct}")
    print(f"📄 Test Examples Analyzed: {len(all_predictions)}")
    print(f"💾 Detailed results saved to: {detailed_file}")
    print(f"💾 Predictions saved to: {predictions_file}")
    print("✅ Testing completed!")

if __name__ == "__main__":
    main() 