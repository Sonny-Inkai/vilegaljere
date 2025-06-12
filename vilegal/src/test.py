import omegaconf
import hydra
import torch
import json
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pl_data_modules import VietnameseLegalPLDataModule
from pl_modules import VietnameseLegalPLModule
from utils import extract_vietnamese_legal_triplets
from score import score


def test(conf: omegaconf.DictConfig) -> None:
    """Test the Vietnamese legal relation extraction model"""
    
    # Load model and tokenizer
    config = AutoConfig.from_pretrained(conf.model_name_or_path)
    
    # Vietnamese legal domain special tokens
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": domain_special_tokens,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # Load trained checkpoint if available
    if conf.checkpoint_path:
        pl_module = VietnameseLegalPLModule.load_from_checkpoint(
            conf.checkpoint_path, 
            conf=conf, 
            config=config, 
            tokenizer=tokenizer, 
            model=model
        )
    else:
        pl_module = VietnameseLegalPLModule(conf, config, tokenizer, model)

    # Data module
    pl_data_module = VietnameseLegalPLDataModule(conf, tokenizer, model)
    pl_data_module.setup("test")

    # Trainer for testing
    trainer = pl.Trainer(
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False
    )

    # Run test
    results = trainer.test(pl_module, datamodule=pl_data_module)
    
    print("Test Results:")
    print(f"Test Loss: {results[0]['test_loss_epoch']:.4f}")
    print(f"Test Precision: {results[0]['test_precision_epoch']:.4f}")
    print(f"Test Recall: {results[0]['test_recall_epoch']:.4f}")
    print(f"Test F1: {results[0]['test_f1_epoch']:.4f}")
    
    return results


def predict_single(text: str, model_path: str):
    """Predict relations for a single text"""
    
    # Load model and tokenizer
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(
        "VietAI/vit5-base",
        additional_special_tokens=domain_special_tokens
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256,
            num_beams=3,
            early_stopping=True
        )
    
    # Decode and extract triplets
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    triplets = extract_vietnamese_legal_triplets(decoded)
    
    return triplets


@hydra.main(version_base=None, config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == '__main__':
    main() 