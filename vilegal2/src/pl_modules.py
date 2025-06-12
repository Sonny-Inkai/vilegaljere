import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    
)
from utils import shift_tokens_left, extract_vietnamese_legal_triplets
from score import score

class VietnameseLegalPLModule(pl.LightningModule):
    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, domain_special_tokens: list):
        super().__init__()
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.domain_special_tokens = domain_special_tokens

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        # The dataloader provides labels already shifted and with pad tokens replaced by -100
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Generate predictions
        generated_tokens = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.val_max_target_length,
            num_beams=self.hparams.num_beams,
        )
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        # Replace -100 in labels to pad_token_id for decoding
        labels = batch['labels']
        labels[labels == -100] = self.tokenizer.pad_token_id
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        # Extract triplets
        pred_triplets = [extract_vietnamese_legal_triplets(text, self.domain_special_tokens) for text in decoded_preds]
        actual_triplets = [extract_vietnamese_legal_triplets(text, self.domain_special_tokens) for text in decoded_labels]
        
        # Calculate scores for the batch
        batch_p, batch_r, batch_f1 = [], [], []
        for pred, actual in zip(pred_triplets, actual_triplets):
            scores = score(pred, actual)['micro']
            batch_p.append(scores['p'])
            batch_r.append(scores['r'])
            batch_f1.append(scores['f1'])

        # Log batch metrics
        self.log('val_f1', torch.tensor(batch_f1).mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_p', torch.tensor(batch_p).mean(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_r', torch.tensor(batch_r).mean(), on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Print samples for the first batch on rank 0
        if batch_idx == 0 and self.trainer.is_global_zero:
            print("\n" + "="*80)
            print(f"VALIDATION SAMPLES (EPOCH {self.current_epoch})")
            for i in range(min(len(decoded_preds), 2)):
                print(f"--- SAMPLE {i+1} ---")
                print(f"PRED: {decoded_preds[i]}")
                print(f"GOLD: {decoded_labels[i]}")
                print(f"EXTRACTED: {pred_triplets[i]}")
            print("="*80 + "\n")

    def test_step(self, batch, batch_idx):
        # Similar to validation_step but for the test set
        generated_tokens = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.val_max_target_length,
            num_beams=self.hparams.num_beams,
        )
        labels = batch['labels']
        labels[labels == -100] = self.tokenizer.pad_token_id
        
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        pred_triplets = [extract_vietnamese_legal_triplets(text, self.domain_special_tokens) for text in decoded_preds]
        actual_triplets = [extract_vietnamese_legal_triplets(text, self.domain_special_tokens) for text in decoded_labels]
        
        batch_f1 = [score(p, a)['micro']['f1'] for p, a in zip(pred_triplets, actual_triplets)]
        self.log('test_f1', torch.tensor(batch_f1).mean(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}} 