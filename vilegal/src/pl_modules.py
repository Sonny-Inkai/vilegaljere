import torch
import json
import pytorch_lightning as pl
import numpy as np
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from torch.nn.utils.rnn import pad_sequence
from utils import shift_tokens_left, extract_vilegal_triplets

class ViLegalJEREModule(pl.LightningModule):
    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.hparams.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            from utils import label_smoothed_nll_loss
            self.loss_fn = label_smoothed_nll_loss

    def forward(self, inputs, labels, **kwargs):
        if self.hparams.label_smoothing == 0:
            if self.hparams.ignore_pad_token_for_loss:
                outputs = self.model(**inputs, use_cache=False, return_dict=True)
                logits = outputs['logits']
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                outputs = self.model(**inputs, labels=labels, use_cache=False, return_dict=True)
                loss = outputs['loss']
                logits = outputs['logits']
        else:
            outputs = self.model(**inputs, use_cache=False, return_dict=True)
            logits = outputs['logits']
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            loss, _ = self.loss_fn(lprobs, labels, self.hparams.label_smoothing, ignore_index=self.config.pad_token_id)
            
        return {'loss': loss, 'logits': logits}

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        
        forward_output = self.forward(batch, labels)
        self.log('train_loss', forward_output['loss'])
        batch["labels"] = labels_original
        return forward_output['loss']

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        
        forward_output = self.forward(batch, labels)
        self.log('val_loss', forward_output['loss'])
        
        # Generate predictions for evaluation
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length or 512,
            "early_stopping": False,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams or 3,
        }
        
        generated_tokens = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            **gen_kwargs,
        )
        
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(torch.where(labels_original != -100, labels_original, self.config.pad_token_id), skip_special_tokens=False)
        
        # Extract triplets for evaluation
        pred_triplets = [extract_vilegal_triplets(rel) for rel in decoded_preds]
        gold_triplets = [extract_vilegal_triplets(rel) for rel in decoded_labels]
        
        batch["labels"] = labels_original
        return {
            'val_loss': forward_output['loss'],
            'pred_triplets': pred_triplets,
            'gold_triplets': gold_triplets,
            'decoded_preds': decoded_preds,
            'decoded_labels': decoded_labels
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss)
        
        # Compute metrics
        all_pred_triplets = []
        all_gold_triplets = []
        for output in outputs:
            all_pred_triplets.extend(output['pred_triplets'])
            all_gold_triplets.extend(output['gold_triplets'])
        
        metrics = self.compute_metrics(all_pred_triplets, all_gold_triplets)
        for key, value in metrics.items():
            self.log(f'val_{key}', value)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)
        
        # Compute metrics
        all_pred_triplets = []
        all_gold_triplets = []
        for output in outputs:
            all_pred_triplets.extend(output['pred_triplets'])
            all_gold_triplets.extend(output['gold_triplets'])
        
        metrics = self.compute_metrics(all_pred_triplets, all_gold_triplets)
        for key, value in metrics.items():
            self.log(f'test_{key}', value)
            
        print(f"\n=== TEST RESULTS ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    def compute_metrics(self, pred_triplets, gold_triplets):
        """Compute Precision, Recall, F1 for triplets"""
        total_pred = sum(len(triplets) for triplets in pred_triplets)
        total_gold = sum(len(triplets) for triplets in gold_triplets)
        total_correct = 0
        
        for pred_list, gold_list in zip(pred_triplets, gold_triplets):
            pred_set = set(pred_list)
            gold_set = set(gold_list)
            total_correct += len(pred_set.intersection(gold_set))
        
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_gold if total_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_pred': total_pred,
            'total_gold': total_gold,
            'total_correct': total_correct
        }

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.hparams.optimizer == "adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters, 
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False
            )
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
            
        if self.hparams.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        elif self.hparams.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        else:
            return optimizer 