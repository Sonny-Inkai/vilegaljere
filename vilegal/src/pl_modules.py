from typing import Any
import json
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from score import score, re_score
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim import AdamW
from transformers.optimization import (
    Adafactor,
    
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from scheduler import get_inverse_square_root_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from utils import BartTripletHead, shift_tokens_left, extract_vietnamese_legal_triplets

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup
}


class VietnameseLegalPLModule(pl.LightningModule):

    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, *args, **kwargs) -> None:
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
            # dynamically import label_smoothed_nll_loss
            from utils import label_smoothed_nll_loss
            self.loss_fn = label_smoothed_nll_loss

    def forward(self, inputs, labels, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        """
        if self.hparams.label_smoothing == 0:
            if self.hparams is not None and self.hparams.ignore_pad_token_for_loss:
                # force training to ignore pad token
                outputs = self.model(**inputs, use_cache=False, return_dict=True)
                logits = outputs['logits']
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                # compute usual loss via models
                outputs = self.model(**inputs, labels=labels, use_cache=False, return_dict=True)
                loss = outputs['loss']
                logits = outputs['logits']
        else:
            # compute label smoothed loss
            outputs = self.model(**inputs, use_cache=False, return_dict=True)
            logits = outputs['logits']
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            loss, _ = self.loss_fn(lprobs, labels, self.hparams.label_smoothing, ignore_index=self.config.pad_token_id)
        
        output_dict = {'loss': loss, 'logits': logits}
        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        forward_output = self.forward(batch, labels)
        self.log('loss', forward_output['loss'])
        batch["labels"] = labels_original
        return forward_output['loss']

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def generate_triples(self, batch, labels):
        """Generate triplets for Vietnamese legal documents"""
        
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache=True,
            **gen_kwargs,
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(
            torch.where(labels != -100, labels, self.config.pad_token_id), 
            skip_special_tokens=False
        )
        
        # Use Vietnamese legal triplet extraction
        return (
            [extract_vietnamese_legal_triplets(rel) for rel in decoded_preds], 
            [extract_vietnamese_legal_triplets(rel) for rel in decoded_labels]
        )

    def validation_step(self, batch: dict, batch_idx: int):
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        forward_output = self.forward(batch, labels)
        
        batch["labels"] = labels_original
        
        loss = forward_output['loss']
        generated_triples, actual_triples = self.generate_triples(batch, batch["labels"])
        
        try:
            scores = []
            for generated, actual in zip(generated_triples, actual_triples):
                score_dict = score(generated, actual, mode='boundaries')
                scores.append(score_dict)
            
            # Aggregate scores
            if scores:
                avg_precision = np.mean([s.get('micro', {}).get('p', 0) for s in scores])
                avg_recall = np.mean([s.get('micro', {}).get('r', 0) for s in scores])
                avg_f1 = np.mean([s.get('micro', {}).get('f1', 0) for s in scores])
            else:
                avg_precision = avg_recall = avg_f1 = 0.0
                
        except Exception as e:
            print(f"Error computing scores: {e}")
            avg_precision = avg_recall = avg_f1 = 0.0

        # Add extensive logging for diagnostics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', avg_precision, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', avg_recall, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', avg_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Print a sample of generated text for debugging on the first batch of every validation epoch
        if batch_idx == 0 and self.trainer.is_global_zero:
            print("\n\n" + "="*80)
            print("VALIDATION SAMPLES (EPOCH " + str(self.current_epoch) + ")")
            print("="*80)
            for i in range(min(2, len(decoded_preds))):
                print(f"SAMPLE {i+1}")
                print(f"  - PRED: {decoded_preds[i]}")
                print(f"  - GOLD: {decoded_labels[i]}")
                print(f"  - EXTRACTED: {generated_triples[i]}")
                print(f"  - ACTUAL:    {actual_triples[i]}")
                print("-"*80)

        return {
            'val_loss': loss,
            'predictions': generated_triples,
            'labels': actual_triples,
            'val_precision': avg_precision,
            'val_recall': avg_recall,
            'val_f1': avg_f1
        }

    def test_step(self, batch: dict, batch_idx: int):
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        forward_output = self.forward(batch, labels)
        
        batch["labels"] = labels_original
        
        loss = forward_output['loss']
        generated_triples, actual_triples = self.generate_triples(batch, batch["labels"])
        
        try:
            scores = []
            for generated, actual in zip(generated_triples, actual_triples):
                score_dict = score(generated, actual, mode='boundaries')
                scores.append(score_dict)
            
            # Aggregate scores
            if scores:
                avg_precision = np.mean([s.get('micro', {}).get('p', 0) for s in scores])
                avg_recall = np.mean([s.get('micro', {}).get('r', 0) for s in scores])
                avg_f1 = np.mean([s.get('micro', {}).get('f1', 0) for s in scores])
            else:
                avg_precision = avg_recall = avg_f1 = 0.0
                
        except Exception as e:
            print(f"Error computing scores: {e}")
            avg_precision = avg_recall = avg_f1 = 0.0

        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.log('test_precision', avg_precision, on_epoch=True, sync_dist=True)
        self.log('test_recall', avg_recall, on_epoch=True, sync_dist=True)
        self.log('test_f1', avg_f1, on_epoch=True, sync_dist=True)

        return {
            'test_loss': loss,
            'predictions': generated_triples,
            'labels': actual_triples,
            'test_precision': avg_precision,
            'test_recall': avg_recall,
            'test_f1': avg_f1
        }

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "inverse_square_root":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler 