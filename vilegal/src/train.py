import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import VietnameseLegalPLDataModule
from pl_modules import VietnameseLegalPLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)
    
    # Configure model for T5
    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id=0,
        early_stopping=False,
        no_repeat_ngram_size=0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )
    
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

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    
    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Data module declaration
    pl_data_module = VietnameseLegalPLDataModule(conf, tokenizer, model)

    # Main module declaration
    pl_module = VietnameseLegalPLModule(conf, config, tokenizer, model)

    # Logger
    wandb_logger = WandbLogger(
        project="vietnamese-legal-re", 
        name=f"{conf.model_name_or_path.split('/')[-1]}-legal"
    )

    callbacks_store = []

    if conf.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience
            )
        )

    callbacks_store.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            dirpath=conf.out_dir,
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.monitor_var_mode
        )
    )
    
    callbacks_store.append(LearningRateMonitor(logging_interval='step'))
    
    # Trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        max_steps=conf.max_steps,
        precision=conf.precision,
        amp_level=conf.amp_level,
        logger=wandb_logger,
        resume_from_checkpoint=conf.checkpoint_path,
        limit_val_batches=conf.val_percent_check
    )

    # Module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == '__main__':
    main() 