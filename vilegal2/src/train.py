import omegaconf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from pl_data_modules import VietnameseLegalPLDataModule
from pl_modules import VietnameseLegalPLModule

def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        conf.model_name_or_path,
        use_fast=conf.use_fast_tokenizer,
        additional_special_tokens=domain_special_tokens,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    data_module = VietnameseLegalPLDataModule(conf, tokenizer)
    pl_module = VietnameseLegalPLModule(conf, tokenizer, model, domain_special_tokens)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=conf.out_dir,
        monitor=conf.monitor_var,
        mode=conf.monitor_var_mode,
        save_top_k=conf.save_top_k,
        filename='{epoch}-{step}-{val_f1:.4f}'
    )
    early_stop_callback = EarlyStopping(
        monitor=conf.monitor_var,
        patience=conf.patience,
        mode=conf.monitor_var_mode
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, lr_monitor]
    if conf.apply_early_stopping:
        callbacks.append(early_stop_callback)

    # Setup Logger
    wandb_logger = WandbLogger(project="vietnamese-legal-re-v2", name=conf.model_name_or_path.split('/')[-1])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=conf.gpus,
        max_steps=conf.max_steps,
        accumulate_grad_batches=conf.gradient_acc_steps,
        val_check_interval=conf.val_check_interval,
        precision=conf.precision,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.fit(pl_module, datamodule=data_module)

@hydra.main(version_base=None, config_path='conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)

if __name__ == '__main__':
    main() 