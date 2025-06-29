import os
import time
import numpy as np
import torch
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
import json

# -----------------------------------------------------------------------------
# -- ViT5 FINE-TUNING TEST CONFIGURATION --
# -----------------------------------------------------------------------------

# --- Cấu hình chung ---
eval_only = False
always_save_checkpoint = True

# Wandb API key
os.environ['WANDB_API_KEY'] = 'bcc183326224decc1f9fee116ccfd509e740fab1'

# --- CẤU HÌNH CHO ViT5 TEST ---
data_path = "/kaggle/input/vietnamese-legal-finetune-dataset" # Nơi chứa file finetune.json
finetune_file_name = "dataset.json"
finetune_dir = '/kaggle/working/vit5_finetune_test'

# ✅ OPTIMIZED: Better hyperparameters for relation extraction fine-tuning
learning_rate = 5e-5    # REBEL uses 5e-5
max_iters = 3000        # Longer training
batch_size = 16       # Smaller batch for ViT5-base
gradient_accumulation_steps = 2  # Maintain effective batch size of 32
weight_decay = 0.01  
eval_interval = 100     # More frequent eval
log_interval = 10    
eval_iters = 100      
warmup_steps = 300      # 10% of max_iters

# wandb logging
wandb_log = True    
wandb_project = 'ViT5_Test'
wandb_run_name = 'vit5_jere_test'

# data
max_source_length = 512  
max_target_length = 200  # Shorter like REBEL (was 256)

# optimizer
optimizer_name = 'adamw'
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
lr_decay_iters = max_iters  
min_lr = 1e-6        
schedule = 'cosine'

# DDP settings for Kaggle T4x2
backend = 'gloo'  

# system
device = 'cuda'  
dtype = 'float16'   
compile = False     

# -----------------------------------------------------------------------------

# Standard imports for ViT5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import utilities (reuse từ utils.py)
from utils import (
    get_num_params,
    get_lr,
    setup_distributed_training,
    setup_training_environment,
)

# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# DDP setup using utils
ddp, master_process, seed_offset, world_size, device = setup_distributed_training(backend)

# ✅ LOAD ViT5 MODEL AND TOKENIZER
if master_process:
    print("🚀 Loading ViT5-base model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")

# Add domain-specific tokens để fair comparison
domain_special_tokens = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<Effective_From>", "<Applicable_In>",
    "<Relates_To>", "<Amended_By>"
]

# Add special tokens
num_added_tokens = tokenizer.add_tokens(domain_special_tokens)
model.resize_token_embeddings(len(tokenizer))

if master_process:
    print(f"✅ Added {num_added_tokens} domain-specific tokens")
    print(f"📊 New vocab size: {len(tokenizer)}")
    for i, token in enumerate(domain_special_tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: {token_id}")

# Calculate total tokens
tokens_per_iter = batch_size * (max_source_length + max_target_length) * gradient_accumulation_steps * world_size
tokens_trained = 0

# Setup training environment using utils
device_type, ctx = setup_training_environment(seed_offset, dtype)

# ✅ REUSE DATA LOADING CODE (same as finetune_vilegaljere.py)
def load_finetune_data():
    """Tải và xử lý dữ liệu từ file finetune.json với validation và cleanup"""
    data_file = os.path.join(data_path, finetune_file_name)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Finetune dataset not found at {data_file}")
    
    processed_data = []
    skipped_count = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, value in data.items():
        source_text = value.get("input_text")
        target_text = value.get("extracted_relations")
        
        if source_text and target_text:
            processed_data.append((source_text, target_text))
        else:
            skipped_count += 1
    
    if master_process:
        print(f"✅ Loaded {len(processed_data)} fine-tuning pairs")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} empty pairs")
    
    return processed_data

# Load data for fine-tuning
if master_process:
    print("Loading fine-tuning data...")

all_data = load_finetune_data()

# Split train/val
split_idx = int(0.9 * len(all_data))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

if master_process:
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

# ✅ REUSE DATASET CLASS
class ViT5JEREDataset(torch.utils.data.Dataset):
    """Standard PyTorch Dataset cho ViT5 fine-tuning"""
    
    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        
        # T5 tokenization
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )
        
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# ✅ REBEL-STYLE DATA COLLATOR
class T5DataCollator:
    """REBEL-style T5 Data Collator with proper decoder_input_ids"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # ✅ REBEL-STYLE: Pad input_ids
        batch_input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        # ✅ REBEL-STYLE: Pad labels
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        # ✅ REBEL KEY: Create decoder_input_ids from labels (before masking!)
        decoder_input_ids = torch.where(
            batch_labels["input_ids"] != self.tokenizer.pad_token_id, 
            batch_labels["input_ids"], 
            self.tokenizer.pad_token_id
        )
        
        # ✅ REBEL-STYLE: Replace padding in labels with -100 for loss calculation
        labels_for_loss = batch_labels["input_ids"].masked_fill(
            batch_labels["input_ids"] == self.tokenizer.pad_token_id, -100
        )
        
        return {
            "input_ids": batch_input_ids["input_ids"],
            "attention_mask": batch_input_ids["attention_mask"],
            "decoder_input_ids": decoder_input_ids,  # ✅ REBEL key addition
            "labels": labels_for_loss
        }

# ✅ CREATE DATASETS AND DATALOADERS
train_dataset = ViT5JEREDataset(train_data, tokenizer, max_source_length, max_target_length)
val_dataset = ViT5JEREDataset(val_data, tokenizer, max_source_length, max_target_length)

data_collator = T5DataCollator(tokenizer)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=True if device_type == 'cuda' else False
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=True if device_type == 'cuda' else False
)

def get_batch(split):
    """Get batch using DataLoader with REBEL-style decoder handling"""
    dataloader = train_dataloader if split == 'train' else val_dataloader
    
    try:
        if not hasattr(get_batch, f'{split}_iter'):
            setattr(get_batch, f'{split}_iter', iter(dataloader))
        
        batch = next(getattr(get_batch, f'{split}_iter'))
    except StopIteration:
        setattr(get_batch, f'{split}_iter', iter(dataloader))
        batch = next(getattr(get_batch, f'{split}_iter'))
    
    if device_type == 'cuda':
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        decoder_input_ids = batch['decoder_input_ids'].to(device, non_blocking=True)  # ✅ REBEL addition
        labels = batch['labels'].to(device, non_blocking=True)
    else:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # ✅ REBEL addition
        labels = batch['labels'].to(device)
    
    return input_ids, labels, attention_mask, decoder_input_ids  # ✅ Return decoder_input_ids

# Move model to device
model.to(device)

# Calculate parameters
param_count = sum(p.numel() for p in model.parameters())
param_count_m = param_count / 1_000_000

if master_process:
    print(f"✅ ViT5-base model initialized with {param_count_m:.1f}M parameters")
    os.makedirs(finetune_dir, exist_ok=True)

# Initialize optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

# Initialize scaler
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Wrap with DDP
if ddp:
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# ✅ ViT5 GENERATION TEST FUNCTION WITH REBEL PARAMETERS
def test_vit5_generation(model, tokenizer, device, master_process=True):
    """Test ViT5 model generation capability with REBEL-style parameters"""
    if not master_process:
        return True
        
    model.eval()
    
    # Test input (same as other tests)
    test_input = "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức."
    # ✅ FIXED: Correct "Mặt trận" instead of "Mặt trần"
    expected_output = "<LEGAL_PROVISION> Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN <ORGANIZATION> Mặt trận Tổ quốc Việt Nam <Relates_To>"
    
    print("\n🧪 TESTING ViT5 MODEL WITH REBEL-STYLE GENERATION:")
    print(f"📥 Input: {test_input[:150]}...")
    print(f"🎯 Expected: {expected_output}")
    
    try:
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # ✅ REBEL-STYLE GENERATION
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,              # REBEL length
                early_stopping=False,       # REBEL key setting
                length_penalty=0,           # REBEL key setting  
                no_repeat_ngram_size=0,     # REBEL key setting
                num_beams=4,                # REBEL beams
                do_sample=False,            # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Generated: {generated_text}")
        
        # Count domain tokens
        domain_tokens_found = []
        for token in domain_special_tokens:
            if token in generated_text:
                domain_tokens_found.append(token)
        
        print(f"🔧 Domain tokens found: {domain_tokens_found}")
        
        # ✅ IMPROVED: Better success criteria
        has_domain_tokens = len(domain_tokens_found) >= 2  # At least 2 domain tokens
        has_content = len(generated_text.strip()) > 20     # Reasonable length
        not_repetitive = not any(word.count(word.split()[0]) > 3 for word in [generated_text] if word.split())
        
        success = has_domain_tokens and has_content and not_repetitive
        
        print(f"✅ Domain tokens: {has_domain_tokens} | Content length: {has_content} | Not repetitive: {not_repetitive}")
        
        if success:
            print("✅ ViT5 REBEL-style generation test passed!")
        else:
            print("❌ ViT5 REBEL-style generation test failed!")
            
        return success
        
    except Exception as e:
        print(f"❌ ViT5 generation test error: {e}")
        return False
    finally:
        model.train()

# Logging setup
if wandb_log and master_process:
    import wandb
    
    try:
        wandb.login()
        print("✅ Đăng nhập wandb thành công!")
    except Exception as e:
        print(f"⚠️ Không thể đăng nhập wandb: {e}")
        wandb.login(anonymous="allow")
    
    wandb_config = {
        'model': 'VietAI/vit5-base',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_iters': max_iters,
        'max_source_length': max_source_length,
        'max_target_length': max_target_length,
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9

# ✅ REBEL-STYLE UTILITY FUNCTIONS (moved to top)
def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the left (REBEL style).
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id
    return shifted_input_ids

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """REBEL-style label smoothed loss"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

# ✅ REBEL-STYLE LOSS FUNCTION
def compute_loss(model, input_ids, attention_mask, decoder_input_ids, labels, label_smoothing=0.0):
    """REBEL-style loss computation with label smoothing support"""
    
    # ✅ REBEL-STYLE: Shift labels for loss calculation
    shifted_labels = shift_tokens_left(labels, -100)
    
    if label_smoothing == 0.0:
        # Standard cross entropy loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=shifted_labels,
        )
        return outputs.loss
    else:
        # Label smoothed loss (REBEL style)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        logits = outputs.logits
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Replace -100 with pad_token_id for label smoothing
        smooth_labels = shifted_labels.clone()
        smooth_labels.masked_fill_(smooth_labels == -100, tokenizer.pad_token_id)
        
        loss, _ = label_smoothed_nll_loss(
            lprobs, smooth_labels, label_smoothing, 
            ignore_index=tokenizer.pad_token_id
        )
        return loss

@torch.no_grad()
def estimate_loss_rebel(model, get_batch_fn, eval_iters, ctx, label_smoothing=0.0):
    """REBEL-style estimate loss with decoder_input_ids and label smoothing"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, labels, attention_mask, decoder_input_ids = get_batch_fn(split)
            with ctx:
                loss = compute_loss(
                    model, input_ids, attention_mask, decoder_input_ids, 
                    labels, label_smoothing
                )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
if master_process:
    print(f"🚀 Starting ViT5 Fine-tuning test...")
    print(f"Training data size: {len(train_data)}, Val data size: {len(val_data)}")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
    print(f"Learning rate: {learning_rate}, Max iters: {max_iters}")

input_ids, labels, attention_mask, decoder_input_ids = get_batch('train')

if master_process:
    print(f"First batch shapes - Input: {input_ids.shape}, Labels: {labels.shape}")
    
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

# ✅ REBEL-STYLE TRAINING PARAMETERS  
label_smoothing = 0.1   # REBEL uses label smoothing

# ✅ REBEL-STYLE GENERATION PARAMETERS  
generation_config = {
    "max_length": 200,
    "early_stopping": False,  # REBEL key setting
    "length_penalty": 0,      # REBEL key setting
    "no_repeat_ngram_size": 0,  # REBEL key setting
    "num_beams": 4,
    "do_sample": False,
}

while True:
    # Set learning rate
    lr = get_lr(iter_num, learning_rate, warmup_steps, lr_decay_iters, min_lr, schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss_rebel(raw_model, get_batch, eval_iters, ctx, label_smoothing)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        print(f"\n🧪 ViT5 EVALUATION (iter {iter_num})")
        
        # Test model generation
        test_success = test_vit5_generation(raw_model, tokenizer, device, master_process)
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "eval/generation_success": 1 if test_success else 0,
            }
            wandb.log(log_dict, step=iter_num)
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"Saving checkpoint to {finetune_dir}")
                raw_model.save_pretrained(finetune_dir, safe_serialization=False)

    if iter_num == 0 and eval_only:
        break

    # Forward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            # ✅ REBEL-STYLE: Use custom loss function with label smoothing
            loss = compute_loss(
                model, input_ids, attention_mask, decoder_input_ids, 
                labels, label_smoothing
            ) / gradient_accumulation_steps
        
        # Get next batch
        input_ids, labels, attention_mask, decoder_input_ids = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_trained += tokens_per_iter

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        
        tokens_per_sec = tokens_per_iter / dt
        tokens_per_sec_M = tokens_per_sec / 1_000_000
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, tps (M) {tokens_per_sec_M:.2f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "train/iter_time_ms": dt * 1000,
                "train/tokens_per_sec_M": tokens_per_sec_M,
            }, step=iter_num)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group() 

# Final test
if master_process and iter_num > 100:
    print(f"\n{'='*60}")
    print("🎯 FINAL ViT5 MODEL TEST")
    print(f"{'='*60}")
    
    test_success = test_vit5_generation(raw_model, tokenizer, device)
    
    if test_success:
        print("✅ ViT5 test completed successfully!")
    else:
        print("❌ ViT5 test failed!")
        
    print(f"{'='*60}")

if master_process:
    print(f"\n🎉 ViT5 Fine-tuning test completed! Model saved to: {finetune_dir}") 