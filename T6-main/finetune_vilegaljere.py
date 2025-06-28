import os
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
import json

# -----------------------------------------------------------------------------
# -- FINE-TUNING CONFIGURATION --
# -----------------------------------------------------------------------------

# --- Cấu hình chung ---
eval_only = False
always_save_checkpoint = True

# Wandb API key
os.environ['WANDB_API_KEY'] = 'bcc183326224decc1f9fee116ccfd509e740fab1'

# --- CẤU HÌNH CHO FINE-TUNING ---
init_from = 'resume' # Bắt buộc phải resume từ model đã pre-trained
data_path = "/kaggle/input/vietnamese-legal-finetune-dataset" # Nơi chứa file finetune.json
finetune_file_name = "dataset.json"
out_dir = '/kaggle/working/vilegaljere_pretrain' # Thư mục chứa checkpoint pre-trained
finetune_dir = '/kaggle/working/vilegaljere_finetune'

# ✅ OPTIMIZED: Better hyperparameters for relation extraction fine-tuning
learning_rate = 1e-4 # Higher learning rate for fine-tuning stability
max_iters = 10000     # Sufficient iterations for fine-tuning convergence
batch_size = 32      # Smaller batch for better gradient stability
gradient_accumulation_steps = 4  # Maintain effective batch size of 64
weight_decay = 0.01  # Standard weight decay for transformer fine-tuning
eval_interval = 500  # More frequent evaluation for monitoring
log_interval = 10    # Keep logging frequency
eval_iters = 300      # Faster evaluation iterations
warmup_iters = 2000   # Shorter warmup for fine-tuning (10% of max_iters)

# wandb logging
wandb_log = True    # Enable for better tracking
wandb_project = 'ViLegalJERE'
wandb_run_name = 'vilegal_jere_finetune_kaggle'

# data
dataset = 'vietnamese_legal'
block_size = 512    # Keep same
max_source_length = 512  # encoder max length
max_target_length = 512  # decoder max length

# model - T5-small architecture (~60M parameters)
n_layer = 6         # T5-small has 6 layers each for encoder/decoder
n_head = 8          # T5-small uses 8 attention heads
head_dim = 64       # 512/8 = 64
rank = 4            # Reasonable CP rank for T6
q_rank = 8          # Reasonable query rank for T6
n_embd = 512        # T5-small hidden size
dropout = 0.1       # Standard dropout for T5
bias = False
using_groupnorm = True

# optimizer
optimizer_name = 'adamw'
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
lr_decay_iters = max_iters  # Match max_iters for proper decay schedule
min_lr = 5e-6        # Higher min_lr to avoid vanishing gradients
schedule = 'cosine'
model_type = 'ViLegalJERE'

# DDP settings for Kaggle T4x2
backend = 'gloo'  # Use gloo instead of nccl for better Kaggle compatibility

# system
device = 'cuda'  
dtype = 'float16'   
compile = False     # Disable compile for Kaggle compatibility
scale_attn_by_inverse_layer_idx = False

# -----------------------------------------------------------------------------

# Import ViLegalJERE model
from model.ViLegalJERE import ViLegalJERE

# Import utilities
from utils import (
    load_custom_tokenizer,
    print_tokenizer_info,
    get_num_params,
    setup_model_with_tokenizer,
    estimate_loss,
    get_lr,
    setup_distributed_training,
    setup_training_environment,
    setup_wandb,
    test_model_generation
)

# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# DDP setup using utils
ddp, master_process, seed_offset, world_size, device = setup_distributed_training(backend)

# Initialize tokenizer with domain tokens (after master_process is defined)
tokenizer = load_custom_tokenizer(master_process)

# Calculate total tokens
tokens_per_iter = batch_size * (max_source_length + max_target_length) * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)
tokens_trained = 0

# Setup training environment using utils
device_type, ctx = setup_training_environment(seed_offset, dtype)

# Data loading for fine-tuning
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
        # ✅ FIXED: Use correct field names from your data format
        source_text = value.get("input_text")
        target_text = value.get("extracted_relations")
        
        # ✅ Load preprocessed data directly
        if source_text and target_text:
            processed_data.append((source_text, target_text))
        else:
            skipped_count += 1
    
    if master_process:
        print(f"✅ Loaded {len(processed_data)} fine-tuning pairs")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} empty pairs")
        
        # ✅ SHOW sample data for verification
        if processed_data:
            print("\n📝 SAMPLE TRAINING DATA:")
            sample_input, sample_target = processed_data[0]
            print(f"📥 Input: {sample_input}...")
            print(f"🎯 Target: {sample_target}...")
    
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

def get_batch(split):
    """Get batch for fine-tuning with structured input/output"""
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError(f"Data split '{split}' is empty. Check data loading.")

    # ✅ IMPROVED FINE-TUNING BATCH PROCESSING
    ix = np.random.randint(len(data), size=(batch_size,))
    batch_pairs = [data[i] for i in ix]
    
    # Extract source and target texts
    source_texts = [pair[0] for pair in batch_pairs]
    target_texts = [pair[1] for pair in batch_pairs]
    
    # ✅ T5 STANDARD: Follow exact T5 documentation format
    # Encode inputs exactly like T5 docs
    input_encoding = tokenizer(
        source_texts,
        padding='longest',
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = input_encoding.input_ids
    attention_mask = input_encoding.attention_mask
    
    # Encode targets exactly like T5 docs  
    target_encoding = tokenizer(
        target_texts,
        padding='longest',
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt"
    )
    labels = target_encoding.input_ids
    
    # ✅ T5 STANDARD: Replace padding token ids with -100 (exact T5 docs method)
    # From docs: "replace padding token id's of the labels by -100"
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] 
        for labels_example in labels
    ]
    labels = torch.tensor(labels)

    # ✅ EFFICIENT GPU transfer
    if device_type == 'cuda':
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        labels = labels.to(device) 
        attention_mask = attention_mask.to(device)
    
    return input_ids, labels, attention_mask

# Model initialization arguments - T5 standard format  
model_args = dict(
    # ✅ T5 standard parameters
    vocab_size=len(tokenizer),
    d_model=n_embd,  # T5 uses d_model instead of n_embd
    num_layers=n_layer,  # T5 uses num_layers instead of n_layer  
    num_heads=n_head,  # T5 uses num_heads instead of n_head
    d_kv=head_dim,  # T5 uses d_kv for key/value dimension
    d_ff=4 * n_embd,  # T5 feed-forward dimension
    dropout_rate=dropout,  # T5 uses dropout_rate instead of dropout
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,
    
    # ✅ ViLegal custom parameters
    rank=rank,
    q_rank=q_rank,
)

# Print tokenizer info using utils
print_tokenizer_info(tokenizer, master_process)

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9

# --- FIXED MODEL INITIALIZATION WITH PROPER EMBEDDING RESIZE ---
if master_process:
    print(f"Resuming training from {out_dir}")

# Check if checkpoint directory exists
if not os.path.exists(out_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}. Cannot resume.")

# Load model from checkpoint
model = ViLegalJERE.from_pretrained(out_dir)

# Setup model with tokenizer using utils
setup_model_with_tokenizer(model, tokenizer, master_process)

model.to(device)

# Calculate parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000

if master_process:
    print(f"Model initialized with {param_count_m:.1f}M parameters")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(finetune_dir, exist_ok=True)

# Initialize scaler and optimizer
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# ✅ FINE-TUNING: Create fresh optimizer (không load từ pre-train)
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

if master_process:
    print("✅ Created fresh optimizer for fine-tuning (reset từ pre-train)")

# Compile model
if compile:
    print("Compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap with DDP
if ddp:
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# estimate_loss function is imported from utils

# get_lr function is imported from utils

# Logging setup
if wandb_log and master_process:
    import wandb
    
    # Tự động đăng nhập wandb
    try:
        wandb.login()
        print("✅ Đăng nhập wandb thành công!")
    except Exception as e:
        print(f"⚠️ Không thể đăng nhập wandb với API key: {e}")
        print("🔄 Chuyển sang chế độ anonymous...")
        wandb.login(anonymous="allow")
        print("✅ Sử dụng wandb ở chế độ anonymous!")
    
    wandb_config = {
        'model_args': model_args,
        'training_args': {
            'batch_size': batch_size,
            'max_source_length': max_source_length,
            'max_target_length': max_target_length,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'max_iters': max_iters,
            'lr_decay_iters': lr_decay_iters,
            'eval_interval': eval_interval,
            'eval_iters': eval_iters,
            'log_interval': log_interval
        },
        'optimizer_args': {
            'optimizer_name': optimizer_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'grad_clip': grad_clip,
            'decay_lr': decay_lr,
            'warmup_iters': warmup_iters,
            'min_lr': min_lr,
            'schedule': schedule
        }
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

# Training loop
if master_process:
    print(f"Starting Fine-tuning ViLegalJERE with {param_count_m:.1f}M parameters...")
    print(f"Training data size: {len(train_data)}, Val data size: {len(val_data)}")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
    print(f"Learning rate: {learning_rate}, Max iters: {max_iters}")

input_ids, labels, attention_mask = get_batch('train')

if master_process:
    print(f"First batch shapes - Input: {input_ids.shape}, Labels: {labels.shape}")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
clip_time = 0

while True:
    # Set learning rate
    lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr, schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation (skip first iteration)
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss(model, get_batch, eval_iters, ctx)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        print(f"\n🧪 VIETNAMESE LEGAL JERE EVALUATION (iter {iter_num})")
        
        # Test model generation
        raw_model = model.module if ddp else model
        test_success = test_model_generation(raw_model, tokenizer, device, master_process)
        
        if test_success:
            print("✅ Model generation test passed!")
        else:
            print("❌ Model generation test failed!")
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            
            log_dict["eval/generation_success"] = 1 if test_success else 0
            
            wandb.log(log_dict, step=iter_num)
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"Saving checkpoint to {finetune_dir}")
                raw_model.save_pretrained(finetune_dir, safe_serialization=False)
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(optimizer_state, os.path.join(finetune_dir, 'optimizer.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            # ✅ T5 STANDARD: Only pass input_ids, attention_mask, and labels
            # T5 automatically creates decoder_input_ids from labels
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            loss = loss / gradient_accumulation_steps
        
        # Get next batch
        input_ids, labels, attention_mask = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_trained += tokens_per_iter
    tokens_trained_B = tokens_trained / 1e9

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.get_num_params() * batch_size * gradient_accumulation_steps * (max_source_length + max_target_length) * 6 / (dt * 1e12)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        tokens_per_sec = tokens_per_iter / dt
        tokens_per_sec_M = tokens_per_sec / 1_000_000
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tps (M) {tokens_per_sec_M:.2f}, tokens trained {tokens_trained:.2f}B")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "train/clip_rate": clip_time / (iter_num + 1),
                "train/grad_norm": total_norm.item() if grad_clip != 0.0 else 0.0,
                "train/iter_time_ms": dt * 1000,
                "train/mfu": running_mfu * 100,
                "train/tokens_per_sec_M": tokens_per_sec_M,
                "train/tokens_trained_B": tokens_trained_B,
                "gpu/memory_allocated_MB": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu/max_memory_allocated_MB": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }, step=iter_num)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group() 

# ✅ Use test_model_generation from utils.py (imported above)

# ✅ Run test after training completed
if master_process and iter_num > 100:
    print(f"\n{'='*60}")
    print("🎯 FINAL MODEL TEST")
    print(f"{'='*60}")
    
    raw_model = model.module if ddp else model
    test_success = test_model_generation(raw_model, tokenizer, device)
    
    if test_success:
        print("✅ Model test completed successfully!")
    else:
        print("❌ Model test failed!")
        
    print(f"{'='*60}")

if master_process:
    print(f"\n🎉 Fine-tuning completed! Model saved to: {finetune_dir}") 