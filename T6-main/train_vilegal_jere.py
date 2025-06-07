import os
import time
import math
import pickle
from contextlib import nullcontext
import importlib
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
import json
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Config optimized for T5-small (~60M params) on Kaggle T4x2 GPUs
# I/O
data_path = "/kaggle/input/vietnamese-legal-dataset"  # Kaggle dataset path
out_dir = '/kaggle/working/out_vilegal_t5small'
resume_dir = '.'
eval_interval = 1  # More frequent eval for shorter training
log_interval = 1   # More frequent logging
eval_iters = 1     # Fewer eval iterations to save time
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = False    # Enable for better tracking
wandb_project = 'ViLegalJERE-T5Small'
wandb_run_name = 'vilegal_t5small_kaggle'
# data
dataset = 'vietnamese_legal'
gradient_accumulation_steps = 1   # Reduced to avoid memory issues
batch_size = 1      # Even smaller for T4 memory constraints
block_size = 512    # Keep same
max_source_length = 256  # encoder max length
max_target_length = 256  # decoder max length
# model - T5-small architecture (~60M parameters)
n_layer = 6         # T5-small has 6 layers each for encoder/decoder
n_head = 8          # T5-small uses 8 attention heads
head_dim = 64       # 512/8 = 64
rank = 4
q_rank = 8
n_embd = 512        # T5-small hidden size
dropout = 0.1       # Standard dropout for T5
bias = False
using_groupnorm = True
# optimizer
optimizer_name = 'adamw'
learning_rate = 1e-4  # Good for T5-small
max_iters = 10     # Reduced for smaller model
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000   # Longer warmup for stability
lr_decay_iters = 50000
min_lr = 1e-6
# DDP settings for Kaggle T4x2
backend = 'gloo'  # Use gloo instead of nccl for better Kaggle compatibility
schedule = 'cosine'
model_type = 'ViLegalJERE'
# system
device = 'cuda'
dtype = 'float16'   # Use float16 for better T4 performance
compile = False     # Disable compile for Kaggle compatibility
scale_attn_by_inverse_layer_idx = False
# -----------------------------------------------------------------------------

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
config = {k: globals()[k] for k in config_keys}

# Import ViLegalJERE model
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')

# Ensure tokenizer has required special tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = len(tokenizer) - 1

# Add extra_id tokens if not present
if not hasattr(tokenizer, 'additional_special_tokens_ids') or len(tokenizer.additional_special_tokens_ids) < 100:
    extra_tokens = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_tokens({'additional_special_tokens': extra_tokens})

def get_num_params(model, non_embedding=False):
    """Return the number of parameters in the model."""
    n_params = sum(p.numel() for p in model.parameters())
    if non_embedding and hasattr(model, 'shared'):
        n_params -= model.shared.weight.numel()
    return n_params

# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    world_size = 1
    # Keep gradient_accumulation_steps as is for single GPU

# Calculate total tokens
tokens_per_iter = batch_size * (max_source_length + max_target_length) * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)
tokens_trained = 0

# Initialize random seed
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Data loading for Vietnamese legal text
def load_legal_data():
    data_file = os.path.join(data_path, 'dataset.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    articles = text.split('Điều ')
    articles = [f"Điều {art.strip()}" for art in articles[1:]][:10]
    tokenized_data = [tokenizer.encode(art, truncation=True, max_length=block_size) for art in articles]
    return tokenized_data

def create_t5_spans(tokens, noise_density=0.15, mean_noise_span_length=3.0):
    """
    Tạo dữ liệu theo kiểu span corruption của T5 với LOGIC ĐÚNG.
    """
    num_tokens = len(tokens)
    num_noise_tokens = int(round(num_tokens * noise_density))
    if num_noise_tokens == 0:
        return tokens, tokens

    # Chọn ngẫu nhiên các vị trí để bắt đầu che
    noise_indices = np.random.choice(range(num_tokens), num_noise_tokens, replace=False)
    noise_mask = np.zeros(num_tokens, dtype=bool)
    noise_mask[noise_indices] = True
    
    # Lấy ID của sentinel token đầu tiên (<extra_id_0>) một cách an toàn
    try:
        sentinel_start_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        if sentinel_start_id == tokenizer.unk_token_id: raise ValueError
    except (KeyError, ValueError):
        sentinel_start_id = len(tokenizer) - 1 # Fallback an toàn
    
    input_ids = []
    labels = []
    
    in_noise_span = False
    sentinel_idx = 0
    
    for i in range(num_tokens):
        if noise_mask[i]:
            if not in_noise_span:
                # Bắt đầu một vùng nhiễu mới
                # DÙNG PHÉP TRỪ để có ID sentinel đúng (10099, 10098, ...)
                sentinel_id = sentinel_start_id - sentinel_idx
                input_ids.append(sentinel_id)
                labels.append(sentinel_id)
                sentinel_idx += 1
            in_noise_span = True
            labels.append(tokens[i])
        else:
            if in_noise_span:
                # Kết thúc vùng nhiễu trước đó
                in_noise_span = False
            input_ids.append(tokens[i])
    
    labels.append(tokenizer.eos_token_id)
    
    return input_ids, labels

# Load data
print("Loading Vietnamese legal data...")
tokenized_data = load_legal_data()
print(f"Loaded {len(tokenized_data)} legal articles")

# Split train/val
split_idx = int(0.95 * len(tokenized_data))
train_data = tokenized_data[:split_idx]
val_data = tokenized_data[split_idx:]

def get_batch(split):
    """
    Lấy batch dữ liệu và tạo attention mask ĐÚNG CÁCH.
    """
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError(f"Data split '{split}' is empty. Check data loading.")

    batch_input_ids = []
    batch_labels = []
    
    for _ in range(batch_size):
        article_tokens = data[np.random.randint(len(data))]
        input_tokens, target_tokens = create_t5_spans(article_tokens)
        
        # Cắt bớt và đệm
        input_padded = input_tokens[:max_source_length] + [tokenizer.pad_token_id] * (max_source_length - len(input_tokens))
        labels_padded = target_tokens[:max_target_length] + [tokenizer.pad_token_id] * (max_target_length - len(target_tokens))
        
        batch_input_ids.append(input_padded)
        batch_labels.append(labels_padded)
    
    # Chuyển thành tensor
    input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    labels = torch.tensor(batch_labels, dtype=torch.long)
    
    # Tạo attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).float()
    
    # T5 tự tạo decoder_input_ids từ labels, nên chúng ta không cần tạo decoder_attention_mask thủ công.
    # Hoặc nếu mô hình cần, ta có thể tạo nó. Để an toàn, chúng ta sẽ tạo nó.
    # LƯU Ý: labels cũng chính là đầu vào cho decoder_input_ids
    temp_decoder_input_ids = torch.cat([torch.full((labels.shape[0], 1), tokenizer.pad_token_id), labels[:, :-1]], dim=-1)
    decoder_attention_mask = (temp_decoder_input_ids != tokenizer.pad_token_id).float()

    if device_type == 'cuda':
        input_ids, labels, attention_mask, decoder_attention_mask = (
            input_ids.pin_memory().to(device, non_blocking=True),
            labels.pin_memory().to(device, non_blocking=True),
            attention_mask.pin_memory().to(device, non_blocking=True),
            decoder_attention_mask.pin_memory().to(device, non_blocking=True)
        )
    else:
        input_ids, labels, attention_mask, decoder_attention_mask = (
            input_ids.to(device), labels.to(device), attention_mask.to(device), decoder_attention_mask.to(device)
        )
    
    # Trả về labels cho cả decoder_input_ids và labels, cùng với attention masks
    return input_ids, labels, labels, attention_mask, decoder_attention_mask

# Initialize tracking
iter_num = 0
best_val_loss = 1e9

# Model initialization
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    head_dim=head_dim, 
    rank=rank, 
    q_rank=q_rank, 
    using_groupnorm=using_groupnorm,
    vocab_size=len(tokenizer),  # Use actual tokenizer size after adding special tokens
    dropout=dropout,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id
)

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

if init_from == 'scratch':
    print("Initializing ViLegalJERE model from scratch")
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)
elif init_from == 'resume':
    print(f"Resuming training from {resume_dir}")
    config_obj = ViLegalConfig.from_json_file(os.path.join(resume_dir, 'config.json'))
    model = ViLegalJERE.from_pretrained(resume_dir, config=config_obj)

# Resize model embeddings to match tokenizer
if hasattr(model, 'resize_token_embeddings'):
    model.resize_token_embeddings(len(tokenizer))

model.to(device)

# Calculate parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000

print(f"Model initialized with {param_count_m:.1f}M parameters")

# Update output directory for Kaggle
if init_from != 'resume':
    wandb_run_name = f"ViLegal_{int(param_count_m)}m_T5small_Kaggle_{current_date}"
    out_dir = f"/kaggle/working/vilegal_{int(param_count_m)}m_checkpoint"

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize scaler and optimizer
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

if init_from == 'resume':
    optimizer_state = torch.load(os.path.join(resume_dir, 'optimizer.pt'), map_location=device)
    optimizer.load_state_dict(optimizer_state['optimizer'])
    iter_num = optimizer_state['iter_num']
    best_val_loss = optimizer_state['best_val_loss']

# Compile model
if compile:
    print("Compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap with DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train/val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch(split)
            with ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, schedule='cosine'):
    """Learning rate scheduler"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Logging setup
if wandb_log and master_process:
    import wandb
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
print(f"Starting training ViLegalJERE with {param_count_m:.1f}M parameters...")
print(f"Training data size: {len(train_data)}, Val data size: {len(val_data)}")
print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
print(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")

input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch('train')
print(f"First batch shapes - Input: {input_ids.shape}, Decoder: {decoder_input_ids.shape}, Labels: {labels.shape}")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
clip_time = 0

while True:
    # Set learning rate
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation (skip first iteration)
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }, step=iter_num)
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"Saving checkpoint to {out_dir}")
                raw_model.save_pretrained(out_dir, safe_serialization=False)
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(optimizer_state, os.path.join(out_dir, 'optimizer.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            loss = loss / gradient_accumulation_steps
        
        # Get next batch
        input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch('train')
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
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tps (M) {tokens_per_sec_M:.2f}, tokens trained {tokens_trained_B:.2f}B")

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