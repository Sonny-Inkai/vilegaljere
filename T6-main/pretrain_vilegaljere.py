import os
import time
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
# Import ViLegalJERE model
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

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
    qualitative_pretrain_test
)

# -----------------------------------------------------------------------------
# -- PRE-TRAINING CONFIGURATION --
# -----------------------------------------------------------------------------

# --- Cấu hình chung ---
eval_only = False
always_save_checkpoint = True

# Wandb API key
os.environ['WANDB_API_KEY'] = 'bcc183326224decc1f9fee116ccfd509e740fab1'

# --- CẤU HÌNH CHO PRE-TRAINING ---
init_from = 'scratch' # 'scratch' or 'resume'
data_path = "/kaggle/input/vietnamese-legal-pretrain-dataset"  # Kaggle dataset path
out_dir = '/kaggle/working/vilegaljere_pretrain'

# ✅ OPTIMIZED: Hyperparameters for pre-training on T4x2
learning_rate = 3e-4  # Standard for T5-small pre-training
max_iters = 1     # Reasonable for T5-small
batch_size = 32       # Safe for T4 memory
gradient_accumulation_steps = 4   # Maintain large effective batch
weight_decay = 1e-2   # Standard T5 weight decay
eval_interval = 1   # Less frequent for pre-training
log_interval = 1     # Reduce logging overhead
eval_iters = 1      # Keep reasonable for evaluation
warmup_iters = 1000   # 10% of max_iters

# wandb logging
wandb_log = True
wandb_project = 'ViLegalJERE'
wandb_run_name = 'vilegal_jere_pretrain_kaggle'

# data
dataset = 'vietnamese_legal'
block_size = 512
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
lr_decay_iters = max_iters
min_lr = 5e-6
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

# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# DDP setup using utils
ddp, master_process, seed_offset, world_size, device = setup_distributed_training(backend)

# Initialize tokenizer with domain tokens (after master_process is defined)
tokenizer = load_custom_tokenizer(master_process)

# Set the test sentence here
QUALITATIVE_TEST_SENTENCE = "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở. 3. Việc phối hợp phải thường xuyên, kịp thời, đồng bộ, chặt chẽ, thống nhất, đúng quy định của pháp luật."

# Calculate total tokens
tokens_per_iter = batch_size * (max_source_length + max_target_length) * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)
tokens_trained = 0

# Setup training environment using utils
device_type, ctx = setup_training_environment(seed_offset, dtype)

# Data loading for Vietnamese legal text (PRE-TRAINING)
def load_legal_data():
    """Load preprocessed Vietnamese legal data for pre-training"""
    data_file = os.path.join(data_path, 'dataset.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # ✅ Load preprocessed articles (one per line)
    articles = [art.strip() for art in text.split('\n') if art.strip()]
    
    if not articles:
        raise ValueError("No articles found in dataset")
        
    return articles

# ✅ T5 span corruption functions for pre-training
def random_spans_helper(inputs_length, noise_density, mean_noise_span_length, 
                       extra_tokens_per_span_inputs=1, extra_tokens_per_span_targets=1):
    """Calculate input and target lengths for span corruption"""
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), tokens_length - 1)
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        
        # Calculate input length (original tokens - noise + sentinel tokens)
        inputs_length = num_nonnoise_tokens + num_noise_spans * extra_tokens_per_span_inputs
        
        # Calculate target length (noise tokens + sentinel tokens)  
        targets_length = num_noise_tokens + num_noise_spans * extra_tokens_per_span_targets
        
        return inputs_length, targets_length
    
    return _tokens_length_to_inputs_length_targets_length(inputs_length)

def create_noise_mask(length, noise_density, mean_noise_span_length):
    """Create random spans noise mask like Google T5"""
    if noise_density == 0.0:
        return [False] * length
    
    # Increase length to avoid degeneracy    
    length = max(length, 2)
    
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = max(1, int(round(num_noise_tokens / mean_noise_span_length)))
    num_nonnoise_tokens = length - num_noise_tokens
    
    def random_segmentation(num_items, num_segments):
        """Partition a sequence randomly into non-empty segments"""
        if num_segments >= num_items:
            return [1] * num_items
        
        # Create random breakpoints
        breaks = sorted(np.random.choice(num_items - 1, num_segments - 1, replace=False))
        breaks = [0] + [b + 1 for b in breaks] + [num_items]
        
        # Calculate segment lengths
        lengths = [breaks[i+1] - breaks[i] for i in range(len(breaks) - 1)]
        return lengths
    
    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans)
    
    # Interleave spans starting with non-noise
    interleaved_span_lengths = []
    for i in range(num_noise_spans):
        interleaved_span_lengths.append(nonnoise_span_lengths[i])
        interleaved_span_lengths.append(noise_span_lengths[i])
    
    # Create mask
    mask = []
    is_noise = False
    for span_length in interleaved_span_lengths:
        mask.extend([is_noise] * span_length)
        is_noise = not is_noise
    
    return mask[:length]

def create_t5_spans(tokens, tokenizer, noise_density=0.15, mean_noise_span_length=3.0):
    """
    ✅ HF CHUẨN: T5 span corruption exactly like DataCollatorForT5MLM
    """
    import numpy as np
    
    num_tokens = len(tokens)
    if num_tokens <= 1:
        return [], []

    # Use same noise mask logic as HF DataCollatorForT5MLM
    noise_mask = create_noise_mask(num_tokens, noise_density, mean_noise_span_length)
    
    tokens = np.array(tokens)
    noise_mask = np.array(noise_mask, dtype=bool)
    
    # ✅ HF STANDARD: Create sentinel IDs exactly like DataCollatorForT5MLM
    # Find start of each noise span
    start_indices = noise_mask.astype(int) - np.roll(noise_mask.astype(int), 1) * noise_mask.astype(int)
    start_indices[0] = noise_mask[0]
    
    # Create cumulative sentinel indices
    sentinel_indices = np.where(start_indices != 0, np.cumsum(start_indices), start_indices)
    
    # Get sentinel token IDs (extra_id_0, extra_id_1, ...)
    sentinel_base_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    input_sentinel_ids = np.where(sentinel_indices != 0, (sentinel_base_id - sentinel_indices + 1), 0)
    
    # Remove consecutive masks after first sentinel
    input_sentinel_ids = input_sentinel_ids - noise_mask.astype(int) + start_indices
    
    # Create input: original tokens with noise spans replaced by sentinels
    input_ids_full = np.where(input_sentinel_ids != 0, input_sentinel_ids, tokens)
    # Keep only non-negative values (removes -1 markers)
    input_ids = input_ids_full[input_ids_full >= 0]
    
    # Create labels: sentinels followed by the original noise tokens
    labels_mask = ~noise_mask  # Invert for labels
    labels_start_indices = noise_mask.astype(int) - np.roll(noise_mask.astype(int), 1) * noise_mask.astype(int)
    labels_start_indices[0] = noise_mask[0]
    
    labels_sentinel_indices = np.where(labels_start_indices != 0, np.cumsum(labels_start_indices), labels_start_indices)
    labels_sentinel_ids = np.where(labels_sentinel_indices != 0, (sentinel_base_id - labels_sentinel_indices + 1), 0)
    
    # For labels: include sentinels and noise tokens, exclude non-noise
    labels_full = np.where(labels_sentinel_ids != 0, labels_sentinel_ids, tokens)
    # Keep sentinel tokens and noise tokens
    keep_mask = (labels_sentinel_ids != 0) | noise_mask
    labels = labels_full[keep_mask]
    
    # Add EOS token
    labels = np.append(labels, tokenizer.eos_token_id)
    
    return input_ids.tolist(), labels.tolist()

# Load data for pre-training
if master_process:
    print("Loading Vietnamese legal data for pre-training...")

all_data = load_legal_data()

if master_process:
    print(f"Loaded {len(all_data)} legal articles")

# Split train/val
split_idx = int(0.9 * len(all_data))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

if master_process:
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

def get_batch(split):
    """
    ✅ HF CHUẨN: Get batch for pre-training working directly with token IDs
    No decode/encode - pure token ID manipulation like HuggingFace DataCollatorForT5MLM
    """
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError(f"Data split '{split}' is empty. Check data loading.")

    # 1. Sample articles and create spans
    ix = np.random.randint(len(data), size=(batch_size,))
    
    batch_input_ids = []
    batch_labels = []
    
    for i in range(batch_size):
        article_text = data[ix[i]]
        
        # Tokenize article
        article_tokens = tokenizer.encode(
            article_text, 
            truncation=True, 
            max_length=block_size, 
            add_special_tokens=False
        )
        
        # Create T5 spans (returns token IDs directly)
        input_ids, labels = create_t5_spans(article_tokens, tokenizer)
        
        if input_ids and labels:
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
    
    # 2. ✅ PYTORCH BUILT-IN: Use efficient pad_sequence function
    from torch.nn.utils.rnn import pad_sequence
    
    # Convert to tensors first
    batch_input_tensors = [torch.tensor(seq, dtype=torch.long) for seq in batch_input_ids]
    batch_label_tensors = [torch.tensor(seq, dtype=torch.long) for seq in batch_labels]
    
    # Use PyTorch's optimized padding
    input_ids = pad_sequence(batch_input_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(batch_label_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # 3. Ensure fixed lengths for T5 (truncate or pad to exact length)
    if input_ids.size(1) > max_source_length:
        input_ids = input_ids[:, :max_source_length]
    elif input_ids.size(1) < max_source_length:
        padding_needed = max_source_length - input_ids.size(1)
        padding = torch.full((input_ids.size(0), padding_needed), tokenizer.pad_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, padding], dim=1)
    
    if labels.size(1) > max_target_length:
        labels = labels[:, :max_target_length]
    elif labels.size(1) < max_target_length:
        padding_needed = max_target_length - labels.size(1)
        padding = torch.full((labels.size(0), padding_needed), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.cat([labels, padding], dim=1)
    
    # Create attention mask (1 for non-pad tokens, 0 for pad tokens)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # ✅ T5 STANDARD: Replace padding token ids in labels with -100
    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)

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

# --- MODEL INITIALIZATION ---
if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)
    
    # Setup model with tokenizer using utils
    setup_model_with_tokenizer(model, tokenizer, master_process)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}")

    model = ViLegalJERE.from_pretrained(out_dir)
    
    # Setup model with tokenizer using utils
    setup_model_with_tokenizer(model, tokenizer, master_process)
    
    # Load optimizer state if available
    optimizer_state_path = os.path.join(out_dir, 'optimizer.pt')
    if os.path.exists(optimizer_state_path):
        checkpoint = torch.load(optimizer_state_path, map_location=device)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        if master_process:
            print(f"✅ Resumed successfully from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")
    else:
        if master_process:
            print(f"⚠️ Warning: optimizer.pt not found in {out_dir}. Starting optimizer from scratch.")

model.to(device)

# Calculate parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000

if master_process:
    print(f"Model initialized with {param_count_m:.1f}M parameters")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize scaler and optimizer
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

# Load optimizer state if resume
if init_from == 'resume':
    optimizer_state_path = os.path.join(out_dir, 'optimizer.pt')
    if os.path.exists(optimizer_state_path):
        checkpoint = torch.load(optimizer_state_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if master_process:
            print("Optimizer state loaded successfully")

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

# Setup wandb logging using utils
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
wandb_initialized = setup_wandb(wandb_log, master_process, wandb_project, wandb_run_name, wandb_config)

# Training loop
if master_process:
    print(f"Starting Pre-training ViLegalJERE with {param_count_m:.1f}M parameters...")
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
        
        if wandb_initialized:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }, step=iter_num)
        
        # CALL THE QUALITATIVE TEST FUNCTION HERE
        qualitative_pretrain_test(raw_model, tokenizer, device, QUALITATIVE_TEST_SENTENCE, create_t5_spans)
        
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

        if wandb_initialized:
            import wandb
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

if master_process:
    print(f"\n🎉 Pre-training completed! Model saved to: {out_dir}") 