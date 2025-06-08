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
# -- CÔNG TẮC CHÍNH --
finetune = False # Đổi thành True khi cháu muốn chạy fine-tuning
# -----------------------------------------------------------------------------

# --- Cấu hình chung ---
eval_only = False
always_save_checkpoint = True

# Trong Kaggle notebook, thêm vào cell đầu tiên:
import os
os.environ['WANDB_API_KEY'] = 'bcc183326224decc1f9fee116ccfd509e740fab1'

# --- Cấu hình riêng cho từng giai đoạn ---
if finetune:
    # --- CẤU HÌNH CHO FINE-TUNING ---
    init_from = 'resume' # Bắt buộc phải resume từ model đã pre-trained
    data_path = "/kaggle/input/vietnamese-legal-dataset-finetuning" # Nơi chứa file finetune.txt
    finetune_file_name = "finetune.json"
    out_dir = '/kaggle/working/out_vilegal_t5small' # Thư mục chứa checkpoint pre-trained
    
    # Siêu tham số cho fine-tuning
    learning_rate = 3e-5 # Learning rate nhỏ hơn nhiều
    max_iters = 10    # Số vòng lặp ít hơn
    batch_size = 1       # Batch size có thể nhỏ hơn
    gradient_accumulation_steps = 2
    weight_decay = 0.01
    eval_interval = 1
    log_interval = 1
    eval_iters = 1
    
else:
    # --- CẤU HÌNH CHO PRE-TRAINING ---
    init_from = 'scratch' # 'scratch' or 'resume'
    data_path = "/kaggle/input/vietnamese-legal-dataset"  # Kaggle dataset path
    out_dir = '/kaggle/working/out_vilegal_t5small'
    
    # Siêu tham số cho pre-training
    learning_rate = 1e-4  # Good for T5-small
    max_iters = 2     # Very small for testing
    batch_size = 1      # Even smaller for T4 memory constraints
    gradient_accumulation_steps = 1   # Reduced to avoid memory issues
    weight_decay = 1e-2
    eval_interval = 1  # More frequent eval for shorter training
    log_interval = 1   # More frequent logging
    eval_iters = 1     # Fewer eval iterations to save time
    
# wandb logging
wandb_log = True    # Enable for better tracking
wandb_project = 'ViLegalJERE-T5Small'
wandb_run_name = 'vilegal_t5small_kaggle'
# data
dataset = 'vietnamese_legal'
block_size = 512    # Keep same
max_source_length = 512  # encoder max length
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
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.9
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
dtype = 'float16'   # Use float32 for kaggle t4x2
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
    """Tải dữ liệu cho pre-training"""
    data_file = os.path.join(data_path, 'dataset.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    articles = [art for art in text][:10]
    tokenized_data = [tokenizer.encode(art, truncation=True, max_length=block_size) for art in articles]
    return tokenized_data[:10]

def load_finetune_data():
    """Tải và xử lý dữ liệu từ file finetune.json (JSON) cho fine-tuning"""
    data_file = os.path.join(data_path, finetune_file_name)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Finetune dataset not found at {data_file}")
    
    processed_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, value in data.items():
        source_text = value.get("formatted_context_sent", "")
        target_text = value.get("extracted_relations_text", "")
        
        # Chỉ lấy các cặp dữ liệu có cả input và output
        if source_text and target_text:
            processed_data.append((source_text, target_text))
    
    if master_process:
        print(f"Loaded {len(processed_data)} fine-tuning pairs")
    return processed_data[:10]

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

# Load data based on mode
if master_process: 
    print(f"Running in '{'Fine-tuning' if finetune else 'Pre-training'}' mode.")

if finetune:
    print("Loading fine-tuning data...")
    all_data = load_finetune_data()
else:
    print("Loading Vietnamese legal data for pre-training...")
    all_data = load_legal_data()
    print(f"Loaded {len(all_data)} legal articles")

# Split train/val
split_idx = int(0.9 * len(all_data))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]
print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

def get_batch(split):
    """
    HÀM GET_BATCH ĐA NĂNG - Hỗ trợ cả pre-training và fine-tuning
    """
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError(f"Data split '{split}' is empty. Check data loading.")

    if finetune:
        # --- LẤY BATCH CHO FINE-TUNING ---
        ix = np.random.randint(len(data), size=(batch_size,))
        batch_pairs = [data[i] for i in ix]
        
        # Lấy các cặp (context, relations)
        source_texts = [pair[0] for pair in batch_pairs]
        target_texts = [pair[1] for pair in batch_pairs]
        
        # Tokenize source và target
        input_encodings = tokenizer(source_texts, padding=True, truncation=True, 
                                  max_length=max_source_length, return_tensors="pt")
        target_encodings = tokenizer(target_texts, padding=True, truncation=True, 
                                   max_length=max_target_length, return_tensors="pt")
        
        input_ids = input_encodings.input_ids
        attention_mask = input_encodings.attention_mask
        labels = target_encodings.input_ids
        
        # Tạo decoder_attention_mask
        # This mask is based on shifted labels to correctly mask padding in the decoder's self-attention.
        temp_decoder_input_ids = torch.cat([torch.full((labels.shape[0], 1), tokenizer.pad_token_id), labels[:, :-1]], dim=-1)
        decoder_attention_mask = (temp_decoder_input_ids != tokenizer.pad_token_id).float()
        
    else:
        # --- LẤY BATCH CHO PRE-TRAINING (logic cũ) ---
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
        
        # Tạo decoder_attention_mask cho pre-training
        temp_decoder_input_ids = torch.cat([torch.full((labels.shape[0], 1), tokenizer.pad_token_id), labels[:, :-1]], dim=-1)
        decoder_attention_mask = (temp_decoder_input_ids != tokenizer.pad_token_id).float()

    # Chuyển lên GPU
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
    
    # Trả về dữ liệu theo format model cần
    return input_ids, labels, labels, attention_mask, decoder_attention_mask

# Model initialization arguments
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

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9

# --- KHỐI KHỞI TẠO MODEL VÀ RESUME ĐÃ SỬA LẠI ---
if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    # Khởi tạo model từ đầu với các tham số đã định nghĩa
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    
    # Kiểm tra xem thư mục checkpoint có tồn tại không
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}. Cannot resume.")

    # Tải lại model từ checkpoint đã lưu. 
    # from_pretrained sẽ tự động tải cả config và trọng số
    model = ViLegalJERE.from_pretrained(out_dir)
    
    # Tải lại trạng thái của optimizer và các biến tiến trình
    optimizer_state_path = os.path.join(out_dir, 'optimizer.pt')
    if os.path.exists(optimizer_state_path):
        checkpoint = torch.load(optimizer_state_path, map_location=device)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        if master_process:
            print(f"Resumed successfully from iteration {iter_num} with best_val_loss {best_val_loss:.4f}")
    else:
        # Nếu không tìm thấy file optimizer, bắt đầu từ đầu nhưng vẫn dùng model đã tải
        print(f"Warning: optimizer.pt not found in {out_dir}. Starting optimizer from scratch.")
        # iter_num và best_val_loss sẽ giữ giá trị mặc định là 0 và 1e9

# Fix vocabulary size mismatch between model and tokenizer
if model.config.vocab_size != len(tokenizer):
    if master_process:
        print(f"Vocab size mismatch: model={model.config.vocab_size}, tokenizer={len(tokenizer)}")
        print("Resizing model embeddings to match tokenizer...")
    model.resize_token_embeddings(len(tokenizer))
    if master_process:
        print(f"Model embeddings resized to {len(tokenizer)}")

model.to(device)

# Calculate parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000

print(f"Model initialized with {param_count_m:.1f}M parameters")

# Update output directory for Kaggle
if init_from != 'resume':
    wandb_run_name = f"ViLegal_{int(param_count_m)}m_T5small_Kaggle_{current_date}"
    out_dir = f"/kaggle/working/out_vilegal_t5small"

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize scaler and optimizer
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

# Tải lại trạng thái optimizer nếu resume và file tồn tại
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
    
    # Tự động đăng nhập wandb
    try:
        # Thử đăng nhập bằng API key từ environment hoặc file config
        wandb.login()
        print("✅ Đăng nhập wandb thành công!")
    except Exception as e:
        print(f"⚠️ Không thể đăng nhập wandb với API key: {e}")
        print("🔄 Chuyển sang chế độ anonymous...")
        # Fallback sang chế độ anonymous nếu không có API key
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
mode_text = "Fine-tuning" if finetune else "Pre-training"
print(f"Starting {mode_text} ViLegalJERE with {param_count_m:.1f}M parameters...")
print(f"Training data size: {len(train_data)}, Val data size: {len(val_data)}")
print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
print(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
print(f"Mode: {mode_text}, Learning rate: {learning_rate}, Max iters: {max_iters}")

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