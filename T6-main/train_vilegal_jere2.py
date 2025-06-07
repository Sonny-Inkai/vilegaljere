import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
from transformers import AutoTokenizer

from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

# -----------------------------------------------------------------------------
# Cấu hình mặc định
# I/O
data_path = "/kaggle/input/vietnamese-legal-dataset"
out_dir = '/kaggle/working/output'
resume_dir = '.'
eval_interval = 1
log_interval = 1
eval_iters = 1
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb
wandb_log = False
wandb_project = 'ViLegalJERE'
wandb_run_name = 'vilegal-run'
# data
dataset = 'vietnamese_legal'
gradient_accumulation_steps = 1
batch_size = 1
block_size = 256
max_source_length = 128
max_target_length = 128
# model
n_layer = 6
n_head = 8
head_dim = 64
rank = 2
q_rank = 4
n_embd = 512
dropout = 0.1
bias = False
using_groupnorm = True
# optimizer
optimizer_name = 'adamw'
learning_rate = 3e-4
max_iters = 1
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
# learning rate decay
decay_lr = True
warmup_iters = 1
lr_decay_iters = 1
min_lr = 3e-5
# DDP
backend = 'nccl'
# system
device = 'cuda'
dtype = 'float16'
compile = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
config = {k: globals()[k] for k in config_keys}

# --- DDP SETUP ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
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

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# --- TOKENIZER & MODEL SETUP ---
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
# Tính toán vocab_size thực tế cần thiết
all_special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.additional_special_tokens]
max_token_id = max([tokenizer.vocab_size - 1] + all_special_token_ids)
actual_vocab_size = max_token_id + 1
print(f"Tokenizer loaded. Calculated vocab_size: {actual_vocab_size}")


# --- DATA LOADING AND PREPROCESSING ---
def load_legal_data():
    data_file = os.path.join(data_path, 'dataset.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    articles = text.split('Điều ')
    articles = [f"Điều {art.strip()}" for art in articles[1:] if len(art.strip()) > 50]
    tokenized_data = [tokenizer.encode(art, truncation=True, max_length=block_size) for art in articles]
    return tokenized_data

# =================================================================================
# == HÀM QUAN TRỌNG NHẤT CẦN SỬA LẠI LOGIC TẠO SENTINEL TOKEN
# =================================================================================
def create_t5_spans(tokens, noise_density=0.15, mean_noise_span_length=3.0):
    num_tokens = len(tokens)
    num_noise_tokens = int(round(num_tokens * noise_density))
    if num_noise_tokens == 0:
        return tokens, tokens + [tokenizer.eos_token_id]

    # Chọn ngẫu nhiên các vị trí để bắt đầu che (mask)
    noise_indices = sorted(np.random.choice(range(num_tokens), num_noise_tokens, replace=False))

    input_tokens = []
    target_tokens = []
    
    # --- LOGIC ĐÃ SỬA ---
    # 1. Lấy ID của <extra_id_0> một cách tường minh. Đây là ID cao nhất trong dải sentinel.
    try:
        extra_id_0_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        if extra_id_0_id == tokenizer.unk_token_id:
            raise ValueError
    except (KeyError, ValueError):
        # Fallback an toàn nếu tokenizer không có extra_id
        print("CẢNH BÁO: Không tìm thấy <extra_id_0>. Dùng fallback.")
        extra_id_0_id = actual_vocab_size - 1

    prev_idx = -1
    sentinel_idx = 0
    for idx in noise_indices:
        if idx > prev_idx:
            # Nếu có khoảng trống giữa các vùng nhiễu, thêm sentinel token
            if prev_idx != -1:
                target_tokens.extend(tokens[prev_idx+1:idx])
            
            # Thêm sentinel token vào target
            # Dùng phép trừ để có ID đúng: 10099, 10098, 10097,...
            target_sentinel_id = extra_id_0_id - sentinel_idx
            target_tokens.append(target_sentinel_id)
            sentinel_idx += 1
        
        input_tokens.extend(tokens[prev_idx+1:idx])
        prev_idx = idx

    # Thêm các token còn lại sau vùng nhiễu cuối cùng
    input_tokens.extend(tokens[prev_idx+1:])
    target_tokens.extend(tokens[prev_idx+1:])
    target_tokens.append(tokenizer.eos_token_id)

    # Thay thế các vùng nhiễu trong input bằng sentinel token
    final_input = []
    in_masked_span = False
    sentinel_idx = 0
    for i in range(num_tokens):
        if i in noise_indices:
            if not in_masked_span:
                # Dùng phép trừ để có ID đúng: 10099, 10098, 10097,...
                input_sentinel_id = extra_id_0_id - sentinel_idx
                final_input.append(input_sentinel_id)
                sentinel_idx += 1
                in_masked_span = True
        else:
            final_input.append(tokens[i])
            in_masked_span = False

    return final_input, target_tokens

if master_process:
    print("Loading data...")
all_data = load_legal_data()
split_idx = int(0.95 * len(all_data))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]
if master_process:
    print(f"Data loaded: {len(train_data)} train, {len(val_data)} val samples.")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError("Data split is empty. Check data loading and splitting.")
        
    ix = torch.randint(len(data), (batch_size,))
    
    batch_inputs, batch_targets = [], []
    for i in ix:
        input_tokens, target_tokens = create_t5_spans(data[i])
        
        input_padded = input_tokens[:max_source_length] + [tokenizer.pad_token_id] * (max_source_length - len(input_tokens))
        target_padded = target_tokens[:max_target_length] + [tokenizer.pad_token_id] * (max_target_length - len(target_tokens))

        batch_inputs.append(input_padded)
        batch_targets.append(target_padded)
        
    x = torch.tensor(batch_inputs, dtype=torch.long)
    y = torch.tensor(batch_targets, dtype=torch.long)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    # Decoder input is the same as labels, but shifted right
    # T5 models handle this internally with decoder_start_token_id
    # We pass 'y' to both decoder_input_ids and labels
    return x, y

# --- Model init ---
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=actual_vocab_size, dropout=dropout,
    head_dim=head_dim, rank=rank, q_rank=q_rank, using_groupnorm=using_groupnorm,
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id # T5 often starts with pad
)

if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    config = ViLegalConfig(**model_args)
    model = ViLegalJERE(config)
else:
    # resume training logic
    pass

model.to(device)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    # find_unused_parameters=True có thể cần thiết nếu một số tham số của
    # mô hình không được sử dụng trong forward pass (ví dụ trong cross-attention)
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# Lấy con trỏ đến mô hình gốc (unwrapped model) để lưu checkpoint
raw_model = model.module if ddp else model

# --- HÀM ĐÁNH GIÁ VÀ LỊCH TRÌNH LEARNING RATE ---

@torch.no_grad()
def estimate_loss():
    """Ước tính loss trên tập train và val"""
    out = {}
    model.eval() # Chuyển model sang chế độ đánh giá
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # Lấy batch dữ liệu và attention masks
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
    model.train() # Chuyển model trở lại chế độ huấn luyện
    return out

def get_lr(it, schedule='cosine'):
    """Lịch trình giảm learning rate"""
    # 1. Giai đoạn warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2. Giai đoạn đã huấn luyện xong
    if it > lr_decay_iters:
        return min_lr
    # 3. Giai đoạn giảm dần
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Cosine decay
    return min_lr + coeff * (learning_rate - min_lr)

# --- THIẾT LẬP LOGGING (NẾU DÙNG WANDB) ---
if wandb_log and master_process:
    import wandb
    wandb_config = {k: v for k, v in config.items() if k != 'model_args'}
    wandb_config.update(config['model_args'])
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

def get_num_params(model, non_embedding=False):
    """Return the number of parameters in the model."""
    n_params = sum(p.numel() for p in model.parameters())
    if non_embedding and hasattr(model, 'shared'):
        n_params -= model.shared.weight.numel()
    return n_params

iter_num = 0
# --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---
print(f"Bắt đầu huấn luyện ViLegalJERE với {get_num_params(raw_model)/1e6:.1f}M tham số...")
# Lấy batch đầu tiên
input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
clip_time = 0

while True:
    # Xác định learning rate cho iteration hiện tại
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Đánh giá mô hình theo định kỳ
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100, # MFU (Model Flops Utilization)
            })
        
        # Lưu checkpoint nếu có val loss tốt hơn hoặc luôn luôn lưu
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': config, # Lưu lại toàn bộ config
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"Đang lưu checkpoint vào {out_dir}")
                # Sửa lỗi lưu file bằng cách thêm safe_serialization=False
                raw_model.save_pretrained(out_dir, safe_serialization=False)
                # Lưu state của optimizer riêng
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(optimizer_state, os.path.join(out_dir, 'optimizer.pt'))


    if iter_num == 0 and eval_only:
        break

    # Vòng lặp forward và backward, có tích lũy gradient
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # bật/tắt đồng bộ gradient giữa các GPU
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
            loss = loss / gradient_accumulation_steps # Chia loss cho các bước tích lũy
        
        # Lấy batch tiếp theo ngay lập tức để tiết kiệm thời gian chờ
        input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask = get_batch('train')
        # lan truyền ngược
        scaler.scale(loss).backward()

    # Cắt gradient để tránh bùng nổ gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1

    # Cập nhật trọng số
    scaler.step(optimizer)
    scaler.update()
    # Reset gradient về 0
    optimizer.zero_grad(set_to_none=True)

    # Đo lường thời gian và ghi log
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
        if wandb_log:
             wandb.log({
                "iter": iter_num,
                "train/loss_step": lossf,
                "lr_step": lr,
             })

    iter_num += 1
    local_iter_num += 1

    # Kết thúc vòng lặp nếu đã đạt max_iters
    if iter_num > max_iters:
        break

# --- DỌN DẸP ---
if ddp:
    destroy_process_group()