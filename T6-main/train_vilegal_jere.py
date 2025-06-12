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
finetune = True
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
    
    # ✅ FIXED: Better hyperparameters for relation extraction fine-tuning
    learning_rate = 3e-4 # ✅ FIXED: Higher learning rate for fine-tuning stability
    max_iters = 3000     # ✅ FIXED: Sufficient iterations for fine-tuning convergence
    batch_size = 16      # ✅ FIXED: Smaller batch for better gradient stability
    gradient_accumulation_steps = 4  # ✅ FIXED: Maintain effective batch size of 64
    weight_decay = 0.01  # ✅ FIXED: Standard weight decay for transformer fine-tuning
    eval_interval = 100  # ✅ FIXED: More frequent evaluation for monitoring
    log_interval = 10    # ✅ FIXED: Keep logging frequency
    eval_iters = 50      # ✅ FIXED: Faster evaluation iterations
    warmup_iters = 300   # ✅ FIXED: Shorter warmup for fine-tuning (10% of max_iters)
    
else:
    # --- CẤU HÌNH CHO PRE-TRAINING ---
    init_from = 'scratch' # 'scratch' or 'resume'
    data_path = "/kaggle/input/vietnamese-legal-dataset"  # Kaggle dataset path
    out_dir = '/kaggle/working/out_vilegal_t5small'
    
    # ✅ FIXED: Siêu tham số cho pre-training tối ưu cho T4x2
    learning_rate = 3e-4  # ✅ FIXED: Standard for T5-small pre-training
    max_iters = 10000     # ✅ FIXED: Reasonable for T5-small
    batch_size = 32       # ✅ FIXED: Safe for T4 memory
    gradient_accumulation_steps = 4   # ✅ FIXED: Maintain large effective batch
    weight_decay = 1e-2   # ✅ FIXED: Standard T5 weight decay
    eval_interval = 500  # ✅ FIXED: Less frequent for pre-training
    log_interval = 10     # ✅ FIXED: Reduce logging overhead
    eval_iters = 200      # ✅ FIXED: Keep reasonable for evaluation
    
# wandb logging
wandb_log = True    # Enable for better tracking
wandb_project = 'ViLegalJERE-T5Small'
wandb_run_name = 'vilegal_t5small_kaggle'
# data
dataset = 'vietnamese_legal'
block_size = 512    # Keep same
max_source_length = 512  # encoder max length
max_target_length = 512  # decoder max length
# model - T5-small architecture (~60M parameters)
n_layer = 6         # T5-small has 6 layers each for encoder/decoder
n_head = 8          # T5-small uses 8 attention heads
head_dim = 64       # 512/8 = 64
rank = 4            # ✅ GOOD: Reasonable CP rank for T6
q_rank = 8          # ✅ GOOD: Reasonable query rank for T6
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
lr_decay_iters = max_iters  # ✅ FIXED: Match max_iters for proper decay schedule
min_lr = 5e-6        # ✅ FIXED: Higher min_lr to avoid vanishing gradients (was 1e-6)
# DDP settings for Kaggle T4x2
backend = 'gloo'  # Use gloo instead of nccl for better Kaggle compatibility
schedule = 'cosine'
model_type = 'ViLegalJERE'
# system
device = 'cuda'  
dtype = 'float16'   
compile = False     # Disable compile for Kaggle compatibility
scale_attn_by_inverse_layer_idx = False
# -----------------------------------------------------------------------------

# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
# config = {k: globals()[k] for k in config_keys}

# Import ViLegalJERE model
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE

# ✅ FIXED: Custom tokenizer with domain tokens
def load_custom_tokenizer():
    """Load custom trained tokenizer with domain-specific tokens"""
    from transformers import AutoTokenizer
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    
    # ✅ ADD domain-specific tokens for Vietnamese Legal JERE
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    # Add special tokens to tokenizer
    special_tokens_dict = {'additional_special_tokens': domain_special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"✅ Added {num_added_toks} domain-specific tokens")
    print(f"📊 New vocab size: {len(tokenizer)}")
    
    # ✅ VERIFY tokens were added correctly
    for token in domain_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: {token_id}")
    
    return tokenizer

# Initialize tokenizer with domain tokens
tokenizer = load_custom_tokenizer()

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

# ✅ FIXED: Memory optimization for T4 GPU
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear cache before training
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass

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

    articles = text.split('\n')
    tokenized_data = [tokenizer.encode(art, truncation=True, max_length=block_size) for art in articles]
    return tokenized_data

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
        source_text = value.get("formatted_context_sent", "")
        target_text = value.get("extracted_relations_text", "")
        
        # ✅ ENHANCED: Validation và cleanup
        if source_text and target_text:
            # Clean and validate source text
            source_text = source_text.strip()
            target_text = target_text.strip()
            
            # ✅ VALIDATION: Check if target contains required domain tokens
            required_tokens = ["<ORGANIZATION>", "<LOCATION>", "<LEGAL_PROVISION>", "<RIGHT/DUTY>", "<PERSON>"]
            has_domain_tokens = any(token in target_text for token in required_tokens)
            
            # ✅ VALIDATION: Check reasonable length
            if (50 <= len(source_text) <= 2000 and 
                20 <= len(target_text) <= 1000 and 
                has_domain_tokens):
                processed_data.append((source_text, target_text))
            else:
                skipped_count += 1
                if master_process and skipped_count <= 3:
                    print(f"⚠️ Skipping invalid pair:")
                    print(f"   Source len: {len(source_text)}, Target len: {len(target_text)}")
                    print(f"   Has tokens: {has_domain_tokens}")
                    print(f"   Target: {target_text[:100]}...")
        else:
            skipped_count += 1
    
    if master_process:
        print(f"✅ Loaded {len(processed_data)} valid fine-tuning pairs")
        print(f"⚠️ Skipped {skipped_count} invalid pairs")
        
        # ✅ SHOW sample data for verification
        if processed_data:
            print("\n📝 SAMPLE TRAINING DATA:")
            sample_input, sample_target = processed_data[0]
            print(f"📥 Input: {sample_input[:150]}...")
            print(f"🎯 Target: {sample_target[:150]}...")
            
            # ✅ CHECK tokenization
            input_tokens = tokenizer.tokenize(sample_input)
            target_tokens = tokenizer.tokenize(sample_target)
            print(f"📊 Input tokens: {len(input_tokens)}, Target tokens: {len(target_tokens)}")
    
    return processed_data

# ✅ Helper functions cho T5 span corruption theo chuẩn Google
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

def create_t5_spans(tokens, noise_density=0.15, mean_noise_span_length=3.0):
    """
    Tạo dữ liệu theo kiểu span corruption của T5 chuẩn Google.
    """
    import numpy as np
    
    num_tokens = len(tokens)
    if num_tokens <= 1:
        return tokens, tokens
        
    # ✅ Tạo noise mask theo chuẩn Google T5
    noise_mask = create_noise_mask(num_tokens, noise_density, mean_noise_span_length)
    
    # ✅ Lấy sentinel token theo chuẩn Google T5
    try:
        sentinel_start_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        if sentinel_start_id == tokenizer.unk_token_id:
            raise ValueError("Sentinel token not found")
    except (KeyError, ValueError):
        # ✅ CORRECT FALLBACK: Hard-coded based on tokenizer config
        sentinel_start_id = 10099  # <extra_id_0> confirmed by tokenizer analysis
        if master_process:
            print(f"Warning: Using hard-coded sentinel_start_id = {sentinel_start_id}")
    
    # ✅ Create input sequence with sentinels (noise_span_to_unique_sentinel)
    input_ids = []
    labels = []
    
    prev_token_is_noise = False
    sentinel_idx = 0
    
    for i, token in enumerate(tokens):
        is_noise = noise_mask[i] if i < len(noise_mask) else False
        
        if is_noise:
            if not prev_token_is_noise:
                # First token of a noise span - add sentinel to input
                sentinel_id = sentinel_start_id - sentinel_idx
                input_ids.append(sentinel_id)
                labels.append(sentinel_id)  # Start target with same sentinel
                sentinel_idx += 1
            # Add noise token to target only
            labels.append(token)
        else:
            # Non-noise token goes to input
            input_ids.append(token)
            
        prev_token_is_noise = is_noise
    
    # ✅ Thêm EOS vào cuối labels
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
    ✅ ENHANCED GET_BATCH - Properly handles both pre-training and fine-tuning with T5 format
    """
    data = train_data if split == 'train' else val_data
    if not data:
        raise ValueError(f"Data split '{split}' is empty. Check data loading.")

    if finetune:
        # --- ✅ IMPROVED FINE-TUNING BATCH PROCESSING ---
        ix = np.random.randint(len(data), size=(batch_size,))
        batch_pairs = [data[i] for i in ix]
        
        # Extract source and target texts
        source_texts = [pair[0] for pair in batch_pairs]
        target_texts = [pair[1] for pair in batch_pairs]
        
        # ✅ CRITICAL FIX: Add T5-style task prefix for relation extraction
        source_texts = [f"extract relations: {text}" for text in source_texts]
        
        # ✅ PROPER T5 tokenization with appropriate max lengths
        input_encodings = tokenizer(
            source_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_source_length, 
            return_tensors="pt",
            add_special_tokens=True
        )
        
        target_encodings = tokenizer(
            target_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_target_length, 
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids = input_encodings.input_ids
        attention_mask = input_encodings.attention_mask.to(torch.bool)
        labels = target_encodings.input_ids
        
        # ✅ CRITICAL FIX: Proper T5 decoder input construction
        # T5 decoder starts with pad token, then target sequence (shifted right)
        decoder_input_ids = torch.full((labels.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long)
        decoder_input_ids = torch.cat([decoder_input_ids, labels[:, :-1]], dim=-1)
        
        # ✅ PROPER decoder attention mask
        decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id)
        
        # ✅ ENHANCED: Debug info for fine-tuning
        if master_process and np.random.random() < 0.01:  # 1% chance to show debug
            print(f"\n🔍 BATCH DEBUG INFO:")
            print(f"📥 Sample input: {tokenizer.decode(input_ids[0][:50])}")
            print(f"🎯 Sample target: {tokenizer.decode(labels[0][:50])}")
            print(f"🔄 Sample decoder_input: {tokenizer.decode(decoder_input_ids[0][:50])}")
        
    else:
        # --- ✅ MAINTAINED PRE-TRAINING LOGIC ---
        batch_input_ids = []
        batch_labels = []
        
        for _ in range(batch_size):
            article_tokens = data[np.random.randint(len(data))]
            
            # ✅ ENHANCED: Better error handling for span corruption
            if len(article_tokens) < 10:  # Skip very short articles
                article_tokens = data[np.random.randint(len(data))]
            
            input_tokens, target_tokens = create_t5_spans(article_tokens)
            
            # Pad/truncate to fixed lengths
            input_padded = (input_tokens[:max_source_length] + 
                          [tokenizer.pad_token_id] * max(0, max_source_length - len(input_tokens)))[:max_source_length]
            labels_padded = (target_tokens[:max_target_length] + 
                           [tokenizer.pad_token_id] * max(0, max_target_length - len(target_tokens)))[:max_target_length]
            
            batch_input_ids.append(input_padded)
            batch_labels.append(labels_padded)
        
        # Convert to tensors
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        labels = torch.tensor(batch_labels, dtype=torch.long)
        
        # Create attention masks
        attention_mask = (input_ids != tokenizer.pad_token_id)
        
        # ✅ PROPER T5 decoder input for pre-training
        decoder_input_ids = torch.full((labels.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long)
        decoder_input_ids = torch.cat([decoder_input_ids, labels[:, :-1]], dim=-1)
        decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id)

    # ✅ EFFICIENT GPU transfer
    if device_type == 'cuda':
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        decoder_attention_mask = decoder_attention_mask.pin_memory().to(device, non_blocking=True)
        decoder_input_ids = decoder_input_ids.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        labels = labels.to(device) 
        attention_mask = attention_mask.to(device)
        decoder_attention_mask = decoder_attention_mask.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
    
    return input_ids, decoder_input_ids, labels, attention_mask, decoder_attention_mask

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
    pad_token_id=tokenizer.pad_token_id,      # 0 - Correct
    eos_token_id=tokenizer.eos_token_id,      # 3 - Correct (not 1!)
    decoder_start_token_id=tokenizer.eos_token_id  # 3 - Use EOS as decoder start
)

print(f"🔧 TOKENIZER DEBUG INFO:")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
print(f"  EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"  UNK token: '{tokenizer.unk_token}' (id: {tokenizer.unk_token_id})")

# ✅ Verify sentinel tokens
try:
    sentinel_test = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    print(f"  Sentinel <extra_id_0>: {sentinel_test}")
except:
    print(f"  ⚠️ Could not find <extra_id_0> token!")

print(f"🎯 MODEL CONFIG:")
print(f"  decoder_start_token_id: {tokenizer.eos_token_id} (should match EOS)")

# Initialize tracking variables
iter_num = 0
best_val_loss = 1e9

# --- ✅ FIXED MODEL INITIALIZATION WITH PROPER EMBEDDING RESIZE ---
if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    # Initialize model with original vocab size first
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)
    
    # ✅ CRITICAL FIX: Resize embeddings for domain tokens AFTER model creation
    if len(tokenizer) != model.config.vocab_size:
        if master_process:
            print(f"🔧 Resizing embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        if master_process:
            print(f"✅ Model embeddings resized to {len(tokenizer)}")

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    
    # Check if checkpoint directory exists
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {out_dir}. Cannot resume.")

    # Load model from checkpoint
    model = ViLegalJERE.from_pretrained(out_dir)
    
    # ✅ CRITICAL FIX: Always resize embeddings when resuming to match current tokenizer
    if len(tokenizer) != model.config.vocab_size:
        if master_process:
            print(f"🔧 Resizing embeddings during resume: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        if master_process:
            print(f"✅ Model embeddings resized to {len(tokenizer)}")
    
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

# ✅ VERIFICATION: Check final model configuration
if master_process:
    print(f"\n🔍 FINAL MODEL VERIFICATION:")
    print(f"📊 Model vocab size: {model.config.vocab_size}")
    print(f"📊 Tokenizer vocab size: {len(tokenizer)}")
    print(f"📊 Model embedding shape: {model.shared.weight.shape}")
    
    # Test tokenization of domain tokens
    test_tokens = ["<ORGANIZATION>", "<LOCATION>", "<Relates_To>"]
    for token in test_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"🧪 {token}: id={token_id}, valid={token_id < model.config.vocab_size}")

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
                # ✅ FIXED: Pass encoder attention mask correctly in evaluation
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # Encoder attention mask
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,  # Decoder attention mask
                    labels=labels
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, schedule='cosine'):
    """Learning rate scheduler với warmup và decay cải thiện"""
    if it < warmup_iters:
        # ✅ Linear warmup
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    
    # ✅ Decay phase  
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    if schedule == 'cosine':
        # Cosine annealing
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    elif schedule == 'linear':
        # Linear decay
        coeff = 1.0 - decay_ratio
    elif schedule == 'constant':
        # Constant after warmup
        coeff = 1.0
    else:
        # Default to cosine
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
            # ✅ FIXED: Pass encoder attention mask correctly for T5 architecture
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Encoder attention mask
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,  # Decoder attention mask
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

# ✅ ENHANCED: Test model function for debugging
def test_model_generation(model, tokenizer, device):
    """Test the trained model with a sample input"""
    model.eval()
    
    # Test input with relation extraction prefix
    test_input = "extract relations: Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam"
    
    print(f"\n🧪 TESTING MODEL GENERATION:")
    print(f"📥 Input: {test_input}")
    
    # Tokenize input
    inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        # Generate using enhanced method
        with torch.no_grad():
            if hasattr(model, 'generate_relations'):
                outputs = model.generate_relations(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=256,
                    num_beams=3,
                    early_stopping=True,
                    length_penalty=1.0
                )
            else:
                # Fallback to standard generate
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=256,
                )
        
        # Decode result
        result = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"🤖 Generated: {result}")
        
        # Clean result
        clean_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🧹 Clean output: {clean_result}")
        
        # Check for domain tokens
        domain_tokens = ["<ORGANIZATION>", "<LOCATION>", "<LEGAL_PROVISION>", "<RIGHT/DUTY>", "<PERSON>", "<Relates_To>"]
        found_tokens = [token for token in domain_tokens if token in result]
        print(f"🏷️  Domain tokens found: {found_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

# ✅ Run test if we're in fine-tuning mode and training completed
if finetune and master_process and iter_num > 100:
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

print(f"\n🎉 Training completed! Model saved to: {out_dir}") 