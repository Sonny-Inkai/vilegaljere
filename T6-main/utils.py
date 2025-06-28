"""
Utilities for ViLegalJERE Training
Contains common functions shared between pretrain and finetune scripts
"""

import os
import time
import math
import torch
import numpy as np
from contextlib import nullcontext
from datetime import datetime
from transformers import AutoTokenizer

# ===============================================================================
# TOKENIZER UTILITIES
# ===============================================================================

def load_custom_tokenizer(master_process=True):
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
    
    # Only print on master process to avoid duplicate logs
    if master_process:
        print(f"✅ Added {num_added_toks} domain-specific tokens")
        print(f"📊 New vocab size: {len(tokenizer)}")
        
        # ✅ VERIFY tokens were added correctly
        for token in domain_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: {token_id}")
    
    return tokenizer

def print_tokenizer_info(tokenizer, master_process=True):
    """Print tokenizer configuration info"""
    if master_process:
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

# ===============================================================================
# MODEL UTILITIES
# ===============================================================================

def get_num_params(model, non_embedding=False):
    """Return the number of parameters in the model."""
    n_params = sum(p.numel() for p in model.parameters())
    if non_embedding and hasattr(model, 'shared'):
        n_params -= model.shared.weight.numel()
    return n_params

def get_model_args(tokenizer, config):
    """Create model arguments dictionary"""
    model_args = dict(
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_embd=config['n_embd'], 
        block_size=config['block_size'],
        bias=config['bias'], 
        head_dim=config['head_dim'], 
        rank=config['rank'], 
        q_rank=config['q_rank'], 
        using_groupnorm=config['using_groupnorm'],
        vocab_size=len(tokenizer),
        dropout=config['dropout'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id
    )
    return model_args

def setup_model_with_tokenizer(model, tokenizer, master_process=True):
    """Setup model with proper tokenizer size and verification"""
    # ✅ Resize embeddings for domain tokens
    if len(tokenizer) != model.config.vocab_size:
        if master_process:
            print(f"🔧 Resizing embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        if master_process:
            print(f"✅ Model embeddings resized to {len(tokenizer)}")
    
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

# ===============================================================================
# TRAINING UTILITIES
# ===============================================================================

@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_iters, ctx):
    """Estimate loss on train/val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, labels, attention_mask = get_batch_fn(split)
            with ctx:
                # ✅ T5 STANDARD: Only pass input_ids, attention_mask, and labels
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr, schedule='cosine'):
    """Learning rate scheduler with warmup and decay"""
    if iter_num < warmup_iters:
        # Linear warmup
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    
    # Decay phase  
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
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

# ===============================================================================
# SETUP UTILITIES
# ===============================================================================

def setup_distributed_training(backend='gloo'):
    """Setup distributed training parameters"""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
        from torch.distributed import init_process_group
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
        device = 'cuda'
    
    return ddp, master_process, seed_offset, world_size, device

def setup_training_environment(seed_offset=0, dtype='float16'):
    """Setup training environment with optimizations"""
    # Initialize random seed
    torch.manual_seed(5000 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ✅ Memory optimization for T4 GPU
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass

    # Setup autocast context
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    
    return device_type, ctx

def setup_wandb(wandb_log, master_process, wandb_project, wandb_run_name, config):
    """Setup Weights & Biases logging"""
    if wandb_log and master_process:
        import wandb
        
        # Auto wandb login
        try:
            wandb.login()
            print("✅ Wandb login successful!")
        except Exception as e:
            print(f"⚠️ Cannot login wandb with API key: {e}")
            print("🔄 Using anonymous mode...")
            wandb.login(anonymous="allow")
            print("✅ Using wandb in anonymous mode!")
        
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        return True
    return False

def test_model_generation(model, tokenizer, device, master_process=True):
    """Test the trained model with Vietnamese legal sample input"""
    if not master_process:
        return True
        
    model.eval()
    
    # ✅ UPDATED: Use your actual data format
    test_input = "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở. 3. Việc phối hợp phải thường xuyên, kịp thời, đồng bộ, chặt chẽ, thống nhất, đúng quy định của pháp luật."
    expected_output = "<LEGAL_PROVISION> Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN <ORGANIZATION> Mặt trần Tổ quốc Việt Nam <Relates_To>"
    
    print(f"\n🧪 TESTING VIETNAMESE LEGAL JERE MODEL:")
    print(f"📥 Input: {test_input[:100]}...")
    print(f"🎯 Expected: {expected_output}")
    
    # Tokenize input
    inputs = tokenizer(test_input, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        # Generate using standard HuggingFace method
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,
                num_beams=3,
                early_stopping=True,
                length_penalty=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode result
        result = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"🤖 Generated: {result}")
        
        # Clean result
        clean_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🧹 Clean output: {clean_result}")
        
        # ✅ NEW: Extract and validate Vietnamese legal triplets
        extracted_triplets = extract_vietnamese_legal_triplets(clean_result)
        print(f"🏷️  Extracted triplets: {len(extracted_triplets)} found")
        for i, triplet in enumerate(extracted_triplets):
            print(f"   {i+1}. {triplet['head_type']}: '{triplet['head']}' → {triplet['tail_type']}: '{triplet['tail']}' ({triplet['relation']})")
        
        # Check for domain tokens
        domain_tokens = ["<ORGANIZATION>", "<LOCATION>", "<LEGAL_PROVISION>", "<RIGHT/DUTY>", "<PERSON>", "<Relates_To>", "<Applicable_In>"]
        found_tokens = [token for token in domain_tokens if token in result]
        print(f"🔧 Domain tokens found: {found_tokens}")
        
        return len(extracted_triplets) > 0  # Success if any valid triplets extracted
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def qualitative_pretrain_test(model, tokenizer, device, test_sentence, create_spans_fn=None):
    """
    Performs a qualitative "fill-in-the-blanks" test to see how the model is learning.
    """
    print("\n" + "="*80)
    print("🔬 QUALITATIVE PRE-TRAINING TEST")
    print(f"📖 Original sentence: '{test_sentence}'")

    model.eval()  # Switch model to evaluation mode

    # 1. Create input/target pairs from the test sentence
    article_tokens = tokenizer.encode(test_sentence, add_special_tokens=False)
    
    # Use the provided create_spans_fn or try to import it
    if create_spans_fn is not None:
        try:
            # ✅ HF CHUẨN: create_t5_spans now returns token IDs directly
            input_ids_list, target_ids_list = create_spans_fn(article_tokens, tokenizer)
            
            # Convert to strings for display
            input_string = tokenizer.decode(input_ids_list, skip_special_tokens=False)
            target_string = tokenizer.decode(target_ids_list, skip_special_tokens=False)
        except Exception as e:
            print(f"⚠️ Span creation failed: {e}, using original tokens")
            input_ids_list = article_tokens
            target_ids_list = article_tokens
            input_string = tokenizer.decode(article_tokens)
            target_string = tokenizer.decode(article_tokens)
    else:
        # Try to import create_t5_spans from the global scope
        try:
            import sys
            import importlib
            # Try to get the function from the caller's globals
            frame = sys._getframe(1)
            if 'create_t5_spans' in frame.f_globals:
                create_t5_spans = frame.f_globals['create_t5_spans']
                input_ids_list, target_ids_list = create_t5_spans(article_tokens, tokenizer)
                input_string = tokenizer.decode(input_ids_list, skip_special_tokens=False)
                target_string = tokenizer.decode(target_ids_list, skip_special_tokens=False)
            else:
                # Simple fallback - just use the original tokens
                input_ids_list = article_tokens
                target_ids_list = article_tokens
                input_string = tokenizer.decode(article_tokens)
                target_string = tokenizer.decode(article_tokens)
        except:
            # Fallback if import fails
            input_ids_list = article_tokens
            target_ids_list = article_tokens
            input_string = tokenizer.decode(article_tokens)
            target_string = tokenizer.decode(article_tokens)

    # 2. Prepare input for the model with proper attention mask
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    
    # 3. Decode to see what the masked input and expected target look like
    corrupted_input_text = input_string
    expected_target_text = target_string
    print(f"❓ Masked Input: {corrupted_input_text}")
    print(f"🎯 Expected Target: {expected_target_text}")

    # 4. Have the model generate the output
    with torch.no_grad():
        try:
            # ✅ FIXED: Use standard HuggingFace generate() with correct parameters
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=512,
                do_sample=False,  # Greedy decoding for deterministic results
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 5. Decode the model's output and compare
            model_output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            print(f"🤖 Model Prediction: {model_output_text}")
            
            # 6. Also show a clean version
            clean_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"🧹 Clean Prediction: {clean_output}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print(f"🤖 Model Prediction: [Generation Error]")
    
    print("="*80 + "\n")

    model.train()  # Switch the model back to training mode

# ===============================================================================
# VIETNAMESE LEGAL JERE UTILITIES  
# ===============================================================================

def extract_vietnamese_legal_triplets(text):
    """
    Extract triplets from Vietnamese legal format: <Head_Type> Head_Text <Tail_Type> Tail_Text <Relation_Type>
    
    Example input: "<LEGAL_PROVISION> Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN <ORGANIZATION> Mặt trận Tổ quốc Việt Nam <Relates_To>"
    Output: [{'head': 'Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN', 'head_type': 'LEGAL_PROVISION', 
              'tail': 'Mặt trận Tổ quốc Việt Nam', 'tail_type': 'ORGANIZATION', 'relation': 'Relates_To'}]
    """
    # Define Vietnamese legal domain tokens
    LEGAL_ENTITY_TYPES = {
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>"
    }
    
    LEGAL_RELATION_TYPES = {
        "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
    }
    
    triplets = []
    tokens = text.replace("</s>", "").replace("<s>", "").replace("<pad>", "").split()
    
    i = 0
    while i < len(tokens):
        if tokens[i] in LEGAL_ENTITY_TYPES:
            # Found head type
            head_type = tokens[i].strip('<>')
            i += 1
            
            # Extract head text until next type token
            head_text = []
            while i < len(tokens) and tokens[i] not in (LEGAL_ENTITY_TYPES | LEGAL_RELATION_TYPES):
                head_text.append(tokens[i])
                i += 1
            
            if i < len(tokens) and tokens[i] in LEGAL_ENTITY_TYPES:
                # Found tail type
                tail_type = tokens[i].strip('<>')
                i += 1
                
                # Extract tail text until relation type
                tail_text = []
                while i < len(tokens) and tokens[i] not in LEGAL_RELATION_TYPES:
                    tail_text.append(tokens[i])
                    i += 1
                
                if i < len(tokens) and tokens[i] in LEGAL_RELATION_TYPES:
                    # Found relation
                    relation_type = tokens[i].strip('<>')
                    
                    triplets.append({
                        'head': ' '.join(head_text).strip(),
                        'head_type': head_type,
                        'tail': ' '.join(tail_text).strip(),
                        'tail_type': tail_type,
                        'relation': relation_type
                    })
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return triplets

def evaluate_vietnamese_legal_jere(predicted_triplets, gold_triplets, mode="strict"):
    """
    Evaluate Vietnamese legal JERE predictions
    
    Args:
        predicted_triplets: List of lists of predicted triplets for each sentence
        gold_triplets: List of lists of gold triplets for each sentence  
        mode: "strict" (exact match) or "partial" (entity overlap)
    """
    assert mode in ["strict", "partial"], "Mode must be 'strict' or 'partial'"
    
    total_tp = total_fp = total_fn = 0
    relation_scores = {}
    
    # Initialize relation scores
    LEGAL_RELATIONS = ["Effective_From", "Applicable_In", "Relates_To", "Amended_By"]
    for rel in LEGAL_RELATIONS:
        relation_scores[rel] = {"tp": 0, "fp": 0, "fn": 0}
    
    for pred_sent, gold_sent in zip(predicted_triplets, gold_triplets):
        if mode == "strict":
            # Exact matching including entity types
            pred_set = {(t['head'], t['head_type'], t['tail'], t['tail_type'], t['relation']) 
                       for t in pred_sent}
            gold_set = {(t['head'], t['head_type'], t['tail'], t['tail_type'], t['relation']) 
                       for t in gold_sent}
        else:
            # Partial matching - only entity text and relation
            pred_set = {(t['head'], t['tail'], t['relation']) for t in pred_sent}
            gold_set = {(t['head'], t['tail'], t['relation']) for t in gold_sent}
        
        # Calculate metrics for this sentence
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Per-relation metrics
        for rel in LEGAL_RELATIONS:
            if mode == "strict":
                pred_rel = {(t['head'], t['head_type'], t['tail'], t['tail_type']) 
                           for t in pred_sent if t['relation'] == rel}
                gold_rel = {(t['head'], t['head_type'], t['tail'], t['tail_type']) 
                           for t in gold_sent if t['relation'] == rel}
            else:
                pred_rel = {(t['head'], t['tail']) for t in pred_sent if t['relation'] == rel}
                gold_rel = {(t['head'], t['tail']) for t in gold_sent if t['relation'] == rel}
            
            relation_scores[rel]["tp"] += len(pred_rel & gold_rel)
            relation_scores[rel]["fp"] += len(pred_rel - gold_rel)
            relation_scores[rel]["fn"] += len(gold_rel - pred_rel)
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate per-relation metrics
    for rel in relation_scores:
        tp = relation_scores[rel]["tp"]
        fp = relation_scores[rel]["fp"] 
        fn = relation_scores[rel]["fn"]
        
        rel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rel_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0
        
        relation_scores[rel].update({
            "precision": rel_precision,
            "recall": rel_recall, 
            "f1": rel_f1
        })
    
    return {
        "overall": {"precision": precision, "recall": recall, "f1": f1},
        "per_relation": relation_scores,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn}
    }

def test_vietnamese_legal_triplet_extraction():
    """Test the Vietnamese legal triplet extraction function"""
    print("\n" + "="*60)
    print("🧪 TESTING VIETNAMESE LEGAL TRIPLET EXTRACTION")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            "input": "<LEGAL_PROVISION> Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN <ORGANIZATION> Mặt trận Tổ quốc Việt Nam <Relates_To>",
            "expected": [
                {
                    'head': 'Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN',
                    'head_type': 'LEGAL_PROVISION', 
                    'tail': 'Mặt trận Tổ quốc Việt Nam',
                    'tail_type': 'ORGANIZATION',
                    'relation': 'Relates_To'
                }
            ]
        },
        {
            "input": "<ORGANIZATION> Chính phủ <LEGAL_PROVISION> Nghị định số 15/2021/NĐ-CP <Amended_By>",
            "expected": [
                {
                    'head': 'Chính phủ',
                    'head_type': 'ORGANIZATION',
                    'tail': 'Nghị định số 15/2021/NĐ-CP', 
                    'tail_type': 'LEGAL_PROVISION',
                    'relation': 'Amended_By'
                }
            ]
        },
        {
            "input": "<LEGAL_PROVISION> Luật số 67/2020/QH14 <DATE/TIME> ngày 17 tháng 6 năm 2021 <Effective_From>",
            "expected": [
                {
                    'head': 'Luật số 67/2020/QH14',
                    'head_type': 'LEGAL_PROVISION',
                    'tail': 'ngày 17 tháng 6 năm 2021',
                    'tail_type': 'DATE/TIME',
                    'relation': 'Effective_From'
                }
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Input: {test_case['input']}")
        
        extracted = extract_vietnamese_legal_triplets(test_case['input'])
        print(f"Extracted: {extracted}")
        print(f"Expected: {test_case['expected']}")
        
        # Check if extraction matches expected
        success = len(extracted) == len(test_case['expected'])
        if success and extracted:
            for ext, exp in zip(extracted, test_case['expected']):
                if (ext['head'] != exp['head'] or ext['head_type'] != exp['head_type'] or
                    ext['tail'] != exp['tail'] or ext['tail_type'] != exp['tail_type'] or
                    ext['relation'] != exp['relation']):
                    success = False
                    break
        
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    print("\n" + "="*60)
    return True