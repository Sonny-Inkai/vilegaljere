#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE ANALYSIS OF VILEGALJERE vs GOOGLE T5 STANDARDS
================================================================
Chú Tony Stark's definitive analysis of the model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
import os

# Add model path
sys.path.append('model')
from ViLegalJERE import ViLegalConfig, ViLegalJERE

print("🔍 COMPREHENSIVE VILEGALJERE ANALYSIS")
print("="*60)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
config = ViLegalConfig()
model = ViLegalJERE(config)

print(f"📊 TOKENIZER ANALYSIS")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
print(f"  EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"  UNK token: '{tokenizer.unk_token}' (id: {tokenizer.unk_token_id})")

# Check sentinel tokens structure
print(f"\n🏷️  SENTINEL TOKENS STRUCTURE")
try:
    extra_id_0 = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    extra_id_1 = tokenizer.convert_tokens_to_ids('<extra_id_1>')
    extra_id_99 = tokenizer.convert_tokens_to_ids('<extra_id_99>')
    
    print(f"  <extra_id_0>: {extra_id_0}")
    print(f"  <extra_id_1>: {extra_id_1}")
    print(f"  <extra_id_99>: {extra_id_99}")
    
    # Check if they're in decreasing order (T5 standard)
    if extra_id_0 > extra_id_1 > extra_id_99:
        print("  ✅ Sentinel tokens in correct decreasing order")
    else:
        print("  ❌ Sentinel tokens NOT in decreasing order")
        
except Exception as e:
    print(f"  ❌ Error accessing sentinel tokens: {e}")

print(f"\n⚙️  MODEL CONFIG ANALYSIS")
print(f"  Model vocab_size: {config.vocab_size}")
print(f"  Tokenizer vocab_size: {len(tokenizer)}")
print(f"  Config pad_token_id: {config.pad_token_id}")
print(f"  Config eos_token_id: {config.eos_token_id}")
print(f"  Config decoder_start_token_id: {config.decoder_start_token_id}")

# Match check
if config.vocab_size == len(tokenizer):
    print("  ✅ Vocab sizes match")
else:
    print(f"  ⚠️  Vocab size mismatch: model={config.vocab_size}, tokenizer={len(tokenizer)}")

if config.eos_token_id == tokenizer.eos_token_id:
    print("  ✅ EOS token IDs match")
else:
    print(f"  ❌ EOS mismatch: model={config.eos_token_id}, tokenizer={tokenizer.eos_token_id}")

if config.decoder_start_token_id == tokenizer.eos_token_id:
    print("  ✅ Decoder start token correctly set to EOS")
else:
    print(f"  ❌ Decoder start should be EOS: got {config.decoder_start_token_id}, expected {tokenizer.eos_token_id}")

print(f"\n🧪 LOSS COMPUTATION ANALYSIS")
print("-"*40)

# Test data following Google T5 standards
batch_size = 2
seq_len = 8

# Create test batch - simulating T5 style input
input_ids = torch.tensor([
    [100, 200, 10099, 400, 500, 0, 0, 0],  # <extra_id_0> for corrupted span
    [600, 700, 800, 10098, 900, 0, 0, 0]   # <extra_id_1> for corrupted span
])

labels = torch.tensor([
    [10099, 300, 3, 0, 0, 0, 0, 0],  # <extra_id_0>, recovered token, EOS, pad
    [10098, 350, 850, 3, 0, 0, 0, 0]  # <extra_id_1>, recovered tokens, EOS, pad
])

# Create decoder_input_ids following T5 standard: [EOS] + labels[:-1]
decoder_input_ids = torch.cat([
    torch.full((labels.shape[0], 1), tokenizer.eos_token_id), 
    labels[:, :-1]
], dim=-1)

print(f"Input IDs: {input_ids}")
print(f"Labels: {labels}")
print(f"Decoder input: {decoder_input_ids}")
print(f"Decoder starts with: {decoder_input_ids[:, 0]} (should all be {tokenizer.eos_token_id})")

# Test model forward
attention_mask = (input_ids != tokenizer.pad_token_id).float()
decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).float()

print(f"\n🔬 FORWARD PASS TESTING")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels
    )

print(f"  Forward pass successful: ✅")
print(f"  Loss: {outputs['loss']:.4f}")
print(f"  Logits shape: {outputs['logits'].shape}")

# Analyze loss computation step by step
print(f"\n🧮 DETAILED LOSS ANALYSIS")
logits = outputs['logits']

# Following T5 standard loss computation
shift_logits = logits[..., :-1, :].contiguous()  # Remove last position
shift_labels = labels[..., 1:].contiguous()      # Remove first position (BOS)

print(f"  Original logits shape: {logits.shape}")
print(f"  Shifted logits shape: {shift_logits.shape}")
print(f"  Shifted labels shape: {shift_labels.shape}")

# Check label masking
shift_labels_masked = shift_labels.clone()
shift_labels_masked[shift_labels_masked == config.pad_token_id] = -100

print(f"  Labels before masking: {shift_labels}")
print(f"  Labels after masking: {shift_labels_masked}")

# Manual loss computation for verification
loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
manual_loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels_masked.view(-1))

print(f"  Model computed loss: {outputs['loss']:.6f}")
print(f"  Manual computed loss: {manual_loss:.6f}")
print(f"  Loss computation matches: {'✅' if abs(outputs['loss'] - manual_loss) < 1e-6 else '❌'}")

print(f"\n🎯 TENSOR PRODUCT ATTENTION ANALYSIS")
print("-"*40)

# Analyze parameter reduction
total_params = model.get_num_params(non_embedding=False)
embedding_params = model.shared.weight.numel() + model.lm_head.weight.numel()
attention_params = sum(p.numel() for name, p in model.named_parameters() if 'c_qkv' in name or 'c_kv' in name)

print(f"  Total parameters: {total_params:,}")
print(f"  Embedding parameters: {embedding_params:,}")
print(f"  Attention parameters: {attention_params:,}")

# Calculate theoretical attention params for standard transformer
n_head, head_dim, n_embd = config.n_head, config.head_dim, config.n_embd
n_layer = config.n_layer

# Standard attention: 3 * (n_embd * n_embd) per layer for QKV
# CP attention: much smaller due to rank decomposition
standard_attention_params = n_layer * 3 * (n_embd * n_embd)  # Encoder
standard_attention_params += n_layer * 4 * (n_embd * n_embd)  # Decoder (self + cross attention)

print(f"  Standard attention would need: {standard_attention_params:,} params")
print(f"  Actual attention params: {attention_params:,}")
reduction_factor = standard_attention_params / attention_params if attention_params > 0 else 0
print(f"  Parameter reduction: {reduction_factor:.1f}x")

print(f"\n🎨 ARCHITECTURE COMPLIANCE CHECK")
print("-"*40)

# Check if model follows T5 architecture principles
checks = []

# 1. Shared embeddings
input_emb = model.get_input_embeddings()
output_emb = model.get_output_embeddings()
checks.append(("Shared input/output embeddings", input_emb.weight is output_emb.weight))

# 2. Encoder-decoder structure
checks.append(("Has encoder blocks", len(model.encoder_blocks) > 0))
checks.append(("Has decoder blocks", len(model.decoder_blocks) > 0))
checks.append(("Equal encoder/decoder layers", len(model.encoder_blocks) == len(model.decoder_blocks)))

# 3. Cross attention in decoder
has_cross_attention = any(hasattr(block, 'cross_attn') for block in model.decoder_blocks)
checks.append(("Decoder has cross-attention", has_cross_attention))

# 4. Layer normalization
has_encoder_ln = hasattr(model, 'encoder_ln')
has_decoder_ln = hasattr(model, 'decoder_ln')
checks.append(("Has encoder layer norm", has_encoder_ln))
checks.append(("Has decoder layer norm", has_decoder_ln))

print("  Architecture checks:")
for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"    {status} {check_name}")

print(f"\n📈 DATA FLOW VERIFICATION")
print("-"*40)

# Test encoding
encoder_output = model.encode(input_ids, attention_mask)
print(f"  Encoder output shape: {encoder_output.last_hidden_state.shape}")

# Test decoding with encoder states
decoder_output = model.decode(
    decoder_input_ids,
    encoder_output.last_hidden_state,
    decoder_attention_mask,
    attention_mask
)
print(f"  Decoder output shape: {decoder_output.logits.shape}")

# Check generation capability
print(f"\n🚀 GENERATION TEST")
try:
    test_input = torch.tensor([[100, 200, 300, 0, 0]])
    test_attention = (test_input != 0).float()
    
    generated = model.generate(
        input_ids=test_input,
        attention_mask=test_attention,
        max_length=10,
        do_sample=False
    )
    print(f"  Generation successful: ✅")
    print(f"  Generated sequence: {generated}")
    print(f"  Generated shape: {generated.shape}")
except Exception as e:
    print(f"  Generation failed: ❌ {e}")

print(f"\n🏆 FINAL ASSESSMENT")
print("="*60)

issues_found = []

# Critical issues
if config.eos_token_id != tokenizer.eos_token_id:
    issues_found.append("❌ CRITICAL: EOS token ID mismatch")

if config.decoder_start_token_id != tokenizer.eos_token_id:
    issues_found.append("❌ CRITICAL: Decoder start token should be EOS")

if config.vocab_size != len(tokenizer):
    issues_found.append("⚠️  WARNING: Vocab size mismatch")

# Check attention mask types
if attention_mask.dtype != torch.bool:
    issues_found.append("⚠️  MINOR: Attention masks should be boolean")

if abs(outputs['loss'] - manual_loss) > 1e-6:
    issues_found.append("❌ CRITICAL: Loss computation error")

# Summary
if not issues_found:
    print("🎉 PERFECT! No issues found. Model is 100% compliant with T5 standards.")
    compliance_score = 100
else:
    print("Issues found:")
    for issue in issues_found:
        print(f"  {issue}")
    
    critical_issues = len([i for i in issues_found if "CRITICAL" in i])
    warning_issues = len([i for i in issues_found if "WARNING" in i])
    minor_issues = len([i for i in issues_found if "MINOR" in i])
    
    compliance_score = 100 - (critical_issues * 30) - (warning_issues * 10) - (minor_issues * 5)
    compliance_score = max(0, compliance_score)

print(f"\n📊 COMPLIANCE SCORE: {compliance_score}%")

if compliance_score >= 95:
    print("🟢 EXCELLENT: Model will train correctly")
elif compliance_score >= 80:
    print("🟡 GOOD: Minor fixes recommended")
else:
    print("🔴 NEEDS WORK: Critical issues must be fixed")

print(f"\n💡 RECOMMENDATIONS:")
if compliance_score < 100:
    print("  1. Fix all CRITICAL issues immediately")
    print("  2. Address WARNING issues for optimal performance")
    print("  3. Consider MINOR issues for best practices")
else:
    print("  🎯 Model is ready for training!")
    print("  🚀 Consider tuning hyperparameters for best results")

print("\n" + "="*60)
print("Analysis complete! 🔍✨") 