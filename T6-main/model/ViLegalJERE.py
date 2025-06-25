import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from transformers import T5Config  # Thêm T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().type_as(x)
            self.sin_cached = freqs.sin().type_as(x)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CPLinear(nn.Module):
    # Bilinear form of x using CP decomposition
    def __init__(self, in_features, n_head, head_dim, rank: int = 2, q_rank: int = 8):
        super(CPLinear, self).__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank
        self.q_rank = q_rank

        self.W_A_q = nn.Linear(in_features, n_head * q_rank, bias=False)
        self.W_A_k = nn.Linear(in_features, n_head * rank, bias=False)
        self.W_A_v = nn.Linear(in_features, n_head * rank, bias=False)

        self.W_B_q = nn.Linear(in_features, q_rank * head_dim, bias=False)
        self.W_B_k = nn.Linear(in_features, rank * head_dim, bias=False)
        self.W_B_v = nn.Linear(in_features, rank * head_dim, bias=False)
        
        # ✅ FIXED: Only add rotary for self-attention, NOT cross-attention
        self.rotary = Rotary(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        W_A_q_tensor = self.W_A_q.weight.view(self.in_features, self.n_head, self.q_rank)
        W_A_k_tensor = self.W_A_k.weight.view(self.in_features, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.in_features, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_q_tensor)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_q.weight.data = W_A_q_tensor.view_as(self.W_A_q.weight)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_q_tensor = self.W_B_q.weight.view(self.in_features, self.q_rank, self.head_dim)
        W_B_k_tensor = self.W_B_k.weight.view(self.in_features, self.rank, self.head_dim)
        W_B_v_tensor = self.W_B_v.weight.view(self.in_features, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_q_tensor)
        nn.init.xavier_uniform_(W_B_k_tensor)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_q.weight.data = W_B_q_tensor.view_as(self.W_B_q.weight)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)
        
    def forward(self, x, apply_rope=True):
        # ✅ FIXED: Add apply_rope parameter to control RoPE usage
        batch_size, seq_len, _ = x.size()

        A_q = self.W_A_q(x).view(batch_size, seq_len, self.n_head, self.q_rank)
        A_k = self.W_A_k(x).view(batch_size, seq_len, self.n_head, self.rank)
        A_v = self.W_A_v(x).view(batch_size, seq_len, self.n_head, self.rank)

        B_q = self.W_B_q(x).view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(x).view(batch_size, seq_len, self.rank, self.head_dim)
        B_v = self.W_B_v(x).view(batch_size, seq_len, self.rank, self.head_dim)
        
        # ✅ FIXED: Only apply RoPE when requested (for self-attention)
        if apply_rope:
            cos, sin = self.rotary(B_q)
            B_q, B_k = apply_rotary_emb(B_q, cos, sin), apply_rotary_emb(B_k, cos, sin)
        
        A_q = A_q.view(batch_size * seq_len, self.n_head, self.q_rank)
        A_k = A_k.view(batch_size * seq_len, self.n_head, self.rank)
        A_v = A_v.view(batch_size * seq_len, self.n_head, self.rank)

        B_q = B_q.view(batch_size * seq_len, self.q_rank, self.head_dim)
        B_k = B_k.view(batch_size * seq_len, self.rank, self.head_dim)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.head_dim)
        
        q = torch.bmm(A_q, B_q).div_(self.q_rank).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank).view(batch_size, seq_len, self.n_head, self.head_dim)

        return q, k, v

class ViLegalSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, is_causal=False):
        super().__init__()
        # ✅ Use T5 standard attribute names
        self.n_head = config.num_heads  # T5 uses num_heads
        self.head_dim = config.d_kv     # T5 uses d_kv for key/value dimension
        self.n_embd = config.d_model    # T5 uses d_model
        self.rank = config.rank
        self.q_rank = config.q_rank
        self.is_cross_attention = is_cross_attention
        self.is_causal = is_causal

        # ✅ FIXED: Proper Q/K/V setup for cross-attention
        if is_cross_attention:
            # Cross-attention: Q from decoder, K/V from encoder
            self.c_q = CPLinear(self.n_embd, self.n_head, self.head_dim, self.q_rank, self.q_rank)
            self.c_kv = CPLinear(self.n_embd, self.n_head, self.head_dim, self.rank, self.rank)
            # ✅ NO RoPE for cross-attention
        else:
            # Self-attention: Q/K/V from same input
            self.c_qkv = CPLinear(self.n_embd, self.n_head, self.head_dim, self.rank, self.q_rank)

        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        
        # ✅ Safely get using_groupnorm with default value True
        self.using_groupnorm = getattr(config, 'using_groupnorm', True)
        if self.using_groupnorm:
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, hidden_states, attention_mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False, mask=None, **kwargs):
        """
        T5-compatible forward method for ViLegalSelfAttention
        """
        # Map T5 parameters to our parameters
        x = hidden_states
        encoder_hidden_states = key_value_states
        encoder_attention_mask = mask if mask is not None else attention_mask
        
        B, T, C = x.size()

        if self.is_cross_attention and encoder_hidden_states is not None:
            # ✅ Cross-attention: Q from decoder, K/V from encoder
            q, _, _ = self.c_q(x, apply_rope=False)  # Query from decoder input, NO RoPE
            _, k, v = self.c_kv(encoder_hidden_states, apply_rope=False)  # Key/Value from encoder, NO RoPE
            # Use encoder_attention_mask for cross-attention
            mask_to_use = encoder_attention_mask
        else:
            # Self-attention with RoPE
            q, k, v = self.c_qkv(x, apply_rope=True)  # Apply RoPE for self-attention
            # Use attention_mask for self-attention
            mask_to_use = attention_mask

        # ✅ Proper attention mask handling
        attn_mask_for_spda = None
        if mask_to_use is not None:
            # Convert boolean mask to additive mask for scaled_dot_product_attention
            # True = keep token, False = mask token
            if mask_to_use.dtype == torch.bool:
                attn_mask_for_spda = ~mask_to_use  # Invert for SDPA (True = mask)
            else:
                attn_mask_for_spda = mask_to_use == 0  # 0 = mask, 1 = keep
            
            # Ensure proper shape for SDPA
            if attn_mask_for_spda.dim() == 2:
                attn_mask_for_spda = attn_mask_for_spda.unsqueeze(1).unsqueeze(1)
        
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, n_head, T, head_dim)
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask_for_spda,
            is_causal=self.is_causal and not self.is_cross_attention
        )
        
        if self.using_groupnorm:
            y = self.subln(y)
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        
        # ✅ Return T5-compatible output format
        # T5 expects (hidden_states, attention_weights, position_bias)
        attention_weights = None  # We don't compute attention weights
        position_bias = None      # We don't use position bias
        
        if use_cache:
            return (y, attention_weights, position_bias, past_key_value)
        else:
            return (y, attention_weights, position_bias)





class ViLegalConfig(T5Config):
    model_type = "vilegal_jere"
    
    def __init__(self, rank=4, q_rank=8, **kwargs):
        """
        Khởi tạo ViLegalConfig.
        
        Args:
            rank (int, optional): Rank cho Key và Value trong CPLinear. Mặc định là 4.
            q_rank (int, optional): Rank cho Query trong CPLinear. Mặc định là 8.
            **kwargs: Tất cả các tham số tiêu chuẩn khác của T5
                      (ví dụ: vocab_size, d_model, num_heads,...)
                      sẽ được tự động truyền vào đây.
        """
        # 1. Gọi hàm __init__ của lớp cha (T5Config) trước tiên.
        #    **kwargs sẽ tự động thu thập và truyền tất cả các tham số T5 chuẩn
        #    (như d_model, n_head, vocab_size...) vào cho lớp cha.
        #    Lớp cha sẽ xử lý tất cả các tham số đó cho chúng ta.
        super().__init__(**kwargs)
        
        # 2. Bây giờ, chúng ta chỉ cần thêm các thuộc tính MỚI và RIÊNG BIỆT
        #    của ViLegalJERE mà T5Config không có.
        self.rank = rank
        self.q_rank = q_rank

class ViLegalJERE(T5ForConditionalGeneration):
    config_class = ViLegalConfig
    base_model_prefix = "vilegal_jere"
    supports_gradient_checkpointing = True

    def __init__(self, config: ViLegalConfig):
        # Bước 1: Gọi hàm __init__ của lớp cha (T5ForConditionalGeneration).
        # Lệnh 'super' này sẽ tự động xây dựng toàn bộ kiến trúc T5 chuẩn
        # bao gồm encoder, decoder, embedding, và lm_head cho bạn.
        super().__init__(config)

        # Bước 2: "Độ" lại kiến trúc chuẩn bằng cách thay thế các lớp
        # Self-Attention gốc bằng lớp ViLegalSelfAttention tùy chỉnh của bạn.
        
        # Vòng lặp để thay thế các khối trong Encoder
        if hasattr(self, 'encoder'):
            for i in range(len(self.encoder.block)):
                # Truy cập vào lớp SelfAttention trong khối và thay thế nó
                self.encoder.block[i].layer[0].SelfAttention = ViLegalSelfAttention(config, is_causal=False)

        # Vòng lặp để thay thế các khối trong Decoder
        if hasattr(self, 'decoder'):
            for i in range(len(self.decoder.block)):
                # Thay thế Self-Attention (lớp 0) trong khối decoder
                self.decoder.block[i].layer[0].SelfAttention = ViLegalSelfAttention(config, is_causal=True)
                
                # Thay thế Cross-Attention (lớp 1) trong khối decoder
                self.decoder.block[i].layer[1].EncDecAttention = ViLegalSelfAttention(config, is_cross_attention=True)

        # Trọng số của lm_head sẽ tự động được chia sẻ với self.shared (lớp embedding)
        # bởi hàm __init__ của lớp cha, nên chúng ta không cần làm lại.




    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings to match new vocabulary size"""
        # ✅ Use parent class method for robust resizing
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        
        # Update config vocab_size
        self.config.vocab_size = new_num_tokens
        
        return new_embeddings







    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'shared'):
            n_params -= self.shared.weight.numel()
        return n_params 