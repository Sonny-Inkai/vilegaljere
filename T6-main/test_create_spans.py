#!/usr/bin/env python3
"""
Test file để kiểm tra hàm create_t5_spans
Kiểm tra xem có bị mất token đầu tiên của cụm bị che không
"""

import numpy as np
from transformers import AutoTokenizer
from pretrain_vilegaljere import create_t5_spans, create_noise_mask


def test_create_spans():
    """Test hàm create_t5_spans với ví dụ cụ thể"""
    print("🧪 Testing create_t5_spans function...")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    except:
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
    # Test case 1: Câu tiếng Việt đơn giản
    test_text = "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở. 3. Việc phối hợp phải thường xuyên, kịp thời, đồng bộ, chặt chẽ, thống nhất, đúng quy định của pháp luật."

    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    
    print(f"\n📝 Original text: {test_text}")
    print(f"🔢 Original tokens: {tokens}")
    print(f"📖 Token strings: {[tokenizer.decode([t]) for t in tokens]}")
    
    # Set seed để có kết quả nhất quán
    np.random.seed(42)
    
    # Tạo spans
    input_ids, labels = create_t5_spans(tokens, tokenizer)
    
    print(f"\n✅ Input IDs: {input_ids}")
    print(f"🎯 Labels: {labels}")
    
    # Decode để xem kết quả
    input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    labels_text = tokenizer.decode(labels, skip_special_tokens=False)
    
    print(f"\n🔍 Input decoded: {input_text}")
    print(f"🔍 Labels decoded: {labels_text}")
    
    # Kiểm tra chi tiết noise mask
    noise_mask = create_noise_mask(len(tokens), 0.15, 3.0)
    print(f"\n🎭 Noise mask: {noise_mask}")
    
    # Phân tích chi tiết từng token
    print(f"\n📊 Detailed analysis:")
    for i, (token_id, is_noise) in enumerate(zip(tokens, noise_mask)):
        token_str = tokenizer.decode([token_id])
        status = "NOISE" if is_noise else "KEEP"
        print(f"  Token {i}: {token_id:5d} '{token_str:15s}' -> {status}")
    
    # Test case 2: Câu ngắn khác
    print(f"\n" + "="*60)
    print("🧪 Test case 2:")
    
    test_text2 = "Tôi đang học về luật pháp Việt Nam."
    tokens2 = tokenizer.encode(test_text2, add_special_tokens=False)
    
    print(f"📝 Original text: {test_text2}")
    print(f"🔢 Original tokens: {tokens2}")
    
    np.random.seed(123)  # Seed khác
    input_ids2, labels2 = create_t5_spans(tokens2, tokenizer)
    
    input_text2 = tokenizer.decode(input_ids2, skip_special_tokens=False)
    labels_text2 = tokenizer.decode(labels2, skip_special_tokens=False)
    
    print(f"🔍 Input decoded: {input_text2}")
    print(f"🔍 Labels decoded: {labels_text2}")
    
    # Verify không có token nào bị mất
    print(f"\n✅ Verification:")
    print(f"Original tokens count: {len(tokens2)}")
    
    # Đếm số token trong labels (trừ sentinel và eos)
    sentinel_count = sum(1 for token_id in labels2 if token_id >= tokenizer.convert_tokens_to_ids('<extra_id_0>') - 100)
    eos_count = 1  # EOS token
    actual_content_tokens = len(labels2) - sentinel_count - eos_count
    
    noise_mask2 = create_noise_mask(len(tokens2), 0.15, 3.0)
    expected_noise_tokens = sum(noise_mask2)
    
    print(f"Expected noise tokens: {expected_noise_tokens}")
    print(f"Actual content tokens in labels: {actual_content_tokens}")
    print(f"Match: {expected_noise_tokens == actual_content_tokens}")

if __name__ == "__main__":
    test_create_spans() 