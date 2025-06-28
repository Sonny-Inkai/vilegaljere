#!/usr/bin/env python3
"""
Test file để kiểm tra hàm create_t5_spans
Kiểm tra xem có bị mất token đầu tiên của cụm bị che không
"""

import numpy as np
from transformers import AutoTokenizer

# Copy các hàm từ pretrain_vilegaljere.py
def create_noise_mask(length: int, noise_density: float, mean_noise_span_length: float) -> np.ndarray:
    """Tạo noise mask theo đúng thuật toán của Google T5."""
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
    num_noise_spans = max(num_noise_spans, 1)
    
    if num_noise_tokens >= length:
        return np.ones(length, dtype=bool)

    def _random_segmentation(num_items, num_segments):
        if num_items < num_segments:
            return np.array([1] * num_items + [0] * (num_segments - num_items), dtype=np.int64)

        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    num_nonnoise_tokens = length - np.sum(noise_span_lengths)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = 1
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise

def create_t5_spans(tokens: list, tokenizer) -> (list, list):
    """
    PHIÊN BẢN CHUẨN VÀ ĐÁNG TIN CẬY NHẤT.
    Thực hiện T5 span corruption bằng vòng lặp Python đơn giản, dễ hiểu và chính xác.
    """
    noise_mask = create_noise_mask(len(tokens), 0.15, 3.0)
    
    # Lấy ID của sentinel token đầu tiên để làm cơ sở
    sentinel_base_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    input_ids_list = []
    labels_ids_list = []
    
    in_noise_span = False
    sentinel_idx = 0
    
    for i, token_id in enumerate(tokens):
        is_noise = noise_mask[i]
        
        if is_noise:
            # Nếu một token nằm trong vùng nhiễu
            if not in_noise_span:
                # Nếu đây là token đầu tiên của một cụm nhiễu mới
                # 1. Đánh dấu bắt đầu một cụm nhiễu
                in_noise_span = True
                # 2. Lấy sentinel ID tiếp theo
                sentinel_id = sentinel_base_id - sentinel_idx
                # 3. Thêm sentinel ID vào cả input và labels
                input_ids_list.append(sentinel_id)
                labels_ids_list.append(sentinel_id)
                sentinel_idx += 1
            
            # 4. Thêm token gốc (bị che) vào labels
            labels_ids_list.append(token_id)
            
        else:
            # Nếu một token không nằm trong vùng nhiễu
            if in_noise_span:
                # Nếu token ngay trước đó là nhiễu, đánh dấu kết thúc cụm nhiễu
                in_noise_span = False
            
            # Thêm token gốc vào input
            input_ids_list.append(token_id)

    # Thêm token kết thúc chuỗi vào cuối labels
    if not labels_ids_list:
        # Xử lý trường hợp không có token nào bị che
        return [], []
    else:
        labels_ids_list.append(tokenizer.eos_token_id)

    return input_ids_list, labels_ids_list

def test_create_spans():
    """Test hàm create_t5_spans với ví dụ cụ thể"""
    print("🧪 Testing create_t5_spans function...")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
    except:
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
    # Test case 1: Câu tiếng Việt đơn giản
    test_text = "Điều 14 02/2014/TT-BNNPTTN quy định trình tự, thủ tục cấp và thu hồi giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi Nội dung, trình tự các phiên họp Hội đồng 1. Phiên họp thứ nhất: a) Thư ký hành chính (là đại diện của cơ quan thường trực, Bộ Nông nghiệp và Phát triển nông thôn) giới thiệu đại biểu, tuyên bố lý do phiên họp và báo cáo tóm tắt về tính hợp lệ của hồ sơ; b) Chủ tịch Hội đồng chủ trì phiên họp, phân công 01 (một) Thư ký hội đồng và 02 (hai) thành viên phản biện hồ sơ. Trường hợp cần thiết, Hội đồng có thể kiến nghị bổ sung từ 2-3 chuyên gia phản biện độc lập là các nhà khoa học có kinh nghiệm trong lĩnh vực liên quan; c) Hội đồng bầu ban kiểm phiếu, gồm 03 (ba) thành viên là ủy viên Hội đồng, trong đó có 01 (một) Trưởng ban; d) Hội đồng thống nhất thời gian phiên họp thứ hai và kế hoạch làm việc. 2. Phiên họp thứ hai: a) Thư ký hành chính đọc báo cáo tổng hợp ý kiến công chúng về hồ sơ đăng ký cấp Giấy xác nhận và nhận xét của chuyên gia phản biện độc lập theo mẫu quy định tại Phụ lục 7 của Thông tư này (nếu có); b) Ủy viên của Hội đồng nhận xét hồ sơ đăng ký theo biểu mẫu quy định tại Phụ lục 7 của Thông tư này; c) Hội đồng trao đổi, thảo luận về hồ sơ đăng ký cấp Giấy xác nhận theo các yêu cầu quy định tại Thông tư này; d) Hội đồng bỏ phiếu đánh giá hồ sơ đăng ký theo biểu mẫu quy định tại Phụ lục 8; đ) Ban kiểm phiếu tổng hợp và báo cáo kết quả kiểm phiếu theo biểu mẫu quy định tại Phụ lục 9 của Thông tư này; e) Hồ sơ đăng ký đạt yêu cầu là hồ sơ đạt ít nhất ¾ (ba phần tư) số phiếu đánh giá “đạt yêu cầu” của thành viên Hội đồng tham dự phiên họp; g) Hội đồng thảo luận, kết luận và kiến nghị những điểm bổ sung, sửa đổi cần thiết về những nội dung đã nêu trong hồ sơ (nếu có) và thông qua Biên bản cuộc họp theo quy định tại Phụ lục 10 của Thông tư này."
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