import os
from sympy import quo
import torch
from model.ViLegalJERE import ViLegalJERE
from utils import load_custom_tokenizer

def load_finetuned_model(finetune_dir, device='cuda'):
    """Load finetuned ViLegalJERE model from checkpoint directory"""
    if not os.path.exists(finetune_dir):
        raise FileNotFoundError(f"Finetune directory not found: {finetune_dir}")
    
    print(f"🔄 Loading finetuned model from {finetune_dir}")
    
    # Load model from pretrained checkpoint
    model = ViLegalJERE.from_pretrained(finetune_dir)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print("✅ Finetuned model loaded successfully!")
    return model

def generate_relations(model, tokenizer, device, context_text, max_length=512):
    """
    Generate relations from Vietnamese legal context text
    
    Args:
        model: Finetuned ViLegalJERE model
        tokenizer: Custom Vietnamese legal tokenizer
        device: torch device (cuda/cpu)
        context_text: Input Vietnamese legal text
        max_length: Maximum generation length
        
    Returns:
        str: Generated relations text
    """
    model.eval()
    
    with torch.no_grad():
        # Tokenize input text
        inputs = tokenizer(
            context_text,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Generate with T5 standard parameters
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,  # Target max length for relations
            min_length=10,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.2,
            length_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )
        
        return generated_text.strip()

def test_single_case(model, tokenizer, device, case_name, input_text, expected_target):
    """Test a single case and return results"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST CASE: {case_name}")
    print(f"{'='*80}")
    
    print("\n📝 INPUT TEXT:")
    print("-" * 50)
    print(input_text[:200] + "..." if len(input_text) > 200 else input_text)
    
    print("\n🎯 EXPECTED TARGET:")
    print("-" * 50)
    print(expected_target)
    
    print("\n🚀 Generating relations...")
    
    # Generate relations
    result = generate_relations(
        model=model,
        tokenizer=tokenizer, 
        device=device,
        context_text=input_text,
        max_length=512
    )
    
    print("\n✨ GENERATED OUTPUT:")
    print("-" * 50)
    print(result)
    
    print("\n📊 COMPARISON:")
    print("-" * 50)
    print(f"Expected:  {expected_target}")
    print(f"Generated: {result}")
    
    # Simple accuracy check
    is_match = expected_target.strip() == result.strip()
    if is_match:
        print("\n✅ PERFECT MATCH!")
    else:
        print("\n❓ Different output - analyze model performance")
    
    return is_match, result

def main():
    """Multiple test cases for comprehensive model evaluation"""
    # Configuration
    finetune_dir = '/kaggle/input/vilegaljere/vilegaljere_finetune'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    print("📥 Loading custom tokenizer...")
    tokenizer = load_custom_tokenizer(master_process=True)
    
    # Load finetuned model
    model = load_finetuned_model(finetune_dir, device)
    
    # Test cases from training data
    test_cases = [
        {
            "name": "legal_28 - Điều 5 BNNPTTN",
            "input": "Điều 5 02/2014/TT-BNNPTTN quy định trình tự, thủ tục cấp và thu hồi giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi Các trường hợp phải đăng ký cấp Giấy xác nhận 1. Thực vật biến đổi gen mang sự kiện chuyển gen đơn lẻ (single transformation event) là kết quả của quá trình chuyển một gen quy định một tính trạng mong muốn bằng công nghệ chuyển gen. 2. Thực vật biến đổi gen mang sự kiện chuyển gen tổ hợp (vector stacked transformation event) là kết quả của quá trình chuyển từ hai hoặc nhiều gen quy định một hoặc nhiều tính trạng mong muốn bằng công nghệ chuyển gen.",
            "target": "<LEGAL_PROVISION> Điều 5 02/2014/TT-BNNPTTN <LEGAL_PROVISION> giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi Các trường hợp phải đăng ký cấp Giấy xác nhận <Relates_To>"
        },
        {
            "name": "legal_26 - Điều 3 BNNPTTN with ORGANIZATION",
            "input": "Điều 3 02/2014/TT-BNNPTTN quy định trình tự, thủ tục cấp và thu hồi giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi Giải thích từ ngữ Trong Thông tư này các từ ngữ dưới đây được hiểu như sau: 1. Thực vật biến đổi gen là thực vật, mẫu vật di truyền, sản phẩm trực tiếp của thực vật mang một hoặc nhiều gen mới được tạo ra bằng công nghệ ADN tái tổ hợp. 2. Đánh giá rủi ro của thực vật biến đổi gen đối với sức khỏe con người và vật nuôi (sau đây gọi tắt là đánh giá rủi ro) là các hoạt động nhằm xác định nguy cơ tiềm ẩn và khả năng xảy ra rủi ro của thực vật biến đổi gen khi sử dụng làm thực phẩm, thức ăn chăn nuôi. 3. Sự kiện chuyển gen là kết quả của quá trình tái tổ hợp ADN mục tiêu vào một vị trí nhất định trong hệ gen của một loài cây để tạo ra một cây tương ứng mang gen mục tiêu. 4. Nước phát triển là nước có nền công nghệ sinh học tiên tiến trong nhóm các nước thuộc Tổ chức hợp tác và Phát triển kinh tế - OECD và nhóm các nước có nền kinh tế lớn G20. 5. Mã nhận diện duy nhất là mã do Tổ chức hợp tác và Phát triển kinh tế xác định cho từng sự kiện chuyển gen.",
            "target": "<LEGAL_PROVISION> Điều 3 02/2014/TT-BNNPTTN <ORGANIZATION> Tổ chức hợp tác và Phát triển kinh tế <Relates_To>"
        },
        {
            "name": "legal_29 - Điều 6 BNNPTTN Simple",
            "input": "Điều 6 02/2014/TT-BNNPTTN quy định trình tự, thủ tục cấp và thu hồi giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi Điều kiện cấp Giấy xác nhận Thực vật biến đổi gen được cấp Giấy xác nhận phải đáp ứng một trong các điều kiện sau: 1. Thực vật biến đổi gen được ít nhất 05 (năm) nước phát triển cho phép sử dụng làm thực phẩm, thức ăn chăn nuôi và chưa xảy ra rủi ro ở các nước đó.",
            "target": "<LEGAL_PROVISION> 02/2014/TT-BNNPTTN <ORGANIZATION> BNNPTTN <Relates_To>"
        },
        {
            "name": "legal_30 - Điều 7 BNNPTTN Document Reference",
            "input": "Điều 7 02/2014/TT-BNNPTTN quy định trình tự, thủ tục cấp và thu hồi giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi Hồ sơ đăng ký cấp Giấy xác nhận 1. Số lượng hồ sơ: 03 (ba) bộ, gồm 01 (một) bản chính và 02 (hai) bản sao. 2. Trường hợp đăng ký cấp Giấy xác nhận cho đối tượng quy định tại khoản 1 Điều 6 của Thông tư này, hồ sơ bao gồm: a) Đơn đăng ký cấp Giấy xác nhận theo mẫu quy định tại Phụ lục 1 của Thông tư này;",
            "target": "<LEGAL_PROVISION> 02/2014/TT-BNNPTTN <LEGAL_PROVISION> Thông tư này <Relates_To>"
        },
        {
            "name": "legal_31 - Complex Multiple Relations",
            "input": "Điều 7 02/2014/TT-BNNPTTN quy định trình tự, thủ tục cấp và thu hồi giấy xác nhận thực vật biến đổi gen đủ điều kiện sử dụng làm thực phẩm, thức ăn chăn nuôi 4. Trường hợp đăng ký cấp Giấy xác nhận cho đối tượng quy định tại khoản 2 Điều 5 của Thông tư này, hồ sơ bao gồm: a) Các tài liệu quy định tại khoản 2 Điều này (trường hợp đăng ký cấp Giấy xác nhận cho đối tượng quy định tại khoản 1 Điều 6 của Thông tư này); b) Các tài liệu quy định tại khoản 3 Điều này (trường hợp đăng ký cấp Giấy xác nhận cho đối tượng quy định tại khoản 2 Điều 6 của Thông tư này);",
            "target": "<LEGAL_PROVISION> Điều 7 02/2014/TT-BNNPTTN <LEGAL_PROVISION> Thông tư này <Relates_To> <LEGAL_PROVISION> Điều 7 02/2014/TT-BNNPTTN <LEGAL_PROVISION> Thông tư này <Relates_To> <LEGAL_PROVISION> Điều 7 02/2014/TT-BNNPTTN <LEGAL_PROVISION> Thông tư này <Relates_To>"
        }
    ]
    
    print("="*80)
    print("🧪 VIETNAMESE LEGAL JERE COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Track results
    results = []
    total_cases = len(test_cases)
    perfect_matches = 0
    
    # Run all test cases
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔥 Running Test {i}/{total_cases}")
        
        is_match, generated = test_single_case(
            model, tokenizer, device,
            case["name"],
            case["input"], 
            case["target"]
        )
        
        results.append({
            "name": case["name"],
            "match": is_match,
            "expected": case["target"],
            "generated": generated
        })
        
        if is_match:
            perfect_matches += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Perfect Matches: {perfect_matches}/{total_cases} ({perfect_matches/total_cases*100:.1f}%)")
    print(f"❓ Different Outputs: {total_cases - perfect_matches}/{total_cases}")
    
    print("\n📋 DETAILED RESULTS:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "✅ MATCH" if result["match"] else "❌ DIFF"
        print(f"{i}. {result['name'][:40]:40} | {status}")
    
    print(f"\n{'='*80}")
    
    if perfect_matches == total_cases:
        print("🎉 ALL TESTS PASSED! Model is performing perfectly!")
    elif perfect_matches > total_cases * 0.7:
        print("👍 Model performing well! Some minor differences to investigate.")
    else:
        print("⚠️ Model needs improvement. Check training data and hyperparameters.")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
