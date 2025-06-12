import torch
import json
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalJERE

def load_model_and_tokenizer(model_path="/kaggle/working/out_vilegal_t5small"):
    """Load finetuned model and tokenizer"""
    try:
        model = ViLegalJERE.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
        
        # ✅ FIX: Thêm tất cả special tokens cần thiết cho Vietnamese Legal JERE
        domain_special_tokens = [
            "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
            "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
            "<Relates_To>", "<Amended_By>"
        ]
        
        # ✅ FIX: Không cần triplet tokens vì cháu dùng format riêng
        # triplet_tokens = ["<triplet>", "<subj>", "<obj>"]  # Bỏ dòng này
        
        # ✅ FIX: Chỉ dùng domain special tokens  
        all_special_tokens = domain_special_tokens
        
        # ✅ FIX: Thêm vào tokenizer
        tokenizer.add_tokens(all_special_tokens, special_tokens=True)
        
        print(f"✅ Added {len(all_special_tokens)} special tokens to tokenizer")
        print(f"New tokenizer vocab size: {len(tokenizer)}")
        
        # ✅ FIX: Model phải resize để match tokenizer
        if model.config.vocab_size != len(tokenizer):
            print(f"🔧 Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Model loaded successfully on {device}")
        print(f"Model vocab size: {model.config.vocab_size}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def extract_vietnamese_legal_relations(text):
    """
    Trích xuất relations từ generated text theo format Vietnamese Legal JERE
    Format: <ENTITY_TYPE> entity_text <ENTITY_TYPE> entity_text <RELATION_TYPE>
    """
    relations = []
    
    # Làm sạch text
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
    
    # ✅ FIX: Parse theo format của cháu: <HEAD_TYPE> head_text <TAIL_TYPE> tail_text <RELATION>
    tokens = text.split()
    
    # Entity và relation types
    entity_types = ["<ORGANIZATION>", "<LOCATION>", "<LEGAL_PROVISION>", "<RIGHT/DUTY>", "<PERSON>", "<DATE/TIME>"]
    relation_types = ["<Relates_To>", "<Effective_From>", "<Applicable_In>", "<Amended_By>"]
    
    i = 0
    while i < len(tokens):
        # Tìm head entity
        if tokens[i] in entity_types:
            head_type = tokens[i]
            head_text = ""
            i += 1
            
            # Collect head text until next entity type
            while i < len(tokens) and tokens[i] not in entity_types and tokens[i] not in relation_types:
                head_text += " " + tokens[i]
                i += 1
            
            # Tìm tail entity
            if i < len(tokens) and tokens[i] in entity_types:
                tail_type = tokens[i]
                tail_text = ""
                i += 1
                
                # Collect tail text until relation type
                while i < len(tokens) and tokens[i] not in entity_types and tokens[i] not in relation_types:
                    tail_text += " " + tokens[i]
                    i += 1
                
                # Tìm relation
                if i < len(tokens) and tokens[i] in relation_types:
                    relation = tokens[i]
                    
                    # Tạo triplet
                    if head_text.strip() and tail_text.strip():
                        relations.append({
                            'head': head_text.strip(),
                            'head_type': head_type.replace('<', '').replace('>', ''),
                            'tail': tail_text.strip(),
                            'tail_type': tail_type.replace('<', '').replace('>', ''),
                            'relation': relation.replace('<', '').replace('>', '')
                        })
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return relations

def generate_relations(model, tokenizer, device, context_text, max_length=512):
    """Generate relation extraction from context"""
    # Tokenize input (encoder input)
    inputs = tokenizer(
        context_text,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # ✅ FIX: Sử dụng generation parameters tốt hơn
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            do_sample=False,  # ✅ Sử dụng deterministic generation để debug
            num_beams=3,      # ✅ Beam search cho output ổn định hơn
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2  # ✅ Tránh lặp lại
        )
    
    # Decode output (skip the start token)
    generated_text = tokenizer.decode(outputs[0, 1:], skip_special_tokens=False)
    
    # ✅ Trích xuất relations theo format Vietnamese Legal
    relations = extract_vietnamese_legal_relations(generated_text)
    
    return generated_text, relations

def test_model():
    """Test model with 3 sample cases"""
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Test cases from your data
    test_cases = [
        {
            "id": "54/2019/QH14__Dieu51",
            "context": "Điều 51: Tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam 1. Nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài khi tham gia đầu tư, hoạt động trên thị trường chứng khoán Việt Nam tuân thủ quy định về tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư theo quy định của pháp luật về chứng khoán và thị trường chứng khoán. 2. Chính phủ quy định chi tiết tỷ lệ sở hữu nước ngoài, điều kiện, trình tự, thủ tục đầu tư, việc tham gia của nhà đầu tư nước ngoài, tổ chức kinh tế có vốn đầu tư nước ngoài trên thị trường chứng khoán Việt Nam.",
            "expected": "<ORGANIZATION> tổ chức kinh tế có vốn đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To> <LEGAL_PROVISION> pháp luật về chứng khoán và thị trường chứng khoán <LOCATION> thị trường chứng khoán Việt Nam <Relates_To> <ORGANIZATION> Chính phủ <ORGANIZATION> tổ chức kinh tế có vốn đầu tư nước ngoài <Relates_To> <ORGANIZATION> tổ chức kinh tế có vốn đầu tư nước ngoài <LOCATION> thị trường chứng khoán Việt Nam <Relates_To>"
        },
        {
            "id": "59/2020/QH14__Dieu173", 
            "context": "Điều 173: Trách nhiệm của Kiểm soát viên 1. Tuân thủ đúng pháp luật, Điều lệ công ty, nghị quyết Đại hội đồng cổ đông và đạo đức nghề nghiệp trong thực hiện quyền và nghĩa vụ được giao. 2. Thực hiện quyền và nghĩa vụ được giao một cách trung thực, cẩn trọng, tốt nhất nhằm bảo đảm lợi ích hợp pháp tối đa của công ty. 3. Trung thành với lợi ích của công ty và cổ đông; không lạm dụng địa vị, chức vụ và sử dụng thông tin, bí quyết, cơ hội kinh doanh, tài sản khác của công ty để tư lợi hoặc phục vụ lợi ích của tổ chức, cá nhân khác. 4. Nghĩa vụ khác theo quy định của Luật này và Điều lệ công ty. 5. Trường hợp vi phạm quy định tại các khoản 1, 2, 3 và 4 Điều này mà gây thiệt hại cho công ty hoặc người khác thì Kiểm soát viên phải chịu trách nhiệm cá nhân hoặc liên đới bồi thường thiệt hại đó. Thu nhập và lợi ích khác mà Kiểm soát viên có được do vi phạm phải hoàn trả cho công ty. 6. Trường hợp phát hiện có Kiểm soát viên vi phạm trong thực hiện quyền và nghĩa vụ được giao thì phải thông báo bằng văn bản đến Ban kiểm soát; yêu cầu người có hành vi vi phạm chấm dứt hành vi vi phạm và khắc phục hậu quả.",
            "expected": "<RIGHT/DUTY> Tuân thủ đúng pháp luật, Điều lệ công ty, nghị quyết Đại hội đồng cổ đông và đạo đức nghề nghiệp trong thực hiện quyền và nghĩa vụ được giao <LEGAL_PROVISION> Điều 173 <Relates_To> <RIGHT/DUTY> Thực hiện quyền và nghĩa vụ được giao một cách trung thực, cẩn trọng, tốt nhất nhằm bảo đảm lợi ích hợp pháp tối đa của công ty <LEGAL_PROVISION> Điều 173 <Relates_To>"
        },
        {
            "id": "54/2019/QH14__Dieu63",
            "context": "Điều 63: Bừ trừ và thanh toán giao dịch chứng khoán 1. Hoạt động bù trừ, xác định nghĩa vụ thanh toán tiền và chứng khoán được thực hiện thông qua Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam. 2. Thanh toán chứng khoán được thực hiện trên hệ thống tài khoản lưu ký tại Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam, thanh toán tiền giao dịch chứng khoán được thực hiện qua ngân hàng thanh toán và phải tuân thủ nguyên tắc chuyển giao chứng khoán đồng thời với thanh toán tiền. 3. Bộ trưởng Bộ Tài chính quy định các biện pháp xử lý trong trường hợp thành viên của Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam tạm thời mất khả năng thanh toán giao dịch chứng khoán.",
            "expected": "<ORGANIZATION> Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam <LEGAL_PROVISION> Điều 63 <Relates_To> <ORGANIZATION> Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam <ORGANIZATION> ngân hàng thanh toán <Relates_To> <ORGANIZATION> Bộ Tài chính <ORGANIZATION> Tổng công ty lưu ký và bù trừ chứng khoán Việt Nam <Relates_To>"
        }
    ]
    
    print("=" * 80)
    print("TESTING FINETUNED ViLegalJERE MODEL")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}: {test_case['id']}")
        print("-" * 60)
        
        print(f"📝 INPUT CONTEXT:")
        print(f"{test_case['context'][:200]}...")
        
        print(f"\n🎯 EXPECTED RELATIONS:")
        print(f"{test_case['expected'][:150]}...")
        
        print(f"\n🤖 MODEL GENERATED:")
        try:
            generated_text, relations = generate_relations(model, tokenizer, device, test_case['context'])
            print(f"{generated_text}")
            
            # Simple evaluation
            if generated_text and len(generated_text) > 10:
                print(f"✅ Generation successful ({len(generated_text)} chars)")
                
                # Check if output contains expected patterns
                has_entities = any(tag in generated_text for tag in ["<ORGANIZATION>", "<LOCATION>", "<RIGHT/DUTY>", "<LEGAL_PROVISION>"])
                has_relations = "<Relates_To>" in generated_text
                
                if has_entities and has_relations:
                    print(f"✅ Output format looks correct (has entities and relations)")
                    print(f"🎯 Extracted {len(relations)} relations:")
                    for i, rel in enumerate(relations[:3]):  # Show first 3 relations
                        print(f"   {i+1}. {rel['head']} ({rel['head_type']}) --{rel['relation']}--> {rel['tail']} ({rel['tail_type']})")
                else:
                    print(f"⚠️ Output format may be incorrect")
                    print(f"🎯 Extracted {len(relations)} relations")
            else:
                print("❌ Generation failed or too short")
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    test_model()
