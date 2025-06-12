import torch
import json
from transformers import AutoTokenizer
from model.ViLegalJERE import ViLegalJERE, ViLegalConfig
import os

def load_model_and_tokenizer(model_path="/kaggle/working/out_vilegal_t5small"):
    """Load finetuned model and tokenizer"""
    try:
        # ✅ FIX: Setup tokenizer với special tokens trước
        print("🔧 Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('sonny36/vilegaljere')
        
        domain_special_tokens = [
            "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
            "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
            "<Relates_To>", "<Amended_By>"
        ]
        
        tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
        print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
        
        # ✅ FIX: Check if model path exists
        if not os.path.exists(model_path):
            print(f"❌ Model path not found: {model_path}")
            
            # Try alternative paths
            alt_paths = ["out_vilegal_t5small", "./out_vilegal_t5small"]
            model_path = None
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"✅ Found model at: {model_path}")
                    break
            
            if model_path is None:
                print("🔧 No trained model found. Creating new model...")
                print("⚠️  WARNING: This will be an UNTRAINED model!")
                config = ViLegalConfig(vocab_size=len(tokenizer))
                model = ViLegalJERE(config)
                return model, tokenizer
        
        # ✅ FIX: Try to load with proper error handling
        try:
            print(f"🔧 Loading model from: {model_path}")
            model = ViLegalJERE.from_pretrained(model_path)
            print(f"✅ Model loaded successfully")
            
            # ✅ FIX: Check vocab size mismatch
            if model.shared.num_embeddings != len(tokenizer):
                print(f"🔧 VOCAB SIZE MISMATCH!")
                print(f"   Model: {model.shared.num_embeddings}")
                print(f"   Tokenizer: {len(tokenizer)}")
                print(f"   Resizing model embeddings...")
                model.resize_token_embeddings(len(tokenizer))
                print(f"✅ Resized to {len(tokenizer)}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading saved model: {e}")
            print("🔧 Creating new model with correct vocab size...")
            print("⚠️  WARNING: This will be an UNTRAINED model!")
            
            config = ViLegalConfig(vocab_size=len(tokenizer))
            model = ViLegalJERE(config)
            return model, tokenizer
        
    except Exception as e:
        print(f"❌ Fatal error in model loading: {e}")
        raise e

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
    
    # ✅ FIXED: Beam search works now!
    print(f"🔧 Input shape: {inputs['input_ids'].shape}")
    print(f"🔧 Input token range: [{inputs['input_ids'].min().item()}, {inputs['input_ids'].max().item()}]")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=200,   # ✅ Longer for better output
            do_sample=False,  # ✅ Deterministic generation
            num_beams=3,      # ✅ Beam search works now!
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2  # ✅ Avoid repetition
        )
    
    print(f"🔧 Output shape: {outputs.shape}")
    print(f"🔧 Output token range: [{outputs.min().item()}, {outputs.max().item()}]")
    
    # Decode output (skip the start token)
    generated_text = tokenizer.decode(outputs[0, 1:], skip_special_tokens=False)
    
    # ✅ Trích xuất relations theo format Vietnamese Legal
    relations = extract_vietnamese_legal_relations(generated_text)
    
    return generated_text, relations

def test_generation():
    """Test generation với các test cases từ file test.json"""
    print("🧪 TESTING VIETNAMESE LEGAL JERE GENERATION")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # ✅ FIX: Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on: {device}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load test data
    test_data_path = "../split_data/test.json"
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found: {test_data_path}")
        print("🔧 Creating simple test case...")
        test_cases = [{
            'id': 'simple_test',
            'context': 'Điều 1: Phạm vi điều chỉnh Bộ luật lao động quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động.',
            'relations': '<RIGHT/DUTY> quyền, nghĩa vụ, trách nhiệm của người lao động <LEGAL_PROVISION> Bộ luật lao động <Relates_To>'
        }]
    else:
        import json
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_cases = []
        for key, value in list(test_data.items())[:3]:  # Take first 3 cases
            test_cases.append({
                'id': key,
                'context': value.get('formatted_context_sent', ''),
                'relations': value.get('extracted_relations_text', '')
            })
    
    # ✅ FIX: Warning about untrained model
    print("\n⚠️  IMPORTANT NOTE:")
    if model.config.vocab_size != 7110:  # If not the original trained model
        print("🔴 THIS IS AN UNTRAINED MODEL!")
        print("   - Output will be random/nonsensical")
        print("   - You need to train the model first using train_vilegal_jere.py")
        print("   - This test only verifies that generation works without errors")
    else:
        print("✅ Using trained model - expecting meaningful output")
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}: {test_case['id']}")
        print("-" * 60)
        print(f"📝 INPUT CONTEXT:")
        print(f"{test_case['context'][:200]}{'...' if len(test_case['context']) > 200 else ''}")
        
        print(f"\n🎯 EXPECTED RELATIONS:")
        print(f"{test_case['relations'][:200]}{'...' if len(test_case['relations']) > 200 else ''}")
        
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
            import traceback
            traceback.print_exc()
        
        print("=" * 60)

if __name__ == "__main__":
    test_generation()
