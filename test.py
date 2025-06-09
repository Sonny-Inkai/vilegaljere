import os
import json

def load_finetune_data():
    """Tải và xử lý dữ liệu từ file finetune.json (JSON) cho fine-tuning"""
    data_file = os.path.join('./split_data/test.json')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Finetune dataset not found at {data_file}")
    
    processed_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key, value in data.items():
        source_text = value.get("formatted_context_sent", "")
        target_text = value.get("extracted_relations_text", "")
        
        # Chỉ lấy các cặp dữ liệu có cả input và output
        if source_text and target_text:
            processed_data.append((source_text, target_text))
    
   
    return processed_data

print(len(load_finetune_data()))
print(load_finetune_data()[0][1])