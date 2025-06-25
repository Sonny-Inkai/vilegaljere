# Test script for new ViLegalConfig implementation
from model.ViLegalJERE import ViLegalConfig, ViLegalJERE
import torch

# Test config creation with T5 parameters
n_embd = 512
n_layer = 6 
n_head = 8
head_dim = 64
rank = 4
q_rank = 8
dropout = 0.1

model_args = dict(
    vocab_size=10100,
    d_model=n_embd,
    num_layers=n_layer,
    num_heads=n_head,
    d_kv=head_dim,
    d_ff=4 * n_embd,
    dropout_rate=dropout,
    pad_token_id=0,
    eos_token_id=1,
    decoder_start_token_id=0,
    rank=rank,
    q_rank=q_rank,
)

try:
    config_obj = ViLegalConfig(**model_args)
    model = ViLegalJERE(config_obj)
    
    print('✅ Config and Model created successfully!')
    print(f'Config: vocab_size={config_obj.vocab_size}, d_model={config_obj.d_model}')
    print(f'T5 attrs: num_layers={config_obj.num_layers}, num_heads={config_obj.num_heads}')
    print(f'ViLegal attrs: rank={config_obj.rank}, q_rank={config_obj.q_rank}')
    print(f'Model parameters: {model.get_num_params():,}')
    
    # Test generation
    input_ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=20, num_beams=1, do_sample=False)
    print(f'✅ Generation test successful! Output shape: {outputs.shape}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc() 