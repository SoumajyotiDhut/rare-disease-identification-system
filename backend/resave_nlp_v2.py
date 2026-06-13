import torch
import numpy as np

print('Re-saving NLP V2 model...')
ckpt = torch.load(
    'models/improved_nlp_v2.pt',
    map_location='cpu',
    weights_only=False
)

def convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict):
        return {convert(k): convert(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    return obj

clean = {
    'model_state_dict': ckpt['model_state_dict'],
    'label_remap'     : convert(ckpt['label_remap']),
    'reverse_remap'   : convert(ckpt['reverse_remap']),
    'num_classes'     : int(ckpt['num_classes']),
    'metrics'         : convert(ckpt.get('metrics', {})),
    'architecture'    : ckpt.get('architecture', 'BioBERTClassifierV2')
}

torch.save(clean, 'models/improved_nlp_v2_clean.pt')
print('Clean model saved')
print(f'Classes: {clean["num_classes"]}')