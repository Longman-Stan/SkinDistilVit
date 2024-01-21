import torch
from transformers import ViTModel, ViTForImageClassification
import os
import sys
sys.path.append("..")
from model.fcvit.modeling_fcvit import FCViTForImageClassificationProbs
from model.distilvit.modeling_distilvit import DistilViTForImageClassification

KEEP_LAYERS = 1

model_checkpoint = f"../../experiments/matryoshka/MDistilViT_{KEEP_LAYERS+1}"
checkpoint_weights_dest = f"../../experiments/matryoshka/MDistilViT_{KEEP_LAYERS}_untrained/pytorch_model.bin"
os.makedirs( checkpoint_weights_dest.rsplit('/',1)[0], exist_ok=True)

model = DistilViTForImageClassification.from_pretrained(model_checkpoint)

state_dict = model.state_dict()
compressed_state_dict = {}

teacher_selected_layers = list(range(KEEP_LAYERS))

crt_layer = None
crt_layer_idx = -1

for key in state_dict:
    if "classifier" not in key:
        compressed_key = f"distilvit.{key.split('.',1)[1]}"
    else:
        compressed_key = key
    if ".layer." in key:
        layer_l, layer_r = compressed_key.split(".layer.",1)
        left_part = f"{layer_l}.layer."
        layer_num, layer_r = layer_r.split('.',1)
        right_part = f".{layer_r}"
        layer_num = int(layer_num)

        if layer_num in teacher_selected_layers:
            if layer_num != crt_layer:
                crt_layer = layer_num
                crt_layer_idx += 1
            compressed_state_dict[f"{left_part}{crt_layer_idx}{right_part}"] = state_dict[key]
    else:
        compressed_state_dict[compressed_key] = state_dict[key]

print(f"N layers selected for distillation: {crt_layer_idx+1}")
print(f"Number of params transferred for distillation: {len(compressed_state_dict.keys())}")

print(f"Save transferred checkpoint to {checkpoint_weights_dest}.")
torch.save(compressed_state_dict, checkpoint_weights_dest)
