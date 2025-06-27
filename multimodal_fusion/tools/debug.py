
#######################
### text_encoder.py ###
#######################

'''
import torch
from text_encoder import build_text_encoder

# Dummy param object for testing
class DummyParams:
    def __init__(self, text_encoder_name, text_encoder_path, model_dim, text_dropout=0.1, freeze=True):
        self.text_encoder_name = text_encoder_name
        self.text_encoder_path = text_encoder_path
        self.model_dim = model_dim
        self.text_dropout = text_dropout
        self.freeze_text_encoder = freeze

# Define test samples
sample_texts = [
    "The structure is composed of edge-sharing TiO6 octahedra.",
    "Fe2O3 crystallizes in the corundum structure.",
    "SiO2 forms a three-dimensional network of SiO4 tetrahedra.",
]

# Define encoder configs to test
test_configs = [
    {
        "name": "MatSciBERT",
        "params": DummyParams(
            text_encoder_name="matscibert",
            text_encoder_path="/home/yangjw/crystalformer/llm/matscibert",
            model_dim=128
        )
    },
    {
        "name": "Generic BERT (bert-base-uncased)",
        "params": DummyParams(
            text_encoder_name="bert",
            text_encoder_path="/home/yangjw/crystalformer/llm/bert-base-uncased",
            model_dim=128
        )
    }
]

# Run test
for config in test_configs:
    print(f"\n🔍 Testing encoder: {config['name']}")
    encoder = build_text_encoder(config["params"])
    encoder.eval()

    with torch.no_grad():
        output = encoder(sample_texts)

    print("✅ Output shape:", output.shape)
    print("✅ Sample output (first row):", output[0])
'''
'''
### matscibert

from transformers import AutoModel, AutoTokenizer
import torch

model_name = "m3rg-iitd/matscibert"
save_dir = "/home/yangjw/crystalformer/llm/matscibert"

# save
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("✅ MatSciBERT saved locally at:", save_dir)

# check
print("🔍 Testing local MatSciBERT...")
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModel.from_pretrained(save_dir)

text = "The structure consists of edge-sharing SiO4 tetrahedra."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs).last_hidden_state
print("✅ MatSciBERT forward success, shape:", outputs.shape)



### bert

model_name = "google-bert/bert-base-uncased"
save_dir = "/home/yangjw/crystalformer/llm/bert-base-uncased"

# save
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("✅ BERT-Base saved locally at:", save_dir)

# check
print("🔍 Testing local BERT-Base...")
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModel.from_pretrained(save_dir)

text = "The crystal exhibits a layered hexagonal structure."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs).last_hidden_state
print("✅ BERT-Base forward success, shape:", outputs.shape)
'''

############################
### test fusion_block.py ###
############################
'''
import torch
from fusion_block import FusionBlock

# Simulated input: batch_size = 4, model_dim = 128
B = 4
D = 128
struct_feat = torch.randn(B, D)
text_feat = torch.randn(B, D)

# List all fusion types to be tested
fusion_types = ['sum', 'concat', 'gated']

print("=== Fusion Block Output Shapes ===")
for fusion_type in fusion_types:
    try:
        fusion = FusionBlock(fusion_type=fusion_type, model_dim=D)
        output = fusion(struct_feat, text_feat)
        print(f"{fusion_type:<10}: {tuple(output.shape)}")
    except Exception as e:
        print(f"{fusion_type:<10}: ❌ ERROR: {e}")
'''

#########################
### dataset_fusion.py ###
#########################

# test_dataset_fusion.py

'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch_geometric.loader import DataLoader
from dataset_fusion import FusionDatasetMP_Latticeformer

# ====== Config ======
dataset = FusionDatasetMP_Latticeformer(
    target_split='train',
    target_field='fe',
    root_dir=None,
    use_text_mask=True,
    text_mask_prob=0.0,
    use_struct_mask=True,
    struct_mask_prob=0.0,
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ====== Test first 3 samples ======
print(f"🔍 Total samples in dataset: {len(dataset)}\n")

for i, batch in enumerate(loader):
    print(f"📦 Sample {i + 1}")

    print("🧬 Material ID:", batch.material_id[0])
    print("📖 Raw Text:", batch.text[0])
    print("📖 Masked Text:", batch.text_masked[0])

    print("🧱 Atom feature shape:", batch.atom_fea[0].shape)
    print("🧱 Atom feature (masked) shape:", batch.atom_fea_masked[0].shape)

    print("🎯 Target y:", batch.y[0].item())
    print("-" * 80)

    if i >= 2:
        break
'''


######## matscibert size?
'''
import torch
from transformers import AutoModel

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / 1024**2  # float32 每个参数4字节
    size_gb = size_mb / 1024
    if total_params >= 1e9:
        scale = f"{total_params/1e9:.2f}B"
    else:
        scale = f"{total_params/1e6:.2f}M"
    return total_params, scale, size_mb, size_gb

def test_model(model_name, device='cuda'):
    print(f"\nLoading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    total_params, scale, size_mb, size_gb = get_model_size(model)
    print(f"✅ Total Parameters: {scale}")
    print(f"💾 Estimated Memory (fp32): {size_mb:.2f} MB ≈ {size_gb:.2f} GB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU total memory: {gpu_mem:.2f} GB")
        if size_gb < gpu_mem * 0.8:
            print("✅ Safe to load and fine-tune on a single GPU")
        elif size_gb < gpu_mem:
            print("⚠️  Might need gradient checkpointing or lower batch size")
        else:
            print("❌ Not enough memory — consider LoRA or model parallelism")
    else:
        print("⚠️  No GPU detected — running on CPU only")

if __name__ == "__main__":
    test_model("m3rg-iitd/matscibert")
'''


#### check .pt dim?
'''
import torch
import os

# 修改为你的文件路径
pt_file = "/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018/text_embedding/matscibert_eform_train.pt"

# 加载缓存
embeddings = torch.load(pt_file)

# 检查维度
for key, emb in embeddings.items():
    print(f"{key}: shape = {tuple(emb.shape)}")

'''

### 检查token-level数据的dim
'''import torch
import pickle

pkl_path = "/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018/data_list/e_form_token_train.pkl"
with open(pkl_path, "rb") as f:
    data_list = pickle.load(f)

print(type(data_list[0].text_emb))  # Tensor
print(data_list[0].text_emb.shape)  # 应该是 [L, D]
'''

##

import torch
import os

# ==== Replace with your real path ====
embedding_path = "/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018/text_embedding/matscibert_token_e_form_train.pt"

# ==== Load the .pt file ====
assert os.path.exists(embedding_path), f"File not found: {embedding_path}"
emb_dict = torch.load(embedding_path)
print(f"✅ Loaded embedding file: {embedding_path}")
print(f"🔢 Total samples: {len(emb_dict)}")

# ==== Check shape of first 5 samples ====
count_bad = 0
for i, (mid, emb) in enumerate(emb_dict.items()):
    if not isinstance(emb, torch.Tensor):
        print(f"[❌] {mid} is not a Tensor!")
        count_bad += 1
        continue
    if emb.dim() == 2:
        print(f"[✔️] {mid}: shape = {emb.shape} ✅ [L, D] token-level")
    elif emb.dim() == 1:
        print(f"[⚠️] {mid}: shape = {emb.shape} ❌ [D] pooled-level")
        count_bad += 1
    else:
        print(f"[❌] {mid}: unexpected shape = {emb.shape}")
        count_bad += 1
    if i >= 4:
        break

print(f"\n🔍 Summary: {count_bad} bad samples in first 5")

# ==== Optional: check full set ====
# bad = [mid for mid, emb in emb_dict.items() if emb.dim() != 2]
# print(f"\n[‼️] Found {len(bad)} samples not in token-level format.")
