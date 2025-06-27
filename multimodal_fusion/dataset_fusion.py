### FusionDatasetMP_Latticeformer

# dataset_fusion.py

import os
import csv
import sys
import json
import random
import torch
import pickle  # for preprocessing data saving

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dataloaders.common import generate_site_species_vector
from torch.nn.utils.rnn import pad_sequence  ## cross_attn

from multimodal_fusion.mask_strategy import apply_modal_dropout  ### mask training strategy

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ROOT = os.path.join(BASE_DIR, "multimodal_fusion", "data", "crysmmnet_dataset", "mp_2018")

class FusionDatasetMP_Latticeformer(Dataset):
    """
    Multimodal dataset class for Latticeformer + text fusion.
    Provides structure tokens, masked/unmasked text, and regression targets.
    """

    def __init__(self,
                 target_split='train',
                 structure_encoder='latticeformer',
                 use_text_mask=False,
                 use_struct_mask=False,
                 text_mask_prob=0.0,
                 struct_mask_prob=0.0,
                 root_dir=None,
                 cif_folder='cif',
                 description_file='description.csv',
                 target_field='e_form',
                 freeze_text_encoder=True,
                 fusion_type="concat",  ## cross-attn的话需要对应修改数据传入的dim
                 modal_dropout_prob=0.0,  ##
                 ):
        super().__init__()

        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(__file__), "data", "crysmmnet_dataset", "mp_2018")

        self.root_dir = root_dir
        self.cif_dir = os.path.join(self.root_dir, cif_folder)
        self.description_path = os.path.join(self.root_dir, description_file)
        self.split_file = os.path.join(self.root_dir, f"{target_split}.json")

        self.structure_encoder = structure_encoder
        self.use_text_mask = use_text_mask
        self.use_struct_mask = use_struct_mask
        self.text_mask_prob = text_mask_prob
        self.struct_mask_prob = struct_mask_prob
        self.target_field = target_field
        self.ATOM_NUM_UPPER = 98  # from CrystalFormer
        self.id2text = self._load_description()

        with open(self.split_file, 'r') as f:
            all_samples = json.load(f)

        # keep the one with text description
        self.samples = [s for s in all_samples if s['id'] in self.id2text]

        ##
        self.fusion_type = fusion_type.lower()
        self.use_token_emb = (self.fusion_type == "cross_attn")
        print(f"[FusionDataset] fusion_type = {self.fusion_type} | use_token_emb = {self.use_token_emb}")

        ## normalization部分重新设置在dataloader中
        self.target_mean, self.target_std = self._compute_target_stats()
        print(f"[FusionDataset] Target mean: {self.target_mean:.4f} | std: {self.target_std:.4f}")

        print(f"[FusionDataset] Loaded {len(self.samples)} samples from {target_split}.json")
        print(f"[FusionDataset] Description count: {len(self.id2text)}")
        print(f"[FusionDataset] text_mask_prob = {self.text_mask_prob} | struct_mask_prob = {self.struct_mask_prob}")

        ### text embedding在param frozen时用提前处理的
        self.freeze_text_encoder = freeze_text_encoder
        if self.freeze_text_encoder == True:
            prefix = "matscibert_token" if self.use_token_emb else "matscibert"  ###
            emb_file = f"{prefix}_{self.target_field}_{target_split}.pt"
            emb_path = os.path.join(self.root_dir, "text_embedding", emb_file)

            if os.path.exists(emb_path):
                self.text_embedding_cache = torch.load(emb_path)
                print(f"[FusionDataset] Loaded text embedding cache from {emb_path}")
            else:
                self.text_embedding_cache = None
        else:
            self.text_embedding_cache = None

        ### structure preprocessing cache
        # Structure input cache
        self.struct_input_cache = None
        struct_input_path = os.path.join(self.root_dir, "struct_input", f"struct_input_{target_split}.pt")
        if os.path.exists(struct_input_path):
            self.struct_input_cache = torch.load(struct_input_path)
            print(f"[FusionDataset] Loaded structure input cache from {struct_input_path}")

        ### 预处理的结构数据根据id打包成list, (pkl)
        ### pkl
        ### 先根据fusion_type设置好储存的dir等
        pkl_dir = os.path.join(self.root_dir, "data_list")
        os.makedirs(pkl_dir, exist_ok=True)
        if self.use_token_emb:
            pkl_name = f"{self.target_field}_token_{target_split}"
        else:
            pkl_name = f"{self.target_field}_{target_split}"
        pkl_path = os.path.join(pkl_dir, pkl_name + ".pkl")


        self.pkl_cache_path = pkl_path

        if os.path.exists(self.pkl_cache_path):
            print(f"[FusionDataset] Loading cached data_list from {self.pkl_cache_path}")
            with open(self.pkl_cache_path, 'rb') as f:
                self.data_list = pickle.load(f)
        else:
            print("[FusionDataset] No cached data_list found, building...")
            self.data_list = self._build_data_list()
            # save
            with open(self.pkl_cache_path, 'wb') as f:
                pickle.dump(self.data_list, f)
        
        # self.data_list = self._build_data_list()

        ### mask strategy
        self.modal_dropout_prob = modal_dropout_prob
        
    def _load_description(self):
        # description 太长了超过字数了
        csv.field_size_limit(sys.maxsize)
        maxlen = 0
        id2text = {}
        print(f"\n{self.description_path}\n") ## debug

        with open(self.description_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get('Id') or row.get('id')
                desc = row.get('Description') or row.get('text')
                if mid and desc:
                    id2text[mid] = desc
                if len(desc) > maxlen:
                    maxlen = len(desc)
            print(f"[FusionDataset] Max text length: {maxlen} chars")
        return id2text

    def __len__(self):
        return len(self.samples)

    def _mask_text(self, text: str) -> str:
        if not self.use_text_mask or self.text_mask_prob <= 0:
            return text
        tokens = text.split()
        masked = [t if random.random() > self.text_mask_prob else '[MASK]' for t in tokens]
        return ' '.join(masked)

    def _mask_struct(self, atom_fea: torch.Tensor) -> torch.Tensor:
        if not self.use_struct_mask or self.struct_mask_prob <= 0:
            return atom_fea
        mask = torch.rand(atom_fea.size(0)) > self.struct_mask_prob
        return atom_fea * mask.unsqueeze(1).float()
    
    ## normalization部分重新设置在dataloader中
    def _compute_target_stats(self):
        targets = []
        for item in self.samples:
            val = item.get(self.target_field)
            # Fallback: support field alias
            if val is None:
                if self.target_field == "e_form":
                    val = item.get("fe")
                elif self.target_field == "bandgap":
                    val = item.get("bgap")
            if val is None:
                continue
            if isinstance(val, str):
                val = float(val)
            targets.append(val)
        if not targets:
            print(f"[Warning] No valid targets found for field '{self.target_field}' in dataset! Returning NaN.")
            return float("nan"), float("nan")
        targets = torch.tensor(targets, dtype=torch.float32)
        return targets.mean().item(), targets.std().item()

    ### 预处理结构数据 打包
    def _build_data_list(self):
        data_list = []
        for item in self.samples:
            material_id = item.get("material_id") or item.get("file_id") or item.get("id")

            # structure & mask
            if self.struct_input_cache and material_id in self.struct_input_cache:
                struct = self.struct_input_cache[material_id]
                x = self._mask_struct(struct["x"])
                pos = struct["pos"]
                trans_vec = struct["trans_vec"]
                sizes = struct["sizes"]
            else:
                raise ValueError(f"Structure input not found for material_id {material_id}")

            # text & mask
            text = self.id2text[material_id]
            text_masked = self._mask_text(text)

            # target value
            target_val = item.get(self.target_field) or item.get("fe") or item.get("bgap")
            target_val = float(target_val)
            y = torch.tensor([target_val], dtype=torch.float32)

            # text embedding
            emb = None
            if self.freeze_text_encoder and self.text_embedding_cache and material_id in self.text_embedding_cache:
                emb = self.text_embedding_cache[material_id]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb, dtype=torch.float32)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                elif emb.dim() != 2:
                    raise ValueError(f"Expected text_emb of shape [L, D] or [D], got {emb.shape}")
                emb = emb.float()

            # create Data object
            data = Data()
            data.x = x
            data.pos = pos
            data.trans_vec = trans_vec
            data.sizes = sizes
            data.material_id = material_id
            data.text = text
            data.text_masked = text_masked
            data.y = y
            setattr(data, self.target_field, y)
            if emb is not None:
                data.text_emb = emb.unsqueeze(0)  ### collate_fn_fusion_tokenlevel用不进去 所以直接加一个dim
                # data.text_emb = emb
                
            data_list.append(data)
        return data_list

    '''    
    ### new __getitem__ 只根据id从list中抽取data
    def __getitem__(self, idx):
        return self.data_list[idx]
    '''

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # Step 1: Decide whether to drop modalities (only for training)
        if getattr(self, "target_split", "train") == "train" and self.modal_dropout_prob > 0:
            use_struct_mask, use_text_mask = apply_modal_dropout(self.modal_dropout_prob, self.text_mask_prob)
        else:
            use_struct_mask, use_text_mask = False, False

        # Debug print
        # print(f"[MASK DEBUG] idx={idx} | struct_mask={use_struct_mask} | text_mask={use_text_mask}")

        # Step 2: Apply structure masking (drop struct)
        if use_struct_mask and hasattr(data, "x") and data.x is not None:
            data.x = torch.zeros_like(data.x)
            # print(f"[MASK DEBUG] Struct masked: {data.x.shape} -> all zeros: {data.x.abs().sum().item()==0}")


        # Step 3: Apply text masking (drop text)
        if use_text_mask:
            if hasattr(data, "text_emb"):
                data.text_emb = torch.zeros_like(data.text_emb)
                # print(f"[MASK DEBUG] Text embedding masked: {data.text_emb.shape} -> all zeros: {data.text_emb.abs().sum().item()==0}")

            else:
                data.text_masked = "[MASK]"
                # print("[MASK DEBUG] Text masked at raw input level")

        return data

'''
    def __getitem__(self, idx):
        item = self.samples[idx]
        material_id = item.get("material_id") or item.get("file_id") or item.get("id")
        
        ### preprocess了结构数据所以不用一个一个读取cif文件了
        #################

        # cif_file = os.path.join(self.cif_dir, f"{material_id}.cif")
        # #print(f"Loading CIF file for material {material_id}: {cif_file}")  ## debug
        # if not os.path.exists(cif_file):
        #     raise FileNotFoundError(f"Missing .cif for {material_id}")
        
        # # load structure
        # structure = CifParser(cif_file).get_structures()[0]
        # # print("======================")
        # # print(f"structure1: {structure}")
        # structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        # # print("======================")
        # # print(f"structure2: {structure}")
        # # input()
        # atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        # atom_fea = generate_site_species_vector(structure, self.ATOM_NUM_UPPER)

        # ## debug
        
        # if atom_fea is None or atom_fea.shape[0] == 0:
        #     raise ValueError(f"[Invalid atom_fea] for {material_id}")
        # elif atom_fea.shape[0] == 0:
        #     print(f"[Error] atom_fea has 0 elements for material {material_id}")
        # # else:
        # #     print(f"[Success] atom_fea shape for material {material_id}: {atom_fea.shape}")
        
        #################

        # prepare data
        data = Data()
        # data.atom_fea = atom_fea
        # data.atom_fea_masked = self._mask_struct(atom_fea)
        # data.pos = atom_pos
        # data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0)
        # data.sizes = torch.tensor([len(structure)], dtype=torch.long)
        data.material_id = material_id

        ### preprocessing
        if self.struct_input_cache and material_id in self.struct_input_cache:
            struct = self.struct_input_cache[material_id]
            data.x = self._mask_struct(struct["x"])
            data.pos = struct["pos"]
            data.trans_vec = struct["trans_vec"]
            data.sizes = struct["sizes"]
        else:
            raise ValueError(f"Structure input not found for material_id {material_id}. Check struct_input cache.")

        # data.x = atom_fea  ## de
        # setattr(data, 'x', atom_fea)  ## de

        #print(f"data for material {material_id}: {data}")  ## debug

        # text + mask
        data.text = self.id2text[material_id]
        data.text_masked = self._mask_text(data.text)

        # regression target
        target_val = item.get(self.target_field)
        if target_val is None:
            # fallback 机制：兼容默认字段名
            if self.target_field == "e_form":
                target_val = item.get("fe")
            elif self.target_field == "bandgap":
                target_val = item.get("bgap")
            else:
                raise ValueError(f"Missing target field '{self.target_field}' for material_id {material_id}")

        if isinstance(target_val, str):
            target_val = float(target_val)

        target_tensor = torch.tensor([target_val], dtype=torch.float32)
        data.y = torch.tensor([target_val], dtype=torch.float32)
        setattr(data, self.target_field, torch.tensor([target_val], dtype=torch.float))
        data[self.target_field] = target_tensor
        #print(f"data.keys: {data.keys}")  ## debug

        ### text embedding在frozen时用提前处理好的
        if self.text_embedding_cache and material_id in self.text_embedding_cache:
            emb = self.text_embedding_cache[material_id]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)  # add batch dim
            setattr(data, 'text_emb', emb)

            
        return data
    '''


'''
def collate_fn_fusion_tokenlevel(batch):
    # 直接 batch 是 List[Data]，从中提取 text_emb
    print("*******************************************")
    input()
    input()
    batch_graph = Batch.from_data_list(batch, follow_batch=['x'])  # batch 是 List[Data]

    text_emb_list = []
    for data in batch:
        emb = getattr(data, "text_emb", None)
        if emb is None:
            raise ValueError(f"[Collate Error] text_emb missing in sample {data.material_id}")
        text_emb_list.append(emb)  # 每个 emb 是 [L_i, D]

    # Pad to [B, L_max, D]
    pad_emb = pad_sequence(text_emb_list, batch_first=True)  # ✅ [B, L, D]
    print(f"[Collate Debug] text_emb_list lens: {[e.shape for e in text_emb_list]}")
    print(f"[Collate Debug] Padded text_emb shape = {pad_emb.shape}")

    # 生成对应的 mask
    lengths = [e.size(0) for e in text_emb_list]
    max_len = max(lengths)
    text_mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        text_mask[i, :l] = 1

    # 设置属性到 batch 对象中
    setattr(batch_graph, 'text_emb', pad_emb)
    setattr(batch_graph, 'text_mask', text_mask)

    return batch_graph
'''
# def collate_fn_fusion_tokenlevel(batch):
    
#     data_objs = [item["data"] for item in batch]
#     text_emb_list = [item["text_emb"] for item in batch]

#     # batch_graph = Batch.from_data_list(batch, follow_batch=['x'])  # PyG方式
#     batch_graph = Batch.from_data_list(data_objs, follow_batch=['x'])

#     # text_emb_list = []
#     # D = None
#     D = text_emb_list[0].size(-1)
#     pad_emb = pad_sequence(text_emb_list, batch_first=True)  # [B, L, D]

#     # for data in batch:
#     #     emb = getattr(data, "text_emb", None)
#     #     if emb is None:
#     #         if D is None:
#     #             raise ValueError("Missing text_emb and D is unknown")
#     #         emb = torch.zeros(1, D)
#     #     else:
#     #         D = emb.size(-1)
#     #     text_emb_list.append(emb)
#     # pad_emb = pad_sequence(text_emb_list, batch_first=True)  # [B, L_max, D]
#     print(f"[Collate] Padded text_emb shape: {pad_emb.shape}")
    
#     lengths = [e.size(0) for e in text_emb_list]
#     max_len = max(lengths)
#     text_mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
#     for i, l in enumerate(lengths):
#         text_mask[i, :l] = 1

#     # batch_graph.text_emb = pad_emb
#     # batch_graph.text_mask = text_mask  # future use in attention masking
#     setattr(batch_graph, 'text_emb', pad_emb)
#     setattr(batch_graph, 'text_mask', text_mask)

#     return batch_graph
