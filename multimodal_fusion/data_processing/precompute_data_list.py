
import os
import sys
import torch
import pickle
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from multimodal_fusion.dataset_fusion import FusionDatasetMP_Latticeformer

def generate_cached_data_list(split, target_field, root_dir, fusion_type):
    dataset = FusionDatasetMP_Latticeformer(
        target_split=split,
        target_field=target_field,
        root_dir=root_dir,
        freeze_text_encoder=True,
        use_struct_mask=False,
        use_text_mask=False,
        fusion_type=fusion_type,
    )
    print(f"[Done] Cached dataset for split='{split}' with {len(dataset)} samples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--target_field", type=str, default="e_form")
    parser.add_argument("--fusion_type", type=str, default="concat")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        generate_cached_data_list(split, args.target_field, args.root_dir, args.fusion_type)
