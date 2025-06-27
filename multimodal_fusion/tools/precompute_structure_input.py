
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import json
import torch
from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from dataloaders.common import generate_site_species_vector

def precompute_structure_inputs(
    split_json_path,
    cif_folder,
    save_path,
    atom_num_upper=98,
):
    """
    Precompute structure input tensors and save as a dict {material_id: input_dict}.
    Each input_dict contains:
      - x: atom features
      - pos: atom positions
      - trans_vec: lattice matrix
      - sizes: number of atoms
    """
    with open(split_json_path, 'r') as f:
        samples = json.load(f)

    struct_input_dict = {}

    for sample in tqdm(samples, desc=f"Processing {os.path.basename(split_json_path)}"):
        material_id = sample.get("material_id") or sample.get("id") or sample.get("file_id")
        cif_file = os.path.join(cif_folder, f"{material_id}.cif")

        if not os.path.exists(cif_file):
            print(f"[Warning] Missing CIF file: {material_id}")
            continue

        try:
            structure = CifParser(cif_file).get_structures()[0]
            structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()

            atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
            atom_fea = generate_site_species_vector(structure, atom_num_upper)
            if atom_fea is None or atom_fea.shape[0] == 0:
                print(f"[Skip] Invalid atom_fea for {material_id}")
                continue

            lattice_mat = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0)
            n_atoms = torch.tensor([atom_fea.shape[0]], dtype=torch.long)

            struct_input_dict[material_id] = {
                "x": atom_fea,
                "pos": atom_pos,
                "trans_vec": lattice_mat,
                "sizes": n_atoms
            }

        except Exception as e:
            print(f"[Error] Failed to parse {material_id}: {e}")
            continue

    torch.save(struct_input_dict, save_path)
    print(f"[Done] Saved {len(struct_input_dict)} structures to {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Precompute structure inputs for multimodal crystal model")
    parser.add_argument('--split_json', type=str, required=True, help='Path to train/val/test.json')
    parser.add_argument('--cif_folder', type=str, required=True, help='Path to folder containing .cif files')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output .pt file')
    args = parser.parse_args()

    precompute_structure_inputs(
        split_json_path=args.split_json,
        cif_folder=args.cif_folder,
        save_path=args.save_path
    )
