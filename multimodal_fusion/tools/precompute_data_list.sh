#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
root_dir="/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018"
target_field1="bandgap"
target_field2="e_form"
fusion_type1="concat"
fusion_type2="cross_attn"

rm -rf ${root_dir}/data_list/*.pkl

echo ">>> Starting data_list precomputation for ${target_field} under ${root_dir} ..."
python3.10 /home/yangjw/crystalformer/multimodal_fusion/tools/precompute_data_list.py \
    --root_dir ${root_dir} \
    --target_field ${target_field1} \
    --fusion_type ${fusion_type1}

python3.10 /home/yangjw/crystalformer/multimodal_fusion/tools/precompute_data_list.py \
    --root_dir ${root_dir} \
    --target_field ${target_field1} \
    --fusion_type ${fusion_type2}

python3.10 /home/yangjw/crystalformer/multimodal_fusion/tools/precompute_data_list.py \
    --root_dir ${root_dir} \
    --target_field ${target_field2} \
    --fusion_type ${fusion_type1}

python3.10 /home/yangjw/crystalformer/multimodal_fusion/tools/precompute_data_list.py \
    --root_dir ${root_dir} \
    --target_field ${target_field2} \
    --fusion_type ${fusion_type2}

echo ">>> Done. Cached data_list_{split}.pkl saved under ${root_dir}"
