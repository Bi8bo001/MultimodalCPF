
#!/bin/bash

cd /home/yangjw/crystalformer
export PYTHONPATH=$PYTHONPATH:/home/yangjw/crystalformer

BASE_PATH=/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018

for SPLIT in train val test
do
  CUDA_VISIBLE_DEVICES=1 python multimodal_fusion/tools/precompute_structure_input.py \
    --split_json ${BASE_PATH}/${SPLIT}.json \
    --cif_folder ${BASE_PATH}/cif \
    --save_path ${BASE_PATH}/struct_input/struct_input_${SPLIT}.pt
done
