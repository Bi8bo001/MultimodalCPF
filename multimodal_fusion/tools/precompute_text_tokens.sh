#!/bin/bash

# TARGET=e_form  # or bandgap
TARGET=bandgap
BASE_PATH=/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018

for SPLIT in train val test
do
  CUDA_VISIBLE_DEVICES=1 python /home/yangjw/crystalformer/multimodal_fusion/tools/precompute_text_tokens.py \
    --input_json ${BASE_PATH}/${SPLIT}.json \
    --output_path ${BASE_PATH}/text_embedding/matscibert_token_${TARGET}_${SPLIT}.pt \
    --model_name m3rg-iitd/matscibert \
    --output_dim 128 \
    --device cuda
done
