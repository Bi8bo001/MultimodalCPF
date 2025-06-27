#!/usr/bin/bash

# path
save_path="/home/yangjw/crystalformer/multimodal_fusion/result/latticeformer/concat"
# save_path="/home/yangjw/crystalformer/multimodal_fusion/result/latticeformer"
data_root="/home/yangjw/crystalformer/multimodal_fusion/data/crysmmnet_dataset/mp_2018"
param_json="/home/yangjw/crystalformer/multimodal_fusion/default_fusion.json"

gpu=0
targets="e_form"
exp_name="baseline_mp2018"
layer=4
reproduciblity_state=42
text_encoder="matscibert"

# start training
CUDA_VISIBLE_DEVICES=${gpu} python3.10 train_fusion.py -p ${param_json} \
    --seed 123 \
    --save_path ${save_path} \
    --experiment_name ${exp_name}/${targets} \
    --targets ${targets} \
    --encoder_name latticeformer \
    --num_layers ${layer} \
    --domain real \
    --batch_size 64 \
    --fusion_type cross_attn \
    --text_encoder_name ${text_encoder} \
    --text_encoder_path m3rg-iitd/matscibert \
    --freeze_text_encoder true \
    --text_mask_prob 0.0 \
    --struct_mask_prob 0.0 \
    --normalize_targets scale_bias \
    --reproduciblity_state ${reproduciblity_state} \
    --log_every_n_steps 50 \
    --use_amp false