#!/bin/bash
# Script để chạy URPC với Boundary-Aware Loss + Multi-Scale Attention Fusion (PP4 + PP1)
# Author: KhangPX

cd /teamspace/studios/this_studio/code

echo "================================================================"
echo "URPC with Boundary-Aware Loss + Multi-Scale Attention Fusion"
echo "PP4 (SDM Integration) + PP1 (Attention Fusion)"
echo "Author: KhangPX"
echo "================================================================"

# Chạy với cấu hình mặc định
python khangpx_improvement.py \
    --root_path ../data/ACDC \
    --exp ACDC/URPC_BoundaryAware_AttentionFusion \
    --model unet_urpc \
    --num_classes 4 \
    --labeled_num 7 \
    --batch_size 24 \
    --labeled_bs 12 \
    --max_iterations 30000 \
    --base_lr 0.01 \
    --boundary_weight 1.0 \
    --sdm_sigma 5.0

echo "Training completed!"
