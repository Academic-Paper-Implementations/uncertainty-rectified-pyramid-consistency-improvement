#!/bin/bash
# ================================================================
# Batch Hyperparameter Tuning for PP4 (Boundary-Aware Loss)
# Chạy nhiều configs và upload weights lên Azure/VPS sau khi xong
# Author: KhangPX
# ================================================================

cd /teamspace/studios/this_studio/code

# ============ ĐỌC CREDENTIALS TỪ .env ============
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)
    echo "Loaded credentials from .env"
else
    echo "WARNING: .env not found. Upload sẽ bị tắt."
    UPLOAD_ENABLED=false
fi
# ==================================================

UPLOAD_ENABLED=${UPLOAD_ENABLED:-false}
UPLOAD_METHOD=${UPLOAD_METHOD:-"scp"}
MODEL_BASE_DIR="../model"

run_experiment() {
    local EXP_NAME=$1
    local BOUNDARY_WEIGHT=$2
    local SDM_SIGMA=$3
    local BOUNDARY_MODE=$4
    local MAX_ITER=${5:-10000}

    echo ""
    echo "================================================================"
    echo "Running: $EXP_NAME"
    echo "  boundary_weight=$BOUNDARY_WEIGHT, sdm_sigma=$SDM_SIGMA"
    echo "  boundary_mode=$BOUNDARY_MODE, max_iter=$MAX_ITER"
    echo "================================================================"

    python khangpx_improvement.py \
        --root_path ../data/ACDC/ACDC \
        --exp "ACDC/$EXP_NAME" \
        --model unet_urpc \
        --num_classes 4 \
        --labeled_num 7 \
        --batch_size 24 \
        --labeled_bs 12 \
        --max_iterations $MAX_ITER \
        --base_lr 0.01 \
        --boundary_weight $BOUNDARY_WEIGHT \
        --sdm_sigma $SDM_SIGMA \
        --boundary_mode $BOUNDARY_MODE

    echo "Finished: $EXP_NAME"

    # Upload weights nếu được bật
    if [ "$UPLOAD_ENABLED" = true ]; then
        upload_weights "$EXP_NAME"
    fi
}

upload_weights() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_BASE_DIR/ACDC/${EXP_NAME}_7_labeled/unet_urpc"

    echo "Uploading weights for $EXP_NAME..."

    if [ "$UPLOAD_METHOD" = "scp" ]; then
        # Tạo thư mục trên server
        ssh ${SCP_USER}@${SCP_HOST} "mkdir -p ${SCP_DEST_PATH}/${EXP_NAME}"
        # Upload toàn bộ weights
        scp -r "$WEIGHT_DIR"/*.pth \
            ${SCP_USER}@${SCP_HOST}:${SCP_DEST_PATH}/${EXP_NAME}/
        echo "Uploaded via SCP to ${SCP_HOST}:${SCP_DEST_PATH}/${EXP_NAME}/"

    elif [ "$UPLOAD_METHOD" = "azure" ]; then
        # Upload lên Azure Blob Storage
        az storage blob upload-batch \
            --account-name "$AZURE_ACCOUNT" \
            --destination "$AZURE_CONTAINER" \
            --source "$WEIGHT_DIR" \
            --destination-path "$EXP_NAME" \
            --sas-token "$AZURE_SAS_TOKEN" \
            --pattern "*.pth"
        echo "Uploaded to Azure: $AZURE_CONTAINER/$EXP_NAME/"
    fi
}

# ================================================================
# DANH SÁCH EXPERIMENTS
# Baseline đã biết: boundary_weight=1.0, sigma=5.0 → Dice=0.84
# ================================================================

# Config 1: Tăng boundary_weight (focus boundary mạnh hơn)
run_experiment "PP4_w1.5_s5.0_both" 1.5 5.0 "both" 10000

# Config 2: Tăng boundary_weight mạnh hơn nữa
run_experiment "PP4_w2.0_s5.0_both" 2.0 5.0 "both" 10000

# Config 3: Giảm sigma (boundary hẹp hơn, focus hơn)
run_experiment "PP4_w1.0_s3.0_both" 1.0 3.0 "both" 10000

# Config 4: Kết hợp tăng weight + giảm sigma
run_experiment "PP4_w1.5_s3.0_both" 1.5 3.0 "both" 10000

# Config 5: Chỉ dùng boundary CE (không dùng boundary Dice)
run_experiment "PP4_w1.0_s5.0_ce_only" 1.0 5.0 "ce_only" 10000

# Config 6: Best config từ trên → chạy 30k iterations
# (Uncomment sau khi biết config nào tốt nhất từ 10k)
# run_experiment "PP4_BEST_30k" 1.5 3.0 "both" 30000

echo ""
echo "================================================================"
echo "All experiments completed!"
echo "Results summary:"
echo "================================================================"

# In kết quả từ log files
for exp_dir in $MODEL_BASE_DIR/ACDC/PP4_*_7_labeled; do
    exp_name=$(basename $exp_dir)
    log_file="$exp_dir/unet_urpc/log.txt"
    if [ -f "$log_file" ]; then
        best_dice=$(grep "mean_dice" "$log_file" | awk '{print $NF}' | sort -n | tail -1)
        echo "  $exp_name: best_dice=$best_dice"
    fi
done
