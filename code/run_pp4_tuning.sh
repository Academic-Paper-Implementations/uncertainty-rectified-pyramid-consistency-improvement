#!/bin/bash
# ================================================================
# Batch Hyperparameter Tuning for PP4 (Boundary-Aware Loss)
# Tune: boundary_weight, sdm_sigma
# Upload weights lên Azure Blob Storage sau khi xong
# Author: KhangPX
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============ ĐỌC CREDENTIALS TỪ .env ============
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key// /}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
    done < "$ENV_FILE"
    echo "Loaded credentials from $ENV_FILE"
else
    echo "WARNING: .env not found. Upload sẽ bị tắt."
    UPLOAD_ENABLED=false
fi
# ==================================================

UPLOAD_ENABLED=${UPLOAD_ENABLED:-false}
MODEL_BASE_DIR="$SCRIPT_DIR/../model"

# ============ UPLOAD LÊN AZURE (curl + SAS token, không cần az CLI) ============
upload_weights() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_BASE_DIR/ACDC/${EXP_NAME}_7_labeled/unet_urpc"

    if [ "$UPLOAD_ENABLED" != "true" ]; then
        return
    fi

    if [ ! -d "$WEIGHT_DIR" ]; then
        echo "WARNING: Weight dir not found: $WEIGHT_DIR"
        return
    fi

    # Lấy AccountName từ connection string
    local ACCT=$(echo "$AZURE_CONNECTION_STRING" | grep -oP 'AccountName=\K[^;]+')
    local CONTAINER="${AZURE_CONTAINER:-ssl4mis-weights}"
    local SAS="${AZURE_SAS_TOKEN}"

    if [ -z "$ACCT" ] || [ -z "$SAS" ]; then
        echo "ERROR: Cần AZURE_CONNECTION_STRING và AZURE_SAS_TOKEN trong .env"
        echo "  Tạo SAS token: Azure Portal → Storage Account → Shared access signature"
        return
    fi

    local UPLOAD_OK=0
    local UPLOAD_FAIL=0

    echo "Uploading $EXP_NAME → Azure ($ACCT/$CONTAINER)..."
    for pth_file in "$WEIGHT_DIR"/*.pth; do
        [ -f "$pth_file" ] || continue
        local FNAME=$(basename "$pth_file")
        local URL="https://${ACCT}.blob.core.windows.net/${CONTAINER}/${EXP_NAME}/${FNAME}?${SAS}"

        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
            -H "x-ms-blob-type: BlockBlob" \
            -H "Content-Type: application/octet-stream" \
            --data-binary "@$pth_file" \
            "$URL")

        if [ "$HTTP_CODE" = "201" ]; then
            echo "  ✓ $FNAME"
            UPLOAD_OK=$((UPLOAD_OK+1))
        else
            echo "  ✗ $FNAME [HTTP $HTTP_CODE]"
            UPLOAD_FAIL=$((UPLOAD_FAIL+1))
        fi
    done

    echo "Upload done: $UPLOAD_OK OK, $UPLOAD_FAIL FAIL"
}
# =============================================================================

run_experiment() {
    local EXP_NAME=$1
    local BOUNDARY_WEIGHT=$2
    local SDM_SIGMA=$3
    local MAX_ITER=${4:-10000}

    echo ""
    echo "================================================================"
    echo "Running: $EXP_NAME"
    echo "  boundary_weight=$BOUNDARY_WEIGHT, sdm_sigma=$SDM_SIGMA, max_iter=$MAX_ITER"
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
        --sdm_sigma $SDM_SIGMA

    echo "Finished: $EXP_NAME"
    upload_weights "$EXP_NAME"
}

# ================================================================
# DANH SÁCH EXPERIMENTS
# Baseline đã biết: boundary_weight=1.0, sigma=5.0 → Dice=0.84
# ================================================================

run_experiment "PP4_w1.5_s5.0" 1.5 5.0 10000
run_experiment "PP4_w2.0_s5.0" 2.0 5.0 10000
run_experiment "PP4_w1.0_s3.0" 1.0 3.0 10000
run_experiment "PP4_w1.5_s3.0" 1.5 3.0 10000
run_experiment "PP4_w0.5_s5.0" 0.5 5.0 10000

# Best config → 30k iterations (uncomment sau khi biết best từ 10k)
# run_experiment "PP4_BEST_30k" 1.5 3.0 30000

echo ""
echo "================================================================"
echo "All experiments completed!"
echo "================================================================"
for exp_dir in "$MODEL_BASE_DIR"/ACDC/PP4_*_7_labeled; do
    exp_name=$(basename "$exp_dir")
    log_file="$exp_dir/unet_urpc/log.txt"
    if [ -f "$log_file" ]; then
        best_dice=$(grep "mean_dice" "$log_file" | awk '{print $NF}' | sort -n | tail -1)
        echo "  $exp_name: best_dice=$best_dice"
    fi
done
