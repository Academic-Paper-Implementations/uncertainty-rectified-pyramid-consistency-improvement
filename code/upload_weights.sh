#!/bin/bash
# ================================================================
# Upload weights lên Azure Blob Storage hoặc VPS qua SCP
# Dùng độc lập sau khi training xong
# Author: KhangPX
# ================================================================
# 
# Cách dùng:
#   bash upload_weights.sh scp <exp_name>        # Upload 1 experiment qua SCP
#   bash upload_weights.sh azure <exp_name>      # Upload 1 experiment lên Azure
#   bash upload_weights.sh scp all               # Upload tất cả experiments
#   bash upload_weights.sh azure all             # Upload tất cả lên Azure
#
# Ví dụ:
#   bash upload_weights.sh scp PP4_w1.5_s3.0_both
#   bash upload_weights.sh azure all
# ================================================================

METHOD=${1:-"scp"}
EXP_TARGET=${2:-"all"}

MODEL_BASE_DIR="../model/ACDC"

# ============ ĐỌC CREDENTIALS TỪ .env ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        value="${value%%#*}" # Strip comments from value
        key="${key// /}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
    done < "$ENV_FILE"
    echo "Loaded credentials from $ENV_FILE"
else
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Run: cp .env.example .env  then fill in your credentials"
    exit 1
fi
# ==================================================

upload_via_scp() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_BASE_DIR/${EXP_NAME}_7_labeled/unet_urpc"

    if [ ! -d "$WEIGHT_DIR" ]; then
        echo "ERROR: Directory not found: $WEIGHT_DIR"
        return 1
    fi

    echo "Uploading $EXP_NAME via SCP to ${SCP_HOST}..."

    # Build SSH options
    SSH_OPTS="-p $SCP_PORT"
    if [ -n "$SCP_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SCP_KEY"
    fi

    # Tạo thư mục đích trên server
    ssh $SSH_OPTS ${SCP_USER}@${SCP_HOST} \
        "mkdir -p ${SCP_DEST_PATH}/${EXP_NAME}"

    # Upload tất cả .pth files
    scp $SSH_OPTS "$WEIGHT_DIR"/*.pth \
        ${SCP_USER}@${SCP_HOST}:${SCP_DEST_PATH}/${EXP_NAME}/

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Uploaded $EXP_NAME to ${SCP_HOST}:${SCP_DEST_PATH}/${EXP_NAME}/"
    else
        echo "ERROR: Upload failed for $EXP_NAME"
    fi
}

upload_via_azure() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_BASE_DIR/${EXP_NAME}_7_labeled/unet_urpc"

    if [ ! -d "$WEIGHT_DIR" ]; then
        echo "ERROR: Directory not found: $WEIGHT_DIR"
        return 1
    fi

    echo "Uploading $EXP_NAME to Azure Blob Storage..."

    # Build auth options
    AUTH_OPTS=""
    if [ -n "$AZURE_SAS_TOKEN" ]; then
        AUTH_OPTS="--sas-token \"$AZURE_SAS_TOKEN\""
    elif [ "$AZURE_USE_LOGIN" = true ]; then
        AUTH_OPTS="--auth-mode login"
    fi

    # Upload
    az storage blob upload-batch \
        --account-name "$AZURE_ACCOUNT" \
        --destination "$AZURE_CONTAINER" \
        --source "$WEIGHT_DIR" \
        --destination-path "$EXP_NAME" \
        --pattern "*.pth" \
        $AUTH_OPTS

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Uploaded $EXP_NAME to Azure: $AZURE_CONTAINER/$EXP_NAME/"
    else
        echo "ERROR: Azure upload failed for $EXP_NAME"
    fi
}

do_upload() {
    local EXP_NAME=$1
    if [ "$METHOD" = "scp" ]; then
        upload_via_scp "$EXP_NAME"
    elif [ "$METHOD" = "azure" ]; then
        upload_via_azure "$EXP_NAME"
    else
        echo "ERROR: Unknown method '$METHOD'. Use 'scp' or 'azure'."
        exit 1
    fi
}

# ============ MAIN ============
if [ "$EXP_TARGET" = "all" ]; then
    echo "Uploading ALL experiments via $METHOD..."
    for exp_dir in $MODEL_BASE_DIR/*_7_labeled; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir" "_7_labeled")
            do_upload "$exp_name"
        fi
    done
else
    do_upload "$EXP_TARGET"
fi

echo "Done!"
