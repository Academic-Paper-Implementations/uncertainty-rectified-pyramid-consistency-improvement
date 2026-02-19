#!/bin/bash
# ================================================================
# PP4 Test Pipeline
# 1. Tải weights từ Azure
# 2. Chạy test_2D_fully.py cho từng experiment
# 3. Tạo bảng thống kê CSV
# 4. Upload bảng lên Azure
# Author: KhangPX
# ================================================================
# Chạy: bash code/run_pp4_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============ ĐỌC .env ============
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
    echo "Loaded .env"
else
    echo "ERROR: .env not found"
    exit 1
fi
# ==================================

ACCT=$(echo "$AZURE_CONNECTION_STRING" | grep -oP 'AccountName=\K[^;]+')
CONTAINER="${AZURE_CONTAINER:-ssl4mis-weights}"
SAS="${AZURE_SAS_TOKEN}"
MODEL_DIR="$SCRIPT_DIR/../model/ACDC"
RESULT_CSV="$SCRIPT_DIR/../pp4_test_results.csv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -z "$ACCT" ] || [ -z "$SAS" ]; then
    echo "ERROR: Thiếu AZURE_CONNECTION_STRING hoặc AZURE_SAS_TOKEN"
    exit 1
fi

BASE_URL="https://${ACCT}.blob.core.windows.net/${CONTAINER}"

# ================================================================
# DANH SÁCH 5 EXPERIMENTS TRÊN AZURE
# ================================================================
EXPERIMENTS=(
    "PP4_w0.5_s5.0"
    "PP4_w1.0_s3.0"
    "PP4_w1.5_s3.0"
    "PP4_w1.5_s5.0"
    "PP4_w2.0_s5.0"
)

# ============ HÀM: DOWNLOAD WEIGHTS TỪ AZURE ============
download_weights() {
    local EXP_NAME=$1
    local LOCAL_DIR="$MODEL_DIR/${EXP_NAME}_7_labeled/unet_urpc"
    mkdir -p "$LOCAL_DIR"

    echo "Downloading $EXP_NAME từ Azure..."

    # List blobs trong folder experiment
    LIST_URL="${BASE_URL}?restype=container&comp=list&prefix=${EXP_NAME}/&${SAS}"
    BLOB_LIST=$(curl -s "$LIST_URL")

    # Extract tên các file .pth
    PTH_FILES=$(echo "$BLOB_LIST" | grep -oP "<Name>\K[^<]+" | grep "\.pth$")

    if [ -z "$PTH_FILES" ]; then
        echo "  WARNING: Không tìm thấy .pth file cho $EXP_NAME trên Azure"
        return 1
    fi

    local COUNT=0
    while IFS= read -r blob_path; do
        local FNAME=$(basename "$blob_path")
        local LOCAL_PATH="$LOCAL_DIR/$FNAME"

        HTTP_CODE=$(curl -s -o "$LOCAL_PATH" -w "%{http_code}" \
            "${BASE_URL}/${blob_path}?${SAS}")

        if [ "$HTTP_CODE" = "200" ]; then
            echo "  ✓ $FNAME ($(du -h "$LOCAL_PATH" | cut -f1))"
            COUNT=$((COUNT+1))
        else
            echo "  ✗ $FNAME [HTTP $HTTP_CODE]"
        fi
    done <<< "$PTH_FILES"

    echo "  Downloaded $COUNT file(s) → $LOCAL_DIR"
    return 0
}

# ============ HÀM: CHẠY TEST ============
run_test() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_DIR/${EXP_NAME}_7_labeled/unet_urpc"

    # Kiểm tra best model tồn tại
    local BEST_MODEL="$WEIGHT_DIR/unet_urpc_best_model.pth"
    if [ ! -f "$BEST_MODEL" ]; then
        echo "  ERROR: Không tìm thấy $BEST_MODEL"
        return 1
    fi

    echo "  Testing với best model: $(basename $BEST_MODEL)"

    # Chạy test_2D_fully.py, capture output
    TEST_OUTPUT=$(python test_2D_fully.py \
        --root_path ../data/ACDC/ACDC \
        --exp "ACDC/$EXP_NAME" \
        --model unet_urpc \
        --num_classes 4 \
        --labeled_num 7 2>&1)

    echo "$TEST_OUTPUT"
    echo "$TEST_OUTPUT"  # return for parsing
}

# ============ KHỞI TẠO CSV ============
echo "experiment,RV_dice,RV_hd95,RV_asd,Myo_dice,Myo_hd95,Myo_asd,LV_dice,LV_hd95,LV_asd,mean_dice,mean_hd95,mean_asd" > "$RESULT_CSV"

# ================================================================
# MAIN LOOP
# ================================================================
echo ""
echo "================================================================"
echo "PP4 Test Pipeline"
echo "================================================================"

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo "Processing: $EXP"
    echo "----------------------------------------------------------------"

    # Step 1: Download
    download_weights "$EXP"
    if [ $? -ne 0 ]; then
        echo "  SKIP testing (download failed)"
        echo "$EXP,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" >> "$RESULT_CSV"
        continue
    fi

    # Step 2: Test
    echo "  Running test..."
    TEST_OUT=$(python test_2D_fully.py \
        --root_path ../data/ACDC/ACDC \
        --exp "ACDC/$EXP" \
        --model unet_urpc \
        --num_classes 4 \
        --labeled_num 7 2>&1)

    echo "$TEST_OUT"

    # Step 3: Parse kết quả từ output
    # Output format: [array([dice, hd95, asd]), array([...]), array([...])]
    METRICS=$(echo "$TEST_OUT" | grep -E "^\[array" | tail -1)

    if [ -n "$METRICS" ]; then
        # Parse 9 số từ metrics array: [array([d,h,a]), array([d,h,a]), array([d,h,a])]
        NUMS=$(echo "$METRICS" | grep -oP '[0-9]+\.[0-9]+' | head -9)
        NUM_ARR=($NUMS)

        if [ ${#NUM_ARR[@]} -ge 9 ]; then
            RV_D=${NUM_ARR[0]}; RV_H=${NUM_ARR[1]}; RV_A=${NUM_ARR[2]}
            MY_D=${NUM_ARR[3]}; MY_H=${NUM_ARR[4]}; MY_A=${NUM_ARR[5]}
            LV_D=${NUM_ARR[6]}; LV_H=${NUM_ARR[7]}; LV_A=${NUM_ARR[8]}

            MEAN_D=$(echo "scale=6; ($RV_D + $MY_D + $LV_D) / 3" | bc)
            MEAN_H=$(echo "scale=6; ($RV_H + $MY_H + $LV_H) / 3" | bc)
            MEAN_A=$(echo "scale=6; ($RV_A + $MY_A + $LV_A) / 3" | bc)

            echo "$EXP,$RV_D,$RV_H,$RV_A,$MY_D,$MY_H,$MY_A,$LV_D,$LV_H,$LV_A,$MEAN_D,$MEAN_H,$MEAN_A" >> "$RESULT_CSV"
            echo "  → mean_dice=$MEAN_D  mean_hd95=$MEAN_H  mean_asd=$MEAN_A"
        else
            echo "  WARNING: Không parse được metrics đủ (got ${#NUM_ARR[@]} numbers)"
            echo "$EXP,PARSE_ERROR,,,,,,,,,," >> "$RESULT_CSV"
        fi
    else
        echo "  WARNING: Không tìm thấy metrics output"
        echo "$EXP,NO_OUTPUT,,,,,,,,,," >> "$RESULT_CSV"
    fi
done

# ================================================================
# IN BẢNG KẾT QUẢ
# ================================================================
echo ""
echo "================================================================"
echo "KẾT QUẢ CUỐI CÙNG"
echo "================================================================"
echo ""
# In bảng đẹp từ CSV dùng python (thay vì column không có sẵn)
python3 -c "
import csv, sys
rows = list(csv.reader(open('$RESULT_CSV')))
if not rows: sys.exit()
widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
for row in rows:
    print('  ' + '  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
"

# ================================================================
# UPLOAD KẾT QUẢ LÊN AZURE
# ================================================================
echo ""
echo "Uploading results to Azure..."
RESULT_BLOB="results/pp4_test_results_${TIMESTAMP}.csv"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
    -H "x-ms-blob-type: BlockBlob" \
    -H "Content-Type: text/csv" \
    --data-binary "@$RESULT_CSV" \
    "${BASE_URL}/${RESULT_BLOB}?${SAS}")

if [ "$HTTP_CODE" = "201" ]; then
    echo "✓ Uploaded: Azure → $CONTAINER/$RESULT_BLOB"
    echo "  URL: https://${ACCT}.blob.core.windows.net/${CONTAINER}/${RESULT_BLOB}"
else
    echo "✗ Upload failed [HTTP $HTTP_CODE]"
    echo "  Kết quả vẫn được lưu tại: $RESULT_CSV"
fi

echo ""
echo "Done! CSV lưu tại: $RESULT_CSV"
