#!/bin/bash
# ================================================================
# PP4 Multi-Seed Benchmark Script
# ================================================================
# Chạy URPC + Boundary-Aware Loss 4 lần với 4 random seeds khác nhau,
# mỗi lần 10000 iterations. Tự động chọn best checkpoint theo val Dice,
# upload lên Azure, chạy test, và thống kê mean ± std cuối cùng.
#
# Tính năng:
#   - Resume: Nếu run đã hoàn thành (có best model), tự động skip
#   - Upload Azure: Dùng curl + SAS token (không cần az CLI)
#   - Test: Chạy test_2D_fully.py và parse Dice/HD95/ASD
#   - Benchmark: Tính mean ± std trên 4 seeds → CSV + JSON
#
# Cách chạy:
#   bash code/run_pp4_multiseed_benchmark.sh
#
# Author: KhangPX
# ================================================================

set -euo pipefail

# ============ CẤU HÌNH ============

# 4 random seeds cho 4 lần chạy
SEEDS=(1337 2024 42 12345)

# Hyperparameters cố định (best config từ PP4 tuning)
BOUNDARY_WEIGHT=1.5
SDM_SIGMA=3.0
BOUNDARY_MODE="both"
MAX_ITER=10000
LABELED_NUM=7
BATCH_SIZE=24
LABELED_BS=12
BASE_LR=0.01
NUM_CLASSES=4
MODEL="unet_urpc"

# Đường dẫn
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ROOT_PATH="../data/ACDC/ACDC"
MODEL_BASE_DIR="$SCRIPT_DIR/../model/ACDC"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output files
RESULT_CSV="$SCRIPT_DIR/../pp4_multiseed_benchmark.csv"
RESULT_JSON="$SCRIPT_DIR/../pp4_multiseed_benchmark.json"
PER_RUN_CSV="$SCRIPT_DIR/../pp4_multiseed_per_run.csv"

# ============ ĐỌC .env (Azure credentials) ============
ENV_FILE="$SCRIPT_DIR/../.env"
UPLOAD_ENABLED=false
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        value="${value%%#*}"
        key="${key// /}"
        value="${value%"${value##*[![:space:]]}"}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
    done < "$ENV_FILE"
    echo "[ENV] Loaded credentials from $ENV_FILE"

    # Kiểm tra Azure config
    ACCT=$(echo "${AZURE_CONNECTION_STRING:-}" | grep -oP 'AccountName=\K[^;]+' || true)
    CONTAINER="${AZURE_CONTAINER:-ssl4mis-weights}"
    SAS="${AZURE_SAS_TOKEN:-}"
    if [ -n "$ACCT" ] && [ -n "$SAS" ]; then
        UPLOAD_ENABLED=true
        BASE_URL="https://${ACCT}.blob.core.windows.net/${CONTAINER}"
        echo "[AZURE] Upload enabled → $ACCT/$CONTAINER"
    else
        echo "[AZURE] Thiếu AZURE_CONNECTION_STRING hoặc AZURE_SAS_TOKEN → upload tắt"
    fi
else
    echo "[ENV] .env not found → upload tắt"
fi

# ================================================================
# HÀM TIỆN ÍCH
# ================================================================

# Hàm: Upload best checkpoint lên Azure
upload_best_to_azure() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_BASE_DIR/${EXP_NAME}_${LABELED_NUM}_labeled/${MODEL}"

    if [ "$UPLOAD_ENABLED" != "true" ]; then
        echo "  [UPLOAD] Skipped (upload disabled)"
        return 0
    fi

    if [ ! -d "$WEIGHT_DIR" ]; then
        echo "  [UPLOAD] ERROR: Directory not found: $WEIGHT_DIR"
        return 1
    fi

    echo "  [UPLOAD] Uploading $EXP_NAME → Azure..."
    local UPLOAD_OK=0
    local UPLOAD_FAIL=0

    for pth_file in "$WEIGHT_DIR"/*.pth; do
        [ -f "$pth_file" ] || continue
        local FNAME=$(basename "$pth_file")
        local URL="${BASE_URL}/${EXP_NAME}/${FNAME}?${SAS}"

        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
            -H "x-ms-blob-type: BlockBlob" \
            -H "Content-Type: application/octet-stream" \
            --data-binary "@$pth_file" \
            "$URL")

        if [ "$HTTP_CODE" = "201" ]; then
            echo "    ✓ $FNAME"
            UPLOAD_OK=$((UPLOAD_OK+1))
        else
            echo "    ✗ $FNAME [HTTP $HTTP_CODE]"
            UPLOAD_FAIL=$((UPLOAD_FAIL+1))
        fi
    done
    echo "  [UPLOAD] Done: $UPLOAD_OK OK, $UPLOAD_FAIL FAIL"
}

# Hàm: Chạy test và parse metrics
# Trả về: sets global vars RV_D, RV_H, RV_A, MY_D, MY_H, MY_A, LV_D, LV_H, LV_A
run_test_and_parse() {
    local EXP_NAME=$1
    local WEIGHT_DIR="$MODEL_BASE_DIR/${EXP_NAME}_${LABELED_NUM}_labeled/${MODEL}"
    local BEST_MODEL="$WEIGHT_DIR/${MODEL}_best_model.pth"

    # Reset global metric vars
    RV_D=""; RV_H=""; RV_A=""
    MY_D=""; MY_H=""; MY_A=""
    LV_D=""; LV_H=""; LV_A=""
    MEAN_D=""; MEAN_H=""; MEAN_A=""
    TEST_SUCCESS=false

    if [ ! -f "$BEST_MODEL" ]; then
        echo "  [TEST] ERROR: Best model not found: $BEST_MODEL"
        return 1
    fi

    echo "  [TEST] Running test_2D_fully.py for $EXP_NAME..."

    # Chạy test, capture stdout+stderr
    local TEST_LOG="$WEIGHT_DIR/test_output.log"
    python test_2D_fully.py \
        --root_path "$ROOT_PATH" \
        --exp "ACDC/$EXP_NAME" \
        --model "$MODEL" \
        --num_classes "$NUM_CLASSES" \
        --labeled_num "$LABELED_NUM" 2>&1 | tee "$TEST_LOG"

    # Parse kết quả: format cuối cùng là dòng chứa 3 array
    # [array([dice, hd95, asd]), array([dice, hd95, asd]), array([dice, hd95, asd])]
    local METRICS=$(grep -E "^\[array" "$TEST_LOG" | tail -1)

    if [ -z "$METRICS" ]; then
        echo "  [TEST] WARNING: Không parse được metrics từ output"
        return 1
    fi

    # Extract 9 số (Dice/HD95/ASD × 3 classes: RV, Myo, LV)
    local NUMS=$(echo "$METRICS" | grep -oP '[0-9]+\.[0-9]+' | head -9)
    local NUM_ARR=($NUMS)

    if [ ${#NUM_ARR[@]} -lt 9 ]; then
        echo "  [TEST] WARNING: Chỉ parse được ${#NUM_ARR[@]}/9 metrics"
        return 1
    fi

    # Gán vào global vars
    RV_D=${NUM_ARR[0]}; RV_H=${NUM_ARR[1]}; RV_A=${NUM_ARR[2]}
    MY_D=${NUM_ARR[3]}; MY_H=${NUM_ARR[4]}; MY_A=${NUM_ARR[5]}
    LV_D=${NUM_ARR[6]}; LV_H=${NUM_ARR[7]}; LV_A=${NUM_ARR[8]}

    # Tính mean (dùng python để tránh lỗi bc)
    MEAN_D=$(python3 -c "print(round(($RV_D + $MY_D + $LV_D) / 3, 6))")
    MEAN_H=$(python3 -c "print(round(($RV_H + $MY_H + $LV_H) / 3, 6))")
    MEAN_A=$(python3 -c "print(round(($RV_A + $MY_A + $LV_A) / 3, 6))")

    TEST_SUCCESS=true
    echo "  [TEST] ✓ mean_dice=$MEAN_D  mean_hd95=$MEAN_H  mean_asd=$MEAN_A"
    return 0
}

# ================================================================
# HEADER
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PP4 Multi-Seed Benchmark                                  ║"
echo "║  URPC + Boundary-Aware Loss (SDM)                          ║"
echo "║  Author: KhangPX                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Seeds:            ${SEEDS[*]}"
echo "║  boundary_weight:  $BOUNDARY_WEIGHT"
echo "║  sdm_sigma:        $SDM_SIGMA"
echo "║  boundary_mode:    $BOUNDARY_MODE"
echo "║  max_iterations:   $MAX_ITER"
echo "║  labeled_num:      $LABELED_NUM"
echo "║  batch_size:       $BATCH_SIZE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============ KHỞI TẠO PER-RUN CSV ============
echo "seed,experiment,RV_dice,RV_hd95,RV_asd,Myo_dice,Myo_hd95,Myo_asd,LV_dice,LV_hd95,LV_asd,mean_dice,mean_hd95,mean_asd" > "$PER_RUN_CSV"

# ================================================================
# MAIN LOOP: Chạy 4 seeds
# ================================================================
RUN_INDEX=0
TOTAL_RUNS=${#SEEDS[@]}

for SEED in "${SEEDS[@]}"; do
    RUN_INDEX=$((RUN_INDEX+1))
    EXP_NAME="PP4_SEED${SEED}_w${BOUNDARY_WEIGHT}_s${SDM_SIGMA}"
    SNAPSHOT_PATH="$MODEL_BASE_DIR/${EXP_NAME}_${LABELED_NUM}_labeled/${MODEL}"
    BEST_MODEL_PATH="$SNAPSHOT_PATH/${MODEL}_best_model.pth"

    echo ""
    echo "================================================================"
    echo "  RUN $RUN_INDEX/$TOTAL_RUNS — Seed=$SEED"
    echo "  Experiment: $EXP_NAME"
    echo "================================================================"

    # ────────────────────────────────────────────────────────────
    # STEP 1: TRAIN (với resume support)
    # ────────────────────────────────────────────────────────────
    if [ -f "$BEST_MODEL_PATH" ]; then
        echo "  [TRAIN] ✓ SKIP — Best model đã tồn tại: $BEST_MODEL_PATH"
        echo "          (Xóa file này nếu muốn train lại seed=$SEED)"
    else
        echo "  [TRAIN] Starting training (seed=$SEED, max_iter=$MAX_ITER)..."
        TRAIN_START=$(date +%s)

        python khangpx_improvement.py \
            --root_path "$ROOT_PATH" \
            --exp "ACDC/$EXP_NAME" \
            --model "$MODEL" \
            --num_classes "$NUM_CLASSES" \
            --labeled_num "$LABELED_NUM" \
            --batch_size "$BATCH_SIZE" \
            --labeled_bs "$LABELED_BS" \
            --max_iterations "$MAX_ITER" \
            --base_lr "$BASE_LR" \
            --seed "$SEED" \
            --boundary_weight "$BOUNDARY_WEIGHT" \
            --sdm_sigma "$SDM_SIGMA" \
            --boundary_mode "$BOUNDARY_MODE"

        TRAIN_END=$(date +%s)
        TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
        echo "  [TRAIN] ✓ Completed in ${TRAIN_DURATION}s"

        # Kiểm tra best model được lưu
        if [ ! -f "$BEST_MODEL_PATH" ]; then
            echo "  [TRAIN] ✗ ERROR: Best model không được tạo! Kiểm tra log."
            echo "$SEED,$EXP_NAME,TRAIN_FAILED,,,,,,,,,,," >> "$PER_RUN_CSV"
            continue
        fi
    fi

    # ────────────────────────────────────────────────────────────
    # STEP 2: UPLOAD best checkpoint lên Azure
    # ────────────────────────────────────────────────────────────
    upload_best_to_azure "$EXP_NAME"

    # ────────────────────────────────────────────────────────────
    # STEP 3: TEST — chạy test_2D_fully.py và parse metrics
    # ────────────────────────────────────────────────────────────
    if run_test_and_parse "$EXP_NAME"; then
        echo "$SEED,$EXP_NAME,$RV_D,$RV_H,$RV_A,$MY_D,$MY_H,$MY_A,$LV_D,$LV_H,$LV_A,$MEAN_D,$MEAN_H,$MEAN_A" >> "$PER_RUN_CSV"
    else
        echo "$SEED,$EXP_NAME,TEST_FAILED,,,,,,,,,,," >> "$PER_RUN_CSV"
    fi

    echo ""
    echo "  ──── RUN $RUN_INDEX/$TOTAL_RUNS DONE ────"
done

# ================================================================
# TỔNG HỢP: Tính mean ± std từ 4 runs
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  COMPUTING BENCHMARK STATISTICS (mean ± std)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Export biến cho Python script
export PER_RUN_CSV RESULT_CSV RESULT_JSON
export BOUNDARY_WEIGHT SDM_SIGMA BOUNDARY_MODE MAX_ITER LABELED_NUM MODEL

python3 << 'PYTHON_SCRIPT'
import csv
import json
import os
import sys
import numpy as np

# ── Đọc per-run CSV ──
per_run_csv = os.environ.get("PER_RUN_CSV", "../pp4_multiseed_per_run.csv")
result_csv  = os.environ.get("RESULT_CSV",  "../pp4_multiseed_benchmark.csv")
result_json = os.environ.get("RESULT_JSON", "../pp4_multiseed_benchmark.json")

rows = []
with open(per_run_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Bỏ qua các run failed
        if "FAILED" in row.get("RV_dice", ""):
            print(f"  [SKIP] seed={row['seed']} — {row['RV_dice']}")
            continue
        try:
            rows.append({
                "seed": int(row["seed"]),
                "experiment": row["experiment"],
                "RV_dice": float(row["RV_dice"]),
                "RV_hd95": float(row["RV_hd95"]),
                "RV_asd":  float(row["RV_asd"]),
                "Myo_dice": float(row["Myo_dice"]),
                "Myo_hd95": float(row["Myo_hd95"]),
                "Myo_asd":  float(row["Myo_asd"]),
                "LV_dice": float(row["LV_dice"]),
                "LV_hd95": float(row["LV_hd95"]),
                "LV_asd":  float(row["LV_asd"]),
                "mean_dice": float(row["mean_dice"]),
                "mean_hd95": float(row["mean_hd95"]),
                "mean_asd":  float(row["mean_asd"]),
            })
        except (ValueError, KeyError) as e:
            print(f"  [SKIP] seed={row.get('seed','?')} — parse error: {e}")

if len(rows) == 0:
    print("\n  ✗ Không có run nào thành công! Không thể tính benchmark.")
    sys.exit(1)

print(f"\n  Số runs thành công: {len(rows)}/{len(rows)}")

# ── Tính mean ± std cho từng metric ──
metric_keys = [
    "RV_dice", "RV_hd95", "RV_asd",
    "Myo_dice", "Myo_hd95", "Myo_asd",
    "LV_dice", "LV_hd95", "LV_asd",
    "mean_dice", "mean_hd95", "mean_asd",
]

stats = {}
for key in metric_keys:
    values = [r[key] for r in rows]
    stats[key] = {
        "mean": round(float(np.mean(values)), 6),
        "std":  round(float(np.std(values)), 6),
        "min":  round(float(np.min(values)), 6),
        "max":  round(float(np.max(values)), 6),
        "values": [round(v, 6) for v in values],
    }

# ── In bảng tổng hợp ──
print("\n  ┌─────────────┬────────────────────────────┐")
print("  │ Metric      │ Mean ± Std                 │")
print("  ├─────────────┼────────────────────────────┤")
for key in metric_keys:
    s = stats[key]
    val_str = f"{s['mean']:.4f} ± {s['std']:.4f}"
    print(f"  │ {key:<11} │ {val_str:<26} │")
print("  └─────────────┴────────────────────────────┘")

# ── In per-run detail ──
print("\n  Per-run results:")
print(f"  {'Seed':<8} {'mean_dice':<12} {'mean_hd95':<12} {'mean_asd':<12}")
print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*12}")
for r in rows:
    print(f"  {r['seed']:<8} {r['mean_dice']:<12.4f} {r['mean_hd95']:<12.4f} {r['mean_asd']:<12.4f}")

# ── Ghi benchmark CSV ──
with open(result_csv, "w", newline="") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["metric", "mean", "std", "min", "max"] + [f"seed_{r['seed']}" for r in rows])
    for key in metric_keys:
        s = stats[key]
        writer.writerow([key, s["mean"], s["std"], s["min"], s["max"]] + s["values"])
    # Thêm per-run summary rows
    writer.writerow([])
    writer.writerow(["# Per-run details"])
    writer.writerow(["seed", "experiment"] + metric_keys)
    for r in rows:
        writer.writerow([r["seed"], r["experiment"]] + [r[k] for k in metric_keys])

print(f"\n  ✓ Benchmark CSV saved: {result_csv}")

# ── Ghi benchmark JSON ──
output = {
    "description": "PP4 Multi-Seed Benchmark (URPC + Boundary-Aware Loss)",
    "config": {
        "boundary_weight": float(os.environ.get("BOUNDARY_WEIGHT", "1.5")),
        "sdm_sigma": float(os.environ.get("SDM_SIGMA", "3.0")),
        "boundary_mode": os.environ.get("BOUNDARY_MODE", "both"),
        "max_iterations": int(os.environ.get("MAX_ITER", "10000")),
        "labeled_num": int(os.environ.get("LABELED_NUM", "7")),
        "model": os.environ.get("MODEL", "unet_urpc"),
        "seeds": [r["seed"] for r in rows],
    },
    "benchmark": {},
    "per_run": rows,
}
for key in metric_keys:
    output["benchmark"][key] = {
        "mean": stats[key]["mean"],
        "std": stats[key]["std"],
        "formatted": f"{stats[key]['mean']:.4f} ± {stats[key]['std']:.4f}",
    }

with open(result_json, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"  ✓ Benchmark JSON saved: {result_json}")
PYTHON_SCRIPT

# ================================================================
# UPLOAD KẾT QUẢ TỔNG HỢP LÊN AZURE
# ================================================================
if [ "$UPLOAD_ENABLED" = "true" ]; then
    echo ""
    echo "[UPLOAD] Uploading benchmark results to Azure..."

    for UPLOAD_FILE in "$RESULT_CSV" "$RESULT_JSON" "$PER_RUN_CSV"; do
        if [ -f "$UPLOAD_FILE" ]; then
            FNAME=$(basename "$UPLOAD_FILE")
            BLOB_PATH="results/multiseed_benchmark/${TIMESTAMP}_${FNAME}"
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
                -H "x-ms-blob-type: BlockBlob" \
                -H "Content-Type: text/plain" \
                --data-binary "@$UPLOAD_FILE" \
                "${BASE_URL}/${BLOB_PATH}?${SAS}")

            if [ "$HTTP_CODE" = "201" ]; then
                echo "  ✓ $FNAME → Azure/$BLOB_PATH"
            else
                echo "  ✗ $FNAME [HTTP $HTTP_CODE]"
            fi
        fi
    done
fi

# ================================================================
# IN BẢNG PER-RUN CSV ĐẸP
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PER-RUN RESULTS TABLE                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
python3 -c "
import csv, sys
rows = list(csv.reader(open('$PER_RUN_CSV')))
if not rows: sys.exit()
widths = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(len(rows[0]))]
for row in rows:
    print('  ' + '  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
"

# ================================================================
# HOÀN TẤT
# ================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ ALL DONE!                                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Output files:                                             ║"
echo "║  • Per-run:    $PER_RUN_CSV"
echo "║  • Benchmark:  $RESULT_CSV"
echo "║  • JSON:       $RESULT_JSON"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
