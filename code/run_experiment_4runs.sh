#!/bin/bash
# =============================================================================
# Script chạy thí nghiệm tự động 4 lần với URPC + Boundary-Aware Loss
# Author: KhangPX
# Usage: bash run_experiment_4runs.sh
# =============================================================================

set -e  # Exit on error

# ======================== CONFIGURATION ========================
EXP_NAME="ACDC/URPC_Boundary_Aware_Experiment"
MODEL="unet_urpc"
MAX_ITERATIONS=10000
NUM_RUNS=4
LABELED_NUM=7
BATCH_SIZE=24
LABELED_BS=12
BASE_LR=0.01
BOUNDARY_WEIGHT=1.0
SDM_SIGMA=5.0
ROOT_PATH="../data/ACDC/ACDC"

# Output directories
RESULTS_DIR="../results/${EXP_NAME//\//_}"
CSV_FILE="${RESULTS_DIR}/all_runs_results.csv"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
# ===============================================================

cd /teamspace/studios/this_studio/code

echo "=============================================================="
echo "🚀 URPC + Boundary-Aware Loss - Multi-Run Experiment"
echo "=============================================================="
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL}"
echo "Max Iterations: ${MAX_ITERATIONS}"
echo "Number of Runs: ${NUM_RUNS}"
echo "Labeled Patients: ${LABELED_NUM}"
echo "Results Directory: ${RESULTS_DIR}"
echo "=============================================================="

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Remove old CSV file if exists
if [ -f "${CSV_FILE}" ]; then
    rm "${CSV_FILE}"
    echo "Removed old results file: ${CSV_FILE}"
fi

# Run experiments
for RUN_ID in $(seq 1 ${NUM_RUNS}); do
    echo ""
    echo "=============================================================="
    echo "🔄 Starting Run ${RUN_ID}/${NUM_RUNS}"
    echo "=============================================================="
    
    # Unique experiment name for this run
    RUN_EXP_NAME="${EXP_NAME}_run${RUN_ID}"
    
    # Different seed for each run to ensure different initialization
    SEED=$((1337 + RUN_ID * 100))
    
    echo "📝 Config:"
    echo "   - Experiment: ${RUN_EXP_NAME}"
    echo "   - Seed: ${SEED}"
    echo "   - Output: ${CSV_FILE}"
    
    # ==================== TRAINING ====================
    echo ""
    echo "🏋️ [Run ${RUN_ID}] Training started..."
    START_TIME=$(date +%s)
    
    python khangpx_improvement.py \
        --root_path "${ROOT_PATH}" \
        --exp "${RUN_EXP_NAME}" \
        --model "${MODEL}" \
        --num_classes 4 \
        --labeled_num ${LABELED_NUM} \
        --batch_size ${BATCH_SIZE} \
        --labeled_bs ${LABELED_BS} \
        --max_iterations ${MAX_ITERATIONS} \
        --base_lr ${BASE_LR} \
        --seed ${SEED}
    
    END_TIME=$(date +%s)
    TRAIN_DURATION=$((END_TIME - START_TIME))
    echo "✅ [Run ${RUN_ID}] Training completed in ${TRAIN_DURATION}s"
    
    # ==================== TESTING ====================
    echo ""
    echo "🧪 [Run ${RUN_ID}] Testing started..."
    
    python test_2D_to_csv.py \
        --root_path "${ROOT_PATH}" \
        --exp "${RUN_EXP_NAME}" \
        --model "${MODEL}" \
        --num_classes 4 \
        --labeled_num ${LABELED_NUM} \
        --run_id ${RUN_ID} \
        --csv_output "${CSV_FILE}"
    
    echo "✅ [Run ${RUN_ID}] Testing completed"
    
    # ==================== COPY BEST MODEL ====================
    SRC_MODEL="../model/${RUN_EXP_NAME}_${LABELED_NUM}_labeled/${MODEL}/${MODEL}_best_model.pth"
    DST_MODEL="${RESULTS_DIR}/best_model_run${RUN_ID}.pth"
    
    if [ -f "${SRC_MODEL}" ]; then
        cp "${SRC_MODEL}" "${DST_MODEL}"
        echo "📁 [Run ${RUN_ID}] Best model saved to: ${DST_MODEL}"
    fi
    
    echo ""
    echo "=============================================================="
    echo "✅ Run ${RUN_ID}/${NUM_RUNS} COMPLETED"
    echo "=============================================================="
done

# ==================== SUMMARIZE RESULTS ====================
echo ""
echo "=============================================================="
echo "📊 Generating Summary..."
echo "=============================================================="

python summarize_results.py \
    --input_csv "${CSV_FILE}" \
    --output_csv "${SUMMARY_FILE}"

echo ""
echo "=============================================================="
echo "🎉 ALL ${NUM_RUNS} RUNS COMPLETED!"
echo "=============================================================="
echo ""
echo "📁 Results saved to:"
echo "   - All runs: ${CSV_FILE}"
echo "   - Summary:  ${SUMMARY_FILE}"
echo "   - Models:   ${RESULTS_DIR}/best_model_run*.pth"
echo ""
echo "=============================================================="
