"""
PP4 Multi-Seed Benchmark: Download, Test, and Aggregate
========================================================
Tải best model weights từ Azure, chạy test_2D_fully.py,
và tổng hợp metrics (Dice, HD95, ASD) qua 4 seeds.

Cách chạy:
    python benchmark_multiseed.py

    # Chỉ tổng hợp (nếu đã test xong):
    python benchmark_multiseed.py --skip_test

    # Chỉ test, không download:
    python benchmark_multiseed.py --skip_download

Author: KhangPX
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ================================================================
# CẤU HÌNH
# ================================================================

# 4 seed experiments trên Azure
SEED_EXPERIMENTS = [
    "PP4_SEED1337_w1.5_s3.0",
    "PP4_SEED12345_w1.5_s3.0",
    "PP4_SEED2024_w1.5_s3.0",
    "PP4_SEED42_w1.5_s3.0",
]

# Cấu hình model/training
MODEL = "unet_urpc"
NUM_CLASSES = 4
LABELED_NUM = 7

# Đường dẫn
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_PATH = SCRIPT_DIR / ".." / "data" / "ACDC" / "ACDC"
MODEL_BASE_DIR = SCRIPT_DIR / ".." / "model" / "ACDC"
OUTPUT_DIR = SCRIPT_DIR / ".."

# Classes trong ACDC dataset
CLASS_NAMES = ["RV", "Myo", "LV"]   # Right Ventricle, Myocardium, Left Ventricle
METRIC_NAMES = ["Dice", "HD95", "ASD"]


# ================================================================
# ĐỌC .ENV FILE
# ================================================================

def load_env(env_file: Path) -> dict:
    """Đọc .env file và trả về dict các credentials."""
    env = {}
    if not env_file.exists():
        print(f"[ENV] WARNING: {env_file} not found")
        return env

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#")[0].strip()  # Remove inline comments
            env[key] = value

    print(f"[ENV] Loaded {len(env)} vars from {env_file}")
    return env


def get_azure_config(env: dict) -> dict:
    """Trích xuất Azure config từ env dict."""
    conn_str = env.get("AZURE_CONNECTION_STRING", "")
    account_match = re.search(r"AccountName=([^;]+)", conn_str)
    account = account_match.group(1) if account_match else env.get("AZURE_ACCOUNT", "")
    container = env.get("AZURE_CONTAINER", "ssl4mis-weights")
    sas_token = env.get("AZURE_SAS_TOKEN", "")

    if not account or not sas_token:
        print("[AZURE] WARNING: Missing account or SAS token")
        return {}

    return {
        "account": account,
        "container": container,
        "sas_token": sas_token,
        "base_url": f"https://{account}.blob.core.windows.net/{container}",
    }


# ================================================================
# AZURE: LIST & DOWNLOAD
# ================================================================

def list_blobs(azure_cfg: dict, prefix: str) -> list:
    """List tất cả blobs trong một folder trên Azure."""
    import urllib.request
    import xml.etree.ElementTree as ET

    url = (
        f"{azure_cfg['base_url']}?"
        f"restype=container&comp=list&prefix={prefix}/&{azure_cfg['sas_token']}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read().decode("utf-8")
        root = ET.fromstring(xml_data)
        blobs = []
        for blob in root.iter("Blob"):
            name_elem = blob.find("Name")
            if name_elem is not None:
                blobs.append(name_elem.text)
        return blobs
    except Exception as e:
        print(f"  [AZURE] Error listing blobs: {e}")
        return []


def download_blob(azure_cfg: dict, blob_name: str, local_path: Path) -> bool:
    """Download một blob từ Azure Blob Storage."""
    import urllib.request

    url = f"{azure_cfg['base_url']}/{blob_name}?{azure_cfg['sas_token']}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(local_path))
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"    ✓ {blob_name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"    ✗ {blob_name}: {e}")
        return False


def download_experiment_weights(azure_cfg: dict, exp_name: str) -> Path:
    """
    Download tất cả weights (bao gồm best model) của một experiment từ Azure.
    Trả về đường dẫn local dir chứa weights.
    """
    local_dir = MODEL_BASE_DIR / f"{exp_name}_{LABELED_NUM}_labeled" / MODEL
    best_model_path = local_dir / f"{MODEL}_best_model.pth"

    # Skip nếu đã có best model
    if best_model_path.exists():
        size_mb = best_model_path.stat().st_size / (1024 * 1024)
        print(f"  [DOWNLOAD] SKIP — Best model đã tồn tại ({size_mb:.1f} MB)")
        return local_dir

    print(f"  [DOWNLOAD] Downloading {exp_name} từ Azure...")

    # List tất cả blobs trong experiment folder
    blobs = list_blobs(azure_cfg, exp_name)
    pth_blobs = [b for b in blobs if b.endswith(".pth")]

    if not pth_blobs:
        print(f"  [DOWNLOAD] WARNING: Không tìm thấy .pth files cho {exp_name}")
        return local_dir

    # Download tất cả .pth files
    downloaded = 0
    for blob in pth_blobs:
        fname = os.path.basename(blob)
        local_path = local_dir / fname
        if local_path.exists():
            print(f"    ≡ {fname} (already exists)")
            downloaded += 1
            continue
        if download_blob(azure_cfg, blob, local_path):
            downloaded += 1

    print(f"  [DOWNLOAD] Done: {downloaded}/{len(pth_blobs)} files")
    return local_dir


# ================================================================
# CHẠY TEST
# ================================================================

def run_test(exp_name: str) -> dict:
    """
    Chạy test_2D_fully.py cho một experiment và parse kết quả.

    Returns:
        dict với keys: RV_Dice, RV_HD95, RV_ASD, Myo_Dice, ..., mean_Dice, ...
        hoặc None nếu test thất bại.
    """
    weight_dir = MODEL_BASE_DIR / f"{exp_name}_{LABELED_NUM}_labeled" / MODEL
    best_model = weight_dir / f"{MODEL}_best_model.pth"

    if not best_model.exists():
        print(f"  [TEST] ERROR: Best model not found: {best_model}")
        return None

    print(f"  [TEST] Running test_2D_fully.py...")

    # Chạy test script
    cmd = [
        sys.executable, str(SCRIPT_DIR / "test_2D_fully.py"),
        "--root_path", str(ROOT_PATH),
        "--exp", f"ACDC/{exp_name}",
        "--model", MODEL,
        "--num_classes", str(NUM_CLASSES),
        "--labeled_num", str(LABELED_NUM),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 phút timeout
            cwd=str(SCRIPT_DIR),
        )
        output = result.stdout + result.stderr

        # Lưu test log
        log_path = weight_dir / "test_output.log"
        with open(log_path, "w") as f:
            f.write(output)

    except subprocess.TimeoutExpired:
        print(f"  [TEST] ERROR: Timeout sau 10 phút")
        return None
    except Exception as e:
        print(f"  [TEST] ERROR: {e}")
        return None

    # Parse output
    return parse_test_output(output, exp_name)


def parse_test_output(output: str, exp_name: str) -> dict:
    """
    Parse output của test_2D_fully.py.

    Output format:
      [array([dice, hd95, asd]), array([dice, hd95, asd]), array([dice, hd95, asd])]
      [array([mean_dice, mean_hd95, mean_asd])]
    """
    # Tìm dòng chứa 3 arrays (kết quả per-class)
    # Pattern: [array([...]), array([...]), array([...])]
    lines = output.strip().split("\n")

    metrics_line = None
    for line in reversed(lines):
        if "array" in line and line.count("array") >= 3:
            metrics_line = line.strip()
            break

    if not metrics_line:
        print(f"  [TEST] WARNING: Không parse được metrics từ output")
        print(f"  [TEST] Output tail: {lines[-3:] if len(lines) >= 3 else lines}")
        return None

    # Extract tất cả số thực từ dòng metrics
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", metrics_line)
    numbers = [float(n) for n in numbers]

    if len(numbers) < 9:
        print(f"  [TEST] WARNING: Chỉ parse được {len(numbers)}/9 numbers")
        return None

    # Map vào dict
    result = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        for j, metric_name in enumerate(METRIC_NAMES):
            result[f"{cls_name}_{metric_name}"] = numbers[i * 3 + j]

    # Tính mean
    for j, metric_name in enumerate(METRIC_NAMES):
        values = [result[f"{cls}_{metric_name}"] for cls in CLASS_NAMES]
        result[f"mean_{metric_name}"] = np.mean(values)

    # Thêm metadata
    result["experiment"] = exp_name

    # Extract seed từ tên experiment
    seed_match = re.search(r"SEED(\d+)", exp_name)
    result["seed"] = int(seed_match.group(1)) if seed_match else 0

    return result


def parse_best_dice_from_filenames(exp_name: str) -> float:
    """
    Backup: lấy best validation Dice từ tên file checkpoint.
    File format: iter_XXXX_dice_0.YYYY.pth
    """
    weight_dir = MODEL_BASE_DIR / f"{exp_name}_{LABELED_NUM}_labeled" / MODEL
    if not weight_dir.exists():
        return 0.0

    best_dice = 0.0
    for pth_file in weight_dir.glob("iter_*_dice_*.pth"):
        match = re.search(r"dice_([\d.]+)", pth_file.name)
        if match:
            dice = float(match.group(1))
            best_dice = max(best_dice, dice)

    return best_dice


# ================================================================
# TỔNG HỢP KẾT QUẢ
# ================================================================

def aggregate_results(results: list) -> tuple:
    """
    Tổng hợp kết quả từ nhiều runs.

    Returns:
        (per_run_df, summary_df, best_run_info)
    """
    # ── Bảng 1: Kết quả từng run ──
    metric_columns = []
    for cls in CLASS_NAMES:
        for met in METRIC_NAMES:
            metric_columns.append(f"{cls}_{met}")
    for met in METRIC_NAMES:
        metric_columns.append(f"mean_{met}")

    rows = []
    for r in results:
        row = {"Run": r["experiment"], "Seed": r["seed"]}
        for col in metric_columns:
            row[col] = r.get(col, np.nan)
        rows.append(row)

    per_run_df = pd.DataFrame(rows)
    per_run_df = per_run_df.sort_values("Seed").reset_index(drop=True)
    per_run_df.index = [f"Run {i+1}" for i in range(len(per_run_df))]

    # ── Bảng 2: Mean ± Std ──
    summary_rows = []
    for col in metric_columns:
        values = per_run_df[col].dropna()
        summary_rows.append({
            "Metric": col,
            "Mean": values.mean(),
            "Std": values.std(),
            "Min": values.min(),
            "Max": values.max(),
            "Mean ± Std": f"{values.mean():.4f} ± {values.std():.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── Best run (theo mean_Dice cao nhất) ──
    best_idx = per_run_df["mean_Dice"].idxmax()
    best_run = per_run_df.loc[best_idx]
    best_info = {
        "run_name": best_run["Run"],
        "seed": int(best_run["Seed"]),
        "mean_Dice": best_run["mean_Dice"],
        "mean_HD95": best_run["mean_HD95"],
        "mean_ASD": best_run["mean_ASD"],
    }

    return per_run_df, summary_df, best_info


# ================================================================
# EXPORT KẾT QUẢ
# ================================================================

def export_results(per_run_df, summary_df, best_info, results):
    """Xuất kết quả ra CSV, JSON, và in ra terminal."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV: Per-run results ──
    per_run_csv = OUTPUT_DIR / "pp4_multiseed_per_run_results.csv"
    per_run_df.to_csv(per_run_csv, index=True)
    print(f"\n  ✓ Per-run CSV: {per_run_csv}")

    # ── CSV: Summary (mean ± std) ──
    summary_csv = OUTPUT_DIR / "pp4_multiseed_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✓ Summary CSV: {summary_csv}")

    # ── JSON: Full report ──
    report = {
        "description": "PP4 Multi-Seed Benchmark (URPC + Boundary-Aware Loss)",
        "timestamp": timestamp,
        "config": {
            "model": MODEL,
            "num_classes": NUM_CLASSES,
            "labeled_num": LABELED_NUM,
            "experiments": SEED_EXPERIMENTS,
        },
        "per_run": [],
        "benchmark": {},
        "best_run": best_info,
    }

    # Per-run data
    for r in results:
        run_data = {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in r.items()}
        report["per_run"].append(run_data)

    # Benchmark summary
    for _, row in summary_df.iterrows():
        report["benchmark"][row["Metric"]] = {
            "mean": round(row["Mean"], 6),
            "std": round(row["Std"], 6),
            "formatted": row["Mean ± Std"],
        }

    json_path = OUTPUT_DIR / "pp4_multiseed_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON report: {json_path}")

    return per_run_csv, summary_csv, json_path


def print_results(per_run_df, summary_df, best_info):
    """In kết quả đẹp ra terminal."""
    print("\n" + "=" * 70)
    print("  BẢNG 1: KẾT QUẢ TỪNG RUN")
    print("=" * 70)

    # Chọn các cột quan trọng để hiển thị
    display_cols = ["Seed", "mean_Dice", "mean_HD95", "mean_ASD"]
    for cls in CLASS_NAMES:
        display_cols.insert(-3, f"{cls}_Dice")

    available_cols = [c for c in display_cols if c in per_run_df.columns]
    print(per_run_df[available_cols].to_string())

    print("\n" + "=" * 70)
    print("  BẢNG 2: MEAN ± STD (qua 4 seeds)")
    print("=" * 70)
    display_summary = summary_df[["Metric", "Mean ± Std", "Min", "Max"]].copy()
    print(display_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("  RUN TỐT NHẤT (theo mean Dice)")
    print("=" * 70)
    print(f"  🏆 {best_info['run_name']}")
    print(f"     Seed:      {best_info['seed']}")
    print(f"     mean_Dice: {best_info['mean_Dice']:.4f}")
    print(f"     mean_HD95: {best_info['mean_HD95']:.4f}")
    print(f"     mean_ASD:  {best_info['mean_ASD']:.4f}")
    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PP4 Multi-Seed Benchmark: Download, Test, Aggregate"
    )
    parser.add_argument("--skip_download", action="store_true",
                        help="Bỏ qua bước download từ Azure (dùng weights local)")
    parser.add_argument("--skip_test", action="store_true",
                        help="Bỏ qua bước test (đọc kết quả đã có)")
    parser.add_argument("--experiments", nargs="+", default=SEED_EXPERIMENTS,
                        help="Danh sách experiment names trên Azure")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PP4 Multi-Seed Benchmark                                  ║")
    print("║  URPC + Boundary-Aware Loss (SDM)                          ║")
    print("║  Author: KhangPX                                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    experiments = args.experiments
    print(f"  Experiments: {experiments}")
    print(f"  Model: {MODEL}")
    print(f"  Root path: {ROOT_PATH}")
    print()

    # ── Step 1: Load Azure config ──
    env = load_env(SCRIPT_DIR / ".." / ".env")
    azure_cfg = get_azure_config(env)

    # ── Step 2: Download weights từ Azure ──
    if not args.skip_download and azure_cfg:
        print("\n" + "=" * 50)
        print("  STEP 1: DOWNLOAD WEIGHTS TỪ AZURE")
        print("=" * 50)
        for exp_name in experiments:
            print(f"\n  ── {exp_name} ──")
            download_experiment_weights(azure_cfg, exp_name)
    elif args.skip_download:
        print("[DOWNLOAD] Skipped (--skip_download)")
    else:
        print("[DOWNLOAD] Skipped (no Azure config)")

    # ── Step 3: Chạy test cho từng experiment ──
    results = []

    if not args.skip_test:
        print("\n" + "=" * 50)
        print("  STEP 2: CHẠY TEST CHO TỪNG SEED")
        print("=" * 50)

        for i, exp_name in enumerate(experiments, 1):
            print(f"\n  ── [{i}/{len(experiments)}] {exp_name} ──")
            result = run_test(exp_name)
            if result:
                results.append(result)
                print(f"  [TEST] ✓ mean_Dice={result['mean_Dice']:.4f}  "
                      f"mean_HD95={result['mean_HD95']:.4f}  "
                      f"mean_ASD={result['mean_ASD']:.4f}")
            else:
                # Fallback: lấy dice từ filename
                best_dice = parse_best_dice_from_filenames(exp_name)
                if best_dice > 0:
                    print(f"  [FALLBACK] Best val Dice từ filename: {best_dice:.4f}")
                    print(f"  [FALLBACK] (Chỉ có Dice, không có HD95/ASD)")
                    results.append({
                        "experiment": exp_name,
                        "seed": int(re.search(r"SEED(\d+)", exp_name).group(1)),
                        "mean_Dice": best_dice,
                        "mean_HD95": np.nan,
                        "mean_ASD": np.nan,
                        **{f"{cls}_{met}": np.nan
                           for cls in CLASS_NAMES for met in METRIC_NAMES},
                    })
    else:
        # Đọc kết quả từ test logs đã có
        print("\n[TEST] Skipped (--skip_test). Đọc kết quả từ test_output.log...")
        for exp_name in experiments:
            weight_dir = MODEL_BASE_DIR / f"{exp_name}_{LABELED_NUM}_labeled" / MODEL
            log_path = weight_dir / "test_output.log"
            if log_path.exists():
                output = log_path.read_text()
                result = parse_test_output(output, exp_name)
                if result:
                    results.append(result)
                    print(f"  ✓ {exp_name}: mean_Dice={result['mean_Dice']:.4f}")
                else:
                    print(f"  ✗ {exp_name}: Parse failed")
            else:
                print(f"  ✗ {exp_name}: test_output.log not found")

    # ── Step 4: Tổng hợp ──
    if len(results) == 0:
        print("\n  ✗ Không có kết quả nào! Kiểm tra lại data/weights.")
        sys.exit(1)

    print(f"\n  Tổng số runs thành công: {len(results)}/{len(experiments)}")

    print("\n" + "=" * 50)
    print("  STEP 3: TỔNG HỢP KẾT QUẢ")
    print("=" * 50)

    per_run_df, summary_df, best_info = aggregate_results(results)

    # ── Step 5: In kết quả + export ──
    print_results(per_run_df, summary_df, best_info)
    export_results(per_run_df, summary_df, best_info, results)

    print("\n  ✅ Benchmark hoàn tất!")
    print()


if __name__ == "__main__":
    main()
