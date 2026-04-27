# Uncertainty Rectified Pyramid Consistency (URPC) — Nhánh cải tiến (DSP391m)

Repo này là một nhánh thực hành/đồ án tập trung vào **URPC (Uncertainty Rectified Pyramid Consistency)** cho bài toán **phân đoạn ảnh y khoa bán giám sát (semi-supervised segmentation)**. Trọng tâm hiện tại là **URPC 2D trên dataset ACDC** và các cải tiến được triển khai/thử nghiệm thông qua các file `*improvement.py`.

## Repo đang làm gì?

- **Baseline URPC 2D**: `code/train_uncertainty_rectified_pyramid_consistency_2D.py`
- **Các biến thể/cải tiến URPC 2D** (đang dùng để thí nghiệm):
  - `code/thucnv_improvement.py`: multi-scale **attention fusion** cho 4 nhánh URPC
  - `code/thucnv_improvement_2.py`: attention fusion + **target sharpening** + **entropy minimization** trên unlabeled
  - `code/khangpx_improvement.py`: **boundary-aware supervised loss** dựa trên **Signed Distance Map (SDM)**
  - `code/hiepdv_improvement.py`: ý tưởng **dynamic confidence thresholding** (hiện đang để dạng *commented-out*, chưa chạy trực tiếp)

## Cấu trúc code liên quan

- **Attention fusion module**: `code/networks/attention_fusion.py`
- **Dataloader (đọc `.h5`)**: `code/dataloaders/dataset.py`
- **TensorBoard/Checkpoint**: lưu theo `snapshot_path` trong từng script train

## Cài đặt

```bash
pip install -r requirements.txt
```

## Dữ liệu ACDC — Format mà repo đang đọc

Các script train 2D dùng `BaseDataSets` tại `code/dataloaders/dataset.py`. Vì vậy `--root_path` phải trỏ tới thư mục có tối thiểu cấu trúc sau:

```
<ROOT_PATH>/
  train_slices.list
  val.list
  data/
    slices/
      <case_id>.h5
      ...
    <case_id>.h5
    ...
```

- **Train** đọc: `data/slices/<case_id>.h5`
- **Val** đọc: `data/<case_id>.h5`

Repo có `download_acdc.py` để tải ACDC từ Google Drive, nhưng hiện **hard-code đường dẫn kiểu Linux** (ví dụ `/teamspace/...`). Nếu bạn chạy Windows, cần sửa `PROJECT_DIR/DATA_DIR` trước khi dùng.

## Cách chạy

Khuyến nghị chạy từ thư mục `code/` để import nội bộ đúng.

### 1) Baseline URPC 2D

```bash
cd code
python train_uncertainty_rectified_pyramid_consistency_2D.py --root_path ../data/ACDC
```

Output mặc định được lưu dưới `../model/...` (tức thư mục `model/` ở root repo).

### 2) `thucnv_improvement.py` — Multi-Scale Attention Fusion

Ý tưởng: thay vì fusion bằng trung bình \( (p_1+p_2+p_3+p_4)/4 \), dùng attention học trọng số \( w_i \) (spatial hoặc lightweight). Có tuỳ chọn dùng uncertainty (entropy) để điều hướng trọng số.

```bash
cd code
python thucnv_improvement.py --root_path ../data/ACDC --output_path ../model
```

Checkpoint được lưu dạng dict, bao gồm:
- `model_state_dict`
- `attention_fusion_state_dict`
- metadata (`iteration`, `best_performance`, ...)

### 3) `thucnv_improvement_2.py` — Attention Fusion + Sharpening + Entropy loss

Ý tưởng bổ sung:
- **Target sharpening** cho pseudo-target của unlabeled (`--sharpen_T`)
- **Entropy minimization** trên unlabeled (`--entropy_weight`)

```bash
cd code
python thucnv_improvement_2.py --root_path ../data/ACDC --output_path ../model
```

### 4) `khangpx_improvement.py` — URPC + Boundary-Aware Loss (SDM) (PP4)

Ý tưởng: thêm boundary-aware supervised loss dựa SDM để tăng trọng số lỗi gần biên. Tham số chính:
- `--boundary_weight`
- `--sdm_sigma`
- `--boundary_mode` (`ce_only|dice_only|both`)

```bash
cd code
python khangpx_improvement.py --root_path ../data/ACDC/ACDC
```

Lưu ý: script này hiện **không dùng `--output_path`**, output mặc định lưu theo `../model/...` giống baseline.

### 5) `hiepdv_improvement.py` — Dynamic Confidence Thresholding (nháp)

File mô tả ý tưởng “chỉ tính consistency loss cho pixel unlabeled có confidence > threshold, threshold tăng dần theo training (curriculum)”, nhưng hiện **toàn bộ code đang bị comment**, nên chưa thể chạy trực tiếp.

## Output/Logging

- Log file: `log.txt` trong thư mục output (snapshot).
- TensorBoard: `.../log/`
- Baseline + `khangpx_improvement.py`: lưu model bằng `state_dict()` thuần.
- `thucnv_improvement.py` và `thucnv_improvement_2.py`: lưu checkpoint dạng dict (model + attention module + metadata).

## Ghi chú nhanh khi chạy trên Windows

- Kiểm tra kỹ `--root_path` đúng format mục “Dữ liệu ACDC”.
- Nếu dùng `download_acdc.py` cần sửa đường dẫn cho Windows.
