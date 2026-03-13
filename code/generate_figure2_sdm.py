"""
Generate Figure 2 (SDM Visualization) for report4_methodology.tex.

Figure layout:
  (a) Ground-truth label map
  (b) Signed Distance Map (SDM) of one selected class
  (c) Boundary weight map from SDM

This script follows the same SDM definition used in khangpx_improvement.py:
  weight = 1 + exp(-min(|SDM|)^2 / (2 * sigma^2))
"""

import argparse
import os
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from scipy.ndimage import distance_transform_edt
except ModuleNotFoundError:
    distance_transform_edt = None


def ensure_optional_dependencies():
    missing = []
    if h5py is None:
        missing.append("h5py")
    if plt is None:
        missing.append("matplotlib")
    if distance_transform_edt is None:
        missing.append("scipy")

    if missing:
        pkg_text = " ".join(missing)
        raise ModuleNotFoundError(
            "Missing required packages: "
            f"{', '.join(missing)}. "
            "Install them in your active environment with: "
            f"pip install {pkg_text}"
        )


def compute_sdm_single_class(binary_mask: np.ndarray) -> np.ndarray:
    """Compute signed distance map for one binary mask."""
    binary_mask = binary_mask.astype(np.uint8)

    if binary_mask.sum() == 0:
        return -distance_transform_edt(np.ones_like(binary_mask)).astype(np.float32)

    if binary_mask.sum() == binary_mask.size:
        return distance_transform_edt(binary_mask).astype(np.float32)

    dist_inside = distance_transform_edt(binary_mask)
    dist_outside = distance_transform_edt(1 - binary_mask)
    sdm = dist_inside - dist_outside
    return sdm.astype(np.float32)


def compute_sdm_label(label_2d: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute SDM stack with shape (C, H, W) from one 2D label map."""
    h, w = label_2d.shape
    sdm = np.zeros((num_classes, h, w), dtype=np.float32)
    for cls in range(num_classes):
        binary = (label_2d == cls).astype(np.uint8)
        sdm[cls] = compute_sdm_single_class(binary)
    return sdm


def compute_boundary_weight_map(sdm: np.ndarray, sigma: float) -> np.ndarray:
    """Compute boundary weight map from SDM stack (C, H, W)."""
    sdm_abs = np.abs(sdm)
    min_sdm_abs = np.min(sdm_abs, axis=0)
    min_sdm_abs = np.clip(min_sdm_abs, a_min=0.0, a_max=sigma * 3)
    boundary = np.exp(-(min_sdm_abs ** 2) / (2 * sigma ** 2))
    return (1.0 + boundary).astype(np.float32)


def pick_case_from_list(root_path: Path, split_list: str) -> str:
    list_path = root_path / split_list
    if not list_path.exists():
        raise FileNotFoundError(
            f"Cannot find list file: {list_path}. "
            "Please pass --case explicitly or fix --root_path."
        )

    with open(list_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    if not names:
        raise RuntimeError(f"No case found in {list_path}")
    return names[0]


def resolve_h5_path(root_path: Path, case_name: str) -> Path:
    slice_path = root_path / "data" / "slices" / f"{case_name}.h5"
    volume_path = root_path / "data" / f"{case_name}.h5"

    if slice_path.exists():
        return slice_path
    if volume_path.exists():
        return volume_path

    raise FileNotFoundError(
        "Cannot find H5 file for case "
        f"'{case_name}'. Tried:\n"
        f"  - {slice_path}\n"
        f"  - {volume_path}\n"
        "Please verify processed ACDC data exists."
    )


def choose_slice(label: np.ndarray, slice_index: int) -> int:
    """Choose slice index for 3D label arrays."""
    if label.ndim != 3:
        return 0

    if slice_index >= 0:
        if slice_index >= label.shape[0]:
            raise ValueError(
                f"slice_index={slice_index} out of range for depth={label.shape[0]}"
            )
        return slice_index

    foreground = (label > 0).sum(axis=(1, 2))
    if np.max(foreground) == 0:
        return label.shape[0] // 2
    return int(np.argmax(foreground))


def load_image_label(h5_path: Path, slice_index: int):
    with h5py.File(h5_path, "r") as h5f:
        image = h5f["image"][:]
        label = h5f["label"][:]

    if label.ndim == 2:
        return image.astype(np.float32), label.astype(np.int32), 0

    if label.ndim == 3:
        idx = choose_slice(label, slice_index)
        return image[idx].astype(np.float32), label[idx].astype(np.int32), idx

    raise ValueError(f"Unsupported label shape: {label.shape}")


def render_figure(
    image: np.ndarray,
    label: np.ndarray,
    sdm: np.ndarray,
    weight_map: np.ndarray,
    class_names,
    num_classes: int,
    class_id: int,
    save_path: Path,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    label_cmap = plt.get_cmap("tab10", num_classes)
    im0 = axes[0].imshow(
        label,
        cmap=label_cmap,
        vmin=-0.5,
        vmax=num_classes - 0.5,
        interpolation="nearest",
    )
    axes[0].set_title("(a) Ground Truth Label")
    axes[0].axis("off")
    cbar0 = plt.colorbar(im0, ax=axes[0], ticks=np.arange(num_classes), fraction=0.046, pad=0.04)
    cbar0.ax.set_yticklabels(class_names)
    cbar0.set_label("Class")

    im1 = axes[1].imshow(sdm[class_id], cmap="RdBu_r", interpolation="nearest")
    axes[1].set_title(f"(b) SDM (Class {class_id})")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(weight_map, cmap="hot", vmin=1.0, vmax=2.0, interpolation="nearest")
    axes[2].set_title("(c) Boundary Weight Map")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figure 2 SDM visualization")
    parser.add_argument(
        "--root_path",
        type=str,
        default="../data/ACDC/ACDC",
        help="Processed ACDC root path containing train_slices.list and data/",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="",
        help="Case name without extension. If empty, use first case in --split_list.",
    )
    parser.add_argument(
        "--split_list",
        type=str,
        default="train_slices.list",
        help="List file used when --case is empty.",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=-1,
        help="Slice index for 3D volume labels. -1 means auto-select foreground-rich slice.",
    )
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument(
        "--class_names",
        type=str,
        default="BG,RV,Myo,LV",
        help="Comma-separated class names for GT legend/colorbar",
    )
    parser.add_argument("--sigma", type=float, default=3.0, help="Sigma in SDM weight equation")
    parser.add_argument(
        "--sdm_class",
        type=int,
        default=2,
        help="Class index visualized in SDM panel (default 2 for Myo)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../figures/sdm_visualization.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--save_npz",
        type=str,
        default="",
        help="Optional path to save intermediate arrays (.npz)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_optional_dependencies()

    root_path = Path(args.root_path).resolve()
    save_path = Path(args.save_path).resolve()

    if args.sdm_class < 0 or args.sdm_class >= args.num_classes:
        raise ValueError(
            f"sdm_class must be in [0, {args.num_classes - 1}], got {args.sdm_class}"
        )

    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    if len(class_names) < args.num_classes:
        class_names = class_names + [f"Class {i}" for i in range(len(class_names), args.num_classes)]
    elif len(class_names) > args.num_classes:
        class_names = class_names[: args.num_classes]

    case_name = args.case.strip() if args.case.strip() else pick_case_from_list(root_path, args.split_list)
    h5_path = resolve_h5_path(root_path, case_name)
    image, label, picked_idx = load_image_label(h5_path, args.slice_index)

    sdm = compute_sdm_label(label, args.num_classes)
    weight_map = compute_boundary_weight_map(sdm, args.sigma)

    render_figure(
        image=image,
        label=label,
        sdm=sdm,
        weight_map=weight_map,
        class_names=class_names,
        num_classes=args.num_classes,
        class_id=args.sdm_class,
        save_path=save_path,
        dpi=args.dpi,
    )

    if args.save_npz:
        npz_path = Path(args.save_npz).resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            case=case_name,
            h5_path=str(h5_path),
            image=image,
            label=label,
            sdm=sdm,
            weight_map=weight_map,
            sigma=float(args.sigma),
            sdm_class=int(args.sdm_class),
            slice_index=int(picked_idx),
        )
        print(f"Saved intermediate arrays to: {npz_path}")

    print("Figure 2 generated successfully")
    print(f"  case        : {case_name}")
    print(f"  h5_path     : {h5_path}")
    print(f"  slice_index : {picked_idx}")
    print(f"  sigma       : {args.sigma}")
    print(f"  sdm_class   : {args.sdm_class}")
    print(f"  weight min  : {weight_map.min():.4f}")
    print(f"  weight max  : {weight_map.max():.4f}")
    print(f"  saved to    : {save_path}")


if __name__ == "__main__":
    main()
