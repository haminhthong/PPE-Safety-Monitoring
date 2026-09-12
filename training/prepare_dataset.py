"""Chuẩn bị dữ liệu huấn luyện PPE: Group Split theo Video và Cắt Person Crop.

Kỹ thuật quan trọng:
1. Group Split theo video/session: Ngăn chặn rò rỉ dữ liệu (data leakage) giữa các frame
   gần nhau của cùng một video trong tập train và test/val.
2. Cắt Person ROI và ánh xạ lại nhãn PPE: Mô hình phát hiện PPE giai đoạn 2 được huấn luyện
   trực tiếp trên ảnh crop của công nhân với nhãn đã được chuẩn hóa theo tọa độ crop.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ppe_detection.crops import PersonCropBuilder
from ppe_detection.detector import read_image, write_image

LOGGER = logging.getLogger("prepare_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PPE_CLASSES = {
    0: "helmet",
    1: "no-helmet",
    2: "vest",
    3: "no-vest",
}


def group_split_by_video(
    samples: list[dict[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Phân chia tập dữ liệu thành train/val/test theo group video_id.

    Tất cả các frame trích xuất từ cùng một video_id hoặc recording_session
    bắt buộc phải nằm trọn vẹn trong cùng 1 tập (train, val hoặc test)
    để tránh rò rỉ hình thái và bối cảnh (context leakage).
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        group_key = sample.get("video_id") or sample.get("recording_session") or sample["sample_id"]
        groups[group_key].append(sample)

    rng = random.Random(seed)
    unique_groups = list(groups.keys())
    rng.shuffle(unique_groups)

    n_total = len(samples)
    target_train = int(n_total * train_ratio)
    target_val = int(n_total * val_ratio)

    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    train_count = 0
    val_count = 0

    for g_key in unique_groups:
        group_samples = groups[g_key]
        g_len = len(group_samples)
        if train_count + g_len <= target_train or (
            train_count < target_train and not splits["train"]
        ):
            splits["train"].extend(group_samples)
            train_count += g_len
        elif val_count + g_len <= target_val or (val_count < target_val and not splits["val"]):
            splits["val"].extend(group_samples)
            val_count += g_len
        else:
            splits["test"].extend(group_samples)

    LOGGER.info(
        "Group split hoàn tất: Train=%d samples, Val=%d samples, Test=%d samples (tổng %d videos)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        len(unique_groups),
    )
    return splits


def remap_ppe_box_to_crop(
    ppe_box_norm: list[float],
    image_shape: tuple[int, int],
    crop_window: tuple[int, int, int, int],
) -> list[float] | None:
    """Chuyển đổi tọa độ bounding box PPE từ ảnh gốc sang hệ tọa độ crop đã chuẩn hóa [0..1]."""
    height, width = image_shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_window
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w <= 0 or crop_h <= 0:
        return None

    cx_norm, cy_norm, w_norm, h_norm = ppe_box_norm
    abs_x1 = (cx_norm - w_norm / 2.0) * width
    abs_y1 = (cy_norm - h_norm / 2.0) * height
    abs_x2 = (cx_norm + w_norm / 2.0) * width
    abs_y2 = (cy_norm + h_norm / 2.0) * height

    # Cắt phần giao giữa bbox PPE và vùng crop người
    clip_x1 = max(abs_x1, float(crop_x1))
    clip_y1 = max(abs_y1, float(crop_y1))
    clip_x2 = min(abs_x2, float(crop_x2))
    clip_y2 = min(abs_y2, float(crop_y2))

    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return None  # PPE nằm ngoài vùng crop

    # Đưa về tọa độ tương đối của crop [0..1]
    new_cx = ((clip_x1 + clip_x2) / 2.0 - crop_x1) / crop_w
    new_cy = ((clip_y1 + clip_y2) / 2.0 - crop_y1) / crop_h
    new_w = (clip_x2 - clip_x1) / crop_w
    new_h = (clip_y2 - clip_y1) / crop_h

    return [new_cx, new_cy, new_w, new_h]


def create_person_crops_dataset(
    manifest_csv: Path,
    output_dir: Path,
    padding: int = 10,
) -> int:
    """Cắt ảnh worker từ ảnh gốc và ánh xạ lại nhãn cho từng split."""
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy manifest file: {manifest_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_builder = PersonCropBuilder(padding=padding)

    with manifest_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    written = 0
    for sample in samples:
        img_path = Path(sample["image_path"])
        split = sample.get("split", "train")
        person_box = [
            float(sample["person_x1"]),
            float(sample["person_y1"]),
            float(sample["person_x2"]),
            float(sample["person_y2"]),
        ]

        img = read_image(img_path)
        if img is None:
            continue

        try:
            cropped, window = crop_builder.crop(img, person_box)
        except ValueError:
            continue

        crop_name = f"{img_path.stem}_worker_{written}.jpg"
        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        out_img_path = out_img_dir / crop_name
        write_image(out_img_path, cropped)

        # Đọc nhãn gốc và chuyển đổi
        label_path = Path(sample["label_path"])
        remapped_lines: list[str] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    new_box = remap_ppe_box_to_crop(
                        coords,
                        img.shape[:2],
                        (window.x1, window.y1, window.x2, window.y2),
                    )
                    if new_box is not None:
                        b = new_box
                        remapped_lines.append(
                            f"{cls_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                        )

        out_lbl_path = out_lbl_dir / f"{out_img_path.stem}.txt"
        out_lbl_path.write_text("\n".join(remapped_lines), encoding="utf-8")
        written += 1

    LOGGER.info("Đã tạo thành công %d mẫu person crop tại %s", written, output_dir)
    return written


def create_data_yaml(dataset_dir: Path, output_yaml: Path) -> None:
    """Tạo file cấu hình data.yaml cho Ultralytics YOLO."""
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": PPE_CLASSES,
    }
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    LOGGER.info("Đã lưu cấu hình dataset tại: %s", output_yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý và chuẩn bị dataset PPE")
    parser.add_argument("--manifest", type=str, help="Đường dẫn file manifest CSV gốc")
    parser.add_argument(
        "--output-dir", type=str, default="data/dataset_ppe", help="Thư mục xuất dataset"
    )
    parser.add_argument("--padding", type=int, default=10, help="Padding cho Person ROI (pixel)")
    parser.add_argument(
        "--data-yaml", type=str, default="training/data.yaml", help="Đường dẫn file data.yaml"
    )
    args = parser.parse_args()

    if args.manifest:
        create_person_crops_dataset(Path(args.manifest), Path(args.output_dir), args.padding)
    create_data_yaml(Path(args.output_dir), Path(args.data_yaml))


if __name__ == "__main__":
    main()
