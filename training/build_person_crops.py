"""Tạo stage-2 person crop sau khi split đã được khóa.

Manifest đầu vào cần thêm ``person_x1``, ``person_y1``, ``person_x2``,
``person_y2`` và label YOLO của ảnh gốc. Script dùng cùng
``PersonCropBuilder`` với serving, remap label sang crop và không tự random
split.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

# Cho phép chạy script trực tiếp từ thư mục repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe_detection.crops import PersonCropBuilder  # noqa: E402

PPE_CLASS_IDS = {"0", "1", "2", "3"}


def _remap_yolo_labels(
    label_path: Path,
    image_shape: tuple[int, ...],
    crop_window: tuple[int, int, int, int],
) -> list[str]:
    """Đổi label YOLO từ ảnh gốc sang hệ tọa độ crop."""
    height, width = image_shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_window
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    remapped: list[str] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Label YOLO sai định dạng tại {label_path}:{line_number}")
        class_id = values[0]
        if class_id not in PPE_CLASS_IDS:
            raise ValueError(f"Class PPE không hợp lệ tại {label_path}:{line_number}")
        center_x, center_y, box_width, box_height = map(float, values[1:])
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
            raise ValueError(f"Label YOLO ngoài khoảng [0,1] tại {label_path}:{line_number}")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"Label YOLO có kích thước bằng 0 tại {label_path}:{line_number}")

        absolute = (
            (center_x - box_width / 2.0) * width,
            (center_y - box_height / 2.0) * height,
            (center_x + box_width / 2.0) * width,
            (center_y + box_height / 2.0) * height,
        )
        clipped = (
            max(absolute[0], crop_x1),
            max(absolute[1], crop_y1),
            min(absolute[2], crop_x2),
            min(absolute[3], crop_y2),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        local_center_x = ((clipped[0] + clipped[2]) / 2.0 - crop_x1) / crop_width
        local_center_y = ((clipped[1] + clipped[3]) / 2.0 - crop_y1) / crop_height
        local_width = (clipped[2] - clipped[0]) / crop_width
        local_height = (clipped[3] - clipped[1]) / crop_height
        remapped.append(
            f"{class_id} {local_center_x:.6f} {local_center_y:.6f} "
            f"{local_width:.6f} {local_height:.6f}"
        )
    return remapped


def build_person_crops(manifest_path: Path, output_dir: Path, padding: int = 10) -> int:
    """Cắt crop theo split có sẵn trong manifest."""
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "sample_id",
            "image_path",
            "label_path",
            "split",
            "person_x1",
            "person_y1",
            "person_x2",
            "person_y2",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest thiếu cột person bbox: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest không có mẫu nào.")

    builder = PersonCropBuilder(padding)
    written = 0
    seen_sample_ids: set[str] = set()
    split_images: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in rows:
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError("sample_id không được rỗng.")
        if sample_id in seen_sample_ids:
            raise ValueError(f"sample_id bị trùng trong manifest: {sample_id}")
        seen_sample_ids.add(sample_id)

        target_split = row["split"].strip().lower()
        if target_split not in {"train", "val", "test"}:
            raise ValueError(f"Split không hợp lệ: {row['split']}")

        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        label_path = Path(row["label_path"])
        if not label_path.is_absolute():
            label_path = (manifest_path.parent / label_path).resolve()
        if not label_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy label: {label_path}")
        person_box = [
            float(row["person_x1"]),
            float(row["person_y1"]),
            float(row["person_x2"]),
            float(row["person_y2"]),
        ]
        crop, window = builder.crop(image, person_box)
        image_dir = output_dir / "images" / target_split
        label_dir = output_dir / "labels" / target_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target_path = image_dir / f"{sample_id}.jpg"
        target_label_path = label_dir / f"{sample_id}.txt"
        remapped_labels = _remap_yolo_labels(
            label_path,
            image.shape,
            (window.x1, window.y1, window.x2, window.y2),
        )
        if not cv2.imwrite(str(target_path), crop):
            raise OSError(f"Không thể ghi crop: {target_path}")
        target_label_path.write_text(
            "\n".join(remapped_labels) + "\n",
            encoding="utf-8",
        )
        split_images[target_split].append(target_path.relative_to(output_dir).as_posix())
        written += 1

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, image_paths in split_images.items():
        manifest_path = manifest_dir / f"{split}.txt"
        content = "\n".join(sorted(image_paths))
        manifest_path.write_text(f"{content}\n" if content else "", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo person crop sau group split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="data/dataset_ppe")
    parser.add_argument("--padding", type=int, default=10)
    args = parser.parse_args()
    count = build_person_crops(Path(args.manifest), Path(args.output_dir), args.padding)
    print(f"Đã tạo {count} person crops")


if __name__ == "__main__":
    main()
