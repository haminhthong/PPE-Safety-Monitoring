"""Xây manifest từ ảnh và annotation thật.

Input là một CSV metadata do người thu thập dữ liệu cung cấp. Script không tự
sinh sample, split, SHA hay pHash. Split phải được gán ở cấp recording session
trước khi tạo person crop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

REQUIRED_COLUMNS = {
    "sample_id",
    "parent_video_id",
    "frame_id",
    "timestamp_ms",
    "camera_id",
    "site_id",
    "recording_session",
    "image_path",
    "label_path",
    "class_labels",
    "split",
    "source_id",
    "annotation_version",
}
OUTPUT_COLUMNS = [
    "sample_id",
    "parent_video_id",
    "frame_id",
    "timestamp_ms",
    "camera_id",
    "site_id",
    "recording_session",
    "image_path",
    "label_path",
    "class_labels",
    "split",
    "source_id",
    "annotation_version",
    "file_sha256",
    "image_phash",
]


def sha256_file(path: Path) -> str:
    """Tính SHA-256 trên bytes thật của file ảnh."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_phash(path: Path) -> str:
    """Tính pHash trên ảnh đã decode, không hash metadata hay tên file."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không thể decode ảnh: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    values = dct
    median = float(np.median(values))
    bits = "".join("1" if value > median else "0" for value in values.flat)
    return f"{int(bits, 2):0{len(bits) // 4}x}"


def _resolve(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _validate_groups(rows: list[dict[str, str]]) -> None:
    session_splits: dict[str, set[str]] = {}
    for row in rows:
        session = row["recording_session"]
        split = row["split"].strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split không hợp lệ: {split}")
        if not session:
            raise ValueError(f"recording_session rỗng ở sample {row['sample_id']}")
        session_splits.setdefault(session, set()).add(split)
    leaked = {session: splits for session, splits in session_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Recording session bị chia chéo split: {leaked}")


def build_manifest(metadata_path: Path, output_path: Path) -> int:
    """Đọc metadata, xác minh file thật và ghi manifest chuẩn."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Metadata thiếu cột: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Metadata không có mẫu nào.")
    _validate_groups(rows)

    output_rows: list[dict[str, str]] = []
    seen_sha: dict[str, str] = {}
    for row in rows:
        for column in REQUIRED_COLUMNS - {"class_labels"}:
            if not row[column].strip():
                raise ValueError(f"Metadata có cột {column} rỗng ở sample {row['sample_id']}")
        image_path = _resolve(row["image_path"], metadata_path.parent)
        label_path = _resolve(row["label_path"], metadata_path.parent)
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy image_path: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy label_path: {label_path}")
        file_sha = sha256_file(image_path)
        duplicate = seen_sha.get(file_sha)
        if duplicate:
            raise ValueError(f"Ảnh trùng SHA: {row['sample_id']} trùng {duplicate}")
        seen_sha[file_sha] = row["sample_id"]

        output_row = {key: row[key].strip() for key in REQUIRED_COLUMNS}
        output_row["image_path"] = os.path.relpath(image_path, output_path.parent.resolve())
        output_row["label_path"] = os.path.relpath(label_path, output_path.parent.resolve())
        output_row["file_sha256"] = file_sha
        output_row["image_phash"] = image_phash(image_path)
        output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xây manifest từ dữ liệu ảnh/annotation thật")
    parser.add_argument("--metadata", required=True, help="CSV metadata có split theo session")
    parser.add_argument("--output", default="data/manifests/dataset.csv")
    args = parser.parse_args()
    count = build_manifest(Path(args.metadata), Path(args.output))
    print(f"Đã ghi {count} mẫu vào {args.output}")


if __name__ == "__main__":
    main()
