"""Fixture manifest tổng hợp cho unit test; không dùng cho production/evaluation.

Fixture cố ý không chứa SHA/pHash của ảnh thật để tránh bị hiểu nhầm là
evidence dataset. Mọi report dùng fixture phải gắn ``synthetic_demo=true``.
"""

from __future__ import annotations

import csv
from pathlib import Path


def generate_synthetic_manifest(output_path: Path, count: int = 4) -> None:
    """Tạo metadata nhỏ phục vụ smoke test."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "sample_id": f"synthetic_{index:04d}",
                    "parent_video_id": "synthetic_video",
                    "frame_id": str(index),
                    "timestamp_ms": str(index * 100),
                    "camera_id": "synthetic_camera",
                    "site_id": "synthetic_site",
                    "recording_session": "synthetic_session",
                    "image_path": "sample.jpg",
                    "label_path": "sample.txt",
                    "class_labels": "helmet;vest",
                    "split": "train",
                    "source_id": "fixture",
                    "annotation_version": "fixture-v1",
                }
            )


if __name__ == "__main__":
    generate_synthetic_manifest(Path("tests/fixtures/synthetic_manifest.csv"))
