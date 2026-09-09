"""Kiểm thử Training Contract và quy trình manifest từ dữ liệu thật."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.audit_dataset import audit_dataset
from training.build_manifest import build_manifest
from training.build_person_crops import _remap_yolo_labels
from training.train import build_train_kwargs


def test_training_config_fields_are_forwarded() -> None:
    """Đảm bảo train.py chuyển tiếp đủ siêu tham số trong config."""
    sample_cfg = {
        "experiment_name": "test_exp",
        "data_config": "training/data.yaml",
        "epochs": 50,
        "image_size": 640,
        "batch_size": 8,
        "seed": 123,
        "optimizer": "AdamW",
        "lr0": 0.005,
        "lrf": 0.001,
        "momentum": 0.9,
        "augmentations": {
            "hsv_h": 0.02,
            "mosaic": 0.5,
        },
    }

    train_kwargs = build_train_kwargs(sample_cfg)

    assert train_kwargs["optimizer"] == "AdamW"
    assert train_kwargs["lr0"] == 0.005
    assert train_kwargs["lrf"] == 0.001
    assert train_kwargs["epochs"] == 50
    assert train_kwargs["hsv_h"] == 0.02
    assert train_kwargs["mosaic"] == 0.5


def test_manifest_builder_uses_real_file_hashes_and_group_split(tmp_path: Path) -> None:
    """Manifest phải hash file thật và từ chối session bị chia chéo split."""
    image_path = tmp_path / "frame.jpg"
    label_path = tmp_path / "frame.txt"
    cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
    label_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata.csv"
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
    rows = [
        {
            "sample_id": "frame-1",
            "parent_video_id": "video-1",
            "frame_id": "1",
            "timestamp_ms": "0",
            "camera_id": "cam-1",
            "site_id": "site-1",
            "recording_session": "session-1",
            "image_path": image_path.name,
            "label_path": label_path.name,
            "class_labels": "helmet",
            "split": "train",
            "source_id": "fixture",
            "annotation_version": "1.0",
        }
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = tmp_path / "dataset.csv"
    assert build_manifest(metadata_path, manifest_path) == 1
    with manifest_path.open(encoding="utf-8", newline="") as file:
        manifest_row = next(csv.DictReader(file))
    assert len(manifest_row["file_sha256"]) == 64
    assert manifest_row["image_phash"]
    audit_report = audit_dataset(manifest_path)
    assert audit_report["evidence_status"] == "measured"

    rows.append({**rows[0], "sample_id": "frame-2", "split": "val"})
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Recording session bị chia chéo split"):
        build_manifest(metadata_path, tmp_path / "invalid.csv")


def test_person_crop_label_remapping_uses_crop_coordinates(tmp_path: Path) -> None:
    """Label PPE phải được đổi từ tọa độ ảnh gốc sang tọa độ person crop."""
    label_path = tmp_path / "frame.txt"
    label_path.write_text("0 0.4 0.5 0.2 0.2\n", encoding="utf-8")
    labels = _remap_yolo_labels(label_path, (100, 100, 3), (10, 20, 60, 80))
    assert labels == ["0 0.600000 0.500000 0.400000 0.333333"]
