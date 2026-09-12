"""Huấn luyện mô hình YOLO PPE và ghi nhận metadata thí nghiệm."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train")


def get_git_commit() -> str:
    """Lấy hash commit git hiện tại nếu có."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def record_environment_metadata(
    output_dir: Path, resolved_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Ghi metadata môi trường và cấu hình huấn luyện."""
    import ultralytics

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "git_commit": get_git_commit(),
        "training_config": resolved_cfg,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = output_dir / "experiment_env.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    resolved_cfg_file = output_dir / "resolved_config.yaml"
    with open(resolved_cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False, allow_unicode=True)

    LOGGER.info("Ghi nhận metadata thí nghiệm tại: %s", output_dir)
    return metadata


def build_train_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Trích xuất tham số huấn luyện cho YOLO."""
    exp_name = cfg.get("experiment_name", "yolov8n_ppe")
    data_cfg = cfg.get("data", "training/data.yaml")
    epochs = int(cfg.get("epochs", 50))
    imgsz = int(cfg.get("imgsz", cfg.get("image_size", 640)))
    batch = int(cfg.get("batch", cfg.get("batch_size", 16)))
    seed = int(cfg.get("seed", 42))

    train_kwargs: dict[str, Any] = {
        "data": data_cfg,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "name": exp_name,
        "project": "runs/train",
        "exist_ok": True,
    }

    for opt_param in (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
    ):
        if opt_param in cfg:
            train_kwargs[opt_param] = cfg[opt_param]

    device = cfg.get("device", "auto")
    if device != "auto":
        train_kwargs["device"] = device

    return train_kwargs


def train_model(
    data: str = "training/data.yaml",
    model_type: str = "yolov8n.pt",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    config_file: str | None = None,
) -> None:
    """Thực thi huấn luyện YOLO PPE."""
    from ultralytics import YOLO

    cfg: dict[str, Any] = {
        "data": data,
        "model_type": model_type,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
    }

    if config_file and Path(config_file).exists():
        with open(config_file, encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
            cfg.update(file_cfg)

    exp_name = cfg.get("experiment_name", "yolov8n_ppe")
    output_dir = Path("runs") / "train" / exp_name
    record_environment_metadata(output_dir, cfg)

    actual_model = cfg.get("model_type", model_type)
    LOGGER.info("Khởi tạo mô hình %s...", actual_model)
    model = YOLO(actual_model)

    train_kwargs = build_train_kwargs(cfg)
    LOGGER.info(
        "Bắt đầu huấn luyện mô hình [%s] trong %d epochs:",
        exp_name,
        train_kwargs["epochs"],
    )
    for k, v in train_kwargs.items():
        LOGGER.info("  - %s: %s", k, v)

    model.train(**train_kwargs)
    LOGGER.info("Hoàn tất huấn luyện!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script huấn luyện YOLO PPE")
    parser.add_argument("--data", default="training/data.yaml", help="Đường dẫn file data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Mô hình base (yolov8n.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--config", default=None, help="File cấu hình YAML (tùy chọn)")
    args = parser.parse_args()

    train_model(
        data=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        config_file=args.config,
    )


if __name__ == "__main__":
    main()
