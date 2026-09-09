"""Huấn luyện YOLO PPE và ghi metadata thí nghiệm."""

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
    config_path: Path, output_dir: Path, resolved_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Ghi metadata môi trường và cấu hình huấn luyện đã giải quyết."""
    import ultralytics

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_file": str(config_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "git_commit": get_git_commit(),
        "resolved_training_contract": resolved_cfg,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = output_dir / "experiment_env.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    resolved_cfg_file = output_dir / "resolved_config.yaml"
    with open(resolved_cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False, allow_unicode=True)

    LOGGER.info("Ghi nhận metadata thí nghiệm và cấu hình giải quyết tại: %s", output_dir)
    return metadata


def build_train_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Trích xuất và kiểm tra toàn bộ tham số huấn luyện thực tế từ Training Contract."""
    exp_name = cfg.get("experiment_name", "yolov8n_ppe_exp")
    data_cfg = cfg.get("data_config", "training/data.yaml")
    epochs = int(cfg.get("epochs", 100))
    imgsz = int(cfg.get("image_size", 640))
    batch = int(cfg.get("batch_size", 16))
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

    # Chuyển tiếp các siêu tham số tối ưu hóa nếu được cấu hình
    for opt_param in (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
    ):
        if opt_param in cfg:
            train_kwargs[opt_param] = cfg[opt_param]

    device = cfg.get("device", "auto")
    if device != "auto":
        train_kwargs["device"] = device

    # Chuyển tiếp chính sách data augmentation chuyên biệt cho PPE
    augs = cfg.get("augmentations", {})
    if isinstance(augs, dict):
        for aug_name, aug_val in augs.items():
            train_kwargs[aug_name] = aug_val

    return train_kwargs


def train_model(config_file: str) -> None:
    """Thực thi huấn luyện YOLO dựa trên file cấu hình Training Contract YAML."""
    from ultralytics import YOLO

    cfg_path = Path(config_file)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_file}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    exp_name = cfg.get("experiment_name", "yolov8n_ppe_exp")
    output_dir = Path("runs") / "train" / exp_name
    record_environment_metadata(cfg_path, output_dir, cfg)

    model_type = cfg.get("model_type", "yolov8n.pt")
    LOGGER.info("Khởi tạo mô hình %s...", model_type)
    model = YOLO(model_type)

    train_kwargs = build_train_kwargs(cfg)
    LOGGER.info(
        "Bắt đầu huấn luyện mô hình [%s] trong %d epochs với tham số đã ánh xạ:",
        exp_name,
        train_kwargs["epochs"],
    )
    for k, v in train_kwargs.items():
        LOGGER.info("  - %s: %s", k, v)

    model.train(**train_kwargs)
    LOGGER.info("Hoàn tất huấn luyện!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script huấn luyện YOLO PPE")
    parser.add_argument(
        "--config",
        default="configs/train_yolov8n.yaml",
        help="Đường dẫn tới file cấu hình YAML Training Contract",
    )
    args = parser.parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
