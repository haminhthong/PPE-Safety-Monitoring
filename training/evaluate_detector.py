"""Đánh giá mô hình phát hiện (Layer 1: PPE Detector & Layer 2: Person Detector).

Chỉ số tính toán:
Layer 1: PPE Detection (mAP50, mAP50-95, per-class Precision, Recall, F1)
Layer 2: Person Detection (Precision, Recall, mAP50, mAP50-95)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate_detector")


def _validate_split(split: str) -> None:
    """Chỉ cho phép đánh giá trên validation hoặc locked test."""
    if split not in {"val", "test"}:
        raise ValueError("split phải là val hoặc test.")


def evaluate_ppe_detector(
    model_path: str,
    data_config: str = "training/data.yaml",
    split: str = "test",
    demo: bool = False,
) -> dict[str, Any]:
    """Đánh giá Layer 1: Hiệu năng phát hiện 4 lớp trang bị bảo hộ PPE."""
    _validate_split(split)
    if demo:
        return {
            "synthetic_demo": True,
            "metrics_available": False,
            "model": model_path,
            "split": split,
        }
    model_file = Path(model_path)
    if not data_config:
        raise FileNotFoundError("PPE evaluation bắt buộc cần --ppe-data chứa annotation PPE.")
    data_file = Path(data_config)
    if not model_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy PPE weights: {model_file}")
    if not data_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset config: {data_file}")

    from ultralytics import YOLO

    LOGGER.info("Đánh giá PPE Detection cho [%s] trên split [%s]...", model_path, split)
    model = YOLO(model_path)
    results = model.val(data=data_config, split=split, verbose=False)

    metrics: dict[str, Any] = {
        "model": model_path,
        "split": split,
        "metrics_available": True,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "mp": float(results.box.mp),
        "mr": float(results.box.mr),
        "class_metrics": {},
    }

    class_names = (
        results.names.items() if isinstance(results.names, dict) else enumerate(results.names)
    )
    for idx, cls_name in class_names:
        if idx < len(results.box.p):
            p = float(results.box.p[idx])
            r = float(results.box.r[idx])
            f1 = 2 * p * r / (p + r + 1e-6)
            map50_95 = float(results.box.maps[idx]) if hasattr(results.box, "maps") else None
            metrics["class_metrics"][cls_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mAP50_95": round(map50_95, 4) if map50_95 is not None else None,
            }

    return metrics


def evaluate_person_detector(
    person_model_path: str,
    split: str = "test",
    demo: bool = False,
    person_data_config: str | None = None,
) -> dict[str, Any]:
    """Đánh giá Layer 2: Khả năng phát hiện người lao động trong bối cảnh công trường."""
    if demo:
        return {
            "synthetic_demo": True,
            "metrics_available": False,
            "model": person_model_path,
            "split": split,
        }
    model_file = Path(person_model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy Person weights: {model_file}")
    _validate_split(split)
    if not person_data_config or not Path(person_data_config).is_file():
        raise FileNotFoundError(
            "Person evaluation bắt buộc cần --person-data chứa annotation full-frame."
        )

    from ultralytics import YOLO

    model = YOLO(str(model_file))
    results = model.val(data=person_data_config, split=split, classes=[0], verbose=False)
    box = results.box
    return {
        "model": str(model_file),
        "split": split,
        "metrics_available": True,
        "person_precision": round(float(box.mp), 4),
        "person_recall": round(float(box.mr), 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Layer 1 & 2: PPE và Person Detector")
    parser.add_argument(
        "--ppe-model", default="models/best.pt", help="Đường dẫn trọng số PPE model"
    )
    parser.add_argument(
        "--person-model", default="models/yolov8n.pt", help="Đường dẫn Person model"
    )
    parser.add_argument(
        "--ppe-data", default="training/data.yaml", help="Dataset config cho PPE person-crop"
    )
    parser.add_argument(
        "--person-data", required=False, help="Dataset config full-frame có class person"
    )
    parser.add_argument("--split", default="test", help="val hoặc test")
    parser.add_argument("--demo", action="store_true", help="Chạy chế độ giả lập benchmark")
    parser.add_argument("--output", default="runs/eval_detector.json", help="File lưu kết quả")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ppe_res = evaluate_ppe_detector(args.ppe_model, args.ppe_data, args.split, demo=args.demo)
    person_res = evaluate_person_detector(
        args.person_model,
        args.split,
        demo=args.demo,
        person_data_config=args.person_data,
    )

    report = {
        "layer_1_ppe_detector": ppe_res,
        "layer_2_person_detector": person_res,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    LOGGER.info("Đã ghi báo cáo đánh giá Detector tại: %s", out_path)


if __name__ == "__main__":
    main()
