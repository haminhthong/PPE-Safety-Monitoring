"""Điểm truy cập đánh giá đa cấp độ cho PPE Safety Surveillance Platform.

Hỗ trợ đánh giá theo 4 tầng kiến trúc:
1. Detection Level (Layer 1 & 2): Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
2. Tracking Level (Layer 3): IDF1, ID switches, Track fragmentation, MOTA
3. Event Level (Layer 4): Event Precision, Event Recall, False alerts/hour, Time-to-alert
4. System Level: Stage-wise funnel và Decision policy ablation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from .evaluate_detector import evaluate_person_detector, evaluate_ppe_detector
    from .evaluate_events import evaluate_violation_events
    from .evaluate_system import run_full_system_evaluation
    from .evaluate_tracking import evaluate_tracking_trajectories
except (ImportError, ValueError):
    from evaluate_detector import evaluate_person_detector, evaluate_ppe_detector
    from evaluate_events import evaluate_violation_events
    from evaluate_system import run_full_system_evaluation
    from evaluate_tracking import evaluate_tracking_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PPE đa cấp độ")
    parser.add_argument(
        "--layer",
        choices=["detector", "tracking", "events", "system"],
        default="system",
        help="Tầng đánh giá cần thực thi (detector, tracking, events, system)",
    )
    parser.add_argument("--model", default="models/best.pt", help="Đường dẫn file trọng số model PPE")
    parser.add_argument("--person-model", default="models/yolov8n.pt", help="Đường dẫn file trọng số Person")
    parser.add_argument("--data", default="training/data.yaml", help="File cấu hình dataset YAML")
    parser.add_argument("--split", default="test", help="Tập dữ liệu đánh giá (val hoặc test)")
    parser.add_argument("--demo", action="store_true", help="Ghi metadata demo; không tạo benchmark giả")
    parser.add_argument(
        "--output", default="runs/eval_results.json", help="File lưu kết quả báo cáo JSON"
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.layer == "detector":
        report = {
            "layer_1_ppe_detector": evaluate_ppe_detector(args.model, args.data, args.split, demo=args.demo),
            "layer_2_person_detector": evaluate_person_detector(
                args.person_model, args.split, demo=args.demo, data_config=args.data
            ),
        }
    elif args.layer == "tracking":
        parser.error("Tầng tracking cần được gọi bằng training/evaluate_tracking.py với GT và prediction.")
    elif args.layer == "events":
        parser.error("Tầng events cần được gọi bằng training/evaluate_events.py với GT và prediction.")
    else:
        report = run_full_system_evaluation(demo=args.demo)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    LOGGER.info("Đánh giá tầng [%s] hoàn tất. Kết quả lưu tại: %s", args.layer, out_path)


if __name__ == "__main__":
    main()
