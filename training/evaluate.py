"""Module đánh giá thống nhất (Unified Evaluation) cho Hệ thống Giám sát PPE.

Hỗ trợ 2 nhiệm vụ đánh giá cốt lõi:
1. Detection Task (`--task detection`):
   Đánh giá mô hình YOLO trên tập test: Precision, Recall, mAP50, mAP50-95
   chi tiết theo từng lớp ('helmet', 'no-helmet', 'vest', 'no-vest').

2. Event Task (`--task event`):
   Đánh giá sự kiện vi phạm an toàn theo thời gian (Spatio-Temporal Event Matching):
   - Event Precision, Event Recall, F1-Score.
   - Tần suất cảnh báo sai (False Alerts / Hour).
   - Độ trễ phát cảnh báo (Time-to-Alert).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LOGGER = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_detection(
    model_path: str,
    data_config: str = "training/data.yaml",
    split: str = "test",
) -> dict[str, Any]:
    """Đánh giá mô hình phát hiện YOLO trên tập validation hoặc test."""
    model_file = Path(model_path)
    data_file = Path(data_config)

    if not model_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file model: {model_file}")
    if not data_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset config: {data_file}")

    from ultralytics import YOLO

    LOGGER.info("Đánh giá mô hình YOLO [%s] trên tập [%s]...", model_path, split)
    model = YOLO(model_path)
    results = model.val(data=data_config, split=split, verbose=False)

    metrics: dict[str, Any] = {
        "model": str(model_file),
        "split": split,
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
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
            map_val = float(results.box.maps[idx]) if hasattr(results.box, "maps") else None
            metrics["class_metrics"][cls_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "mAP50_95": round(map_val, 4) if map_val is not None else None,
            }

    LOGGER.info(
        "Kết quả Detection: mAP50=%.4f, mAP50-95=%.4f, Precision=%.4f, Recall=%.4f",
        metrics["mAP50"],
        metrics["mAP50_95"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics


def evaluate_events(
    events_gt: list[dict[str, Any]],
    events_pred: list[dict[str, Any]],
    duration_hours: float = 1.0,
    time_tolerance_sec: float = 3.0,
    gt_to_pred_map: dict[object, object] | None = None,
) -> dict[str, Any]:
    """Đánh giá sự kiện vi phạm an toàn qua đối chiếu không gian - thời gian."""
    if duration_hours <= 0:
        raise ValueError("duration_hours phải lớn hơn 0.")

    identity_map = {str(k): str(v) for k, v in (gt_to_pred_map or {}).items()}
    tp = 0
    fp = 0
    matched_gt: set[int] = set()
    time_to_alerts: list[float] = []

    def get_interval(event: dict[str, Any]) -> tuple[float, float]:
        start = float(event.get("start_sec", event.get("time_seconds", 0.0)))
        end = float(event.get("end_sec", start))
        return start, end

    def get_alert_time(event: dict[str, Any]) -> float:
        return float(event.get("alert_time_seconds", event.get("time_seconds", 0.0)))

    preds_sorted = sorted(events_pred, key=get_alert_time)
    gts_sorted = sorted(events_gt, key=lambda x: get_interval(x)[0])

    for pred in preds_sorted:
        p_track = str(pred["track_id"])
        p_type = str(pred.get("violation_type", "")).strip().lower()
        p_alert = get_alert_time(pred)
        matched = False

        for idx, gt in enumerate(gts_sorted):
            if idx in matched_gt:
                continue

            gt_track = str(gt["track_id"])
            # Kiểm tra định danh ID (nếu có mapping từ tracker)
            expected_pred_id = identity_map.get(gt_track, gt_track)
            if expected_pred_id != p_track:
                continue

            gt_type = str(gt.get("violation_type", "")).strip().lower()
            if gt_type != p_type:
                continue

            gt_start, gt_end = get_interval(gt)
            # Khớp thời gian trong dung sai cho phép
            if (gt_start - time_tolerance_sec) <= p_alert <= (gt_end + time_tolerance_sec):
                matched = True
                matched_gt.add(idx)
                tp += 1
                latency = max(0.0, p_alert - gt_start)
                time_to_alerts.append(latency)
                break

        if not matched:
            fp += 1

    fn = len(events_gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    median_latency = sorted(time_to_alerts)[len(time_to_alerts) // 2] if time_to_alerts else 0.0
    false_alerts_per_hour = fp / duration_hours

    report = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "false_alerts_per_hour": round(false_alerts_per_hour, 2),
        "median_time_to_alert_sec": round(median_latency, 2),
    }

    LOGGER.info(
        "Kết quả Event Evaluation: Precision=%.4f, Recall=%.4f, F1=%.4f, False Alerts/h=%.2f",
        report["event_precision"],
        report["event_recall"],
        report["event_f1"],
        report["false_alerts_per_hour"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình và sự kiện vi phạm PPE")
    parser.add_argument(
        "--task",
        choices=["detection", "event"],
        default="detection",
        help="Nhiệm vụ đánh giá: 'detection' hoặc 'event'",
    )
    # Tham số cho detection
    parser.add_argument("--model", default="models/best.pt", help="Đường dẫn model weights YOLO")
    parser.add_argument(
        "--data", default="training/data.yaml", help="File cấu hình dataset data.yaml"
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"], help="Tập dữ liệu đánh giá"
    )

    # Tham số cho event
    parser.add_argument("--gt-events", help="Đường dẫn file JSON chứa ground-truth events")
    parser.add_argument("--pred-events", help="Đường dẫn file JSON chứa predicted events")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Tổng số giờ video")
    parser.add_argument("--output", default="runs/eval_results.json", help="File lưu kết quả JSON")

    args = parser.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task == "detection":
        report = evaluate_detection(args.model, args.data, args.split)
    else:
        if not args.gt_events or not args.pred_events:
            parser.error("Đánh giá event yêu cầu cả --gt-events và --pred-events")
        with open(args.gt_events, encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(args.pred_events, encoding="utf-8") as f:
            pred_data = json.load(f)
        report = evaluate_events(gt_data, pred_data, duration_hours=args.duration_hours)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    LOGGER.info("Báo cáo đánh giá đã được lưu tại: %s", out_path)


if __name__ == "__main__":
    main()
