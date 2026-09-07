"""Đánh giá tầng phát hiện sự kiện vi phạm an toàn (Layer 4: Violation Event Evaluation).

Sử dụng cơ chế ghép cặp không-thời gian (Spatio-Temporal Trajectory Matching) thay vì so sánh
bằng ID cơ học thô sơ (vì tracker tự động sinh ID dự đoán độc lập với GT).

Chỉ số tính toán:
- Event Precision & Event Recall.
- Event F1-Score.
- False Alerts per Hour (Tần suất cảnh báo sai mỗi giờ hoạt động).
- Median Time-to-Alert (Độ trễ thời gian từ khi vi phạm thực tế đến lúc hệ thống cảnh báo).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate_events")


def evaluate_violation_events(
    events_gt: list[dict[str, Any]],
    events_pred: list[dict[str, Any]],
    duration_hours: float = 1.0,
    time_tolerance_sec: float = 3.0,
    gt_to_pred_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Đánh giá event interval sau khi identity đã được map từ tracking.

    GT dùng ``start_sec``/``end_sec``; ``time_seconds`` vẫn được chấp nhận
    như point event để tương thích dữ liệu cũ. Không có mapping thì chỉ các
    ID trùng nhau mới được ghép, tuyệt đối không bỏ qua identity check.
    """
    if duration_hours <= 0:
        raise ValueError("duration_hours phải lớn hơn 0.")
    tp = 0
    fp = 0
    matched_gt: set[int] = set()
    time_to_alerts: list[float] = []

    def interval(event: dict[str, Any]) -> tuple[float, float]:
        start = float(
            event.get("start_sec", event.get("event_start_seconds", event.get("time_seconds", 0.0)))
        )
        end = float(event.get("end_sec", event.get("event_end_seconds", start)))
        if end < start:
            raise ValueError(f"Event có end trước start: {event}")
        return start, end

    def alert_time(event: dict[str, Any]) -> float:
        return float(
            event.get(
                "alert_time_sec",
                event.get(
                    "alert_time_seconds",
                    event.get(
                        "alert_time",
                        event.get(
                            "time_seconds",
                            event.get("start_sec", event.get("event_start_seconds", 0.0)),
                        ),
                    ),
                ),
            )
        )

    preds_sorted = sorted(events_pred, key=alert_time)
    gts_sorted = sorted(events_gt, key=lambda x: interval(x)[0])

    for pred in preds_sorted:
        p_track = pred.get("track_id")
        p_type = pred.get("violation_type", "").lower()

        matched_idx = None
        min_time_diff = float("inf")

        for idx, gt in enumerate(gts_sorted):
            if idx in matched_gt:
                continue

            g_track = gt.get("track_id")
            g_type = gt.get("violation_type", "").lower()
            g_start, g_end = interval(gt)

            # Kiểm tra loại vi phạm
            if p_type != g_type:
                continue

            expected_pred_track = gt_to_pred_map.get(g_track) if gt_to_pred_map else g_track
            if expected_pred_track != p_track:
                continue

            p_time = alert_time(pred)
            if g_start - time_tolerance_sec <= p_time <= g_end + time_tolerance_sec:
                dt = max(0.0, p_time - g_start)
                if dt < min_time_diff:
                    min_time_diff = dt
                    matched_idx = idx

        if matched_idx is not None:
            tp += 1
            matched_gt.add(matched_idx)
            time_to_alerts.append(min_time_diff)
        else:
            fp += 1

    fn = len(gts_sorted) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    far_per_hour = fp / max(0.001, duration_hours)
    if time_to_alerts:
        ordered_tta = sorted(time_to_alerts)
        middle = len(ordered_tta) // 2
        median_tta = (
            ordered_tta[middle]
            if len(ordered_tta) % 2
            else (ordered_tta[middle - 1] + ordered_tta[middle]) / 2.0
        )
    else:
        median_tta = 0.0

    return {
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "false_alerts_per_hour": round(far_per_hour, 2),
        "median_time_to_alert_sec": round(median_tta, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": len(gts_sorted),
        "total_pred_events": len(preds_sorted),
        "time_tolerance_sec": time_tolerance_sec,
        "ground_truth_format": "interval",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Layer 4: Violation Events")
    parser.add_argument("--gt-events", default="", help="File JSON danh sách sự kiện Ground Truth")
    parser.add_argument("--pred-events", default="", help="File JSON danh sách sự kiện Predicted")
    parser.add_argument("--gt-to-pred-map", help="JSON object ánh xạ GT track ID sang predicted track ID")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Thời lượng video (giờ)")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Dung sai thời gian (giây)")
    parser.add_argument("--output", default="runs/eval_events.json", help="File lưu báo cáo JSON")
    args = parser.parse_args()

    if not args.gt_events or not args.pred_events:
        parser.error("--gt-events và --pred-events là bắt buộc; không có fallback demo.")
    gt_path = Path(args.gt_events)
    pred_path = Path(args.pred_events)
    if not gt_path.is_file() or not pred_path.is_file():
        parser.error("Không tìm thấy file GT hoặc predicted events.")
    gt_events = json.loads(gt_path.read_text(encoding="utf-8"))
    pred_events = json.loads(pred_path.read_text(encoding="utf-8"))
    id_map = None
    if args.gt_to_pred_map:
        map_path = Path(args.gt_to_pred_map)
        if not map_path.is_file():
            parser.error(f"Không tìm thấy identity map: {map_path}")
        raw_map = json.loads(map_path.read_text(encoding="utf-8"))
        if not isinstance(raw_map, dict):
            parser.error("Identity map phải là JSON object.")
        id_map = {int(key): int(value) for key, value in raw_map.items()}

    metrics = evaluate_violation_events(
        gt_events,
        pred_events,
        duration_hours=args.duration_hours,
        time_tolerance_sec=args.tolerance,
        gt_to_pred_map=id_map,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    LOGGER.info("Báo cáo Layer 4 Events: %s", metrics)


if __name__ == "__main__":
    main()
