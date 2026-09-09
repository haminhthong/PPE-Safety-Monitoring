"""Đánh giá tầng theo dõi đối tượng (Layer 3: Multi-Object Tracking Evaluation).

Tính toán các chỉ số chuẩn của bài toán Tracking:
- ID Switches (Số lần đổi định danh trên cùng một người thực tế).
- Track Fragmentation (Số lần vết bị đứt đoạn).
- IDF1 (Identification F1-score: Độ nhất quán danh tính xuyên suốt quỹ đạo).
- MOTA (Multi-Object Tracking Accuracy).
- Mostly Tracked (MT) & Mostly Lost (ML) ratios.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate_tracking")


def _to_mot_box(box: list[float]) -> list[float]:
    """Đổi box nội bộ ``[x1, y1, x2, y2]`` sang format MOT ``[x, y, w, h]``."""
    if len(box) != 4:
        raise ValueError(f"Bounding box phải có 4 giá trị: {box}")
    x1, y1, x2, y2 = map(float, box)
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Bounding box phải có kích thước dương: {box}")
    return [x1, y1, width, height]


def evaluate_tracking_trajectories(
    gt_trajectories: list[dict[str, Any]],
    pred_trajectories: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Tính metric tracking thật bằng ``motmetrics``.

    Không có annotation trajectory thì metric là không khả dụng và hàm sẽ
    dừng thay vì trả số liệu minh họa.
    """
    if not gt_trajectories:
        raise ValueError("Tracking evaluation bắt buộc cần GT trajectories thật.")

    if pred_trajectories is None:
        raise ValueError("Tracking evaluation bắt buộc cần predicted trajectories.")

    try:
        import motmetrics as mm
    except ImportError as error:
        raise RuntimeError(
            "Thiếu motmetrics. Hãy cài requirements-dev.txt để đánh giá tracking chuẩn."
        ) from error

    # Gom nhóm theo frame_id để tạo accumulator chuẩn MOT.
    gt_by_frame: dict[int, list[dict]] = defaultdict(list)
    pred_by_frame: dict[int, list[dict]] = defaultdict(list)

    for item in gt_trajectories:
        gt_by_frame[item["frame_id"]].append(item)
    for item in pred_trajectories:
        pred_by_frame[item["frame_id"]].append(item)

    accumulator = mm.MOTAccumulator(auto_id=True)
    for fid in sorted(set(gt_by_frame) | set(pred_by_frame)):
        curr_gts = gt_by_frame[fid]
        curr_preds = pred_by_frame[fid]
        gt_ids = [item["track_id"] for item in curr_gts]
        pred_ids = [item["track_id"] for item in curr_preds]
        gt_boxes = [_to_mot_box(item["box"]) for item in curr_gts]
        pred_boxes = [_to_mot_box(item["box"]) for item in curr_preds]
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        accumulator.update(gt_ids, pred_ids, distances, frameid=fid)

    metrics = ["idf1", "id_switches", "fragmentations", "mota", "mostly_tracked", "mostly_lost"]
    summary = mm.metrics.create().compute(accumulator, metrics=metrics, name="evaluation")
    values = summary.loc["evaluation"]

    def metric_float(value: Any) -> float:
        number = float(value)
        return number if math.isfinite(number) else 0.0

    gt_count = len({item["track_id"] for item in gt_trajectories})
    return {
        "metrics_available": True,
        "idf1": round(metric_float(values["idf1"]), 4),
        "id_switches": int(values["id_switches"]),
        "fragmentations": int(values["fragmentations"]),
        "mota": round(metric_float(values["mota"]), 4),
        "mostly_tracked_ratio": round(metric_float(values["mostly_tracked"]) / max(gt_count, 1), 4),
        "mostly_lost_ratio": round(metric_float(values["mostly_lost"]) / max(gt_count, 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Layer 3: Tracking Evaluation")
    parser.add_argument("--gt-tracks", default="", help="File JSON chứa quỹ đạo Ground-Truth")
    parser.add_argument("--pred-tracks", default="", help="File JSON chứa quỹ đạo Predicted")
    parser.add_argument("--output", default="runs/eval_tracking.json", help="File lưu báo cáo JSON")
    args = parser.parse_args()

    if not args.gt_tracks or not args.pred_tracks:
        parser.error("--gt-tracks và --pred-tracks là bắt buộc; không có fallback demo.")
    gt_path = Path(args.gt_tracks)
    pred_path = Path(args.pred_tracks)
    if not gt_path.is_file() or not pred_path.is_file():
        parser.error("Không tìm thấy file GT hoặc predicted trajectories.")
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    pred_data = json.loads(pred_path.read_text(encoding="utf-8"))

    metrics = evaluate_tracking_trajectories(gt_data, pred_data)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    LOGGER.info("Tracking Evaluation Metrics: %s", metrics)


if __name__ == "__main__":
    main()
