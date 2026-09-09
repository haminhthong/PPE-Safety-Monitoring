"""Đánh giá toàn diện hệ thống 4 tầng (End-to-End System Evaluation & Policy Ablation).

Tổng hợp và kết xuất báo cáo thống nhất:
1. Stage-Wise Funnel (Phễu suy luận từng chặng: Person → Track → PPE → Event).
2. Decision Policy Ablation (Thử nghiệm bóc tách đóng góp của từng thành phần chính sách).
3. Pareto Frontier Analysis (So sánh đánh đổi giữa YOLOv8n và YOLOv8s về mAP, Recall và Latency).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    from .evaluate_detector import evaluate_person_detector, evaluate_ppe_detector
    from .evaluate_events import evaluate_violation_events
    from .evaluate_tracking import evaluate_tracking_trajectories
except (ImportError, ValueError):
    from evaluate_detector import evaluate_person_detector, evaluate_ppe_detector
    from evaluate_events import evaluate_violation_events
    from evaluate_tracking import evaluate_tracking_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("evaluate_system")


def calculate_stage_wise_funnel(stage_counts: dict[str, int]) -> dict[str, Any]:
    """Tính funnel từ số đếm trong locked test, không dựng số liệu mẫu."""
    if not stage_counts:
        raise ValueError("Cần stage_counts từ locked test để tính funnel.")
    stages = []
    previous = None
    first = next(iter(stage_counts.values()))
    if first <= 0:
        raise ValueError("Stage đầu tiên phải có ít nhất một mẫu.")
    for stage, count in stage_counts.items():
        if count < 0:
            raise ValueError("Stage count không được âm.")
        if previous is not None and count > previous:
            raise ValueError("Funnel phải không tăng giữa các stage.")
        stage_recall = count / previous if previous is not None else 1.0
        stages.append(
            {
                "stage": stage,
                "count": count,
                "stage_recall": round(stage_recall, 4),
                "cumulative_recall": round(count / first, 4),
            }
        )
        previous = count
    return {"stages": stages, "source": "locked_test_artifacts"}


def calculate_policy_ablation(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Đọc ablation đã chạy thật từ artifacts; không tự sinh benchmark."""
    if not reports:
        raise ValueError("Ablation report trống; không được xuất số liệu giả.")
    return reports


def calculate_pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Đọc kết quả model candidates từ validation artifacts."""
    if not candidates:
        raise ValueError("Chưa có validation artifacts để so sánh model.")
    return candidates


def run_full_system_evaluation(
    ppe_model: str | None = None,
    person_model: str | None = None,
    ppe_data_config: str | None = None,
    person_data_config: str | None = None,
    gt_tracks: list[dict[str, Any]] | None = None,
    pred_tracks: list[dict[str, Any]] | None = None,
    gt_events: list[dict[str, Any]] | None = None,
    pred_events: list[dict[str, Any]] | None = None,
    gt_to_pred_map: dict[object, object] | None = None,
    duration_hours: float | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    """Đánh giá end-to-end từ artifacts thật hoặc trả metadata demo rõ ràng."""
    if demo:
        return {"synthetic_demo": True, "metrics_available": False}
    required = {
        "ppe_model": ppe_model,
        "person_model": person_model,
        "ppe_data_config": ppe_data_config,
        "person_data_config": person_data_config,
        "gt_tracks": gt_tracks,
        "pred_tracks": pred_tracks,
        "gt_events": gt_events,
        "pred_events": pred_events,
        "duration_hours": duration_hours,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Final evaluation thiếu artifact bắt buộc: {', '.join(missing)}")
    return {
        "system_name": "PPE Safety Monitoring",
        "evaluation_protocol": "locked_test_full_video",
        "layer_1_and_2_perception": {
            "ppe_detector": evaluate_ppe_detector(ppe_model, ppe_data_config, "test"),
            "person_detector": evaluate_person_detector(
                person_model, "test", person_data_config=person_data_config
            ),
        },
        "layer_3_tracking": evaluate_tracking_trajectories(gt_tracks, pred_tracks),
        "layer_4_violation_events": evaluate_violation_events(
            gt_events,
            pred_events,
            duration_hours=duration_hours,
            gt_to_pred_map=gt_to_pred_map,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá toàn diện hệ thống PPE 4 tầng")
    parser.add_argument("--ppe-model", required=False, help="PPE weights đã freeze")
    parser.add_argument("--person-model", required=False, help="Person weights đã freeze")
    parser.add_argument("--ppe-data", required=False, help="Dataset config cho PPE person-crop")
    parser.add_argument("--person-data", required=False, help="Dataset config full-frame Person")
    parser.add_argument("--gt-tracks", help="JSON GT trajectories")
    parser.add_argument("--pred-tracks", help="JSON predicted trajectories")
    parser.add_argument("--gt-events", help="JSON GT interval events")
    parser.add_argument("--pred-events", help="JSON predicted events")
    parser.add_argument(
        "--gt-to-pred-map", help="JSON object ánh xạ GT track ID sang predicted track ID"
    )
    parser.add_argument("--duration-hours", type=float, help="Tổng thời lượng locked test")
    parser.add_argument(
        "--demo", action="store_true", help="Chỉ trả metadata demo, không phải benchmark"
    )
    parser.add_argument(
        "--output", default="runs/system_evaluation_report.json", help="File xuất báo cáo"
    )
    args = parser.parse_args()

    def load_json(path: str | None) -> list[dict[str, Any]] | None:
        if not path:
            return None
        file = Path(path)
        if not file.is_file():
            parser.error(f"Không tìm thấy artifact: {file}")
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            parser.error(f"Artifact phải là JSON array: {file}")
        return value

    def load_id_map(path: str | None) -> dict[str, str] | None:
        if not path:
            return None
        file = Path(path)
        if not file.is_file():
            parser.error(f"Không tìm thấy identity map: {file}")
        value = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            parser.error("Identity map phải là JSON object.")
        return {str(key): str(mapped) for key, mapped in value.items()}

    report = run_full_system_evaluation(
        ppe_model=args.ppe_model,
        person_model=args.person_model,
        ppe_data_config=args.ppe_data,
        person_data_config=args.person_data,
        gt_tracks=load_json(args.gt_tracks),
        pred_tracks=load_json(args.pred_tracks),
        gt_events=load_json(args.gt_events),
        pred_events=load_json(args.pred_events),
        gt_to_pred_map=load_id_map(args.gt_to_pred_map),
        duration_hours=args.duration_hours,
        demo=args.demo,
    )
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    LOGGER.info("Đã kết xuất báo cáo đánh giá toàn diện tại: %s", out_p)


if __name__ == "__main__":
    main()
