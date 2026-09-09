"""Điểm vào final evaluation: bắt buộc dùng model, GT và locked test thật."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Cho phép chạy trực tiếp ``python scripts/evaluate_final.py`` từ thư mục repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_system import run_full_system_evaluation  # noqa: E402


def load_json(path: str) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Không tìm thấy evaluation artifact: {file}")
    value = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Artifact phải là JSON array: {file}")
    return value


def load_id_map(path: str | None) -> dict[str, str] | None:
    """Đọc ánh xạ ID sau khi đối chiếu trajectory GT và prediction."""
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Không tìm thấy identity map: {file}")
    value = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Identity map phải là JSON object: {file}")
    return {str(key): str(mapped) for key, mapped in value.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá PPE locked test thật")
    parser.add_argument("--ppe-model", required=True)
    parser.add_argument("--person-model", required=True)
    parser.add_argument("--ppe-data", required=True)
    parser.add_argument("--person-data", required=True)
    parser.add_argument("--gt-tracks", required=True)
    parser.add_argument("--pred-tracks", required=True)
    parser.add_argument("--gt-events", required=True)
    parser.add_argument("--pred-events", required=True)
    parser.add_argument(
        "--gt-to-pred-map", help="JSON object ánh xạ GT track ID sang predicted track ID"
    )
    parser.add_argument("--duration-hours", required=True, type=float)
    parser.add_argument("--output", default="reports/locked_test_metrics.json")
    args = parser.parse_args()

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
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi locked test report: {output}")


if __name__ == "__main__":
    main()
