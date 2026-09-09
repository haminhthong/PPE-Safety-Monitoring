"""Kiểm thử tự động cho module ppe_detection.reporting."""

import json
from pathlib import Path

from ppe_detection.reporting import SessionReport


def test_session_report_counts_and_events():
    """Kiểm tra tổng hợp số liệu đếm vi phạm và thêm sự kiện."""
    report = SessionReport(source="test_video.mp4")
    assert report.counts == {"total": 0, "helmet": 0, "vest": 0, "people": 0}

    report.add_event(track_id=1, kind="helmet", frame_id=10, fps=30.0, snapshot_path="snap1.jpg")
    report.add_event(track_id=1, kind="vest", frame_id=10, fps=30.0, snapshot_path="snap2.jpg")
    report.add_event(track_id=2, kind="helmet", frame_id=20, fps=30.0, snapshot_path="snap3.jpg")

    counts = report.counts
    assert counts["helmet"] == 2
    assert counts["vest"] == 1
    assert counts["total"] == 3
    assert counts["people"] == 2  # Gồm người ID 1 và người ID 2


def test_session_report_save_files(tmp_path: Path):
    """Kiểm tra việc lưu file báo cáo JSON và CSV."""
    report = SessionReport(source="0", resolved_config={"demo_mode": True})
    report.total_frames = 100
    report.unique_track_ids.add(1)
    report.add_event(track_id=1, kind="helmet", frame_id=15, fps=30.0)

    json_path, csv_path = report.save(tmp_path, stem="camera_0")

    assert json_path.is_file()
    assert csv_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total_frames"] == 100
    assert payload["synthetic_demo"] is True
    assert payload["unique_people_tracked"] == 1
    assert len(payload["events"]) == 1
    assert payload["events"][0]["violation_type"] == "helmet"
