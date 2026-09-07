"""Tests kiểm tra cơ chế đánh giá sự kiện vi phạm an toàn bằng Spatio-Temporal Matching."""

from __future__ import annotations

from training.evaluate_events import evaluate_violation_events


def test_event_matching_with_independent_tracker_ids() -> None:
    """Đảm bảo sự kiện vi phạm được ghép cặp chính xác theo loại vi phạm và dung sai thời gian dù ID khác nhau."""
    # GT Person ID = 5, vi phạm mũ ở giây 12.0
    gt_events = [
        {"track_id": 5, "violation_type": "helmet", "time_seconds": 12.0},
    ]

    # Tracker thực tế sinh ID = 107 (khác ID 5), phát hiện ở giây 12.8 (dung sai 0.8s <= 3.0s)
    pred_events = [
        {"track_id": 107, "violation_type": "helmet", "time_seconds": 12.8},
    ]

    metrics = evaluate_violation_events(
        events_gt=gt_events,
        events_pred=pred_events,
        duration_hours=1.0,
        time_tolerance_sec=3.0,
        gt_to_pred_map={5: 107},  # Ánh xạ từ Layer 3 Tracking trajectory
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["event_precision"] == 1.0
    assert metrics["event_recall"] == 1.0
    assert metrics["median_time_to_alert_sec"] == 0.8


def test_event_matching_type_mismatch_fails() -> None:
    """Nếu sai loại vi phạm (ví dụ GT là helmet nhưng Pred là vest) thì tính là FP và FN."""
    gt_events = [{"track_id": 1, "violation_type": "helmet", "time_seconds": 10.0}]
    pred_events = [{"track_id": 1, "violation_type": "vest", "time_seconds": 10.0}]

    metrics = evaluate_violation_events(gt_events, pred_events, duration_hours=1.0)

    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["event_precision"] == 0.0
    assert metrics["event_recall"] == 0.0


def test_interval_event_uses_alert_time_and_identity_map() -> None:
    """Event interval phải ghép đúng cửa sổ thời gian và identity map."""
    metrics = evaluate_violation_events(
        events_gt=[
            {"track_id": 7, "violation_type": "vest", "start_sec": 10.0, "end_sec": 14.0}
        ],
        events_pred=[
            {
                "track_id": 42,
                "violation_type": "vest",
                "alert_time_seconds": 11.2,
            }
        ],
        duration_hours=1.0,
        gt_to_pred_map={7: 42},
    )
    assert metrics["true_positives"] == 1
    assert metrics["median_time_to_alert_sec"] == 1.2
