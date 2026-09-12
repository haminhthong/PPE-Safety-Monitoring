"""Kiểm thử tự động cho module ppe_detection.tracker (IoUTracker & TwoThresholdIoUTracker)."""

from __future__ import annotations

import pytest

from ppe_detection.models import PersonDetection, PPEState, PPEStatus
from ppe_detection.tracker import IoUTracker, TwoThresholdIoUTracker, iou


def test_iou_identical_boxes() -> None:
    """Kiểm tra IoU của 2 bounding box trùng nhau bằng 1.0."""
    box = [0.0, 0.0, 100.0, 100.0]
    assert pytest.approx(iou(box, box), abs=1e-4) == 1.0


def test_iou_disjoint_boxes() -> None:
    """Kiểm tra IoU của 2 bounding box không giao nhau bằng 0.0."""
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    assert iou(box1, box2) == 0.0


def test_iou_partial_overlap() -> None:
    """Kiểm tra tính toán IoU khi 2 box giao nhau một phần."""
    box1 = [0.0, 0.0, 10.0, 10.0]  # diện tích 100
    box2 = [5.0, 0.0, 15.0, 10.0]  # diện tích 100, giao 50, hợp 150 -> IoU = 50/150 = 0.3333
    assert round(iou(box1, box2), 3) == 0.333


def test_iou_tracker_assignment_and_id_persistence() -> None:
    """Kiểm tra tracker gán ID mới và duy trì ID khi di chuyển nhẹ."""
    tracker = IoUTracker(threshold=0.3, max_missed=5)

    det1 = PersonDetection(box=[10.0, 10.0, 50.0, 100.0], confidence=0.9, ppe=PPEStatus())
    tracks_f1 = tracker.update([det1])
    assert len(tracks_f1) == 1
    assigned_id = tracks_f1[0].track_id

    # Frame 2: Box dịch chuyển nhẹ
    det2 = PersonDetection(box=[12.0, 11.0, 52.0, 101.0], confidence=0.91, ppe=PPEStatus())
    tracks_f2 = tracker.update([det2])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == assigned_id


def test_iou_tracker_max_missed_cleanup() -> None:
    """Kiểm tra xóa vết theo dõi khi đối tượng biến mất quá số chu kỳ max_missed."""
    tracker = IoUTracker(threshold=0.3, max_missed=2)

    det = PersonDetection(box=[0.0, 0.0, 10.0, 10.0], confidence=0.8, ppe=PPEStatus())
    tracker.update([det])
    assert len(tracker.active_tracks()) == 1

    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.active_tracks()) == 0


def test_two_threshold_tracker_association() -> None:
    """Kiểm tra TwoThresholdIoUTracker phục hồi track bị che khuất qua detection điểm thấp."""
    tracker = TwoThresholdIoUTracker(
        high_threshold=0.6, match_threshold=0.5, low_match_threshold=0.3
    )

    # Frame 1: Người rõ nét (confidence 0.90) -> gán track ID 1
    det_clear = [PersonDetection(box=[100.0, 100.0, 150.0, 250.0], confidence=0.90)]
    tracks1 = tracker.update(det_clear)
    assert len(tracks1) == 1
    assert tracks1[0].track_id == 1

    # Frame 2: Người bị che khuất một phần, confidence giảm xuống 0.45 (< high_threshold 0.6)
    det_occluded = [PersonDetection(box=[105.0, 102.0, 155.0, 252.0], confidence=0.45)]
    tracks2 = tracker.update(det_occluded)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == 1  # Phục hồi thành công cùng track ID ở giai đoạn 2!


def test_motion_prediction_prevents_box_freeze() -> None:
    """Kiểm tra hàm predict() nội suy chuyển động thay vì giữ box đứng yên."""
    tracker = TwoThresholdIoUTracker()

    # Frame 1 & 2: Người di chuyển sang phải
    tracker.update([PersonDetection(box=[100.0, 100.0, 150.0, 200.0], confidence=0.9)])
    tracker.update([PersonDetection(box=[110.0, 100.0, 160.0, 200.0], confidence=0.9)])

    # Frame 3: Không chạy detector -> gọi predict()
    pred_tracks = tracker.predict()
    assert len(pred_tracks) == 1
    # Bounding box phải dịch chuyển theo vector vận tốc (x1 > 110.0)
    assert pred_tracks[0].box[0] > 110.0


def test_tracker_preserves_last_ppe_evidence() -> None:
    """Kiểm tra nhịp tracking không làm mất bằng chứng PPE đã quan sát trước đó."""
    tracker = IoUTracker(threshold=0.3)
    observed = PPEStatus(helmet_state=PPEState.PRESENT)
    tracker.update([PersonDetection(box=[10.0, 10.0, 50.0, 100.0], confidence=0.9, ppe=observed)])

    # Frame tiếp theo chỉ có person box, chưa inspect PPE
    tracks = tracker.update([PersonDetection(box=[10.0, 10.0, 50.0, 100.0], confidence=0.9)])
    assert tracks[0].ppe.helmet_state is PPEState.PRESENT
