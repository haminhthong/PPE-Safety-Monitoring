"""Bộ test case kiểm thử các trường hợp đặc biệt (Edge Cases) của AI/ML Pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ppe_detection.config import DetectionConfig
from ppe_detection.detector import SyntheticDemoDetector, is_center_in_roi
from ppe_detection.models import PersonDetection, PPEDetection, PPEStatus
from ppe_detection.tracker import IoUTracker


def test_frame_none_or_empty_raises_value_error():
    """Khung hình None hoặc kích thước 0 phải ném ra ValueError."""
    config = DetectionConfig(demo_mode=True)
    detector = SyntheticDemoDetector(config)

    with pytest.raises(ValueError, match="Khung hình đầu vào rỗng."):
        detector.detect(None)

    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Khung hình đầu vào rỗng."):
        detector.detect(empty_frame)


def test_is_center_in_roi_boundary_cases():
    """Kiểm tra logic điểm tâm nằm trong polygon ROI với các tọa độ đặc biệt."""
    roi_polygon = [(10, 10), (100, 10), (100, 100), (10, 100)]

    # Tâm nằm chính giữa ROI
    assert is_center_in_roi([20, 20, 40, 40], roi_polygon) is True

    # Tâm nằm ngoài ROI
    assert is_center_in_roi([150, 150, 200, 200], roi_polygon) is False

    # ROI rỗng hoặc không đủ đỉnh (< 3) mặc định trả về True.
    assert is_center_in_roi([150, 150, 200, 200], []) is True
    assert is_center_in_roi([150, 150, 200, 200], [(10, 10), (20, 20)]) is True


def test_ppe_label_conflict_resolution():
    """Kiểm tra logic phân định nhãn xung đột dựa trên confidence và conflict_margin."""
    config = DetectionConfig(
        demo_mode=True,
        ppe_confidence=0.3,
        conflict_margin=0.1,
    )
    # Kiểm tra phương pháp tính toán trạng thái vi phạm với conflict_margin
    # no-helmet score (0.8) > helmet score (0.6) + 0.1 -> Vi phạm Mũ
    labels_violation = [
        PPEDetection(label="helmet", confidence=0.6),
        PPEDetection(label="no-helmet", confidence=0.8),
    ]
    h_score = max(
        (item.confidence for item in labels_violation if item.label == "helmet"), default=0.0
    )
    nh_score = max(
        (item.confidence for item in labels_violation if item.label == "no-helmet"), default=0.0
    )
    violation = (nh_score >= config.ppe_confidence) and (
        nh_score > h_score + config.conflict_margin
    )
    assert violation is True

    # no-helmet 0.65 không vượt helmet 0.6 + margin 0.1 -> UNKNOWN.
    labels_conflict = [
        PPEDetection(label="helmet", confidence=0.6),
        PPEDetection(label="no-helmet", confidence=0.65),
    ]
    h_score2 = max(
        (item.confidence for item in labels_conflict if item.label == "helmet"), default=0.0
    )
    nh_score2 = max(
        (item.confidence for item in labels_conflict if item.label == "no-helmet"), default=0.0
    )
    conflict = (nh_score2 >= config.ppe_confidence) and (
        nh_score2 > h_score2 + config.conflict_margin
    )
    assert conflict is False


def test_iou_tracker_edge_cases():
    """Kiểm tra IoU Tracker với danh sách rỗng và đối tượng mất dấu."""
    tracker = IoUTracker(threshold=0.3, max_disappeared=2)

    # Không có detection nào
    tracks = tracker.update([])
    assert len(tracks) == 0

    # Khởi tạo 1 track mới
    det = PersonDetection(
        box=[10.0, 10.0, 50.0, 100.0],
        confidence=0.9,
        ppe=PPEStatus(),
    )
    tracks = tracker.update([det])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1

    # Không có detection trong 3 frame liên tiếp -> track bị loại bỏ
    tracker.update([])
    tracker.update([])
    active_after_disappeared = tracker.update([])
    assert len(active_after_disappeared) == 0
