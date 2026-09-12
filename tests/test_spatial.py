"""Kiểm thử tự động cho module ppe_detection.detector (Body-Zone và ROI Spatial Association)."""

from __future__ import annotations

from ppe_detection.detector import (
    is_box_overlapping_roi,
    is_center_in_roi,
    roi_overlap_ratio,
    validate_body_zone,
)
from ppe_detection.models import PPEDetection


def test_ppe_detection_retains_box() -> None:
    """Đảm bảo PPEDetection lưu trữ đầy đủ tọa độ bounding box."""
    det = PPEDetection(label="helmet", confidence=0.91, box=[10.0, 5.0, 50.0, 45.0])
    assert det.box == [10.0, 5.0, 50.0, 45.0]
    assert det.label == "helmet"
    assert det.confidence == 0.91


def test_helmet_must_match_head_zone() -> None:
    """Chấp nhận mũ ở vùng đầu (0 - 35% chiều cao) và loại bỏ mũ ở phần thân/chân."""
    roi_h, roi_w = 200, 100

    # Box 1: Ở vùng đầu (y từ 10 đến 50 -> center y = 30 / 200 = 0.15 <= 0.35) -> HỢP LỆ
    head_box = [10.0, 10.0, 90.0, 50.0]
    assert validate_body_zone("helmet", head_box, roi_h, roi_w) is True
    assert validate_body_zone("no-helmet", head_box, roi_h, roi_w) is True

    # Box 2: Ở chân/đùi do người đứng cạnh dính vào crop -> TỪ CHỐI
    leg_box = [10.0, 150.0, 90.0, 190.0]
    assert validate_body_zone("helmet", leg_box, roi_h, roi_w) is False
    assert validate_body_zone("no-helmet", leg_box, roi_h, roi_w) is False


def test_vest_must_match_torso_zone() -> None:
    """Chấp nhận áo ở vùng thân (30% - 75% chiều cao) và loại bỏ áo ngoài vùng thân."""
    roi_h, roi_w = 200, 100

    # Box 1: Ở thân (tâm y = 100/200 = 0.50 trong khoảng [0.30, 0.75]) -> HỢP LỆ
    torso_box = [5.0, 60.0, 95.0, 140.0]
    assert validate_body_zone("vest", torso_box, roi_h, roi_w) is True
    assert validate_body_zone("no-vest", torso_box, roi_h, roi_w) is True

    # Box 2: Quá cao ở đỉnh đầu (tâm y = 15/200 = 0.075 < 0.30) -> TỪ CHỐI
    top_box = [5.0, 0.0, 95.0, 30.0]
    assert validate_body_zone("vest", top_box, roi_h, roi_w) is False

    # Box 3: Quá thấp ở chân (tâm y = 182.5/200 = 0.91 > 0.75) -> TỪ CHỐI
    feet_box = [5.0, 170.0, 95.0, 195.0]
    assert validate_body_zone("vest", feet_box, roi_h, roi_w) is False


def test_roi_polygon_boundary_and_overlap() -> None:
    """Kiểm tra logic đa giác ROI và tỷ lệ diện tích giao nhau."""
    polygon = [(10, 10), (100, 10), (100, 100), (10, 100)]

    # Tâm nằm trong ROI
    assert is_center_in_roi([20, 20, 40, 40], polygon) is True
    # Tâm nằm ngoài ROI
    assert is_center_in_roi([150, 150, 200, 200], polygon) is False

    # Tính tỷ lệ giao diện tích
    box = [10.0, 10.0, 100.0, 100.0]
    assert roi_overlap_ratio(box, polygon) == 1.0

    # Box giao 50% diện tích
    box_half = [55.0, 10.0, 145.0, 100.0]
    assert is_box_overlapping_roi(box_half, polygon, min_overlap=0.4) is True
    assert is_box_overlapping_roi(box_half, polygon, min_overlap=0.8) is False
