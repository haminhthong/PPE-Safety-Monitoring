"""Tests lưu trữ bounding box và liên kết vùng cơ thể của PPE."""

from __future__ import annotations

from ppe_detection.detector import validate_body_zone
from ppe_detection.models import PPEDetection


def test_ppe_detection_retains_box() -> None:
    """Đảm bảo PPEDetection lưu trữ đầy đủ tọa độ bounding box."""
    det = PPEDetection(label="helmet", confidence=0.91, box=[10.0, 5.0, 50.0, 45.0])
    assert det.box == [10.0, 5.0, 50.0, 45.0]
    assert det.label == "helmet"
    assert det.confidence == 0.91


def test_helmet_must_match_head_zone() -> None:
    """Chấp nhận mũ ở vùng đầu và loại mũ ở phần dưới cơ thể."""
    roi_h, roi_w = 200, 100

    # Box 1: Ở vùng đầu (y từ 10 đến 50 -> center y = 30 / 200 = 0.15 <= 0.35) -> HỢP LỆ
    head_box = [10.0, 10.0, 90.0, 50.0]
    assert validate_body_zone("helmet", head_box, roi_h, roi_w) is True
    assert validate_body_zone("no-helmet", head_box, roi_h, roi_w) is True

    # Box 2: Ở chân/đùi do người đứng cạnh dính vào crop -> TỪ CHỐI.
    leg_box = [10.0, 150.0, 90.0, 190.0]
    assert validate_body_zone("helmet", leg_box, roi_h, roi_w) is False
    assert validate_body_zone("no-helmet", leg_box, roi_h, roi_w) is False


def test_vest_must_match_torso_zone() -> None:
    """Chấp nhận áo ở vùng thân và loại áo ngoài vùng thân."""
    roi_h, roi_w = 200, 100

    # Box 1: Ở thân, tâm y = 0.50 trong [0.30, 0.75] -> HỢP LỆ.
    torso_box = [5.0, 60.0, 95.0, 140.0]
    assert validate_body_zone("vest", torso_box, roi_h, roi_w) is True
    assert validate_body_zone("no-vest", torso_box, roi_h, roi_w) is True

    # Box 2: Quá cao, tâm y = 0.075 < 0.30 -> TỪ CHỐI.
    top_box = [5.0, 0.0, 95.0, 30.0]
    assert validate_body_zone("vest", top_box, roi_h, roi_w) is False

    # Box 3: Quá thấp, tâm y = 0.91 > 0.75 -> TỪ CHỐI.
    feet_box = [5.0, 170.0, 95.0, 195.0]
    assert validate_body_zone("vest", feet_box, roi_h, roi_w) is False
