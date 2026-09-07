"""Kiểm thử tự động cho module ppe_detection.config."""

from pathlib import Path

import pytest

from ppe_detection.config import DetectionConfig


def test_config_default_values():
    """Kiểm tra các giá trị mặc định của DetectionConfig."""
    config = DetectionConfig(demo_mode=True)
    assert config.image_size == 640
    assert config.person_confidence == 0.3
    assert config.ppe_confidence == 0.3
    assert config.violation_confirmations == 2
    assert config.save_snapshots is True


def test_config_validation_invalid_confidence():
    """Kiểm tra ngoại lệ khi tham số confidence nằm ngoài dải [0.0, 1.0]."""
    config = DetectionConfig(demo_mode=True, person_confidence=1.5)
    with pytest.raises(ValueError, match="person_confidence phải nằm trong khoảng"):
        config.validate()


def test_config_validation_missing_model_files(tmp_path: Path):
    """Kiểm tra lỗi FileNotFoundError khi thiếu file model ở chế độ thông thường."""
    non_existent = tmp_path / "not_found.pt"
    config = DetectionConfig(
        person_model_path=non_existent,
        ppe_model_path=non_existent,
        demo_mode=False,
    )
    with pytest.raises(FileNotFoundError):
        config.validate()


def test_config_validation_demo_mode():
    """Kiểm tra bỏ qua kiểm tra file model khi bật demo_mode."""
    config = DetectionConfig(demo_mode=True)
    config.validate()  # Không bắn ra ngoại lệ


def test_runtime_policy_maps_all_decision_fields() -> None:
    """Policy phải điều khiển cadence, zone, overlap và temporal rules."""
    config = DetectionConfig.load_from_policy(Path("configs/runtime_policy.yaml"), demo_mode=True)
    assert config.detection_interval == 1
    assert config.ppe_detection_interval == 4
    assert config.track_ttl_seconds == 5.0
    assert config.head_zone_max == 0.35
    assert config.roi_overlap_threshold == 0.4
    assert config.violation_confirm_seconds == 0.5
