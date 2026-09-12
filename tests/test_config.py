"""Kiểm thử tự động cho module ppe_detection.config."""

from pathlib import Path

import pytest

from ppe_detection.config import DetectionConfig


def test_config_default_values() -> None:
    """Kiểm tra các giá trị mặc định của DetectionConfig."""
    config = DetectionConfig()
    assert config.image_size == 640
    assert config.person_confidence == 0.30
    assert config.ppe_confidence == 0.30
    assert config.violation_confirm_seconds == 0.50
    assert config.save_snapshots is True


def test_config_validation_invalid_confidence() -> None:
    """Kiểm tra ngoại lệ khi tham số confidence nằm ngoài dải [0.0, 1.0]."""
    config = DetectionConfig(person_confidence=1.5)
    with pytest.raises(ValueError, match="person_confidence phải nằm trong khoảng"):
        config.validate(require_models=False)


def test_config_validation_missing_model_files(tmp_path: Path) -> None:
    """Kiểm tra lỗi FileNotFoundError khi thiếu file model."""
    non_existent = tmp_path / "not_found.pt"
    config = DetectionConfig(
        person_model_path=non_existent,
        ppe_model_path=non_existent,
    )
    with pytest.raises(FileNotFoundError):
        config.validate(require_models=True)


def test_load_from_yaml_maps_all_fields() -> None:
    """Kiểm tra nạp cấu hình thành công từ configs/config.yaml."""
    config = DetectionConfig.load_from_yaml(Path("configs/config.yaml"))
    assert config.image_size == 640
    assert config.person_confidence == 0.30
    assert config.ppe_confidence == 0.30
    assert config.tracker_type == "two_threshold_iou"
    assert config.track_ttl_seconds == 5.0
    assert config.head_zone_max == 0.35
    assert config.torso_zone_min == 0.30
    assert config.violation_confirm_seconds == 0.50
    assert config.resolution_confirm_seconds == 1.00
