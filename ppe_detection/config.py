"""Module quản lý cấu hình cho hệ thống phát hiện trang bị bảo hộ (PPE).

Cung cấp dataclass `DetectionConfig` để lưu trữ và kiểm tra tính hợp lệ
của các tham số mô hình, tracking, spatial body-zone và FSM cảnh báo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectionConfig:
    """Lưu trữ toàn bộ tham số cấu hình cho pipeline phát hiện PPE."""

    # Đường dẫn mô hình
    person_model_path: Path | None = None
    ppe_model_path: Path | None = None

    # Tham số suy luận mô hình
    image_size: int = 640
    person_confidence: float = 0.30
    ppe_confidence: float = 0.30
    nms_iou: float = 0.50

    # Chu kỳ suy luận (tiết kiệm tài nguyên)
    detection_interval: int = 1
    ppe_detection_interval: int = 4

    # Tham số Tracker (Two-Threshold IoU / IoU)
    tracker_type: str = "two_threshold_iou"
    tracker_iou: float = 0.50
    high_threshold: float = 0.50
    low_match_threshold: float = 0.30
    max_missed: int = 30
    track_ttl_seconds: float = 5.0

    # Spatial Body-Zone & ROI
    person_roi_padding: int = 10
    enable_body_zone_filter: bool = True
    head_zone_min: float = 0.00
    head_zone_max: float = 0.35
    torso_zone_min: float = 0.30
    torso_zone_max: float = 0.75
    conflict_margin: float = 0.10
    roi_polygon: list[tuple[int, int]] | None = None
    roi_rule: str = "center_or_overlap"
    roi_overlap_threshold: float = 0.40

    # Máy trạng thái thời gian (FSM)
    violation_confirm_seconds: float = 0.50
    resolution_confirm_seconds: float = 1.00
    alert_cooldown_seconds: float = 10.0

    # Đầu ra & Hiển thị
    output_dir: Path = Path("outputs")
    save_output: bool = False
    save_snapshots: bool = True
    show_window: bool = True
    enable_beep: bool = True

    def __post_init__(self) -> None:
        self.person_model_path = self._as_path(self.person_model_path)
        self.ppe_model_path = self._as_path(self.ppe_model_path)
        self.output_dir = Path(self.output_dir)

    @staticmethod
    def _as_path(value: Path | str | None) -> Path | None:
        return Path(value) if value is not None else None

    def validate(self, require_models: bool = True) -> None:
        """Kiểm tra tính hợp lệ của đường dẫn mô hình và các miền giá trị tham số."""
        if require_models:
            if self.person_model_path is None or not self.person_model_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy model phát hiện người: {self.person_model_path}"
                )
            if self.ppe_model_path is None or not self.ppe_model_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy model phát hiện PPE: {self.ppe_model_path}"
                )

        if self.image_size <= 0 or self.detection_interval <= 0 or self.ppe_detection_interval <= 0:
            raise ValueError("image_size và các chu kỳ inference phải lớn hơn 0.")

        if self.person_roi_padding < 0:
            raise ValueError("person_roi_padding không được là số âm.")

        if self.track_ttl_seconds <= 0:
            raise ValueError("track_ttl_seconds phải lớn hơn 0.")
        if self.violation_confirm_seconds < 0 or self.resolution_confirm_seconds < 0:
            raise ValueError("Các ngưỡng thời gian FSM không được là số âm.")
        if self.alert_cooldown_seconds < 0:
            raise ValueError("alert_cooldown_seconds không được là số âm.")

        for name, value in (
            ("person_confidence", self.person_confidence),
            ("ppe_confidence", self.ppe_confidence),
            ("nms_iou", self.nms_iou),
            ("tracker_iou", self.tracker_iou),
            ("high_threshold", self.high_threshold),
            ("low_match_threshold", self.low_match_threshold),
            ("conflict_margin", self.conflict_margin),
            ("roi_overlap_threshold", self.roi_overlap_threshold),
            ("head_zone_min", self.head_zone_min),
            ("head_zone_max", self.head_zone_max),
            ("torso_zone_min", self.torso_zone_min),
            ("torso_zone_max", self.torso_zone_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} phải nằm trong khoảng từ 0.0 đến 1.0.")

        if self.max_missed < 0:
            raise ValueError("max_missed không được là số âm.")

        if self.head_zone_min > self.head_zone_max or self.torso_zone_min > self.torso_zone_max:
            raise ValueError("Giới hạn body-zone không hợp lệ.")
        if self.low_match_threshold > self.tracker_iou:
            raise ValueError("low_match_threshold không được lớn hơn tracker_iou.")
        if self.roi_rule not in {"center", "overlap", "center_or_overlap"}:
            raise ValueError("roi_rule phải là center, overlap hoặc center_or_overlap.")
        if self.tracker_type.lower() not in {"two_threshold_iou", "iou"}:
            raise ValueError(f"tracker_type không được hỗ trợ: {self.tracker_type}")
        if self.roi_polygon is not None:
            if self.roi_polygon and len(self.roi_polygon) < 3:
                raise ValueError("roi_polygon phải có ít nhất 3 đỉnh.")
            for point in self.roi_polygon:
                if len(point) != 2:
                    raise ValueError("Mỗi điểm roi_polygon phải có dạng (x, y).")

    def to_dict(self) -> dict[str, Any]:
        """Chuyển cấu hình thành dict có thể tuần tự hóa JSON."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def load_from_yaml(cls, config_path: Path | str, **overrides: Any) -> DetectionConfig:
        """Nạp cấu hình từ file YAML (hỗ trợ cả cấu trúc lồng nhóm và cấu trúc phẳng)."""
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
        except yaml.YAMLError as error:
            raise ValueError(f"File cấu hình không phải YAML hợp lệ: {config_path}") from error

        if not isinstance(data, dict):
            raise ValueError("File cấu hình YAML phải có định dạng dictionary/object.")

        # Hỗ trợ nhóm 'model'
        model_sec = data.get("model", {})
        # Hỗ trợ nhóm 'tracking'
        tracking_sec = data.get("tracking", {})
        # Hỗ trợ nhóm 'ppe'
        ppe_sec = data.get("ppe", {})
        # Hỗ trợ nhóm 'alert'
        alert_sec = data.get("alert", {})
        # Hỗ trợ nhóm 'output'
        output_sec = data.get("output", {})

        # Tọa độ head / torso zone
        head_zone = ppe_sec.get("head_zone", [0.00, 0.35])
        torso_zone = ppe_sec.get("torso_zone", [0.30, 0.75])

        # Phân tích polygon ROI
        roi_polygon_raw = alert_sec.get("roi_polygon", data.get("roi_polygon"))
        roi_polygon = None
        if roi_polygon_raw is not None and isinstance(roi_polygon_raw, list):
            roi_polygon = [
                (int(p[0]), int(p[1]))
                for p in roi_polygon_raw
                if isinstance(p, list | tuple) and len(p) == 2
            ]

        kwargs: dict[str, Any] = {
            "image_size": model_sec.get("image_size", data.get("image_size", 640)),
            "person_confidence": model_sec.get(
                "person_confidence", data.get("person_confidence", 0.30)
            ),
            "ppe_confidence": model_sec.get("ppe_confidence", data.get("ppe_confidence", 0.30)),
            "nms_iou": model_sec.get("nms_iou", data.get("nms_iou", 0.50)),
            "detection_interval": ppe_sec.get(
                "detection_interval", data.get("detection_interval", 1)
            ),
            "ppe_detection_interval": ppe_sec.get(
                "inspection_interval", data.get("ppe_detection_interval", 4)
            ),
            "person_roi_padding": ppe_sec.get(
                "person_padding_px", data.get("person_roi_padding", 10)
            ),
            "head_zone_min": head_zone[0] if isinstance(head_zone, list | tuple) else 0.0,
            "head_zone_max": head_zone[1] if isinstance(head_zone, list | tuple) else 0.35,
            "torso_zone_min": torso_zone[0] if isinstance(torso_zone, list | tuple) else 0.30,
            "torso_zone_max": torso_zone[1] if isinstance(torso_zone, list | tuple) else 0.75,
            "tracker_type": tracking_sec.get(
                "tracker_type", data.get("tracker_type", "two_threshold_iou")
            ),
            "high_threshold": tracking_sec.get("high_threshold", data.get("high_threshold", 0.50)),
            "tracker_iou": tracking_sec.get(
                "iou_threshold", tracking_sec.get("match_threshold", data.get("tracker_iou", 0.50))
            ),
            "low_match_threshold": tracking_sec.get(
                "low_match_threshold", data.get("low_match_threshold", 0.30)
            ),
            "max_missed": tracking_sec.get("max_missed", data.get("max_missed", 30)),
            "track_ttl_seconds": tracking_sec.get(
                "track_ttl_seconds", data.get("track_ttl_seconds", 5.0)
            ),
            "violation_confirm_seconds": alert_sec.get(
                "violation_seconds",
                alert_sec.get(
                    "violation_confirm_seconds", data.get("violation_confirm_seconds", 0.5)
                ),
            ),
            "resolution_confirm_seconds": alert_sec.get(
                "resolution_seconds",
                alert_sec.get(
                    "resolution_confirm_seconds", data.get("resolution_confirm_seconds", 1.0)
                ),
            ),
            "alert_cooldown_seconds": alert_sec.get(
                "cooldown_seconds",
                alert_sec.get("alert_cooldown_seconds", data.get("alert_cooldown_seconds", 10.0)),
            ),
            "conflict_margin": alert_sec.get("conflict_margin", data.get("conflict_margin", 0.10)),
            "roi_rule": alert_sec.get("roi_rule", data.get("roi_rule", "center_or_overlap")),
            "roi_overlap_threshold": alert_sec.get(
                "roi_overlap_threshold", data.get("roi_overlap_threshold", 0.40)
            ),
            "roi_polygon": roi_polygon,
            "output_dir": output_sec.get("output_dir", data.get("output_dir", "outputs")),
            "save_output": output_sec.get("save_output", data.get("save_output", False)),
            "save_snapshots": output_sec.get("save_snapshots", data.get("save_snapshots", True)),
            "show_window": output_sec.get("show_window", data.get("show_window", True)),
            "enable_beep": output_sec.get("enable_beep", data.get("enable_beep", True)),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # Alias cho caller cũ
    load_from_policy = load_from_yaml
