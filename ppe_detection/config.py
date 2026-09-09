"""Module quản lý cấu hình cho hệ thống phát hiện trang bị bảo hộ (PPE).

Cung cấp dataclass `DetectionConfig` để lưu trữ, kiểm tra tính hợp lệ
của các tham số đầu vào và điều chỉnh hành vi của toàn bộ pipeline theo các hợp đồng kiến trúc.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectionConfig:
    """Lớp lưu trữ toàn bộ cấu hình tham số cho hệ thống phát hiện PPE.

    Attributes:
        person_model_path: Đường dẫn tới file trọng số YOLO phát hiện người.
        ppe_model_path: Đường dẫn tới file trọng số YOLO phát hiện PPE.
        image_size: Kích thước chiều của ảnh đầu vào cho mô hình YOLO (mặc định 640).
        detection_interval: Chu kỳ phát hiện người/tracking (mặc định 1).
        ppe_detection_interval: Chu kỳ inspection PPE độc lập với tracking (mặc định 4).
        person_confidence: Ngưỡng tin cậy phát hiện người (0.0 đến 1.0).
        ppe_confidence: Ngưỡng tin cậy phát hiện PPE (0.0 đến 1.0).
        nms_iou: Ngưỡng IoU cho Non-Maximum Suppression của YOLO.
        tracker_type: Thuật toán tracking ('two_threshold_iou' hoặc 'iou').
        tracker_iou: Ngưỡng IoU tối thiểu để ghép cặp vết theo dõi.
        max_disappeared: Số chu kỳ detector tối đa giữ lại track khi mất dấu.
        max_missed_detections: Số chu kỳ detector bỏ lỡ tối đa trước khi xóa ID khỏi bộ nhớ.
        violation_confirm_seconds: Dwell time xác nhận vi phạm.
        resolution_confirm_seconds: Dwell time xác nhận khắc phục.
        enable_beep: Cho phép phát âm thanh cảnh báo khi có vi phạm mới.
        show_window: Hiển thị cửa sổ xem trực tiếp bằng OpenCV.
        save_output: Lưu video/ảnh kết quả và file báo cáo ra đĩa.
        save_snapshots: Lưu ảnh cắt (ROI snapshot) của đối tượng khi vi phạm.
        output_dir: Thư mục gốc lưu trữ kết quả đầu ra.
        demo_mode: Bật chế độ chạy thử nghiệm (mô phỏng) không cần file trọng số ngoài.
        roi_polygon: Danh sách tọa độ đỉnh (x, y) tạo thành vùng nguy hiểm ROI.
        person_roi_padding: Số pixel padding mở rộng khi cắt ROI người (mặc định 10).
        conflict_margin: Ngưỡng chênh lệch độ tin cậy để giải quyết nhãn xung đột (mặc định 0.1).
        enable_body_zone_filter: Kích hoạt bộ lọc liên kết không gian giải phẫu cơ thể.
        head_zone_min: Tỷ lệ chiều cao tối thiểu cho vùng đầu.
        head_zone_max: Tỷ lệ chiều cao tối đa cho vùng đầu (mặc định 0.35).
        torso_zone_min: Tỷ lệ chiều cao tối thiểu cho vùng thân (mặc định 0.30).
        torso_zone_max: Tỷ lệ chiều cao tối đa cho vùng thân (mặc định 0.75).
    """

    person_model_path: Path | None = None
    ppe_model_path: Path | None = None
    image_size: int = 640
    detection_interval: int = 1
    ppe_detection_interval: int = 4
    person_confidence: float = 0.3
    ppe_confidence: float = 0.3
    nms_iou: float = 0.5
    tracker_type: str = "two_threshold_iou"
    tracker_iou: float = 0.3
    high_threshold: float = 0.5
    low_match_threshold: float = 0.3
    max_disappeared: int = 30
    max_missed_detections: int = 30
    track_ttl_seconds: float = 5.0
    violation_confirmations: int = 2
    resolution_confirmations: int = 3
    violation_confirm_seconds: float = 0.5
    resolution_confirm_seconds: float = 1.0
    alert_cooldown_seconds: float = 10.0
    enable_beep: bool = True
    show_window: bool = True
    save_output: bool = False
    save_snapshots: bool = True
    output_dir: Path = Path("outputs")
    demo_mode: bool = False
    roi_polygon: list[tuple[int, int]] | None = None
    roi_rule: str = "center_or_overlap"
    roi_overlap_threshold: float = 0.4
    person_roi_padding: int = 10
    conflict_margin: float = 0.1
    enable_body_zone_filter: bool = True
    head_zone_min: float = 0.0
    head_zone_max: float = 0.35
    torso_zone_min: float = 0.30
    torso_zone_max: float = 0.75
    policy_path: Path | None = None

    def __post_init__(self) -> None:
        # Đồng bộ alias cũ để các caller hiện tại vẫn chạy đúng.
        if self.max_missed_detections != 30 and self.max_disappeared == 30:
            self.max_disappeared = self.max_missed_detections
        elif self.max_disappeared != 30 and self.max_missed_detections == 30:
            self.max_missed_detections = self.max_disappeared
        self.person_model_path = self._as_path(self.person_model_path)
        self.ppe_model_path = self._as_path(self.ppe_model_path)
        self.output_dir = Path(self.output_dir)
        self.policy_path = self._as_path(self.policy_path)

    @staticmethod
    def _as_path(value: Path | str | None) -> Path | None:
        return Path(value) if value is not None else None

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của đường dẫn và miền giá trị tham số."""
        if not self.demo_mode:
            if self.person_model_path is None or not self.person_model_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy model phát hiện người: {self.person_model_path}"
                )
            if self.ppe_model_path is None or not self.ppe_model_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy model phát hiện PPE: {self.ppe_model_path}"
                )

        if (
            self.image_size <= 0
            or self.detection_interval <= 0
            or self.ppe_detection_interval <= 0
        ):
            raise ValueError("image_size và các chu kỳ inference phải lớn hơn 0.")

        if self.violation_confirmations <= 0:
            raise ValueError("violation_confirmations phải lớn hơn 0.")

        if self.resolution_confirmations <= 0:
            raise ValueError("resolution_confirmations phải lớn hơn 0.")

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

        if self.max_disappeared < 0 or self.max_missed_detections < 0:
            raise ValueError("max_disappeared và max_missed_detections không được là số âm.")

        if self.head_zone_min > self.head_zone_max or self.torso_zone_min > self.torso_zone_max:
            raise ValueError("Giới hạn body-zone không hợp lệ.")
        if self.low_match_threshold > self.tracker_iou:
            raise ValueError("low_match_threshold không được lớn hơn tracker_iou.")
        if self.roi_rule not in {"center", "overlap", "center_or_overlap"}:
            raise ValueError("roi_rule phải là center, overlap hoặc center_or_overlap.")
        if self.tracker_type.lower() not in {"two_threshold_iou", "iou", "bytetrack"}:
            raise ValueError("tracker_type không được hỗ trợ.")
        if self.roi_polygon is not None:
            if self.roi_polygon and len(self.roi_polygon) < 3:
                raise ValueError("roi_polygon phải có ít nhất 3 đỉnh.")
            for point in self.roi_polygon:
                if len(point) != 2:
                    raise ValueError("Mỗi điểm roi_polygon phải có dạng (x, y).")

    def to_dict(self) -> dict[str, Any]:
        """Chuyển cấu hình đã giải quyết thành dữ liệu có thể ghi JSON."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def load_from_policy(cls, policy_path: Path, **overrides: Any) -> DetectionConfig:
        """Nạp toàn bộ runtime policy và fail closed nếu file bị thiếu.

        Việc kiểm tra model được thực hiện ở ``validate`` để caller có thể
        truyền đường dẫn weights sau khi đọc policy.
        """
        policy_path = Path(policy_path)
        if not policy_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy runtime policy: {policy_path}")

        try:
            with policy_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
        except yaml.YAMLError as error:
            raise ValueError(f"Runtime policy không phải YAML hợp lệ: {policy_path}") from error

        if not isinstance(data, dict):
            raise ValueError("Runtime policy phải có cấu trúc YAML dạng object.")

        def section(name: str) -> dict[str, Any]:
            value = data.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"Nhóm cấu hình {name} phải là object.")
            return value

        inference = section("inference_contract")
        tracking = section("tracking_policy")
        spatial = section("spatial_association_policy")
        decision = section("decision_policy")
        head = spatial.get("head_zone", {})
        torso = spatial.get("torso_zone", {})
        if not isinstance(head, dict) or not isinstance(torso, dict):
            raise ValueError("head_zone và torso_zone phải là object.")
        roi_polygon_raw = decision.get("roi_polygon", spatial.get("roi_polygon"))
        roi_polygon = None
        if roi_polygon_raw is not None:
            if not isinstance(roi_polygon_raw, list):
                raise ValueError("roi_polygon phải là danh sách các điểm.")
            try:
                roi_polygon = [
                    (int(point[0]), int(point[1]))
                    for point in roi_polygon_raw
                    if isinstance(point, list | tuple) and len(point) == 2
                ]
            except (TypeError, ValueError, IndexError) as error:
                raise ValueError("roi_polygon phải chứa các điểm [x, y] hợp lệ.") from error
            if len(roi_polygon) != len(roi_polygon_raw):
                raise ValueError("roi_polygon phải chứa các điểm [x, y] hợp lệ.")

        kwargs: dict[str, Any] = {
            "image_size": inference.get("image_size", 640),
            "person_confidence": inference.get("person_confidence", 0.3),
            "ppe_confidence": inference.get("ppe_confidence", 0.3),
            "nms_iou": inference.get("nms_iou", 0.5),
            "person_roi_padding": inference.get("person_roi_padding_px", 10),
            "detection_interval": inference.get("person_detection_interval", 1),
            "ppe_detection_interval": inference.get("ppe_detection_interval", 4),
            "tracker_type": tracking.get("tracker_type", "two_threshold_iou"),
            "high_threshold": tracking.get("high_threshold", 0.5),
            "tracker_iou": tracking.get("match_threshold", 0.3),
            "low_match_threshold": tracking.get("low_match_threshold", 0.3),
            "max_missed_detections": tracking.get("max_missed_detections", 30),
            "track_ttl_seconds": tracking.get("track_ttl_seconds", 5.0),
            "enable_body_zone_filter": spatial.get("enable_body_zone_filter", True),
            "head_zone_min": head.get("y_min_ratio", 0.0),
            "head_zone_max": head.get("y_max_ratio", 0.35),
            "torso_zone_min": torso.get("y_min_ratio", 0.30),
            "torso_zone_max": torso.get("y_max_ratio", 0.75),
            "conflict_margin": decision.get("conflict_margin", 0.1),
            "violation_confirmations": decision.get("violation_confirmations", 2),
            "resolution_confirmations": decision.get("resolution_confirmations", 3),
            "violation_confirm_seconds": decision.get("violation_confirm_seconds", 0.5),
            "resolution_confirm_seconds": decision.get("resolution_confirm_seconds", 1.0),
            "alert_cooldown_seconds": decision.get("alert_cooldown_seconds", 10.0),
            "roi_rule": decision.get("roi_rule", "center_or_overlap"),
            "roi_overlap_threshold": decision.get("roi_overlap_threshold", 0.4),
            "roi_polygon": roi_polygon,
            "policy_path": policy_path,
        }
        kwargs.update(overrides)
        return cls(**kwargs)
