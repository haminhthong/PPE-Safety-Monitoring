"""Suy luận hai giai đoạn (Two-Stage Inference) bằng YOLO
và liên kết không gian (Spatial Association).

Giai đoạn 1: Phát hiện đối tượng người (Person) trên khung hình gốc.
Giai đoạn 2: Cắt vùng ảnh ROI người (với padding) và đưa vào mô hình YOLO thứ 2 để nhận diện PPE
('helmet', 'no-helmet', 'vest', 'no-vest').

Kiểm tra phân vùng giải phẫu cơ thể (Body-Zone Validation):
- Mũ bảo hộ ('helmet', 'no-helmet') nằm ở vùng đầu (Head Zone: y <= 35% chiều cao người).
- Áo phản quang ('vest', 'no-vest') nằm ở vùng thân (Torso Zone: 30% <= y <= 75% chiều cao người).
Giúp giảm triệt để lỗi gán nhầm trang bị khi nhiều công nhân đứng sát nhau.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import DetectionConfig
from .crops import CropWindow, PersonCropBuilder
from .models import PersonDetection, PPEDetection, PPEState, PPEStatus

LOGGER = logging.getLogger(__name__)


def read_image(path: Path | str) -> np.ndarray | None:
    """Đọc ảnh an toàn trên mọi hệ điều hành (kể cả đường dẫn Unicode tiếng Việt trên Windows)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as err:
        LOGGER.error("Lỗi đọc ảnh từ %s: %s", p, err)
        return None


def write_image(path: Path | str, image: np.ndarray) -> bool:
    """Ghi ảnh an toàn trên mọi hệ điều hành (kể cả đường dẫn Unicode tiếng Việt trên Windows)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix if p.suffix else ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    try:
        encoded.tofile(str(p))
        return True
    except Exception as err:
        LOGGER.error("Lỗi ghi ảnh vào %s: %s", p, err)
        return False


def select_device() -> str:
    """Tự động chọn phần cứng tăng tốc inference tốt nhất khả dụng."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_center_in_roi(box: list[float], roi_polygon: list[tuple[int, int]]) -> bool:
    """Kiểm tra tâm của bounding box có nằm trong đa giác ROI hay không."""
    if not roi_polygon or len(roi_polygon) < 3:
        return True

    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    pts = np.array(roi_polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(pts, (float(center_x), float(center_y)), False)
    return res >= 0


def _clip_polygon(
    polygon: list[tuple[float, float]], axis: int, value: float, keep_greater: bool
) -> list[tuple[float, float]]:
    """Cắt đa giác theo một cạnh ngang/dọc của hình chữ nhật."""
    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    previous = polygon[-1]

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= value if keep_greater else point[axis] <= value

    for current in polygon:
        current_inside = inside(current)
        previous_inside = inside(previous)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            ratio = (value - previous[axis]) / denominator if denominator else 0.0
            intersection = (
                previous[0] + ratio * (current[0] - previous[0]),
                previous[1] + ratio * (current[1] - previous[1]),
            )
            result.append(intersection)
        if current_inside:
            result.append(current)
        previous = current
    return result


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Tính diện tích đa giác bằng công thức Shoelace."""
    if len(polygon) < 3:
        return 0.0
    return (
        abs(
            sum(
                polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                for i in range(len(polygon))
            )
        )
        / 2.0
    )


def roi_overlap_ratio(box: list[float], roi_polygon: list[tuple[int, int]]) -> float:
    """Tính tỷ lệ diện tích giao giữa box và đa giác ROI trên diện tích box."""
    if not roi_polygon or len(roi_polygon) < 3:
        return 1.0
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area == 0.0:
        return 0.0
    polygon = [(float(x), float(y)) for x, y in roi_polygon]
    for axis, value, keep_greater in (
        (0, x1, True),
        (0, x2, False),
        (1, y1, True),
        (1, y2, False),
    ):
        polygon = _clip_polygon(polygon, axis, value, keep_greater)
    return min(1.0, _polygon_area(polygon) / box_area)


def is_box_overlapping_roi(
    box: list[float], roi_polygon: list[tuple[int, int]], min_overlap: float = 0.3
) -> bool:
    """Kiểm tra diện tích box nằm trong đa giác ROI có đạt ngưỡng tối thiểu không."""
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap phải nằm trong khoảng từ 0.0 đến 1.0.")
    return roi_overlap_ratio(box, roi_polygon) >= min_overlap


def is_box_in_roi_policy(
    box: list[float],
    roi_polygon: list[tuple[int, int]] | None,
    rule: str,
    min_overlap: float,
) -> bool:
    """Kiểm tra box thỏa mãn quy tắc ROI cấu hình."""
    if not roi_polygon or len(roi_polygon) < 3:
        return True
    if rule not in {"center", "overlap", "center_or_overlap"}:
        raise ValueError(f"roi_rule không hợp lệ: {rule}")
    if rule == "center":
        return is_center_in_roi(box, roi_polygon)
    if rule == "overlap":
        return is_box_overlapping_roi(box, roi_polygon, min_overlap)
    return is_center_in_roi(box, roi_polygon) or is_box_overlapping_roi(
        box, roi_polygon, min_overlap
    )


def validate_body_zone(
    label: str,
    box: list[float],
    roi_h: int,
    roi_w: int,
    head_max: float = 0.35,
    torso_min: float = 0.30,
    torso_max: float = 0.75,
    head_min: float = 0.0,
) -> bool:
    """Kiểm tra phát hiện PPE có nằm đúng phân vùng giải phẫu cơ thể tương ứng không.

    Args:
        label: Tên nhãn ('helmet', 'no-helmet', 'vest', 'no-vest').
        box: Tọa độ bbox [x1, y1, x2, y2] tính theo pixel trong ROI người.
        roi_h: Chiều cao ROI người.
        roi_w: Chiều rộng ROI người.
        head_max: Tỷ lệ chiều cao tối đa cho vùng đầu (mặc định: 35%).
        torso_min: Tỷ lệ chiều cao tối thiểu cho vùng thân (mặc định: 30%).
        torso_max: Tỷ lệ chiều cao tối đa cho vùng thân (mặc định: 75%).
        head_min: Tỷ lệ chiều cao tối thiểu cho vùng đầu (mặc định: 0%).

    Returns:
        True nếu vị trí PPE hợp lệ với giải phẫu cơ thể.
    """
    if roi_h <= 0 or roi_w <= 0:
        return False

    norm_y_center = ((box[1] + box[3]) / 2.0) / float(roi_h)
    clean_label = label.strip().lower().replace("_", "-").replace(" ", "-")

    if clean_label in ("helmet", "no-helmet"):
        return head_min <= norm_y_center <= head_max
    if clean_label in ("vest", "no-vest"):
        return torso_min <= norm_y_center <= torso_max

    return True


class DetectorProtocol(Protocol):
    """Protocol cho các lớp detector."""

    def detect(self, frame: np.ndarray) -> list[PersonDetection]: ...

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]: ...

    def analyze_ppe_for_roi(
        self,
        roi: np.ndarray,
        person_box: list[float] | None = None,
        crop_window: CropWindow | None = None,
    ) -> PPEStatus: ...


class DualModelDetector:
    """Bộ phát hiện hai giai đoạn (Two-Stage Detector) dựa trên Ultralytics YOLO."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.device = select_device()
        from ultralytics import YOLO

        LOGGER.info("Khởi tạo DualModelDetector trên thiết bị: %s", self.device)
        self.person_model = YOLO(str(config.person_model_path))
        self.ppe_model = YOLO(str(config.ppe_model_path))
        self.crop_builder = PersonCropBuilder(config.person_roi_padding)
        self.person_model.to(self.device)
        self.ppe_model.to(self.device)

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Giai đoạn 1: Phát hiện bounding box người trên khung hình gốc."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        results = self.person_model.predict(
            frame,
            imgsz=self.config.image_size,
            conf=self.config.person_confidence,
            iou=self.config.nms_iou,
            classes=[0],  # COCO Class 0 = 'person'
            device=self.device,
            verbose=False,
        )

        height, width = frame.shape[:2]
        persons: list[tuple[list[float], float]] = []
        if not results:
            return persons

        for detection in results[0].boxes:
            raw_box = detection.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = map(int, raw_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            person_box = [float(x1), float(y1), float(x2), float(y2)]
            if not is_box_in_roi_policy(
                person_box,
                self.config.roi_polygon,
                self.config.roi_rule,
                self.config.roi_overlap_threshold,
            ):
                continue

            conf = float(detection.conf[0])
            persons.append((person_box, conf))

        return persons

    def analyze_ppe_for_roi(
        self,
        roi: np.ndarray,
        person_box: list[float] | None = None,
        crop_window: CropWindow | None = None,
    ) -> PPEStatus:
        """Giai đoạn 2: Nhận diện PPE trên ảnh crop người kèm bộ lọc vị trí giải phẫu cơ thể."""
        if roi is None or roi.size == 0:
            return PPEStatus()

        roi_h, roi_w = roi.shape[:2]
        if person_box is None:
            person_box = [0.0, 0.0, float(roi_w), float(roi_h)]
        if crop_window is None:
            crop_window = CropWindow(0, 0, roi_w, roi_h)
        person_h = max(0.0, person_box[3] - person_box[1])
        person_w = max(0.0, person_box[2] - person_box[0])
        labels: list[PPEDetection] = []

        results = self.ppe_model.predict(
            roi,
            imgsz=self.config.image_size,
            conf=self.config.ppe_confidence,
            iou=self.config.nms_iou,
            device=self.device,
            verbose=False,
        )

        if not results:
            return self._build_ppe_status(labels)

        result = results[0]
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(self.ppe_model.names[class_id])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().tolist()
            b_box = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]

            # Map từ tọa độ crop về tọa độ person box gốc trước khi kiểm tra zone
            original_box = [
                b_box[0] + crop_window.x1 - person_box[0],
                b_box[1] + crop_window.y1 - person_box[1],
                b_box[2] + crop_window.x1 - person_box[0],
                b_box[3] + crop_window.y1 - person_box[1],
            ]
            if self.config.enable_body_zone_filter and not validate_body_zone(
                name,
                original_box,
                int(round(person_h)),
                int(round(person_w)),
                head_min=self.config.head_zone_min,
                head_max=self.config.head_zone_max,
                torso_min=self.config.torso_zone_min,
                torso_max=self.config.torso_zone_max,
            ):
                LOGGER.debug(
                    "Bỏ qua [%s] tại vị trí y=%.2f vì không đúng vùng cơ thể.",
                    name,
                    (b_box[1] + b_box[3]) / (2.0 * roi_h),
                )
                continue

            labels.append(PPEDetection(label=name, confidence=conf, box=b_box))

        return self._build_ppe_status(labels)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Quy trình hoàn chỉnh hai giai đoạn trên một khung hình."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        persons = self.detect_persons(frame)
        detections: list[PersonDetection] = []

        for p_box, p_conf in persons:
            roi, crop_window = self.crop_builder.crop(frame, p_box)
            ppe_status = self.analyze_ppe_for_roi(
                roi,
                person_box=p_box,
                crop_window=crop_window,
            )
            detections.append(
                PersonDetection(
                    box=p_box,
                    confidence=p_conf,
                    ppe=ppe_status,
                )
            )

        return detections

    def _build_ppe_status(self, labels: list[PPEDetection]) -> PPEStatus:
        """Phân tích trạng thái PPE dựa trên confidence và conflict_margin."""
        helmet_scores = [
            item.confidence for item in labels if self._normalize(item.label) == "helmet"
        ]
        no_helmet_scores = [
            item.confidence for item in labels if self._normalize(item.label) == "no-helmet"
        ]
        vest_scores = [item.confidence for item in labels if self._normalize(item.label) == "vest"]
        no_vest_scores = [
            item.confidence for item in labels if self._normalize(item.label) == "no-vest"
        ]

        h_score = max(helmet_scores, default=0.0)
        nh_score = max(no_helmet_scores, default=0.0)
        v_score = max(vest_scores, default=0.0)
        nv_score = max(no_vest_scores, default=0.0)

        helmet_state = self._resolve_state(h_score, nh_score)
        vest_state = self._resolve_state(v_score, nv_score)

        return PPEStatus(
            detections=labels,
            helmet_state=helmet_state,
            vest_state=vest_state,
            helmet_score=h_score,
            no_helmet_score=nh_score,
            vest_score=v_score,
            no_vest_score=nv_score,
            helmet_evidence=[
                item for item in labels if self._normalize(item.label) in {"helmet", "no-helmet"}
            ],
            vest_evidence=[
                item for item in labels if self._normalize(item.label) in {"vest", "no-vest"}
            ],
        )

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().lower().replace("_", "-").replace(" ", "-")

    def _resolve_state(self, present_score: float, absent_score: float) -> PPEState:
        """Giải quyết trạng thái PRESENT / ABSENT / UNKNOWN theo ngưỡng và conflict margin."""
        present = present_score >= self.config.ppe_confidence
        absent = absent_score >= self.config.ppe_confidence
        if absent and absent_score > present_score + self.config.conflict_margin:
            return PPEState.ABSENT
        if present and present_score > absent_score + self.config.conflict_margin:
            return PPEState.PRESENT
        return PPEState.UNKNOWN
