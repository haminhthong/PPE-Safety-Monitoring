"""Module thực hiện suy luận (inference) hai giai đoạn bằng YOLO kèm liên kết không gian (Spatial Association).

Giai đoạn 1: Mô hình YOLO phát hiện đối tượng người (Person) trong khung hình.
Giai đoạn 2: Trích xuất vùng ảnh ROI người và đưa vào mô hình YOLO thứ 2 để nhận diện PPE
(Mũ bảo hộ 'helmet'/'no-helmet', Áo phản quang 'vest'/'no-vest').
Đặc biệt: Lưu lại bounding box của PPE và xác thực phân vùng cơ thể (Body-Zone Validation):
- Mũ bảo hộ ('helmet', 'no-helmet') bắt buộc phải nằm ở vùng đầu (Head Zone: y <= 35% ROI).
- Áo phản quang ('vest', 'no-vest') bắt buộc phải nằm ở vùng thân (Torso Zone: 30% <= y <= 75% ROI).
Giảm rủi ro gán nhầm trang bị trong tình huống đám đông đứng sát nhau.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from .config import DetectionConfig
from .crops import CropWindow, PersonCropBuilder
from .models import PPEState, PersonDetection, PPEDetection, PPEStatus

LOGGER = logging.getLogger(__name__)


def select_device() -> str:
    """Tự động chọn phần cứng tăng tốc inference tốt nhất khả dụng."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_center_in_roi(box: list[float], roi_polygon: list[tuple[int, int]]) -> bool:
    """Kiểm tra xem điểm tâm của bounding box có nằm trong vùng nguy hiểm ROI hay không."""
    if not roi_polygon or len(roi_polygon) < 3:
        return True

    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    pts = np.array(roi_polygon, dtype=np.int32)
    res = cv2.pointPolygonTest(pts, (float(center_x), float(center_y)), False)
    return res >= 0


def _clip_polygon(polygon: list[tuple[float, float]], axis: int, value: float, keep_greater: bool) -> list[tuple[float, float]]:
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
    """Tính diện tích đa giác bằng công thức dây giày."""
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


def roi_overlap_ratio(box: list[float], roi_polygon: list[tuple[int, int]]) -> float:
    """Tính diện tích giao giữa box và polygon chia cho diện tích box."""
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
    """Kiểm tra đúng tỷ lệ diện tích box nằm trong polygon ROI."""
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap phải nằm trong khoảng từ 0.0 đến 1.0.")
    return roi_overlap_ratio(box, roi_polygon) >= min_overlap


def is_box_in_roi_policy(
    box: list[float],
    roi_polygon: list[tuple[int, int]] | None,
    rule: str,
    min_overlap: float,
) -> bool:
    """Áp dụng đúng rule ROI đã khai báo trong runtime policy."""
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
    """Kiểm tra xem phát hiện PPE có nằm đúng phân vùng giải phẫu cơ thể tương ứng không.

    Args:
        label: Tên nhãn ('helmet', 'no-helmet', 'vest', 'no-vest').
        box: Tọa độ bbox [x1, y1, x2, y2] tính theo pixel trong ROI.
        roi_h: Chiều cao của vùng ROI người.
        roi_w: Chiều rộng của vùng ROI người.
        head_max: Tỷ lệ chiều cao tối đa cho vùng đầu (mặc định: 35%).
        torso_min: Tỷ lệ chiều cao tối thiểu cho vùng thân (mặc định: 30%).
        torso_max: Tỷ lệ chiều cao tối đa cho vùng thân (mặc định: 75%).

    Returns:
        True nếu phát hiện nằm đúng phân vùng giải phẫu hợp lệ, ngược lại False.
    """
    if roi_h <= 0 or roi_w <= 0:
        return False

    norm_y_center = ((box[1] + box[3]) / 2.0) / float(roi_h)
    clean_label = label.strip().lower().replace("_", "-").replace(" ", "-")

    if clean_label in ("helmet", "no-helmet"):
        # Mũ phải nằm trong vùng đầu của person box gốc.
        return head_min <= norm_y_center <= head_max
    if clean_label in ("vest", "no-vest"):
        # Áo phản quang phải nằm ở vùng thân giữa
        return torso_min <= norm_y_center <= torso_max

    return True


class DetectorProtocol(Protocol):
    """Protocol chuẩn cho các lớp detector trong ứng dụng."""

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Phát hiện danh sách đối tượng người và PPE trên khung hình."""
        ...

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Chỉ phát hiện đối tượng người (phục vụ luồng Track-First)."""
        ...

    def analyze_ppe_for_roi(
        self,
        roi: np.ndarray,
        person_box: list[float] | None = None,
        crop_window: CropWindow | None = None,
    ) -> PPEStatus:
        """Nhận diện PPE và kiểm tra liên kết không gian cho một vùng ROI người."""
        ...


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
        """Phát hiện các bounding box người trên khung hình nguyên bản."""
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
        """Nhận diện trang bị bảo hộ trên vùng ảnh cắt của người kèm lọc không gian giải phẫu."""
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

            # Map từ tọa độ crop về tọa độ person box gốc trước khi kiểm tra zone.
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
                    "Bỏ qua phát hiện [%s] tại vị trí y=%.2f do không đúng phân vùng cơ thể (crowded scene noise).",
                    name,
                    (b_box[1] + b_box[3]) / (2.0 * roi_h),
                )
                continue

            labels.append(PPEDetection(label=name, confidence=conf, box=b_box))

        return self._build_ppe_status(labels)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Phát hiện người và kiểm tra trạng thái trang bị bảo hộ trên khung hình (chuẩn 2 giai đoạn)."""
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
                    # Giữ box người nguyên bản; padding chỉ dùng cho ảnh crop.
                    box=p_box,
                    confidence=p_conf,
                    ppe=ppe_status,
                )
            )

        return detections

    def _build_ppe_status(self, labels: list[PPEDetection]) -> PPEStatus:
        """Phân tích logic vi phạm dựa trên confidence, conflict_margin và scores."""
        helmet_scores = [
            item.confidence
            for item in labels
            if self._normalize(item.label) == "helmet"
        ]
        no_helmet_scores = [
            item.confidence
            for item in labels
            if self._normalize(item.label) == "no-helmet"
        ]
        vest_scores = [
            item.confidence
            for item in labels
            if self._normalize(item.label) == "vest"
        ]
        no_vest_scores = [
            item.confidence
            for item in labels
            if self._normalize(item.label) == "no-vest"
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
            helmet_evidence=[item for item in labels if self._normalize(item.label) in {"helmet", "no-helmet"}],
            vest_evidence=[item for item in labels if self._normalize(item.label) in {"vest", "no-vest"}],
        )

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().lower().replace("_", "-").replace(" ", "-")

    def _resolve_state(self, present_score: float, absent_score: float) -> PPEState:
        """Giải quyết PRESENT/ABSENT/UNKNOWN theo ngưỡng và conflict margin."""
        present = present_score >= self.config.ppe_confidence
        absent = absent_score >= self.config.ppe_confidence
        if absent and absent_score > present_score + self.config.conflict_margin:
            return PPEState.ABSENT
        if present and present_score > absent_score + self.config.conflict_margin:
            return PPEState.PRESENT
        return PPEState.UNKNOWN


class SyntheticDemoDetector:
    """Detector mô phỏng pipeline giả lập phục vụ thử nghiệm và demo không cần weights ngoài."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.frame_counter = 0
        LOGGER.info("Khởi chạy SyntheticDemoDetector (Chế độ mô phỏng pipeline - Zero-Setup).")

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Mô phỏng phát hiện các vị trí người."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        self.frame_counter += 1
        height, width = frame.shape[:2]
        w, h = int(width * 0.2), int(height * 0.5)

        x1_a = int(width * 0.15 + np.sin(self.frame_counter * 0.05) * 15)
        y1_a = int(height * 0.25)

        x1_b = int(width * 0.6 + np.cos(self.frame_counter * 0.05) * 15)
        y1_b = int(height * 0.2)

        candidates = [
            ([float(x1_a), float(y1_a), float(x1_a + w), float(y1_a + h)], 0.92),
            ([float(x1_b), float(y1_b), float(x1_b + w), float(y1_b + h)], 0.88),
        ]

        if self.config.roi_polygon:
            return [
                c
                for c in candidates
                if is_box_in_roi_policy(
                    c[0],
                    self.config.roi_polygon,
                    self.config.roi_rule,
                    self.config.roi_overlap_threshold,
                )
            ]
        return candidates

    def analyze_ppe_for_roi(
        self,
        roi: np.ndarray,
        person_box: list[float] | None = None,
        crop_window: CropWindow | None = None,
    ) -> PPEStatus:
        """Mô phỏng phân tích trạng thái PPE cho một vùng ROI."""
        roi_h, roi_w = (roi.shape[:2]) if (roi is not None and roi.size > 0) else (100, 100)
        # Giả lập mặc định vi phạm theo frame counter chẵn lẻ
        is_violating = (self.frame_counter // 20) % 2 == 1

        if is_violating:
            detections = [
                PPEDetection(
                    label="no-helmet",
                    confidence=0.85,
                    box=[float(roi_w * 0.2), float(roi_h * 0.05), float(roi_w * 0.8), float(roi_h * 0.30)],
                ),
                PPEDetection(
                    label="no-vest",
                    confidence=0.82,
                    box=[float(roi_w * 0.1), float(roi_h * 0.35), float(roi_w * 0.9), float(roi_h * 0.70)],
                ),
            ]
            return PPEStatus(
                detections=detections,
                helmet_state=PPEState.ABSENT,
                vest_state=PPEState.ABSENT,
                no_helmet_score=0.85,
                no_vest_score=0.82,
                helmet_evidence=detections[:1],
                vest_evidence=detections[1:],
            )
        else:
            detections = [
                PPEDetection(
                    label="helmet",
                    confidence=0.89,
                    box=[float(roi_w * 0.2), float(roi_h * 0.05), float(roi_w * 0.8), float(roi_h * 0.30)],
                ),
                PPEDetection(
                    label="vest",
                    confidence=0.86,
                    box=[float(roi_w * 0.1), float(roi_h * 0.35), float(roi_w * 0.9), float(roi_h * 0.70)],
                ),
            ]
            return PPEStatus(
                detections=detections,
                helmet_state=PPEState.PRESENT,
                vest_state=PPEState.PRESENT,
                helmet_score=0.89,
                vest_score=0.86,
                helmet_evidence=detections[:1],
                vest_evidence=detections[1:],
            )

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Tạo đối tượng mô phỏng với bounding box và vi phạm trên ảnh."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")

        self.frame_counter += 1
        height, width = frame.shape[:2]
        detections: list[PersonDetection] = []

        w, h = int(width * 0.2), int(height * 0.5)
        x1_a = int(width * 0.15 + np.sin(self.frame_counter * 0.05) * 15)
        y1_a = int(height * 0.25)
        det_a = PersonDetection(
            box=[float(x1_a), float(y1_a), float(x1_a + w), float(y1_a + h)],
            confidence=0.92,
            ppe=PPEStatus(
                detections=[
                    PPEDetection(label="helmet", confidence=0.89, box=[10.0, 5.0, float(w - 10), float(h * 0.28)]),
                    PPEDetection(label="vest", confidence=0.86, box=[5.0, float(h * 0.35), float(w - 5), float(h * 0.70)]),
                ],
                helmet_state=PPEState.PRESENT,
                vest_state=PPEState.PRESENT,
                helmet_score=0.89,
                vest_score=0.86,
            ),
        )
        detections.append(det_a)

        x1_b = int(width * 0.6 + np.cos(self.frame_counter * 0.05) * 15)
        y1_b = int(height * 0.2)
        det_b = PersonDetection(
            box=[float(x1_b), float(y1_b), float(x1_b + w), float(y1_b + h)],
            confidence=0.88,
            ppe=PPEStatus(
                detections=[
                    PPEDetection(label="no-helmet", confidence=0.85, box=[10.0, 5.0, float(w - 10), float(h * 0.28)]),
                    PPEDetection(label="no-vest", confidence=0.81, box=[5.0, float(h * 0.35), float(w - 5), float(h * 0.70)]),
                ],
                helmet_state=PPEState.ABSENT,
                vest_state=PPEState.ABSENT,
                no_helmet_score=0.85,
                no_vest_score=0.81,
            ),
        )
        detections.append(det_b)

        if self.config.roi_polygon:
            return [
                detection
                for detection in detections
                if is_box_in_roi_policy(
                    detection.box,
                    self.config.roi_polygon,
                    self.config.roi_rule,
                    self.config.roi_overlap_threshold,
                )
            ]
        return detections


MockDetector = SyntheticDemoDetector
