"""Module theo dõi đối tượng với baseline IoU hai ngưỡng.

Duy trì định danh ID ổn định và dự đoán chuyển động mượt mà (Motion Prediction) qua các khung hình:
- `TwoThresholdIoUTracker`: baseline nội bộ phân tách high/low confidence.
  Đây không phải implementation ByteTrack chuẩn của Ultralytics.
- `IoUTracker`: Thuật toán tham lam cổ điển (Greedy IoU) giữ làm baseline nhẹ.
- Hỗ trợ cập nhật chuyển động giữa các frame không chạy detector (chống hiện tượng freeze box).
- Chuẩn hóa ngữ nghĩa thời gian: `max_missed_detections` (chu kỳ detector) và `track_ttl_seconds`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .models import PPEState, PersonDetection, PPEStatus

LOGGER = logging.getLogger(__name__)


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Tính tỉ lệ phần giao trên phần hợp (Intersection over Union) của 2 bounding box.

    Args:
        box_a: Tọa độ [x1, y1, x2, y2] của khung A.
        box_b: Tọa độ [x1, y1, x2, y2] của khung B.

    Returns:
        Giá trị IoU nằm trong khoảng [0.0, 1.0].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _keep_previous_ppe(previous: PPEStatus, current: PPEStatus) -> PPEStatus:
    """Giữ bằng chứng PPE gần nhất khi frame hiện tại chỉ cập nhật person box."""
    has_observation = bool(current.detections) or any(
        state is not PPEState.UNKNOWN
        for state in (current.helmet_state, current.vest_state)
    )
    return current if has_observation else previous


@dataclass
class Track:
    """Đại diện cho trạng thái của một cá nhân đang được hệ thống theo dõi.

    Attributes:
        track_id: Mã ID định danh duy nhất của người.
        box: Tọa độ bounding box hiện tại [x1, y1, x2, y2].
        ppe: Trạng thái kiểm tra PPE hiện tại.
        confidence: Độ tin cậy phát hiện.
        disappeared: Số chu kỳ detector liên tiếp không thấy đối tượng (missed detections).
        updated: Cờ xác định vết này có được cập nhật quan sát ở frame hiện tại không.
        velocity: Vector vận tốc ước lượng [vx1, vy1, vx2, vy2] phục vụ nội suy chuyển động.
        total_observations: Tổng số lần đối tượng được phát hiện và cập nhật.
    """

    track_id: int
    box: list[float]
    ppe: PPEStatus = field(default_factory=PPEStatus)
    confidence: float = 1.0
    disappeared: int = 0
    updated: bool = True
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    total_observations: int = 1
    last_seen_sec: float | None = None

    def predict_next_box(self) -> list[float]:
        """Dự đoán vị trí tiếp theo bằng mô hình vận tốc tuyến tính mượt mà."""
        return [
            self.box[0] + self.velocity[0],
            self.box[1] + self.velocity[1],
            self.box[2] + self.velocity[2],
            self.box[3] + self.velocity[3],
        ]


class TrackerProtocol(Protocol):
    """Protocol chuẩn cho các tracker."""

    def update(
        self, detections: list[PersonDetection], timestamp_sec: float | None = None
    ) -> list[Track]:
        ...

    def predict(self, timestamp_sec: float | None = None) -> list[Track]:
        ...

    def active_tracks(self) -> list[Track]:
        ...


class IoUTracker:
    """Bộ theo dõi đối tượng dựa trên thuật toán ghép cặp Greedy IoU (Baseline)."""

    def __init__(
        self,
        threshold: float = 0.3,
        max_disappeared: int = 30,
        track_ttl_seconds: float | None = None,
    ) -> None:
        """Khởi tạo IoU Tracker.

        Args:
            threshold: Ngưỡng IoU tối thiểu để coi là cùng một đối tượng.
            max_disappeared: Số chu kỳ detector bỏ lỡ tối đa trước khi xóa ID.
        """
        self.threshold = threshold
        self.max_disappeared = max_disappeared
        self.track_ttl_seconds = track_ttl_seconds
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def predict(self, timestamp_sec: float | None = None) -> list[Track]:
        """Cập nhật vị trí dự đoán cho các frame trung gian không chạy detector (chống freeze box)."""
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if track.velocity != [0.0, 0.0, 0.0, 0.0]:
                track.box = track.predict_next_box()
            track.updated = False
            if (
                timestamp_sec is not None
                and track.last_seen_sec is not None
                and self.track_ttl_seconds is not None
                and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
            ):
                del self.tracks[track_id]
        return self.active_tracks()

    def update(
        self, detections: list[PersonDetection], timestamp_sec: float | None = None
    ) -> list[Track]:
        """Cập nhật vị trí vết theo dõi dựa trên danh sách detection ở khung hình mới."""
        if not detections:
            for track_id in list(self.tracks):
                track = self.tracks[track_id]
                track.disappeared += 1
                track.updated = False
                expired = (
                    timestamp_sec is not None
                    and track.last_seen_sec is not None
                    and self.track_ttl_seconds is not None
                    and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
                )
                if track.disappeared > self.max_disappeared or expired:
                    del self.tracks[track_id]
            return self.active_tracks()

        track_ids = list(self.tracks)
        if track_ids:
            scores = np.array(
                [[iou(self.tracks[tid].box, det.box) for det in detections] for tid in track_ids],
                dtype=float,
            )
        else:
            scores = np.empty((0, len(detections)))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        while scores.size > 0 and scores.max(initial=-1.0) >= self.threshold:
            row, col = np.unravel_index(scores.argmax(), scores.shape)
            tid = track_ids[row]
            det = detections[col]

            old_box = self.tracks[tid].box
            # Tính vector dịch chuyển làm vận tốc mượt mà
            velocity = [
                0.7 * self.tracks[tid].velocity[i] + 0.3 * (det.box[i] - old_box[i])
                for i in range(4)
            ]

            self.tracks[tid] = Track(
                track_id=tid,
                box=det.box,
                ppe=_keep_previous_ppe(self.tracks[tid].ppe, det.ppe),
                confidence=det.confidence,
                disappeared=0,
                updated=True,
                velocity=velocity,
                total_observations=self.tracks[tid].total_observations + 1,
                last_seen_sec=(
                    timestamp_sec
                    if timestamp_sec is not None
                    else self.tracks[tid].last_seen_sec
                ),
            )
            matched_tracks.add(tid)
            matched_detections.add(col)

            scores[row, :] = -1.0
            scores[:, col] = -1.0

        # Tạo mới các track chưa khớp
        for col, det in enumerate(detections):
            if col not in matched_detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(
                    track_id=tid,
                    box=det.box,
                    ppe=det.ppe,
                    confidence=det.confidence,
                    disappeared=0,
                    updated=True,
                    last_seen_sec=timestamp_sec,
                )
                matched_tracks.add(tid)

        # Xóa các track quá hạn
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.disappeared += 1
                track.updated = False
                if track.disappeared > self.max_disappeared:
                    del self.tracks[tid]

        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if (
                timestamp_sec is not None
                and track.last_seen_sec is not None
                and self.track_ttl_seconds is not None
                and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
            ):
                del self.tracks[track_id]

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        """Trả về danh sách các track còn hiệu lực sắp xếp theo ID tăng dần."""
        return [self.tracks[tid] for tid in sorted(self.tracks)]


class TwoThresholdIoUTracker:
    """Baseline IoU hai ngưỡng, không gọi nhầm là ByteTrack chuẩn."""

    def __init__(
        self,
        high_threshold: float = 0.5,
        match_threshold: float = 0.6,
        low_match_threshold: float = 0.4,
        max_missed_detections: int = 30,
        track_ttl_seconds: float | None = None,
    ) -> None:
        """Khởi tạo baseline IoU hai ngưỡng.

        Args:
            high_threshold: Ngưỡng phân tách detection độ tin cậy cao và thấp.
            match_threshold: Ngưỡng IoU ghép cặp giai đoạn 1 (high-score).
            low_match_threshold: Ngưỡng IoU ghép cặp giai đoạn 2 (low-score để cứu đối tượng bị che khuất).
            max_missed_detections: Số chu kỳ detector tối đa giữ track trước khi xóa.
        """
        self.high_threshold = high_threshold
        self.match_threshold = match_threshold
        self.low_match_threshold = low_match_threshold
        self.max_missed_detections = max_missed_detections
        self.track_ttl_seconds = track_ttl_seconds
        self.max_disappeared = max_missed_detections  # Alias tương thích ngược
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def predict(self, timestamp_sec: float | None = None) -> list[Track]:
        """Nội suy chuyển động giữa các frame không chạy detector."""
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if track.velocity != [0.0, 0.0, 0.0, 0.0]:
                track.box = track.predict_next_box()
            track.updated = False
            if (
                timestamp_sec is not None
                and track.last_seen_sec is not None
                and self.track_ttl_seconds is not None
                and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
            ):
                del self.tracks[track_id]
        return self.active_tracks()

    def update(
        self, detections: list[PersonDetection], timestamp_sec: float | None = None
    ) -> list[Track]:
        """Cập nhật tracker qua 2 giai đoạn ghép cặp."""
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid].disappeared += 1
                self.tracks[tid].updated = False
                track = self.tracks[tid]
                expired = (
                    timestamp_sec is not None
                    and track.last_seen_sec is not None
                    and self.track_ttl_seconds is not None
                    and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
                )
                if track.disappeared > self.max_missed_detections or expired:
                    del self.tracks[tid]
            return self.active_tracks()

        # Phân tách detections thành 2 nhóm: High Confidence và Low Confidence
        high_dets = [d for d in detections if d.confidence >= self.high_threshold]
        low_dets = [d for d in detections if d.confidence < self.high_threshold]

        track_ids = list(self.tracks)
        matched_tracks: set[int] = set()

        # Giai đoạn 1: Ghép cặp High-confidence detections với các track hiện có
        matched_high_dets: set[int] = set()
        if track_ids and high_dets:
            scores1 = np.array(
                [[iou(self.tracks[tid].box, det.box) for det in high_dets] for tid in track_ids],
                dtype=float,
            )
            while scores1.size > 0 and scores1.max(initial=-1.0) >= self.match_threshold:
                row, col = np.unravel_index(scores1.argmax(), scores1.shape)
                tid = track_ids[row]
                det = high_dets[col]

                old_box = self.tracks[tid].box
                velocity = [
                    0.7 * self.tracks[tid].velocity[i] + 0.3 * (det.box[i] - old_box[i])
                    for i in range(4)
                ]
                self.tracks[tid] = Track(
                    track_id=tid,
                    box=det.box,
                    ppe=_keep_previous_ppe(self.tracks[tid].ppe, det.ppe),
                    confidence=det.confidence,
                    disappeared=0,
                    updated=True,
                    velocity=velocity,
                    total_observations=self.tracks[tid].total_observations + 1,
                    last_seen_sec=(
                        timestamp_sec
                        if timestamp_sec is not None
                        else self.tracks[tid].last_seen_sec
                    ),
                )
                matched_tracks.add(tid)
                matched_high_dets.add(col)
                scores1[row, :] = -1.0
                scores1[:, col] = -1.0

        # Giai đoạn 2: Ghép cặp Low-confidence detections với các track chưa được ghép (cứu worker bị che/xa)
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        if unmatched_tracks and low_dets:
            scores2 = np.array(
                [[iou(self.tracks[tid].box, det.box) for det in low_dets] for tid in unmatched_tracks],
                dtype=float,
            )
            while scores2.size > 0 and scores2.max(initial=-1.0) >= self.low_match_threshold:
                row, col = np.unravel_index(scores2.argmax(), scores2.shape)
                tid = unmatched_tracks[row]
                det = low_dets[col]

                old_box = self.tracks[tid].box
                velocity = [
                    0.7 * self.tracks[tid].velocity[i] + 0.3 * (det.box[i] - old_box[i])
                    for i in range(4)
                ]
                self.tracks[tid] = Track(
                    track_id=tid,
                    box=det.box,
                    ppe=_keep_previous_ppe(self.tracks[tid].ppe, det.ppe),
                    confidence=det.confidence,
                    disappeared=0,
                    updated=True,
                    velocity=velocity,
                    total_observations=self.tracks[tid].total_observations + 1,
                    last_seen_sec=(
                        timestamp_sec
                        if timestamp_sec is not None
                        else self.tracks[tid].last_seen_sec
                    ),
                )
                matched_tracks.add(tid)
                scores2[row, :] = -1.0
                scores2[:, col] = -1.0

        # Khởi tạo track mới từ High-confidence detections còn sót lại
        for col, det in enumerate(high_dets):
            if col not in matched_high_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(
                    track_id=tid,
                    box=det.box,
                    ppe=det.ppe,
                    confidence=det.confidence,
                    disappeared=0,
                    updated=True,
                    last_seen_sec=timestamp_sec,
                )
                matched_tracks.add(tid)

        # Cập nhật số chu kỳ detector bỏ lỡ và xóa track quá hạn
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track.disappeared += 1
                track.updated = False
                expired = (
                    timestamp_sec is not None
                    and track.last_seen_sec is not None
                    and self.track_ttl_seconds is not None
                    and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
                )
                if track.disappeared > self.max_missed_detections or expired:
                    del self.tracks[tid]

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        return [self.tracks[tid] for tid in sorted(self.tracks)]


# Tên cũ giữ để code bên ngoài không hỏng; production dùng tên chính xác ở trên.
ByteTrack = TwoThresholdIoUTracker
