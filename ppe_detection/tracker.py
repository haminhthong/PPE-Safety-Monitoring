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
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from .models import PersonDetection, PPEState, PPEStatus

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
    """Giữ bằng chứng PPE gần nhất cho phần chưa được kiểm tra ở frame này.

    Person tracking và PPE inspection có cadence khác nhau. Một detection người
    không kèm PPE observation không được phép biến trạng thái đã biết thành
    ``UNKNOWN``. Nếu caller chỉ cập nhật một loại PPE, loại còn lại cũng được
    giữ nguyên để tránh làm mất evidence hợp lệ.
    """
    helmet_observed = (
        current.helmet_state is not PPEState.UNKNOWN
        or bool(current.helmet_evidence)
        or current.helmet_score > 0.0
        or current.no_helmet_score > 0.0
    )
    vest_observed = (
        current.vest_state is not PPEState.UNKNOWN
        or bool(current.vest_evidence)
        or current.vest_score > 0.0
        or current.no_vest_score > 0.0
    )
    if helmet_observed and vest_observed:
        return current
    if not helmet_observed and not vest_observed and not current.detections:
        return previous

    return replace(
        current,
        helmet_state=current.helmet_state if helmet_observed else previous.helmet_state,
        helmet_score=current.helmet_score if helmet_observed else previous.helmet_score,
        no_helmet_score=(current.no_helmet_score if helmet_observed else previous.no_helmet_score),
        helmet_evidence=(current.helmet_evidence if helmet_observed else previous.helmet_evidence),
        vest_state=current.vest_state if vest_observed else previous.vest_state,
        vest_score=current.vest_score if vest_observed else previous.vest_score,
        no_vest_score=current.no_vest_score if vest_observed else previous.no_vest_score,
        vest_evidence=current.vest_evidence if vest_observed else previous.vest_evidence,
        detections=current.detections or previous.detections,
    )


def _smooth_velocity(
    previous: Track, new_box: list[float], timestamp_sec: float | None
) -> list[float]:
    """Ước lượng vận tốc theo giây khi tracker nhận timestamp."""
    elapsed = 1.0
    if timestamp_sec is not None:
        previous_time = previous.last_position_sec
        if previous_time is None:
            previous_time = previous.last_seen_sec
        if previous_time is not None:
            elapsed = max(timestamp_sec - previous_time, 1e-6)
    observed_velocity = [(new_box[index] - previous.box[index]) / elapsed for index in range(4)]
    return [0.7 * previous.velocity[index] + 0.3 * observed_velocity[index] for index in range(4)]


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
        last_position_sec: Thời điểm cuối cùng box được cập nhật, kể cả bằng dự đoán.
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
    last_position_sec: float | None = None

    def predict_next_box(self, elapsed_sec: float = 1.0) -> list[float]:
        """Dự đoán vị trí tiếp theo bằng mô hình vận tốc tuyến tính mượt mà."""
        return [self.box[index] + self.velocity[index] * elapsed_sec for index in range(4)]


class TrackerProtocol(Protocol):
    """Protocol chuẩn cho các tracker."""

    def update(
        self, detections: list[PersonDetection], timestamp_sec: float | None = None
    ) -> list[Track]: ...

    def predict(self, timestamp_sec: float | None = None) -> list[Track]: ...

    def active_tracks(self) -> list[Track]: ...


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
        """Dự đoán vị trí ở frame không chạy detector để tránh freeze box."""
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            elapsed_sec = 1.0
            if timestamp_sec is not None:
                previous_time = track.last_position_sec
                if previous_time is None:
                    previous_time = track.last_seen_sec
                if previous_time is not None:
                    elapsed_sec = max(timestamp_sec - previous_time, 0.0)
            if track.velocity != [0.0, 0.0, 0.0, 0.0]:
                track.box = track.predict_next_box(elapsed_sec)
            if timestamp_sec is not None:
                track.last_position_sec = timestamp_sec
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

            # Tính vector dịch chuyển làm vận tốc mượt mà
            velocity = _smooth_velocity(self.tracks[tid], det.box, timestamp_sec)

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
                    timestamp_sec if timestamp_sec is not None else self.tracks[tid].last_seen_sec
                ),
                last_position_sec=(
                    timestamp_sec
                    if timestamp_sec is not None
                    else self.tracks[tid].last_position_sec
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
                    last_position_sec=timestamp_sec,
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
            low_match_threshold: Ngưỡng IoU ghép cặp giai đoạn 2 cho đối tượng bị che.
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
            elapsed_sec = 1.0
            if timestamp_sec is not None:
                previous_time = track.last_position_sec
                if previous_time is None:
                    previous_time = track.last_seen_sec
                if previous_time is not None:
                    elapsed_sec = max(timestamp_sec - previous_time, 0.0)
            if track.velocity != [0.0, 0.0, 0.0, 0.0]:
                track.box = track.predict_next_box(elapsed_sec)
            if timestamp_sec is not None:
                track.last_position_sec = timestamp_sec
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

                velocity = _smooth_velocity(self.tracks[tid], det.box, timestamp_sec)
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
                    last_position_sec=(
                        timestamp_sec
                        if timestamp_sec is not None
                        else self.tracks[tid].last_position_sec
                    ),
                )
                matched_tracks.add(tid)
                matched_high_dets.add(col)
                scores1[row, :] = -1.0
                scores1[:, col] = -1.0

        # Giai đoạn 2: ghép detection điểm thấp với track chưa được ghép.
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        if unmatched_tracks and low_dets:
            scores2 = np.array(
                [
                    [iou(self.tracks[tid].box, det.box) for det in low_dets]
                    for tid in unmatched_tracks
                ],
                dtype=float,
            )
            while scores2.size > 0 and scores2.max(initial=-1.0) >= self.low_match_threshold:
                row, col = np.unravel_index(scores2.argmax(), scores2.shape)
                tid = unmatched_tracks[row]
                det = low_dets[col]

                velocity = _smooth_velocity(self.tracks[tid], det.box, timestamp_sec)
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
                    last_position_sec=(
                        timestamp_sec
                        if timestamp_sec is not None
                        else self.tracks[tid].last_position_sec
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
                    last_position_sec=timestamp_sec,
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
