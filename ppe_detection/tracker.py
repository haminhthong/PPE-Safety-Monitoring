"""Module theo dõi đối tượng (Multi-Object Tracking) bằng Two-Threshold IoU và Greedy IoU.

Duy trì định danh ID của từng công nhân qua các khung hình video và dự đoán vị trí
khi khung hình không chạy detector để tiết kiệm tài nguyên tính toán:
- `TwoThresholdIoUTracker`: Ghép cặp 2 giai đoạn (high-confidence match trước, sau đó
  dùng low-confidence detections để phục hồi các track bị che khuất).
- `IoUTracker`: Ghép cặp tham lam (Greedy IoU) nhẹ nhàng làm baseline.
- Quản lý vòng đời track qua số chu kỳ mất dấu (`max_missed`) và TTL (`track_ttl_seconds`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from .models import PersonDetection, PPEState, PPEStatus

LOGGER = logging.getLogger(__name__)


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Tính tỷ lệ phần giao trên phần hợp (Intersection over Union) giữa 2 bounding box."""
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
    """Giữ bằng chứng PPE gần nhất cho phần chưa được kiểm tra ở frame hiện tại."""
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
    """Ước lượng vận tốc dịch chuyển tuyến tính mượt mà."""
    elapsed = 1.0
    if timestamp_sec is not None:
        previous_time = previous.last_position_sec
        if previous_time is None:
            previous_time = previous.last_seen_sec
        if previous_time is not None:
            elapsed = max(timestamp_sec - previous_time, 1e-6)
    observed_velocity = [(new_box[i] - previous.box[i]) / elapsed for i in range(4)]
    return [0.7 * previous.velocity[i] + 0.3 * observed_velocity[i] for i in range(4)]


@dataclass
class Track:
    """Đại diện cho trạng thái của một người đang được hệ thống theo dõi."""

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
        """Dự đoán vị trí tiếp theo bằng vận tốc tuyến tính."""
        return [self.box[i] + self.velocity[i] * elapsed_sec for i in range(4)]


class TrackerProtocol(Protocol):
    """Protocol cho các tracker."""

    def update(
        self, detections: list[PersonDetection], timestamp_sec: float | None = None
    ) -> list[Track]: ...

    def predict(self, timestamp_sec: float | None = None) -> list[Track]: ...

    def active_tracks(self) -> list[Track]: ...


class IoUTracker:
    """Bộ theo dõi đối tượng dựa trên thuật toán tham lam Greedy IoU (Baseline nhẹ)."""

    def __init__(
        self,
        threshold: float = 0.5,
        max_missed: int = 30,
        track_ttl_seconds: float | None = 5.0,
    ) -> None:
        self.threshold = threshold
        self.max_missed = max_missed
        self.track_ttl_seconds = track_ttl_seconds
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def predict(self, timestamp_sec: float | None = None) -> list[Track]:
        """Dự đoán vị trí ở khung hình không chạy detector để tránh hiện tượng đứng hình."""
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            elapsed_sec = 1.0
            if timestamp_sec is not None:
                prev_t = track.last_position_sec or track.last_seen_sec
                if prev_t is not None:
                    elapsed_sec = max(timestamp_sec - prev_t, 0.0)
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
        """Cập nhật vết theo dõi từ các detection ở khung hình mới."""
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
                if track.disappeared > self.max_missed or expired:
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
                last_seen_sec=timestamp_sec
                if timestamp_sec is not None
                else self.tracks[tid].last_seen_sec,
                last_position_sec=timestamp_sec
                if timestamp_sec is not None
                else self.tracks[tid].last_position_sec,
            )
            matched_tracks.add(tid)
            matched_detections.add(col)

            scores[row, :] = -1.0
            scores[:, col] = -1.0

        # Khởi tạo track mới cho các detection chưa ghép cặp
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

        # Xử lý các track không được match
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
                if track.disappeared > self.max_missed or expired:
                    del self.tracks[tid]

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        return [self.tracks[tid] for tid in sorted(self.tracks)]


class TwoThresholdIoUTracker:
    """Tracker IoU hai ngưỡng phân tách high-confidence và low-confidence detections."""

    def __init__(
        self,
        high_threshold: float = 0.50,
        match_threshold: float = 0.50,
        low_match_threshold: float = 0.30,
        max_missed: int = 30,
        track_ttl_seconds: float | None = 5.0,
    ) -> None:
        self.high_threshold = high_threshold
        self.match_threshold = match_threshold
        self.low_match_threshold = low_match_threshold
        self.max_missed = max_missed
        self.track_ttl_seconds = track_ttl_seconds
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def predict(self, timestamp_sec: float | None = None) -> list[Track]:
        """Nội suy chuyển động giữa các frame không chạy person detector."""
        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            elapsed_sec = 1.0
            if timestamp_sec is not None:
                prev_t = track.last_position_sec or track.last_seen_sec
                if prev_t is not None:
                    elapsed_sec = max(timestamp_sec - prev_t, 0.0)
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
                track = self.tracks[tid]
                track.disappeared += 1
                track.updated = False
                expired = (
                    timestamp_sec is not None
                    and track.last_seen_sec is not None
                    and self.track_ttl_seconds is not None
                    and timestamp_sec - track.last_seen_sec > self.track_ttl_seconds
                )
                if track.disappeared > self.max_missed or expired:
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
                    last_seen_sec=timestamp_sec
                    if timestamp_sec is not None
                    else self.tracks[tid].last_seen_sec,
                    last_position_sec=timestamp_sec
                    if timestamp_sec is not None
                    else self.tracks[tid].last_position_sec,
                )
                matched_tracks.add(tid)
                matched_high_dets.add(col)
                scores1[row, :] = -1.0
                scores1[:, col] = -1.0

        # Giai đoạn 2: Ghép detection điểm thấp với track chưa được ghép (phục hồi khi bị che khuất)
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
                    last_seen_sec=timestamp_sec
                    if timestamp_sec is not None
                    else self.tracks[tid].last_seen_sec,
                    last_position_sec=timestamp_sec
                    if timestamp_sec is not None
                    else self.tracks[tid].last_position_sec,
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

        # Cập nhật số chu kỳ bỏ lỡ và xóa track quá hạn
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
                if track.disappeared > self.max_missed or expired:
                    del self.tracks[tid]

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        return [self.tracks[tid] for tid in sorted(self.tracks)]
