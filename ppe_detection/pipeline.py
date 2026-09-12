"""Module điều phối quy trình xử lý chính (PPE Pipeline).

Luồng xử lý:
1. Frame → 1. Person Detection (YOLO)
2. Bounding boxes → 2. Multi-Object Tracking (Two-Threshold IoU / IoU)
3. Track boxes → 3. Cắt Person ROI (PersonCropBuilder)
4. Crop images → 4. PPE Detection (YOLO nhận diện helmet, no-helmet, vest, no-vest)
5. PPE boxes → 5. Bộ lọc Body-Zone (Head Zone 0-35%, Torso Zone 30-75%)
6. Quan sát PPE → 6. Temporal Confirmation (FSM theo dwell time)
7. Vi phạm xác nhận → 7. Violation Event (lưu snapshot bằng chứng, xuất JSON/CSV)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig
from .crops import PersonCropBuilder
from .detector import DetectorProtocol, DualModelDetector, read_image, write_image
from .models import PersonDetection, PPEState
from .reporting import SessionReport
from .tracker import IoUTracker, Track, TrackerProtocol, TwoThresholdIoUTracker
from .violation_fsm import TemporalViolationFSM
from .visualization import draw_tracks

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def sound_alert() -> None:
    """Phát âm thanh cảnh báo ngắn (Beep) mà không làm gián đoạn chương trình."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1000, 250)
        else:
            print("\a", end="", flush=True)
    except (OSError, RuntimeError):
        LOGGER.warning("Không thể phát âm thanh cảnh báo trên hệ thống này.")


class PPEPipeline:
    """Điều phối toàn bộ quy trình phát hiện người, theo dõi và kiểm tra PPE."""

    def __init__(
        self,
        config: DetectionConfig,
        detector: DetectorProtocol | None = None,
    ) -> None:
        """Khởi tạo pipeline với cấu hình `DetectionConfig` hoặc detector tùy biến (cho test)."""
        self.config = config
        self.crop_builder = PersonCropBuilder(config.person_roi_padding)

        if detector is not None:
            self.detector = detector
        else:
            config.validate(require_models=True)
            self.detector = DualModelDetector(config)

        # Khởi tạo tracker theo cấu hình
        if config.tracker_type.lower() == "two_threshold_iou":
            self.tracker: TrackerProtocol = TwoThresholdIoUTracker(
                high_threshold=config.high_threshold,
                match_threshold=config.tracker_iou,
                low_match_threshold=config.low_match_threshold,
                max_missed=config.max_missed,
                track_ttl_seconds=config.track_ttl_seconds,
            )
        else:
            self.tracker = IoUTracker(
                threshold=config.tracker_iou,
                max_missed=config.max_missed,
                track_ttl_seconds=config.track_ttl_seconds,
            )

        # Máy trạng thái thời gian kiểm soát vi phạm
        self.fsm = TemporalViolationFSM(
            confirm_after_sec=config.violation_confirm_seconds,
            resolve_after_sec=config.resolution_confirm_seconds,
            alert_cooldown_sec=config.alert_cooldown_seconds,
            track_ttl_sec=config.track_ttl_seconds,
        )

    def run(self, source: int | str) -> SessionReport:
        """Thực thi pipeline trên ảnh tĩnh hoặc luồng video/webcam."""
        report = SessionReport(source, resolved_config=self.config.to_dict())
        if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
            self._run_image(Path(source), report)
        else:
            self._run_stream(source, report)
        return report

    def _save_snapshot(self, frame: np.ndarray, track: Track, kind: str, frame_id: int) -> str:
        """Cắt và lưu ảnh snapshot của công nhân vi phạm làm bằng chứng."""
        if not (self.config.save_output and self.config.save_snapshots):
            return ""

        snapshot_dir = self.config.output_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, track.box)
        x1, y1 = max(0, x1 - 15), max(0, y1 - 15)
        x2, y2 = min(w, x2 + 15), min(h, y2 + 15)

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            roi = frame.copy()

        cv2.putText(
            roi,
            f"ID:{track.track_id} {kind.upper()} VIOLATION",
            (5, max(15, roi.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
        )

        filename = f"violation_id{track.track_id}_{kind}_frame{frame_id}_{int(time.time())}.jpg"
        filepath = snapshot_dir / filename
        if not write_image(filepath, roi):
            LOGGER.error("Không thể ghi snapshot bằng chứng: %s", filepath)
            return ""
        return str(filepath)

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        report: SessionReport,
        timestamp_sec: float,
        detect_persons: bool,
    ) -> list[Track]:
        """Xử lý một frame video: cập nhật tracking và định kỳ kiểm tra PPE."""
        if detect_persons:
            person_detections_raw = self.detector.detect_persons(frame)
            person_dets = [
                PersonDetection(box=box, confidence=conf) for box, conf in person_detections_raw
            ]
            tracks = self.tracker.update(person_dets, timestamp_sec=timestamp_sec)
        else:
            tracks = self.tracker.predict(timestamp_sec=timestamp_sec)

        should_inspect_ppe = frame_id == 1 or frame_id % self.config.ppe_detection_interval == 0
        observed_track_ids: set[int] = set()

        for track in tracks:
            report.unique_track_ids.add(track.track_id)
            if not should_inspect_ppe or track.disappeared > 0:
                continue
            try:
                roi, crop_window = self.crop_builder.crop(frame, track.box)
            except ValueError:
                LOGGER.debug(
                    "Bỏ qua kiểm tra PPE do track %d nằm ngoài khung hình.", track.track_id
                )
                continue

            ppe_status = self.detector.analyze_ppe_for_roi(
                roi,
                person_box=track.box,
                crop_window=crop_window,
            )
            track.ppe = ppe_status
            observed_track_ids.add(track.track_id)
            report.ppe_observations += 1
            report.unknown_ppe_observations += int(
                ppe_status.helmet_state is PPEState.UNKNOWN
                or ppe_status.vest_state is PPEState.UNKNOWN
            )

        # Chuyển các quan sát mới vào máy trạng thái FSM
        self._evaluate_fsm(frame, tracks, report, frame_id, timestamp_sec, observed_track_ids)
        self.fsm.clean_inactive_tracks({track.track_id for track in tracks}, timestamp_sec)

        return tracks

    def _evaluate_fsm(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        report: SessionReport,
        frame_id: int,
        timestamp_sec: float,
        observed_track_ids: set[int],
    ) -> None:
        """Đánh giá chuyển trạng thái FSM và kích hoạt cảnh báo vi phạm."""
        should_alert = False

        for track in tracks:
            if track.track_id not in observed_track_ids:
                continue

            for kind, state in (
                ("helmet", track.ppe.helmet_state),
                ("vest", track.ppe.vest_state),
            ):
                transition = self.fsm.update(
                    track_id=track.track_id,
                    violation_type=kind,
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    observation_state=state,
                )

                if transition.should_emit_alert:
                    snapshot_path = self._save_snapshot(frame, track, kind, frame_id)
                    report.add_event(
                        track_id=track.track_id,
                        kind=kind,
                        frame_id=frame_id,
                        time_seconds=timestamp_sec,
                        event_start_seconds=self.fsm.get_state(track.track_id, kind).started_at_sec,
                        snapshot_path=snapshot_path,
                    )
                    log_label = "TÁI PHẠM" if transition.is_recurrence else "XÁC NHẬN VI PHẠM"
                    LOGGER.warning(
                        "%s [%s] - Công nhân ID: %d (Frame %d, %.2fs)",
                        log_label,
                        kind.upper(),
                        track.track_id,
                        frame_id,
                        timestamp_sec,
                    )
                    should_alert = True

        if should_alert and self.config.enable_beep:
            sound_alert()

    def _run_image(self, source: Path, report: SessionReport) -> None:
        """Xử lý nguồn dữ liệu ảnh tĩnh."""
        frame = read_image(source)
        if frame is None:
            raise ValueError(f"Không thể đọc file ảnh: {source}")

        report.total_frames = 1
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, timestamp_sec=0.0)

        for track in tracks:
            report.unique_track_ids.add(track.track_id)
            report.ppe_observations += 1
            report.unknown_ppe_observations += int(
                track.ppe.helmet_state is PPEState.UNKNOWN
                or track.ppe.vest_state is PPEState.UNKNOWN
            )

        annotated = draw_tracks(
            frame,
            tracks,
            report.counts,
            fps=0.0,
            frame_id=1,
            roi_polygon=self.config.roi_polygon,
        )

        if self.config.save_output:
            out_dir = self._prepare_output_dir()
            out_image_path = out_dir / f"{source.stem}_detected{source.suffix}"
            if not write_image(out_image_path, annotated):
                raise OSError(f"Không thể ghi ảnh kết quả: {out_image_path}")
            self._save_report(report, source.stem)
            LOGGER.info("Đã lưu ảnh kết quả: %s", out_image_path)

        if self.config.show_window:
            self._show_image(annotated)

    def _run_stream(self, source: int | str, report: SessionReport) -> None:
        """Xử lý nguồn dữ liệu video file hoặc camera stream."""
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Không thể mở nguồn video/camera: {source}")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if source_fps > 0 else 30.0
        writer = None
        output_stem = self._source_stem(source)
        prev_time = time.perf_counter()
        stream_started_at = time.monotonic()
        smoothed_fps = 0.0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                report.total_frames += 1
                frame_id = report.total_frames
                timestamp_sec = self._frame_timestamp(
                    capture, source, frame_id, source_fps, stream_started_at
                )

                is_detection_frame = frame_id == 1 or frame_id % self.config.detection_interval == 0
                tracks = self._process_frame(
                    frame,
                    frame_id,
                    report,
                    timestamp_sec,
                    detect_persons=is_detection_frame,
                )

                now = time.perf_counter()
                curr_fps = 1.0 / max(now - prev_time, 1e-6)
                smoothed_fps = (
                    curr_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * curr_fps
                )
                prev_time = now

                annotated = draw_tracks(
                    frame,
                    tracks,
                    report.counts,
                    smoothed_fps,
                    frame_id,
                    self.config.roi_polygon,
                )

                if self.config.save_output:
                    if writer is None:
                        writer = self._create_video_writer(annotated, source_fps, output_stem)
                    writer.write(annotated)

                if self.config.show_window:
                    cv2.imshow("Hệ thống Giám sát PPE - OpenCV", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        LOGGER.info("Dừng giám sát theo lệnh người dùng (ESC).")
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.config.show_window:
                cv2.destroyAllWindows()
            if self.config.save_output:
                self._save_report(report, output_stem)

        LOGGER.info(
            "Hoàn tất phiên giám sát: %d khung hình, %d người, %d sự kiện vi phạm.",
            report.total_frames,
            len(report.unique_track_ids),
            report.counts["total"],
        )

    def _prepare_output_dir(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return self.config.output_dir

    def _create_video_writer(self, frame: np.ndarray, fps: float, stem: str) -> cv2.VideoWriter:
        out_path = self._prepare_output_dir() / f"{stem}_detected.mp4"
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Không thể khởi tạo VideoWriter tại: {out_path}")
        LOGGER.info("Đang ghi video kết quả: %s", out_path)
        return writer

    def _save_report(self, report: SessionReport, stem: str) -> None:
        json_p, csv_p = report.save(self._prepare_output_dir(), stem)
        LOGGER.info("Đã xuất báo cáo: JSON (%s), CSV (%s)", json_p, csv_p)

    @staticmethod
    def _source_stem(source: int | str) -> str:
        return f"camera_{source}" if isinstance(source, int) else Path(source).stem

    @staticmethod
    def _frame_timestamp(
        capture: cv2.VideoCapture,
        source: int | str,
        frame_id: int,
        source_fps: float,
        stream_started_at: float,
    ) -> float:
        source_text = str(source).lower()
        is_live = isinstance(source, int) or source_text.startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        if is_live:
            return max(0.0, time.monotonic() - stream_started_at)

        pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms > 0.0:
            return pos_ms / 1000.0
        return (frame_id - 1) / source_fps

    @staticmethod
    def _show_image(frame: np.ndarray) -> None:
        try:
            cv2.imshow("Kết quả Giám sát PPE", frame)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
