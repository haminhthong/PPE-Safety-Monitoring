"""Module điều phối quy trình xử lý chính (PPE Pipeline - Track-First Architecture).

Kết nối các thành phần theo chuẩn kiến trúc Online Canonical:
1. Frame → Person Detector
2. Multi-Object Tracking (baseline IoU hai ngưỡng hoặc IoU kèm Motion Prediction)
3. Trích xuất Person Track ROIs
4. PPE Detector kèm Spatial Body-Zone Association
5. Temporal Violation FSM (Finite State Machine: COMPLIANT → VIOLATING → ALERTED → RESOLVED → VIOLATING)
6. Snapshot bằng chứng, cảnh báo âm thanh và xuất báo cáo đa định dạng JSON/CSV.
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
from .detector import DetectorProtocol, DualModelDetector, SyntheticDemoDetector
from .models import PPEState, PersonDetection
from .reporting import SessionReport
from .tracker import IoUTracker, Track, TrackerProtocol, TwoThresholdIoUTracker
from .violation_fsm import TemporalViolationFSM
from .visualization import draw_tracks

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def sound_alert() -> None:
    """Phát âm thanh cảnh báo ngắn (Beep) mà không làm dừng chương trình."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1000, 250)
        else:
            print("\a", end="", flush=True)
    except (OSError, RuntimeError):
        LOGGER.warning("Không thể phát âm thanh cảnh báo trên hệ thống này.")


class PPEPipeline:
    """Điều phối toàn bộ quy trình nhận diện, theo dõi Track-First và máy trạng thái thời gian."""

    def __init__(self, config: DetectionConfig) -> None:
        """Khởi tạo pipeline với cấu hình `DetectionConfig`."""
        config.validate()
        self.config = config
        self.crop_builder = PersonCropBuilder(config.person_roi_padding)

        if config.demo_mode:
            LOGGER.info("Đang chạy ở chế độ mô phỏng SyntheticDemoDetector.")
            self.detector: DetectorProtocol = SyntheticDemoDetector(config)
        else:
            self.detector = DualModelDetector(config)

        # Khởi tạo tracker theo cấu hình
        if config.tracker_type.lower() in {"bytetrack", "two_threshold_iou"}:
            self.tracker: TrackerProtocol = TwoThresholdIoUTracker(
                high_threshold=config.high_threshold,
                match_threshold=config.tracker_iou,
                low_match_threshold=config.low_match_threshold,
                max_missed_detections=config.max_missed_detections,
                track_ttl_seconds=config.track_ttl_seconds,
            )
        else:
            self.tracker = IoUTracker(
                threshold=config.tracker_iou,
                max_disappeared=config.max_missed_detections,
                track_ttl_seconds=config.track_ttl_seconds,
            )

        # Máy trạng thái hữu hạn kiểm soát vi phạm theo thời gian
        self.fsm = TemporalViolationFSM(
            confirm_after_sec=config.violation_confirm_seconds,
            resolve_after_sec=config.resolution_confirm_seconds,
            alert_cooldown_sec=config.alert_cooldown_seconds,
            track_ttl_sec=config.track_ttl_seconds,
        )

    def run(self, source: int | str) -> SessionReport:
        """Thực thi luồng xử lý tương ứng với loại nguồn đầu vào."""
        report = SessionReport(source, resolved_config=self.config.to_dict())
        if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
            self._run_image(Path(source), report)
        else:
            self._run_stream(source, report)
        return report

    def _save_snapshot(self, frame: np.ndarray, track: Track, kind: str, frame_id: int) -> str:
        """Cắt và lưu ảnh snapshot của cá nhân vi phạm làm bằng chứng."""
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
        if not cv2.imwrite(str(filepath), roi):
            LOGGER.error("Không thể ghi snapshot bằng chứng: %s", filepath)
            return ""
        return str(filepath)

    def _process_track_first_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        report: SessionReport,
        timestamp_sec: float,
    ) -> list[Track]:
        """Detect người, cập nhật track và chỉ inspect PPE khi đến cadence."""
        person_detections_raw = self.detector.detect_persons(frame)

        person_dets: list[PersonDetection] = []
        for box, conf in person_detections_raw:
            person_dets.append(PersonDetection(box=box, confidence=conf))

        # Bước 1 & 2: Cập nhật vị trí vết theo dõi
        tracks = self.tracker.update(person_dets, timestamp_sec=timestamp_sec)

        # Tracking chạy mỗi frame; PPE inference chạy cadence riêng.
        should_inspect_ppe = (
            frame_id == 1 or frame_id % self.config.ppe_detection_interval == 0
        )
        observed_track_ids: set[int] = set()
        for track in tracks:
            report.unique_track_ids.add(track.track_id)
            if not track.updated or not should_inspect_ppe:
                continue
            roi, crop_window = self.crop_builder.crop(frame, track.box)
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

        # Chỉ trạng thái PPE vừa quan sát mới được đưa vào FSM.
        self._evaluate_fsm(
            frame, tracks, report, frame_id, timestamp_sec, observed_track_ids
        )
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
        """Đánh giá chuyển trạng thái máy hữu hạn FSM cho các track."""
        should_alert = False

        for track in tracks:
            if not track.updated or track.track_id not in observed_track_ids:
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
                        event_start_seconds=self.fsm.get_state(
                            track.track_id, kind
                        ).started_at_sec,
                        snapshot_path=snapshot_path,
                    )
                    log_label = "TÁI PHẠM" if transition.is_recurrence else "XÁC NHẬN VI PHẠM"
                    LOGGER.warning(
                        "%s [%s] - Người ID: %d (Frame %d, %.2fs)",
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
        """Xử lý nguồn dữ liệu dạng file ảnh tĩnh."""
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Không thể đọc file ảnh: {source}")

        report.total_frames = 1
        # Ảnh đơn chỉ là kết quả inspection; không đủ temporal evidence để tạo event.
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
            if not cv2.imwrite(str(out_image_path), annotated):
                raise OSError(f"Không thể ghi ảnh đầu ra: {out_image_path}")
            self._save_report(report, source.stem)
            LOGGER.info("Đã lưu ảnh kết quả: %s", out_image_path)

        if self.config.show_window:
            self._show_image(annotated)

    def _run_stream(self, source: int | str, report: SessionReport) -> None:
        """Xử lý nguồn dữ liệu luồng (Video file hoặc Webcam) với Track-First và Motion Prediction."""
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Không thể kết nối nguồn dữ liệu: {source}")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if source_fps > 0 else 30.0
        writer = None
        output_stem = self._source_stem(source)
        prev_time = time.perf_counter()
        smoothed_fps = 0.0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                report.total_frames += 1
                frame_id = report.total_frames

                timestamp_sec = (frame_id - 1) / source_fps
                # Person detector/tracker mặc định chạy mỗi frame; cadence có thể
                # tăng lên chỉ khi debug hoặc tài nguyên hạn chế.
                is_detection_frame = frame_id == 1 or frame_id % self.config.detection_interval == 0
                if is_detection_frame:
                    tracks = self._process_track_first_frame(
                        frame, frame_id, report, timestamp_sec
                    )
                else:
                    # Frame trung gian chỉ dự đoán chuyển động, không tạo PPE evidence.
                    tracks = self.tracker.predict(timestamp_sec=timestamp_sec)

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
                        LOGGER.info("Dừng chương trình theo lệnh người dùng (ESC).")
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

    def _create_video_writer(self, frame: np.ndarray, fps: float, stem: str):
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
            raise OSError(f"Không thể tạo VideoWriter tại: {out_path}")
        LOGGER.info("Đang ghi video đầu ra: %s", out_path)
        return writer

    def _save_report(self, report: SessionReport, stem: str) -> None:
        json_p, csv_p = report.save(self._prepare_output_dir(), stem)
        LOGGER.info("Đã xuất báo cáo: JSON (%s), CSV (%s)", json_p, csv_p)

    @staticmethod
    def _source_stem(source: int | str) -> str:
        return f"camera_{source}" if isinstance(source, int) else Path(source).stem

    @staticmethod
    def _show_image(frame: np.ndarray) -> None:
        try:
            cv2.imshow("Kết quả Phát hiện Trang bị Bảo hộ", frame)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
