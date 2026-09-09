"""Module quản lý và xuất báo cáo thống kê vi phạm an toàn lao động.

Tự động ghi nhận thông tin từng sự kiện vi phạm (ID, loại vi phạm, timestamp, frame,
đường dẫn ảnh bằng chứng snapshot) và xuất báo cáo tổng hợp dạng JSON và CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class ViolationEvent:
    """Thông tin chi tiết của một sự kiện vi phạm an toàn lao động.

    Attributes:
        track_id: ID theo dõi của nhân sự vi phạm.
        violation_type: Loại vi phạm ('helmet' hoặc 'vest').
        frame_id: Thứ tự khung hình phát hiện vi phạm.
        time_seconds: Thời điểm xảy ra vi phạm tính bằng giây.
        detected_at: ISO timestamp thời gian thực phát hiện.
        snapshot_path: Đường dẫn tới ảnh snapshot bằng chứng vi phạm.
    """

    track_id: int
    violation_type: str
    frame_id: int
    time_seconds: float
    detected_at: str
    snapshot_path: str = ""
    event_start_seconds: float = 0.0
    alert_time_seconds: float = 0.0


class SessionReport:
    """Thu thập, tổng hợp số liệu và ghi báo cáo cho một phiên giám sát."""

    def __init__(self, source: int | str, resolved_config: dict | None = None) -> None:
        """Khởi tạo phiên làm việc.

        Args:
            source: Nguồn dữ liệu đầu vào (Camera index hoặc đường dẫn file).
        """
        self.source = str(source)
        self.started_at = datetime.now().astimezone()
        self.events: list[ViolationEvent] = []
        self.total_frames = 0
        self.unique_track_ids: set[int] = set()
        self.ppe_observations = 0
        self.unknown_ppe_observations = 0
        self.resolved_config = resolved_config or {}

    @property
    def counts(self) -> dict[str, int]:
        """Tổng hợp số lượng vi phạm theo từng loại và số người vi phạm.

        Returns:
            Dict chứa số lượng 'helmet', 'vest', 'total' và 'people'.
        """
        helmet_count = sum(event.violation_type == "helmet" for event in self.events)
        vest_count = sum(event.violation_type == "vest" for event in self.events)
        violating_people = len({event.track_id for event in self.events})

        return {
            "total": helmet_count + vest_count,
            "helmet": helmet_count,
            "vest": vest_count,
            "people": violating_people,
        }

    def add_event(
        self,
        track_id: int,
        kind: str,
        frame_id: int,
        fps: float | None = None,
        snapshot_path: str = "",
        time_seconds: float | None = None,
        event_start_seconds: float | None = None,
    ) -> None:
        """Ghi nhận một sự kiện vi phạm mới vào danh sách.

        Args:
            track_id: Mã ID của người vi phạm.
            kind: Loại vi phạm ('helmet' hoặc 'vest').
            frame_id: Số thứ tự frame.
            fps: Tốc độ khung hình (khung/giây).
            snapshot_path: Đường dẫn tới file ảnh snapshot bằng chứng.
        """
        if time_seconds is None:
            time_seconds = round((frame_id - 1) / fps, 3) if fps and fps > 0.0 else 0.0
        time_seconds = round(time_seconds, 3)
        event_start_seconds = (
            round(event_start_seconds, 3) if event_start_seconds is not None else time_seconds
        )
        iso_now = datetime.now().astimezone().isoformat(timespec="seconds")

        self.events.append(
            ViolationEvent(
                track_id=track_id,
                violation_type=kind,
                frame_id=frame_id,
                time_seconds=time_seconds,
                event_start_seconds=event_start_seconds,
                alert_time_seconds=time_seconds,
                detected_at=iso_now,
                snapshot_path=snapshot_path,
            )
        )

    def save(self, output_dir: Path, stem: str) -> tuple[Path, Path]:
        """Xuất dữ liệu báo cáo ra 2 định dạng file JSON tổng hợp và CSV danh sách.

        Args:
            output_dir: Thư mục đầu ra.
            stem: Tên tiền tố cho file báo cáo.

        Returns:
            Tuple chứa (Đường dẫn file JSON, Đường dẫn file CSV).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{stem}_report.json"
        csv_path = output_dir / f"{stem}_events.csv"

        payload = {
            "source": self.source,
            "synthetic_demo": bool(self.resolved_config.get("demo_mode", False)),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "total_frames": self.total_frames,
            "unique_people_tracked": len(self.unique_track_ids),
            "ppe_observations": self.ppe_observations,
            "unknown_ppe_observations": self.unknown_ppe_observations,
            "resolved_config": self.resolved_config,
            "violations_summary": self.counts,
            "events": [asdict(event) for event in self.events],
        }

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "track_id",
                    "violation_type",
                    "frame_id",
                    "time_seconds",
                    "event_start_seconds",
                    "alert_time_seconds",
                    "detected_at",
                    "snapshot_path",
                ],
            )
            writer.writeheader()
            writer.writerows(asdict(event) for event in self.events)

        return json_path, csv_path
