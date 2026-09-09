"""Module định nghĩa cấu trúc dữ liệu cho kết quả phát hiện PPE, Tracking và Trạng thái vi phạm.

Cung cấp các dataclass đại diện cho từng đối tượng phát hiện:
- `PPEDetection`: Trang bị bảo hộ cá nhân kèm bounding box và confidence.
- `PPEStatus`: Tổng hợp trạng thái tuân thủ bảo hộ của một cá nhân kèm phân tích không gian.
- `PersonDetection`: Đối tượng người cùng thông tin vùng ảnh và trạng thái PPE.
- `ViolationState`: Trạng thái máy hữu hạn (FSM) theo thời gian cho từng cá nhân.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PPEState(str, Enum):
    """Trạng thái tri-state của một loại PPE."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class PPEDetection:
    """Kết quả nhận diện một trang bị bảo hộ cá nhân trong vùng ảnh ROI.

    Attributes:
        label: Tên nhãn lớp PPE ('helmet', 'no-helmet', 'vest', 'no-vest').
        confidence: Độ tin cậy dự đoán từ mô hình (từ 0.0 đến 1.0).
        box: Tọa độ [x1, y1, x2, y2] trong crop hoặc chuẩn hóa [0..1].
    """

    label: str
    confidence: float
    box: list[float] | None = None


@dataclass(slots=True)
class PPEStatus:
    """Trạng thái tổng hợp trang bị bảo hộ của một người.

    UNKNOWN được dùng khi detector không có đủ bằng chứng. Nó không được
    tự động suy ra là tuân thủ.
    """

    detections: list[PPEDetection] = field(default_factory=list)
    helmet_state: PPEState = PPEState.UNKNOWN
    vest_state: PPEState = PPEState.UNKNOWN
    helmet_score: float = 0.0
    no_helmet_score: float = 0.0
    vest_score: float = 0.0
    no_vest_score: float = 0.0
    helmet_evidence: list[PPEDetection] = field(default_factory=list)
    vest_evidence: list[PPEDetection] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Chuẩn hóa state khi dữ liệu đến từ JSON hoặc caller cũ."""
        if isinstance(self.helmet_state, str):
            self.helmet_state = PPEState(self.helmet_state.lower())
        if isinstance(self.vest_state, str):
            self.vest_state = PPEState(self.vest_state.lower())

    @property
    def helmet_violation(self) -> bool:
        """Tương thích ngược; trạng thái UNKNOWN vẫn được giữ nguyên."""
        return self.helmet_state is PPEState.ABSENT

    @property
    def vest_violation(self) -> bool:
        """Tương thích ngược; trạng thái UNKNOWN vẫn được giữ nguyên."""
        return self.vest_state is PPEState.ABSENT


@dataclass(slots=True)
class PersonDetection:
    """Thông tin vị trí và trạng thái PPE của một người trước khi đưa vào tracker.

    Attributes:
        box: Tọa độ bounding box [x1, y1, x2, y2] tính theo pixel.
        confidence: Độ tin cậy phát hiện người của mô hình YOLO.
        ppe: Trạng thái kiểm tra trang bị bảo hộ tương ứng.
    """

    box: list[float]
    confidence: float
    ppe: PPEStatus = field(default_factory=PPEStatus)


@dataclass
class ViolationState:
    """Trạng thái máy hữu hạn theo thời gian cho từng cá nhân và loại vi phạm.

    Trạng thái:
        - COMPLIANT: Tuân thủ đầy đủ bảo hộ.
        - VIOLATING: Dấu hiệu vi phạm đang chờ đủ bằng chứng xác nhận.
        - ALERTED: Đã xác nhận vi phạm chính thức và phát cảnh báo/lưu snapshot.
        - RESOLVED: Người đã khắc phục vi phạm (ví dụ đã đội lại mũ).
    """

    state: str = "COMPLIANT"
    consecutive_positive: int = 0
    consecutive_negative: int = 0
    started_at_frame: int = 0
    started_at_sec: float = 0.0
    last_seen_sec: float = 0.0
    event_count: int = 0
    resolved_at_sec: float | None = None
    violation_started_at_sec: float | None = None
    compliance_started_at_sec: float | None = None
    last_alert_at_sec: float | None = None
