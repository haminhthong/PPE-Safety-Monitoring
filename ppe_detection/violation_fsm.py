"""Module máy trạng thái hữu hạn theo thời gian (Temporal Violation FSM) cho hệ thống giám sát PPE.

Quản lý chu kỳ vòng đời của một trạng thái vi phạm:
    COMPLIANT (Tuân thủ)
       ↓ (ABSENT đủ confirm_after_sec)
    ALERTED (Báo động chính thức và lưu snapshot bằng chứng)
       ↓ (PRESENT đủ resolve_after_sec)
    RESOLVED (Đã khắc phục vi phạm)
       ↓ (tái phạm liên tiếp >= confirm_observations)
    ALERTED (Báo động tái phạm - Recurrent Violation Event)

Ngăn báo động giả và cho phép phát hiện công nhân tái phạm sau khi đã khắc phục.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import PPEState, ViolationState

LOGGER = logging.getLogger(__name__)


@dataclass
class FSMTransitionResult:
    """Kết quả chuyển đổi trạng thái của FSM sau mỗi chu kỳ quan sát."""

    track_id: int
    violation_type: str
    previous_state: str
    current_state: str
    should_emit_alert: bool
    is_recurrence: bool = False
    is_resolved: bool = False
    observation_state: PPEState = PPEState.UNKNOWN


class TemporalViolationFSM:
    """Máy trạng thái thời gian kiểm soát việc kích hoạt và gỡ bỏ vi phạm bảo hộ."""

    def __init__(
        self,
        confirm_observations: int | None = None,
        resolve_observations: int | None = None,
        confirm_after_sec: float = 0.5,
        resolve_after_sec: float = 1.0,
        alert_cooldown_sec: float = 10.0,
        track_ttl_sec: float = 5.0,
    ) -> None:
        """Khởi tạo FSM theo thời gian thực.

        ``confirm_observations`` và ``resolve_observations`` chỉ là chế độ
        tương thích cho test/API cũ. Production không truyền hai tham số này;
        quyết định khi đó phụ thuộc dwell time, không phụ thuộc FPS.
        """
        self.confirm_observations = max(1, confirm_observations) if confirm_observations else None
        self.resolve_observations = max(1, resolve_observations) if resolve_observations else None
        if (
            confirm_after_sec < 0
            or resolve_after_sec < 0
            or alert_cooldown_sec < 0
            or track_ttl_sec <= 0
        ):
            raise ValueError("Ngưỡng thời gian FSM không hợp lệ.")
        self.confirm_after_sec = confirm_after_sec
        self.resolve_after_sec = resolve_after_sec
        self.alert_cooldown_sec = alert_cooldown_sec
        self.track_ttl_sec = track_ttl_sec
        self.states: dict[tuple[int, str], ViolationState] = {}

    def get_state(self, track_id: int, violation_type: str) -> ViolationState:
        """Lấy trạng thái hiện tại của một đối tượng và loại vi phạm."""
        key = (track_id, violation_type)
        if key not in self.states:
            self.states[key] = ViolationState(state="COMPLIANT")
        return self.states[key]

    def update(
        self,
        track_id: int,
        violation_type: str,
        is_violated: bool | None = None,
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
        observation_state: PPEState | str | None = None,
    ) -> FSMTransitionResult:
        """Cập nhật quan sát mới từ detector và thực hiện chuyển trạng thái FSM.

        Args:
            track_id: ID theo dõi của người.
            violation_type: Loại vi phạm ('helmet' hoặc 'vest').
            is_violated: API cũ; True/False được chuyển thành ABSENT/PRESENT.
            frame_id: Thứ tự frame hiện tại.
            timestamp_sec: Thời điểm tính bằng giây.
            observation_state: PRESENT, ABSENT hoặc UNKNOWN. Production dùng
                state tri-state; ``is_violated`` chỉ giữ tương thích API cũ.

        Returns:
            `FSMTransitionResult` chứa chỉ dẫn có cần phát cảnh báo hoặc thông báo khắc phục không.
        """
        v_state = self.get_state(track_id, violation_type)
        prev_state = v_state.state

        if observation_state is None:
            observation_state = (
                PPEState.UNKNOWN
                if is_violated is None
                else PPEState.ABSENT
                if is_violated
                else PPEState.PRESENT
            )
        elif isinstance(observation_state, str):
            try:
                observation_state = PPEState(observation_state.lower())
            except ValueError as error:
                raise ValueError(f"PPE state không hợp lệ: {observation_state}") from error

        should_emit = False
        is_recurrence = False
        is_resolved = False
        v_state.last_seen_sec = timestamp_sec

        # UNKNOWN không tăng bằng chứng ở phía nào và không reset event đang mở.
        if observation_state is PPEState.UNKNOWN:
            return FSMTransitionResult(
                track_id=track_id,
                violation_type=violation_type,
                previous_state=prev_state,
                current_state=v_state.state,
                should_emit_alert=False,
                observation_state=PPEState.UNKNOWN,
            )

        if observation_state is PPEState.ABSENT:
            v_state.consecutive_positive += 1
            v_state.consecutive_negative = 0
            v_state.compliance_started_at_sec = None
            if v_state.violation_started_at_sec is None:
                v_state.violation_started_at_sec = timestamp_sec

            elapsed = timestamp_sec - v_state.violation_started_at_sec
            count_ready = (
                self.confirm_observations is not None
                and v_state.consecutive_positive >= self.confirm_observations
            )
            time_ready = elapsed >= self.confirm_after_sec
            if v_state.state in {"COMPLIANT", "VIOLATING", "RESOLVED"}:
                if time_ready or count_ready:
                    cooldown_ok = (
                        v_state.last_alert_at_sec is None
                        or timestamp_sec - v_state.last_alert_at_sec >= self.alert_cooldown_sec
                    )
                    if cooldown_ok:
                        is_recurrence = v_state.event_count > 0
                        v_state.state = "ALERTED"
                        v_state.event_count += 1
                        v_state.started_at_frame = frame_id
                        v_state.started_at_sec = v_state.violation_started_at_sec
                        should_emit = True
                        v_state.last_alert_at_sec = timestamp_sec
                        LOGGER.info("XÁC NHẬN vi phạm ID %d - %s.", track_id, violation_type)
                    else:
                        # Giữ vi phạm ở trạng thái chờ. Khi cooldown kết thúc,
                        # observation ABSENT tiếp theo vẫn có thể phát cảnh báo.
                        v_state.state = "VIOLATING"
                else:
                    v_state.state = "VIOLATING"
        else:
            v_state.consecutive_negative += 1
            v_state.consecutive_positive = 0
            v_state.violation_started_at_sec = None
            if v_state.state == "VIOLATING":
                v_state.state = "COMPLIANT"
                v_state.compliance_started_at_sec = None
            elif v_state.state == "ALERTED":
                if v_state.compliance_started_at_sec is None:
                    v_state.compliance_started_at_sec = timestamp_sec
                elapsed = timestamp_sec - v_state.compliance_started_at_sec
                count_ready = (
                    self.resolve_observations is not None
                    and v_state.consecutive_negative >= self.resolve_observations
                )
                if elapsed >= self.resolve_after_sec or count_ready:
                    v_state.state = "RESOLVED"
                    v_state.resolved_at_sec = timestamp_sec
                    v_state.compliance_started_at_sec = None
                    is_resolved = True
                    LOGGER.info("Đã khắc phục ID %d - %s.", track_id, violation_type)

        return FSMTransitionResult(
            track_id=track_id,
            violation_type=violation_type,
            previous_state=prev_state,
            current_state=v_state.state,
            should_emit_alert=should_emit,
            is_recurrence=is_recurrence,
            is_resolved=is_resolved,
            observation_state=observation_state,
        )

    def clean_inactive_tracks(
        self, active_track_ids: set[int], now_sec: float | None = None
    ) -> None:
        """Dọn track không còn hoạt động hoặc đã quá TTL thời gian."""
        to_delete = []
        for key, state in self.states.items():
            inactive = key[0] not in active_track_ids
            expired = now_sec is not None and now_sec - state.last_seen_sec > self.track_ttl_sec
            if inactive or expired:
                to_delete.append(key)
        for k in to_delete:
            del self.states[k]
