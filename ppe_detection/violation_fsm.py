"""Module máy trạng thái thời gian (Temporal Violation FSM) cho hệ thống giám sát PPE.

Quản lý chu kỳ vòng đời của một trạng thái vi phạm:
    COMPLIANT (Tuân thủ)
       ↓ (ABSENT liên tục đủ confirm_after_sec)
    ALERTED (Xác nhận vi phạm & phát cảnh báo / lưu snapshot)
       ↓ (PRESENT liên tục đủ resolve_after_sec)
    RESOLVED (Đã khắc phục vi phạm)
       ↓ (ABSENT trở lại sau cooldown_seconds)
    ALERTED (Báo động tái phạm - Recurrent Violation Event)

Ngăn báo động giả (False Alarms) từ các frame nhận diện lỗi đơn lẻ (Detector Flicker),
đồng thời trạng thái UNKNOWN được bảo lưu và không làm reset hay advance FSM tùy tiện.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import PPEState, ViolationState

LOGGER = logging.getLogger(__name__)


@dataclass
class FSMTransitionResult:
    """Kết quả chuyển đổi trạng thái FSM sau mỗi quan sát."""

    track_id: int
    violation_type: str
    previous_state: str
    current_state: str
    should_emit_alert: bool
    is_recurrence: bool = False
    is_resolved: bool = False
    observation_state: PPEState = PPEState.UNKNOWN


class TemporalViolationFSM:
    """Máy trạng thái thời gian kiểm soát xác nhận vi phạm và khắc phục."""

    def __init__(
        self,
        confirm_after_sec: float = 0.50,
        resolve_after_sec: float = 1.00,
        alert_cooldown_sec: float = 10.0,
        track_ttl_sec: float = 5.0,
    ) -> None:
        if (
            confirm_after_sec < 0
            or resolve_after_sec < 0
            or alert_cooldown_sec < 0
            or track_ttl_sec <= 0
        ):
            raise ValueError("Các ngưỡng thời gian FSM không hợp lệ.")

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
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
        observation_state: PPEState | str | None = None,
        # Tương thích mềm nếu caller cũ truyền is_violated
        is_violated: bool | None = None,
    ) -> FSMTransitionResult:
        """Cập nhật quan sát mới từ detector và thực hiện chuyển trạng thái FSM.

        Args:
            track_id: ID theo dõi của công nhân.
            violation_type: Loại trang bị kiểm tra ('helmet' hoặc 'vest').
            frame_id: Thứ tự frame hiện tại.
            timestamp_sec: Thời điểm tính bằng giây.
            observation_state: PRESENT, ABSENT hoặc UNKNOWN.
            is_violated: (Tùy chọn) True -> ABSENT, False -> PRESENT.
        """
        v_state = self.get_state(track_id, violation_type)
        prev_state = v_state.state

        if observation_state is None:
            if is_violated is not None:
                observation_state = PPEState.ABSENT if is_violated else PPEState.PRESENT
            else:
                observation_state = PPEState.UNKNOWN
        elif isinstance(observation_state, str):
            try:
                observation_state = PPEState(observation_state.lower())
            except ValueError as err:
                raise ValueError(f"PPE state không hợp lệ: {observation_state}") from err

        should_emit = False
        is_recurrence = False
        is_resolved = False
        v_state.last_seen_sec = timestamp_sec

        # UNKNOWN không tăng bằng chứng ở phía nào và không reset event đang mở
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
            v_state.compliance_started_at_sec = None

            if v_state.violation_started_at_sec is None:
                v_state.violation_started_at_sec = timestamp_sec

            elapsed = timestamp_sec - v_state.violation_started_at_sec
            time_ready = elapsed >= self.confirm_after_sec

            if v_state.state in {"COMPLIANT", "VIOLATING", "RESOLVED"}:
                if time_ready:
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
                        # Trong cooldown, giữ VIOLATING chờ phát lại khi hết cooldown
                        v_state.state = "VIOLATING"
                else:
                    v_state.state = "VIOLATING"
        else:  # PPEState.PRESENT
            v_state.violation_started_at_sec = None

            if v_state.state == "VIOLATING":
                v_state.state = "COMPLIANT"
                v_state.compliance_started_at_sec = None
            elif v_state.state == "ALERTED":
                if v_state.compliance_started_at_sec is None:
                    v_state.compliance_started_at_sec = timestamp_sec
                elapsed = timestamp_sec - v_state.compliance_started_at_sec
                if elapsed >= self.resolve_after_sec:
                    v_state.state = "RESOLVED"
                    v_state.resolved_at_sec = timestamp_sec
                    v_state.compliance_started_at_sec = None
                    is_resolved = True
                    LOGGER.info("ĐÃ KHẮC PHỤC vi phạm ID %d - %s.", track_id, violation_type)

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
        """Dọn dẹp các track không còn hoạt động hoặc đã quá hạn TTL."""
        to_delete = []
        for key, state in self.states.items():
            inactive = key[0] not in active_track_ids
            expired = now_sec is not None and now_sec - state.last_seen_sec > self.track_ttl_sec
            if inactive or expired:
                to_delete.append(key)
        for k in to_delete:
            del self.states[k]
