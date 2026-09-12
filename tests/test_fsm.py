"""Kiểm thử tự động cho TemporalViolationFSM (Máy trạng thái thời gian)."""

from __future__ import annotations

from ppe_detection.models import PPEState
from ppe_detection.violation_fsm import TemporalViolationFSM


def test_fsm_dwell_time_confirmation() -> None:
    """FSM phải có vi phạm kéo dài đủ dwell time (confirm_after_sec) mới phát cảnh báo."""
    fsm = TemporalViolationFSM(confirm_after_sec=0.5, resolve_after_sec=1.0)

    # Thời điểm 0.0s: Phát hiện ABSENT -> chuyển VIOLATING, chưa kích hoạt cảnh báo
    r1 = fsm.update(1, "helmet", frame_id=1, timestamp_sec=0.0, observation_state=PPEState.ABSENT)
    assert r1.current_state == "VIOLATING"
    assert r1.should_emit_alert is False

    # Thời điểm 0.3s (mới 0.3s < 0.5s): Vẫn VIOLATING, chưa ALERTED
    r2 = fsm.update(1, "helmet", frame_id=10, timestamp_sec=0.3, observation_state=PPEState.ABSENT)
    assert r2.current_state == "VIOLATING"
    assert r2.should_emit_alert is False

    # Thời điểm 0.6s (0.6s >= 0.5s): Đủ dwell time -> chuyển sang ALERTED, kích hoạt emit alert!
    r3 = fsm.update(1, "helmet", frame_id=20, timestamp_sec=0.6, observation_state=PPEState.ABSENT)
    assert r3.current_state == "ALERTED"
    assert r3.should_emit_alert is True

    # Thời điểm 0.8s: Vẫn vi phạm -> vẫn ALERTED nhưng không spam lặp cảnh báo
    r4 = fsm.update(1, "helmet", frame_id=25, timestamp_sec=0.8, observation_state=PPEState.ABSENT)
    assert r4.current_state == "ALERTED"
    assert r4.should_emit_alert is False


def test_fsm_resolution_and_recurrence() -> None:
    """FSM chuyển sang RESOLVED khi công nhân tuân thủ và phát hiện TÁI PHẠM khi vi phạm trở lại."""
    fsm = TemporalViolationFSM(
        confirm_after_sec=0.4,
        resolve_after_sec=1.0,
        alert_cooldown_sec=5.0,
    )

    # Xác nhận vi phạm ban đầu
    fsm.update(2, "vest", timestamp_sec=0.0, observation_state=PPEState.ABSENT)
    res_alert = fsm.update(2, "vest", timestamp_sec=0.5, observation_state=PPEState.ABSENT)
    assert res_alert.current_state == "ALERTED"
    assert res_alert.should_emit_alert is True

    # Công nhân bắt đầu mặc áo (PRESENT lúc 1.0s)
    r1 = fsm.update(2, "vest", timestamp_sec=1.0, observation_state=PPEState.PRESENT)
    assert r1.current_state == "ALERTED"
    assert r1.is_resolved is False

    # Mặc áo liên tục đến 2.1s (1.1s >= 1.0s resolve_after_sec) -> RESOLVED
    r2 = fsm.update(2, "vest", timestamp_sec=2.1, observation_state=PPEState.PRESENT)
    assert r2.current_state == "RESOLVED"
    assert r2.is_resolved is True

    # Sau 10 giây (đã hết cooldown 5.0s), công nhân lại cởi áo ra
    fsm.update(2, "vest", timestamp_sec=12.0, observation_state=PPEState.ABSENT)
    r_recurrent = fsm.update(2, "vest", timestamp_sec=12.5, observation_state=PPEState.ABSENT)
    assert r_recurrent.current_state == "ALERTED"
    assert r_recurrent.should_emit_alert is True
    assert r_recurrent.is_recurrence is True


def test_fsm_unknown_preserves_state() -> None:
    """Trạng thái UNKNOWN không được làm mất dấu vết vi phạm đang chờ hoặc reset FSM."""
    fsm = TemporalViolationFSM(confirm_after_sec=0.5, resolve_after_sec=1.0)

    # 1. Bắt đầu có dấu hiệu vi phạm
    fsm.update(3, "helmet", timestamp_sec=0.0, observation_state=PPEState.ABSENT)
    assert fsm.get_state(3, "helmet").state == "VIOLATING"

    # 2. Frame tiếp theo bị mờ/che khuất -> UNKNOWN
    res_unk = fsm.update(3, "helmet", timestamp_sec=0.2, observation_state=PPEState.UNKNOWN)
    assert res_unk.current_state == "VIOLATING"
    assert res_unk.should_emit_alert is False

    # 3. Quan sát ABSENT tiếp tục -> đạt dwell time 0.6s
    res_alert = fsm.update(3, "helmet", timestamp_sec=0.6, observation_state=PPEState.ABSENT)
    assert res_alert.current_state == "ALERTED"
    assert res_alert.should_emit_alert is True


def test_fsm_cooldown_prevents_spam() -> None:
    """Tái phạm trong khoảng thời gian cooldown không được spam cảnh báo ngay lập tức."""
    fsm = TemporalViolationFSM(
        confirm_after_sec=0.2,
        resolve_after_sec=0.2,
        alert_cooldown_sec=10.0,
    )

    fsm.update(4, "helmet", timestamp_sec=0.0, observation_state=PPEState.ABSENT)
    first_alert = fsm.update(4, "helmet", timestamp_sec=0.3, observation_state=PPEState.ABSENT)
    assert first_alert.should_emit_alert is True

    # Khắc phục
    fsm.update(4, "helmet", timestamp_sec=1.0, observation_state=PPEState.PRESENT)
    fsm.update(4, "helmet", timestamp_sec=1.3, observation_state=PPEState.PRESENT)
    assert fsm.get_state(4, "helmet").state == "RESOLVED"

    # Tái phạm ngay tại giây 2.0 (mới trôi qua 1.7s < cooldown 10.0s)
    fsm.update(4, "helmet", timestamp_sec=2.0, observation_state=PPEState.ABSENT)
    pending = fsm.update(4, "helmet", timestamp_sec=2.3, observation_state=PPEState.ABSENT)
    assert pending.should_emit_alert is False
    assert pending.current_state == "VIOLATING"

    # Sau khi vượt qua cooldown (giây 10.5s)
    recurrent = fsm.update(4, "helmet", timestamp_sec=10.5, observation_state=PPEState.ABSENT)
    assert recurrent.should_emit_alert is True
    assert recurrent.is_recurrence is True
