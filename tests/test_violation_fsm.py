"""Tests kiểm tra máy trạng thái hữu hạn thời gian (Temporal Violation FSM)."""

from __future__ import annotations

from ppe_detection.violation_fsm import TemporalViolationFSM


def test_fsm_requires_consecutive_observations_to_alert() -> None:
    """FSM phải đạt đủ số lần quan sát vi phạm liên tiếp mới phát cảnh báo (giảm false alarm)."""
    fsm = TemporalViolationFSM(confirm_observations=3, resolve_observations=3)

    # Lần 1: Vi phạm -> trạng thái chuyển sang VIOLATING, chưa ALERTED
    res1 = fsm.update(
        track_id=1, violation_type="helmet", is_violated=True, frame_id=1, timestamp_sec=0.0
    )
    assert res1.current_state == "VIOLATING"
    assert res1.should_emit_alert is False

    # Lần 2: Vi phạm tiếp -> vẫn VIOLATING, chưa ALERTED
    res2 = fsm.update(
        track_id=1, violation_type="helmet", is_violated=True, frame_id=4, timestamp_sec=0.1
    )
    assert res2.current_state == "VIOLATING"
    assert res2.should_emit_alert is False

    # Lần 3: Đạt 3 lần liên tiếp -> chuyển sang ALERTED, kích hoạt emit alert
    res3 = fsm.update(
        track_id=1, violation_type="helmet", is_violated=True, frame_id=8, timestamp_sec=0.2
    )
    assert res3.current_state == "ALERTED"
    assert res3.should_emit_alert is True

    # Lần 4: Vẫn vi phạm trong trạng thái ALERTED -> không spam emit alert lần nữa
    res4 = fsm.update(
        track_id=1, violation_type="helmet", is_violated=True, frame_id=12, timestamp_sec=0.3
    )
    assert res4.current_state == "ALERTED"
    assert res4.should_emit_alert is False


def test_fsm_resolution_and_recurrence() -> None:
    """FSM chuyển sang RESOLVED khi công nhân tuân thủ và phát hiện TÁI PHẠM khi vi phạm trở lại."""
    fsm = TemporalViolationFSM(confirm_observations=2, resolve_observations=2)

    # Đưa vào trạng thái ALERTED (2 lần vi phạm)
    fsm.update(track_id=2, violation_type="vest", is_violated=True, frame_id=1, timestamp_sec=0.0)
    res_alert = fsm.update(
        track_id=2, violation_type="vest", is_violated=True, frame_id=4, timestamp_sec=0.1
    )
    assert res_alert.current_state == "ALERTED"
    assert res_alert.should_emit_alert is True

    # Công nhân bắt đầu mặc áo (Lần 1 tuân thủ)
    r1 = fsm.update(
        track_id=2, violation_type="vest", is_violated=False, frame_id=8, timestamp_sec=0.2
    )
    assert r1.current_state == "ALERTED"
    assert r1.is_resolved is False

    # Lần 2 tuân thủ liên tiếp -> chuyển sang RESOLVED!
    r2 = fsm.update(
        track_id=2, violation_type="vest", is_violated=False, frame_id=12, timestamp_sec=0.3
    )
    assert r2.current_state == "RESOLVED"
    assert r2.is_resolved is True

    # 10 phút sau, công nhân lại cởi áo ra:
    # Lần 1 tái phạm
    t1 = fsm.update(
        track_id=2, violation_type="vest", is_violated=True, frame_id=200, timestamp_sec=10.0
    )
    assert t1.should_emit_alert is False

    # Lần 2 tái phạm liên tiếp -> kích hoạt ALERTED kèm cờ is_recurrence=True!
    t2 = fsm.update(
        track_id=2, violation_type="vest", is_violated=True, frame_id=204, timestamp_sec=10.1
    )
    assert t2.current_state == "ALERTED"
    assert t2.should_emit_alert is True
    assert t2.is_recurrence is True


def test_fsm_keeps_pending_recurrence_until_cooldown_expires() -> None:
    """Tái phạm trong cooldown vẫn được giữ để phát cảnh báo khi đủ thời gian."""
    fsm = TemporalViolationFSM(
        confirm_observations=1,
        resolve_observations=1,
        confirm_after_sec=0.0,
        resolve_after_sec=0.0,
        alert_cooldown_sec=10.0,
    )

    first = fsm.update(3, "helmet", is_violated=True, timestamp_sec=0.0)
    assert first.should_emit_alert is True

    resolved = fsm.update(3, "helmet", is_violated=False, timestamp_sec=1.0)
    assert resolved.is_resolved is True

    pending = fsm.update(3, "helmet", is_violated=True, timestamp_sec=2.0)
    assert pending.should_emit_alert is False
    assert pending.current_state == "VIOLATING"

    recurrent = fsm.update(3, "helmet", is_violated=True, timestamp_sec=10.1)
    assert recurrent.should_emit_alert is True
    assert recurrent.is_recurrence is True
