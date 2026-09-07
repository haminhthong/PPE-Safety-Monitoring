"""Kiểm thử các bất biến an toàn của tri-state, ROI và FSM."""

from __future__ import annotations

from ppe_detection.detector import is_box_overlapping_roi, roi_overlap_ratio
from ppe_detection.models import PPEState, PPEStatus, PersonDetection
from ppe_detection.tracker import IoUTracker
from ppe_detection.violation_fsm import TemporalViolationFSM


def test_unknown_is_not_compliant() -> None:
    status = PPEStatus()
    assert status.helmet_state is PPEState.UNKNOWN
    assert status.vest_state is PPEState.UNKNOWN
    assert status.helmet_violation is False
    assert status.vest_violation is False


def test_roi_rule_uses_area_overlap() -> None:
    polygon = [(0, 0), (50, 0), (50, 100), (0, 100)]
    box = [25.0, 0.0, 75.0, 100.0]
    assert roi_overlap_ratio(box, polygon) == 0.5
    assert is_box_overlapping_roi(box, polygon, min_overlap=0.4)
    assert not is_box_overlapping_roi(box, polygon, min_overlap=0.6)


def test_unknown_does_not_reset_or_advance_fsm() -> None:
    fsm = TemporalViolationFSM(confirm_after_sec=0.5, resolve_after_sec=1.0)
    first = fsm.update(1, "helmet", frame_id=1, timestamp_sec=0.0, observation_state=PPEState.ABSENT)
    assert first.current_state == "VIOLATING"

    unknown = fsm.update(1, "helmet", frame_id=2, timestamp_sec=5.0, observation_state=PPEState.UNKNOWN)
    assert unknown.current_state == "VIOLATING"
    assert unknown.should_emit_alert is False

    confirmed = fsm.update(1, "helmet", frame_id=3, timestamp_sec=0.6, observation_state=PPEState.ABSENT)
    assert confirmed.current_state == "ALERTED"
    assert confirmed.should_emit_alert is True


def test_missing_observation_fails_closed_as_unknown() -> None:
    """Không có observation cũng không được mặc định là tuân thủ."""
    fsm = TemporalViolationFSM(confirm_after_sec=0.0)
    result = fsm.update(1, "helmet")
    assert result.observation_state is PPEState.UNKNOWN
    assert result.current_state == "COMPLIANT"


def test_tracker_preserves_last_ppe_observation_on_person_only_update() -> None:
    """Cadence tracking không được làm mất PPE evidence đã quan sát trước đó."""
    tracker = IoUTracker(threshold=0.3)
    observed = PPEStatus(helmet_state=PPEState.PRESENT)
    tracker.update(
        [PersonDetection(box=[10.0, 10.0, 50.0, 100.0], confidence=0.9, ppe=observed)]
    )
    tracks = tracker.update(
        [PersonDetection(box=[10.0, 10.0, 50.0, 100.0], confidence=0.9)]
    )
    assert tracks[0].ppe.helmet_state is PPEState.PRESENT
