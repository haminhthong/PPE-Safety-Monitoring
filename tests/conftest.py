"""Các fixtures và mock objects phục vụ kiểm thử tự động (Unit Tests)."""

from __future__ import annotations

import numpy as np
import pytest

from ppe_detection.crops import CropWindow
from ppe_detection.models import PersonDetection, PPEDetection, PPEState, PPEStatus


class MockDetector:
    """Mock detector giả lập trả về detection phục vụ kiểm thử pipeline."""

    def __init__(self, simulate_violation: bool = True) -> None:
        self.simulate_violation = simulate_violation
        self.frame_counter = 0

    def detect_persons(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng.")
        self.frame_counter += 1
        h, w = frame.shape[:2]
        # Tạo 1 người ở giữa khung hình
        person_box = [float(w * 0.2), float(h * 0.1), float(w * 0.6), float(h * 0.9)]
        return [(person_box, 0.92)]

    def analyze_ppe_for_roi(
        self,
        roi: np.ndarray,
        person_box: list[float] | None = None,
        crop_window: CropWindow | None = None,
    ) -> PPEStatus:
        if self.simulate_violation:
            # Giả lập công nhân không đội mũ bảo hộ (ABSENT) nhưng có mặc áo (PRESENT)
            return PPEStatus(
                detections=[
                    PPEDetection(label="no-helmet", confidence=0.88),
                    PPEDetection(label="vest", confidence=0.90),
                ],
                helmet_state=PPEState.ABSENT,
                vest_state=PPEState.PRESENT,
                no_helmet_score=0.88,
                vest_score=0.90,
            )
        return PPEStatus(
            detections=[
                PPEDetection(label="helmet", confidence=0.91),
                PPEDetection(label="vest", confidence=0.90),
            ],
            helmet_state=PPEState.PRESENT,
            vest_state=PPEState.PRESENT,
            helmet_score=0.91,
            vest_score=0.90,
        )

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        persons = self.detect_persons(frame)
        return [
            PersonDetection(
                box=box,
                confidence=conf,
                ppe=self.analyze_ppe_for_roi(frame, person_box=box),
            )
            for box, conf in persons
        ]


@pytest.fixture
def mock_detector() -> MockDetector:
    return MockDetector()
