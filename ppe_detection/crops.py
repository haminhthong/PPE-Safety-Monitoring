"""Hợp đồng cắt ROI người dùng chung cho train, evaluation và serving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CropWindow:
    """Cửa sổ crop theo tọa độ ảnh gốc."""

    x1: int
    y1: int
    x2: int
    y2: int


class PersonCropBuilder:
    """Cắt person ROI với padding cố định và không đổi hệ tọa độ giải phẫu."""

    def __init__(self, padding: int = 10) -> None:
        if padding < 0:
            raise ValueError("padding không được là số âm.")
        self.padding = padding

    def window(self, frame_shape: tuple[int, ...], person_box: list[float]) -> CropWindow:
        """Tạo cửa sổ crop đã cắt vào biên ảnh."""
        if len(frame_shape) < 2 or len(person_box) != 4:
            raise ValueError("frame_shape hoặc person_box không hợp lệ.")
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = map(round, person_box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("person_box phải có chiều rộng và chiều cao lớn hơn 0.")
        window = CropWindow(
            x1=max(0, x1 - self.padding),
            y1=max(0, y1 - self.padding),
            x2=min(width, x2 + self.padding),
            y2=min(height, y2 + self.padding),
        )
        if window.x2 <= window.x1 or window.y2 <= window.y1:
            raise ValueError("person_box không giao với khung hình.")
        return window

    def crop(self, frame: np.ndarray, person_box: list[float]) -> tuple[np.ndarray, CropWindow]:
        """Cắt ROI và trả cả origin để map detection về person box gốc."""
        window = self.window(frame.shape, person_box)
        return frame[window.y1 : window.y2, window.x1 : window.x2], window
