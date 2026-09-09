"""Module hiển thị đồ họa, vẽ bounding box và thông số HUD lên khung hình.

Trực quan hóa kết quả phát hiện người, nhãn PPE, trạng thái vi phạm,
vùng nguy hiểm ROI polygon và bảng chỉ số trực tiếp (Heads-Up Display).
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import PPEState
from .tracker import Track


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    violations: dict[str, int],
    fps: float,
    frame_id: int,
    roi_polygon: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Vẽ bounding box, nhãn thông tin và HUD chỉ số lên khung hình.

    Args:
        frame: Khung hình BGR gốc từ OpenCV.
        tracks: Danh sách vết theo dõi người đang hoạt động.
        violations: Dict chứa thống kê số lượng vi phạm hiện tại.
        fps: Tốc độ xử lý khung hình/giây.
        frame_id: Số thứ tự khung hình hiện tại.
        roi_polygon: Tọa độ vùng nguy hiểm ROI (nếu có).

    Returns:
        Khung hình BGR đã được vẽ bổ sung thông tin.
    """
    output = frame.copy()

    # 1. Vẽ vùng nguy hiểm ROI Polygon nếu được cấu hình
    if roi_polygon and len(roi_polygon) >= 3:
        pts = np.array(roi_polygon, dtype=np.int32)
        # Overlay trong suốt cho vùng ROI
        overlay = output.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 180))
        cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
        cv2.polylines(output, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(
            output,
            "DANGER ZONE (ROI)",
            (pts[0][0] + 5, pts[0][1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # 2. Vẽ thông tin từng đối tượng người
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box)
        has_violation = (
            track.ppe.helmet_state is PPEState.ABSENT or track.ppe.vest_state is PPEState.ABSENT
        )
        has_unknown = (
            track.ppe.helmet_state is PPEState.UNKNOWN or track.ppe.vest_state is PPEState.UNKNOWN
        )

        # Đỏ = vi phạm, cam = chưa đủ bằng chứng, xanh = tuân thủ.
        color = (0, 0, 255) if has_violation else (0, 165, 255) if has_unknown else (0, 200, 0)

        # Vẽ bounding box người
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Vẽ nhãn ID người
        label_id = f"ID: {track.track_id}"
        cv2.putText(
            output,
            label_id,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Vẽ danh sách các nhãn PPE nhận diện được
        for index, item in enumerate(track.ppe.detections):
            text = f"{item.label} {item.confidence:.2f}"
            cv2.putText(
                output,
                text,
                (x1, y1 + 24 + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # 3. Vẽ bảng điều khiển chỉ số tổng hợp (HUD) ở góc trên
    summary = (
        f"FPS: {fps:.1f} | Frame: {frame_id} | People: {len(tracks)} | "
        f"Violators: {violations['people']} | Events: {violations['total']} "
        f"(Helmet: {violations['helmet']}, Vest: {violations['vest']})"
    )

    # Nền mờ cho dải HUD ở phía trên
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        output,
        summary,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return output
