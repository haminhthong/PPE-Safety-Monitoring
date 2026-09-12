"""Kiểm thử tự động cho module ppe_detection.pipeline."""

from pathlib import Path

import numpy as np

from ppe_detection.config import DetectionConfig
from ppe_detection.detector import write_image
from ppe_detection.pipeline import PPEPipeline
from tests.conftest import MockDetector


def test_pipeline_image_processing(tmp_path: Path, mock_detector: MockDetector) -> None:
    """Kiểm tra pipeline xử lý thành công 1 ảnh tĩnh với mock detector."""
    # Tạo 1 ảnh đen kích thước 640x480 bằng helper an toàn
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_frame.jpg"
    assert write_image(img_path, test_img) is True

    out_dir = tmp_path / "outputs"
    config = DetectionConfig(
        show_window=False,
        save_output=True,
        save_snapshots=True,
        output_dir=out_dir,
        enable_beep=False,
    )

    pipeline = PPEPipeline(config, detector=mock_detector)
    report = pipeline.run(str(img_path))

    assert report.total_frames == 1
    assert len(report.unique_track_ids) == 1

    # Kiểm tra file ảnh đầu ra đã được tạo
    out_img = out_dir / "test_frame_detected.jpg"
    assert out_img.is_file()
