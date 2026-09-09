"""Module dịch vụ trung gian (DetectionService) phân tách Backend Inference khỏi CLI và UI.

Tạo cấu trúc lưu trữ đầu ra độc lập theo từng phiên làm việc (Session-based output)
và cung cấp giao diện lập trình xử lý ảnh/video an toàn.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from datetime import datetime
from pathlib import Path

from .config import DetectionConfig
from .pipeline import PPEPipeline
from .reporting import SessionReport

LOGGER = logging.getLogger(__name__)


class DetectionService:
    """Dịch vụ thực thi nhận diện PPE độc lập cho từng phiên làm việc (Session)."""

    def __init__(self, config: DetectionConfig) -> None:
        """Khởi tạo dịch vụ với cấu hình phiên.

        Tự động tạo thư mục đầu ra duy nhất cho session: outputs/YYYYMMDD_HHMMSS_UUID/
        """
        self.raw_config = config
        self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.session_dir = config.output_dir / self.session_id

        # Tạo cấu hình mới gắn với thư mục phiên độc lập
        self.config = dataclasses.replace(config, output_dir=self.session_dir)

    def process(self, source: int | str) -> tuple[SessionReport, Path]:
        """Thực thi pipeline trên nguồn dữ liệu và trả về báo cáo cùng đường dẫn thư mục phiên.

        Args:
            source: Đường dẫn tới file ảnh/video hoặc chỉ số camera.

        Returns:
            Tuple chứa (SessionReport, Path thư mục lưu trữ phiên).
        """
        if self.config.demo_mode:
            LOGGER.warning(
                "[DEMO MODE ACTIVE]: Kết quả mô phỏng bởi SyntheticDemoDetector, "
                "không phải inference AI thật."
            )

        if self.config.save_output:
            self.session_dir.mkdir(parents=True, exist_ok=True)

        pipeline = PPEPipeline(self.config)
        report = pipeline.run(source)
        return report, self.session_dir
