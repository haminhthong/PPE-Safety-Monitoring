"""Chạy demo synthetic riêng biệt với production inference/evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Cho phép chạy trực tiếp ``python scripts/demo_pipeline.py`` từ thư mục repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppe_detection.config import DetectionConfig  # noqa: E402
from ppe_detection.service import DetectionService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo pipeline synthetic, không phải benchmark")
    parser.add_argument("--output-dir", default="outputs/demo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "synthetic_demo.jpg"
    cv2.imwrite(str(image_path), np.zeros((480, 640, 3), dtype=np.uint8))

    config = DetectionConfig(
        demo_mode=True,
        show_window=False,
        save_output=True,
        output_dir=output_dir,
        enable_beep=False,
    )
    report, session_dir = DetectionService(config).process(str(image_path))
    print({"synthetic_demo": True, "session_dir": str(session_dir), "events": len(report.events)})


if __name__ == "__main__":
    main()
