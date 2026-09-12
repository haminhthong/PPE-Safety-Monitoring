"""Đo lường hiệu năng thực tế (Benchmark Throughput) trên file Video.

Đo đạc:
- FPS trung bình của toàn bộ pipeline (Person Detection + Tracking + PPE Detection + FSM).
- Độ trễ xử lý trung bình mỗi khung hình (ms/frame).
- Số lượng công nhân được theo dõi và số sự kiện vi phạm ghi nhận.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from ppe_detection.config import DetectionConfig
from ppe_detection.pipeline import PPEPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("benchmark")


def benchmark_video(video_path: str, config_path: str = "configs/config.yaml") -> dict:
    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    config = DetectionConfig.load_from_yaml(
        config_path,
        save_output=False,
        save_snapshots=False,
        show_window=False,
        enable_beep=False,
    )

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Không thể mở file video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    LOGGER.info(
        "Bắt đầu benchmark video: %s (%dx%d, %d frames, %.1f FPS)",
        video_file.name,
        width,
        height,
        total_frames,
        video_fps,
    )

    pipeline = PPEPipeline(config)
    start_time = time.perf_counter()
    report = pipeline.run(str(video_file))
    elapsed = time.perf_counter() - start_time

    processed_frames = report.total_frames
    actual_fps = processed_frames / max(elapsed, 1e-6)
    avg_latency_ms = (elapsed / max(processed_frames, 1)) * 1000

    results = {
        "video": video_file.name,
        "resolution": f"{width}x{height}",
        "total_frames": processed_frames,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(actual_fps, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "unique_workers": len(report.unique_track_ids),
        "violations": report.counts,
    }

    LOGGER.info("========== KẾT QUẢ BENCHMARK ==========")
    LOGGER.info("Thời gian xử lý: %.2f giây", results["elapsed_seconds"])
    LOGGER.info("Tốc độ xử lý: %.2f FPS", results["fps"])
    LOGGER.info("Độ trễ trung bình: %.2f ms/frame", results["avg_latency_ms"])
    LOGGER.info("Số công nhân theo dõi: %d", results["unique_workers"])
    LOGGER.info("Tổng sự kiện vi phạm: %d", results["violations"]["total"])
    LOGGER.info("=======================================")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hiệu năng pipeline trên video")
    parser.add_argument("--video", required=True, help="Đường dẫn file video đầu vào")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Đường dẫn file cấu hình YAML"
    )
    args = parser.parse_args()

    benchmark_video(args.video, args.config)


if __name__ == "__main__":
    main()
