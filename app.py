"""Giao diện Dòng lệnh (CLI Entry Point) cho Hệ thống Giám sát Trang bị Bảo hộ (PPE).

Ví dụ sử dụng:
    1. Chạy với Webcam và lưu báo cáo kết quả:
       python app.py --source 0 --person-model models/yolov8n.pt --ppe-model models/best.pt --save

    2. Chạy xử lý file Video:
       python app.py --source sample.mp4 --person-model models/yolov8n.pt \\
       --ppe-model models/best.pt --save

    3. Chạy xử lý ảnh tĩnh:
       python app.py --source sample.jpg --person-model models/yolov8n.pt --ppe-model models/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ppe_detection.config import DetectionConfig
from ppe_detection.service import DetectionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("app")


def parse_source(value: str) -> int | str:
    """Chuyển đổi chỉ số camera thành số nguyên hoặc giữ nguyên đường dẫn file."""
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ứng dụng Computer Vision phát hiện người và vi phạm trang bị bảo hộ (PPE)."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Đường dẫn file ảnh/video hoặc chỉ số camera (mặc định: 0)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "config.yaml"),
        help="Đường dẫn file cấu hình YAML (mặc định: configs/config.yaml)",
    )
    parser.add_argument(
        "--person-model",
        default=None,
        help="Đường dẫn file model YOLO phát hiện người (ví dụ: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--ppe-model",
        default=None,
        help="Đường dẫn file model YOLO phát hiện PPE (ví dụ: models/best.pt)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Kích thước ảnh đầu vào cho YOLO inference (mặc định: 640)",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Ngưỡng tin cậy phát hiện người (mặc định: 0.30)",
    )
    parser.add_argument(
        "--ppe-conf",
        type=float,
        default=None,
        help="Ngưỡng tin cậy phát hiện PPE (mặc định: 0.30)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Lưu kết quả (video/ảnh) và file báo cáo JSON/CSV",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Tắt tính năng tự động lưu ảnh snapshot bằng chứng vi phạm",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Thư mục lưu trữ kết quả đầu ra (mặc định: outputs)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Không mở cửa sổ xem trực tiếp OpenCV (phù hợp chạy headless server)",
    )
    parser.add_argument(
        "--no-beep",
        action="store_true",
        help="Tắt âm thanh cảnh báo khi phát hiện vi phạm",
    )
    return parser


def configure_console() -> None:
    """Cấu hình mã hóa UTF-8 cho console hiển thị tiếng Việt chính xác."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_console()
    args = build_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        LOGGER.error("Không tìm thấy file cấu hình: %s", config_path)
        raise SystemExit(1)

    overrides = {
        "show_window": not args.no_display,
        "enable_beep": not args.no_beep,
        "save_output": args.save,
        "save_snapshots": not args.no_snapshots,
        "output_dir": Path(args.output_dir),
    }
    if args.person_model:
        overrides["person_model_path"] = Path(args.person_model)
    if args.ppe_model:
        overrides["ppe_model_path"] = Path(args.ppe_model)
    if args.img_size is not None:
        overrides["image_size"] = args.img_size
    if args.person_conf is not None:
        overrides["person_confidence"] = args.person_conf
    if args.ppe_conf is not None:
        overrides["ppe_confidence"] = args.ppe_conf

    try:
        config = DetectionConfig.load_from_yaml(config_path, **overrides)
    except (FileNotFoundError, ValueError) as err:
        LOGGER.error("Cấu hình không hợp lệ: %s", err)
        raise SystemExit(1) from None

    if not config.person_model_path or not config.ppe_model_path:
        LOGGER.error(
            "CHƯA TRUYỀN MÔ HÌNH: Cần chỉ định --person-model và --ppe-model.\n"
            "Ví dụ chạy: python app.py --source video.mp4 "
            "--person-model models/yolov8n.pt --ppe-model models/best.pt"
        )
        raise SystemExit(1)

    try:
        service = DetectionService(config)
        report, session_dir = service.process(parse_source(args.source))
        if args.save:
            LOGGER.info("Kết quả phiên làm việc được lưu tại: %s", session_dir)
    except (FileNotFoundError, ValueError, OSError) as err:
        LOGGER.error("Lỗi thực thi: %s", err)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
