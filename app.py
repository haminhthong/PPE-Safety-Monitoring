"""Điểm truy cập dòng lệnh (CLI Entry Point) cho ứng dụng phát hiện PPE.

Cung cấp giao diện dòng lệnh với nhiều tùy chọn linh hoạt: nguồn đầu vào (webcam/ảnh/video),
file weights mô hình, các ngưỡng confidence, lưu báo cáo JSON/CSV và ảnh bằng chứng vi phạm.

Ví dụ sử dụng:
    1. Chạy Demo không cần weights:
       python app.py --demo

    2. Chạy với Webcam và lưu báo cáo kết quả:
       python app.py --source 0 --person-model models/yolov8n.pt \\
       --ppe-model models/best.pt --save

    3. Chạy xử lý file Video:
       python app.py --source data/test.mp4 --person-model models/yolov8n.pt \\
       --ppe-model models/best.pt --save
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ppe_detection.config import DetectionConfig

# Cấu hình logging định dạng tiếng Việt chuẩn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("app")


def parse_source(value: str) -> int | str:
    """Chuyển đổi chuỗi chỉ số camera thành số nguyên hoặc giữ nguyên đường dẫn file.

    Args:
        value: Chuỗi truyền vào từ CLI (ví dụ "0" hoặc "video.mp4").

    Returns:
        Số nguyên nếu là chỉ số camera, hoặc chuỗi nếu là đường dẫn file.
    """
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    """Tạo bộ đọc tham số dòng lệnh ArgumentParser.

    Returns:
        Đối tượng ArgumentParser với đầy đủ cờ tùy chọn.
    """
    parser = argparse.ArgumentParser(
        description="Ứng dụng Computer Vision phát hiện người và vi phạm trang bị bảo hộ (PPE)."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Đường dẫn file ảnh/video hoặc chỉ số camera (mặc định: 0)",
    )
    parser.add_argument(
        "--policy",
        default=str(Path(__file__).resolve().parent / "configs" / "runtime_policy.yaml"),
        help="Runtime policy YAML; đây là nguồn cấu hình chính của production.",
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
        "--demo",
        action="store_true",
        help="Bật chế độ Demo giả lập (Zero-Setup) không cần file model ngoài",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Kích thước ảnh đầu vào cho YOLO inference (mặc định: 640)",
    )
    parser.add_argument(
        "--person-interval",
        type=int,
        default=None,
        help="Override chu kỳ phát hiện người (chỉ dùng debug).",
    )
    parser.add_argument(
        "--ppe-interval",
        "--detect-interval",
        dest="ppe_detection_interval",
        type=int,
        default=None,
        help="Override chu kỳ kiểm tra PPE (chỉ dùng debug). --detect-interval là alias cũ.",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Ngưỡng tin cậy tối thiểu cho phát hiện người (mặc định: 0.3)",
    )
    parser.add_argument(
        "--ppe-conf",
        type=float,
        default=None,
        help="Ngưỡng tin cậy tối thiểu cho phát hiện PPE (mặc định: 0.3)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Lưu kết quả (video/ảnh) và file báo cáo JSON/CSV vào thư mục output",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Tắt tính năng tự động lưu ảnh snapshot bằng chứng vi phạm",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Thư mục đầu ra để lưu kết quả và báo cáo (mặc định: outputs)",
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
    """Cấu hình mã hóa UTF-8 cho console để hiển thị tiếng Việt chính xác trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    """Hàm thực thi chính khi khởi chạy ứng dụng từ CLI."""
    configure_console()
    args = build_parser().parse_args()

    is_demo = args.demo
    if not is_demo and (not args.person_model or not args.ppe_model):
        LOGGER.error(
            " CHƯA TRUYỀN MÔ HÌNH: Thiếu --person-model hoặc --ppe-model.\n"
            "   Để chạy chế độ mô phỏng pipeline thử nghiệm, vui lòng truyền cờ '--demo'.\n"
            "   Hoặc truyền đường dẫn model thật: "
            "python app.py --person-model models/yolov8n.pt "
            "--ppe-model models/best.pt"
        )
        raise SystemExit(1)

    try:
        from ppe_detection.service import DetectionService
    except ModuleNotFoundError as error:
        if error.name in {"cv2", "torch", "ultralytics", "numpy"}:
            raise SystemExit(
                f"Thiếu thư viện {error.name!r}. Hãy chạy lệnh: pip install -r requirements.txt"
            ) from None
        raise

    person_p = Path(args.person_model) if args.person_model else None
    ppe_p = Path(args.ppe_model) if args.ppe_model else None
    output_p = Path(args.output_dir)

    overrides = {
        "person_model_path": person_p,
        "ppe_model_path": ppe_p,
        "show_window": not args.no_display,
        "enable_beep": not args.no_beep,
        "save_output": args.save,
        "save_snapshots": not args.no_snapshots,
        "output_dir": output_p,
        "demo_mode": is_demo,
    }
    if args.img_size is not None:
        overrides["image_size"] = args.img_size
    if args.person_interval is not None:
        overrides["detection_interval"] = args.person_interval
    if args.ppe_detection_interval is not None:
        overrides["ppe_detection_interval"] = args.ppe_detection_interval
    if args.person_conf is not None:
        overrides["person_confidence"] = args.person_conf
    if args.ppe_conf is not None:
        overrides["ppe_confidence"] = args.ppe_conf
    try:
        config = DetectionConfig.load_from_policy(Path(args.policy), **overrides)
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("Runtime policy không hợp lệ: %s", error)
        raise SystemExit(1) from None

    try:
        service = DetectionService(config)
        report, session_dir = service.process(parse_source(args.source))
        if args.save:
            LOGGER.info("Kết quả phiên làm việc được lưu tại: %s", session_dir)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.error("Lỗi thực thi: %s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
