"""Script xuất mô hình YOLO PPE sang định dạng tối ưu ONNX / TorchScript / TensorRT."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("export")


def export_model(model_path: str, format_type: str = "onnx", imgsz: int = 640) -> None:
    """Xuất mô hình sang định dạng đầu ra mong muốn."""
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")

    LOGGER.info(
        "Xuất mô hình [%s] sang định dạng [%s] với imgsz=%d...", model_path, format_type, imgsz
    )
    model = YOLO(model_path)
    exported_path = model.export(format=format_type, imgsz=imgsz)
    LOGGER.info("Xuất mô hình thành công: %s", exported_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất mô hình YOLO PPE")
    parser.add_argument(
        "--weights", default="models/best.pt", help="Đường dẫn file trọng số PyTorch (.pt)"
    )
    parser.add_argument(
        "--format", default="onnx", help="Định dạng xuất (onnx, engine, torchscript, openvino)"
    )
    parser.add_argument("--img-size", type=int, default=640, help="Kích thước ảnh (mặc định: 640)")
    args = parser.parse_args()

    try:
        export_model(args.weights, args.format, args.img_size)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as err:
        LOGGER.error("Lỗi xuất mô hình: %s", err)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
