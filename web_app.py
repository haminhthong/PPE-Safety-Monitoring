"""Giao diện Web Trực quan (Streamlit Dashboard) cho Hệ thống Giám sát An toàn PPE.

Các tính năng chính:
1. Tải lên video/ảnh thực tế từ công trường hoặc nhà máy.
2. Tùy chỉnh trực tiếp các ngưỡng phát hiện, chu kỳ inspection và dwell time của FSM.
3. Thực thi nhận diện hai giai đoạn (Person Detection → Tracking → Crop → PPE Detection).
4. Xem video/ảnh đã gắn nhãn bounding box, mã định danh ID và trạng thái an toàn.
5. Xem bảng tổng hợp các sự kiện vi phạm cùng ảnh chụp bằng chứng (Snapshots).

Chạy ứng dụng:
    streamlit run web_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from ppe_detection.config import DetectionConfig
from ppe_detection.detector import read_image
from ppe_detection.service import DetectionService

st.set_page_config(
    page_title="Hệ thống Giám sát Trang bị Bảo hộ PPE",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 200


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()

    st.markdown(
        "<div class='main-header'>🦺 Hệ Thống Giám Sát Tuân Thủ Trang Bị Bảo Hộ (PPE)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sub-header'>Quy trình 2 giai đoạn: Phát hiện người → "
        "Tracking → Crop Worker → Kiểm tra PPE → Temporal FSM</div>",
        unsafe_allow_html=True,
    )

    config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    try:
        defaults = DetectionConfig.load_from_yaml(config_path)
    except Exception as err:
        st.error(f"Không thể đọc file cấu hình {config_path}: {err}")
        return

    # 1. Sidebar - Thiết lập cấu hình
    st.sidebar.header("⚙️ Cấu hình Mô hình & Ngưỡng")

    col_m1, col_m2 = st.sidebar.columns(2)
    with col_m1:
        person_model_str = st.text_input("Model Person", value="models/yolov8n.pt")
    with col_m2:
        ppe_model_str = st.text_input("Model PPE", value="models/best.pt")

    st.sidebar.subheader("🎛️ Ngưỡng Nhận Diện & Tracking")
    person_conf = st.sidebar.slider(
        "Confidence Người", 0.1, 1.0, float(defaults.person_confidence), 0.05
    )
    ppe_conf = st.sidebar.slider("Confidence PPE", 0.1, 1.0, float(defaults.ppe_confidence), 0.05)
    ppe_interval = st.sidebar.slider(
        "Chu kỳ kiểm tra PPE (frame)", 1, 10, defaults.ppe_detection_interval
    )

    st.sidebar.subheader("⏱️ Temporal Confirmation (FSM)")
    confirm_sec = st.sidebar.slider(
        "Thời gian xác nhận vi phạm (giây)",
        0.1,
        5.0,
        float(defaults.violation_confirm_seconds),
        0.1,
    )
    resolve_sec = st.sidebar.slider(
        "Thời gian xác nhận khắc phục (giây)",
        0.1,
        5.0,
        float(defaults.resolution_confirm_seconds),
        0.1,
    )

    save_snapshots = st.sidebar.checkbox("📸 Lưu ảnh bằng chứng vi phạm (Snapshots)", value=True)

    # 2. Upload Dữ liệu
    st.markdown("### 📤 Tải lên Dữ liệu Video hoặc Ảnh")
    uploaded_file = st.file_uploader(
        "Chọn file Ảnh (.jpg, .png) hoặc Video (.mp4, .avi, .mov)",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    if uploaded_file is None:
        st.info("💡 **Gợi ý**: Tải lên video hoặc ảnh công trường/nhà máy để bắt đầu phân tích.")
        return

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"❌ Kích thước file vượt quá giới hạn {MAX_UPLOAD_MB} MB.")
        return

    person_model_path = Path(person_model_str)
    ppe_model_path = Path(ppe_model_str)

    if not person_model_path.is_file() or not ppe_model_path.is_file():
        st.warning(
            f"⚠️ **Chưa tìm thấy file weights**: `{person_model_path}` hoặc `{ppe_model_path}`.\n\n"
            "Vui lòng đặt trọng số vào thư mục `models/` hoặc nhập đúng đường dẫn ở thanh bên trái."
        )
        return

    config = DetectionConfig.load_from_yaml(
        config_path,
        person_model_path=person_model_path,
        ppe_model_path=ppe_model_path,
        person_confidence=person_conf,
        ppe_confidence=ppe_conf,
        ppe_detection_interval=ppe_interval,
        violation_confirm_seconds=confirm_sec,
        resolution_confirm_seconds=resolve_sec,
        show_window=False,
        save_output=True,
        save_snapshots=save_snapshots,
        output_dir=Path("outputs"),
        enable_beep=False,
    )

    st.markdown("---")
    st.markdown("### 🎬 Kết quả Xử lý & Phân tích")

    tmp_path: Path | None = None
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        is_image = suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if is_image:
            test_img = read_image(tmp_path)
            if test_img is None:
                st.error("❌ Không thể đọc file ảnh. Vui lòng kiểm tra định dạng.")
                return
        else:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                cap.release()
                st.error("❌ Không thể mở file video. Vui lòng kiểm tra codec hoặc dùng file .mp4.")
                return
            cap.release()

        with st.spinner("Đang chạy pipeline phát hiện và theo dõi đối tượng..."):
            service = DetectionService(config)
            report, session_dir = service.process(str(tmp_path))

        # Thống kê cards
        counts = report.counts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Công nhân theo dõi", len(report.unique_track_ids))
        col2.metric("⚠️ Số người vi phạm", counts["people"])
        col3.metric("🪖 Vi phạm Mũ", counts["helmet"])
        col4.metric("🦺 Vi phạm Áo", counts["vest"])

        # Hiển thị trực quan
        st.markdown("#### 🖼️ Trực Quan Hóa Đầu Ra")
        st.caption(f"📁 Kết quả được lưu tại: `{session_dir}`")

        if is_image:
            out_img = session_dir / f"{tmp_path.stem}_detected{suffix}"
            if out_img.is_file():
                st.image(str(out_img), caption="Ảnh kết quả giám sát PPE", use_container_width=True)
        else:
            out_vid = session_dir / f"{tmp_path.stem}_detected.mp4"
            if out_vid.is_file():
                st.video(str(out_vid))

        # Bảng sự kiện vi phạm
        if report.events:
            st.markdown("#### 📋 Danh Sách Sự Kiện Vi Phạm (Violation Events)")
            df = pd.DataFrame(
                [
                    {
                        "Worker ID": e.track_id,
                        "Loại Vi Phạm": "Mũ bảo hộ (Helmet)"
                        if e.violation_type == "helmet"
                        else "Áo phản quang (Vest)",
                        "Frame Bắt Đầu": e.frame_id,
                        "Thời Điểm (giây)": e.time_seconds,
                        "Thời Gian Thực": e.detected_at,
                        "Ảnh Bằng Chứng": e.snapshot_path,
                    }
                    for e in report.events
                ]
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.success("✅ Không phát hiện vi phạm an toàn nào kéo dài vượt ngưỡng cảnh báo!")

    except Exception as err:
        st.error(f"❌ Xảy ra lỗi trong quá trình xử lý: {err}")
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
