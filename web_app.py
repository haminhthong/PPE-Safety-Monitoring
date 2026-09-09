"""Giao diện Web Tương tác (Streamlit Web Dashboard) cho Hệ thống PPE Surveillance.

Cho phép tải ảnh/video và tùy chỉnh ngưỡng phát hiện.
và xem trực tiếp kết quả phát hiện vi phạm trang bị bảo hộ kèm biểu đồ phân tích thống kê.

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
from ppe_detection.service import DetectionService

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="PPE Safety Surveillance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 200


def inject_custom_css() -> None:
    """Inject CSS tùy biến để nâng cấp giao diện Web UI sang trọng và chuyên nghiệp."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .demo-warning {
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 10px 15px;
            border-radius: 6px;
            color: #92400E;
            font-weight: 500;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()

    st.markdown(
        "<div class='main-header'>🛡️ Hệ Thống Giám Sát Trang Bị Bảo Hộ AI (PPE)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sub-header'>Ứng dụng YOLO và IoU Tracking phát hiện "
        "vi phạm mũ, áo bảo hộ theo thời gian thực</div>",
        unsafe_allow_html=True,
    )

    # 1. Thanh điều khiển Sidebar
    st.sidebar.header("⚙️ Cấu hình Hệ thống")

    demo_mode = st.sidebar.checkbox(
        "⚡ Chế độ mô phỏng pipeline (Demo Simulation)",
        value=True,
        help=(
            "Bật SyntheticDemoDetector để thử pipeline mà không cần trọng số mô hình."
        ),
    )

    if demo_mode:
        st.markdown(
            "<div class='demo-warning'>⚠️ <b>CHẾ ĐỘ MÔ PHỎNG PIPELINE</b>: "
            "Kết quả do SyntheticDemoDetector tạo, không phải model thật.</div>",
            unsafe_allow_html=True,
        )

    policy_path = Path(__file__).resolve().parent / "configs" / "runtime_policy.yaml"
    try:
        policy_defaults = DetectionConfig.load_from_policy(policy_path, demo_mode=True)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Runtime policy không hợp lệ: {error}")
        return

    col_m1, col_m2 = st.sidebar.columns(2)
    with col_m1:
        person_model_str = st.text_input(
            "Model Person", value="models/yolov8n.pt", disabled=demo_mode
        )
    with col_m2:
        ppe_model_str = st.text_input("Model PPE", value="models/best.pt", disabled=demo_mode)

    st.sidebar.subheader("🎛️ Ngưỡng Phát Hiện")
    person_conf = st.sidebar.slider(
        "Confidence Người", 0.1, 1.0, float(policy_defaults.person_confidence), 0.05
    )
    ppe_conf = st.sidebar.slider(
        "Confidence PPE", 0.1, 1.0, float(policy_defaults.ppe_confidence), 0.05
    )
    person_detection_interval = st.sidebar.slider(
        "Chu kỳ detect người", 1, 10, policy_defaults.detection_interval
    )
    ppe_detection_interval = st.sidebar.slider(
        "Chu kỳ inspect PPE", 1, 10, policy_defaults.ppe_detection_interval
    )
    confirm_seconds = st.sidebar.slider(
        "Thời gian xác nhận vi phạm (giây)",
        0.0,
        10.0,
        float(policy_defaults.violation_confirm_seconds),
        0.1,
    )
    resolve_seconds = st.sidebar.slider(
        "Thời gian xác nhận khắc phục (giây)",
        0.0,
        10.0,
        float(policy_defaults.resolution_confirm_seconds),
        0.1,
    )

    save_snapshots = st.sidebar.checkbox("📸 Lưu bằng chứng vi phạm (Snapshots)", value=True)

    # 2. Khu vực Tải dữ liệu Đầu vào
    st.markdown("### 📤 Tải lên Dữ liệu Giám sát")
    uploaded_file = st.file_uploader(
        "Chọn file Ảnh (.jpg, .png) hoặc Video (.mp4, .avi)",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    if uploaded_file is None:
        st.info(
            "💡 **Gợi ý**: Tải ảnh/video lên để trải nghiệm hoặc bật chế độ "
            "mô phỏng ở thanh bên trái."
        )
        return

    # Kiểm tra kích thước file upload
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"❌ File upload vượt quá giới hạn {MAX_UPLOAD_MB} MB. "
            "Vui lòng chọn file nhỏ hơn."
        )
        return

    # 3. Tiến hành Xử lý Dữ liệu với dọn dẹp an toàn try...finally
    st.markdown("---")
    st.markdown("### 🎬 Kết quả Xử lý & Phân tích")

    person_path = Path(person_model_str) if not demo_mode and person_model_str else None
    ppe_path = Path(ppe_model_str) if not demo_mode and ppe_model_str else None

    try:
        config = DetectionConfig.load_from_policy(
            policy_path,
            person_model_path=person_path,
            ppe_model_path=ppe_path,
            person_confidence=person_conf,
            ppe_confidence=ppe_conf,
            detection_interval=person_detection_interval,
            ppe_detection_interval=ppe_detection_interval,
            violation_confirm_seconds=confirm_seconds,
            resolution_confirm_seconds=resolve_seconds,
            show_window=False,
            save_output=True,
            save_snapshots=save_snapshots,
            output_dir=Path("outputs"),
            demo_mode=demo_mode,
            enable_beep=False,
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Runtime policy không hợp lệ: {error}")
        return

    tmp_path: Path | None = None
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        # Kiểm tra tính khả đọc của OpenCV
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            test_img = cv2.imread(str(tmp_path))
            if test_img is None:
                st.error("❌ OpenCV không thể đọc file ảnh này. Vui lòng định dạng lại file.")
                return
        else:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                cap.release()
                st.error(
                    "❌ OpenCV không thể mở video. Kiểm tra codec hoặc chọn .mp4."
                )
                return
            cap.release()

        with st.spinner("Đang thực thi nhận diện và theo dõi đối tượng trong session..."):
            service = DetectionService(config)
            report, session_dir = service.process(str(tmp_path))

        # 4. Hiển thị Thông số Thống kê Cards
        counts = report.counts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Tổng Số Người Được Theo Dõi", len(report.unique_track_ids))
        col2.metric("⚠️ Số Người Vi Phạm", counts["people"])
        col3.metric("🪖 Vi Phạm Mũ Bảo Hộ", counts["helmet"])
        col4.metric("🦺 Vi Phạm Áo Phản Quang", counts["vest"])

        # 5. Hiển thị Kết quả Đầu ra (Xem ảnh/video trong session_dir)
        st.markdown("#### 🖼️ Xem Trực Quan Đầu Ra")
        st.caption(f"📁 Thư mục lưu trữ kết quả phiên: `{session_dir}`")
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            out_img_path = session_dir / f"{tmp_path.stem}_detected{suffix}"
            if out_img_path.exists():
                st.image(
                    str(out_img_path),
                    caption="Kết quả phát hiện PPE trên ảnh",
                    use_container_width=True,
                )
        else:
            out_vid_path = session_dir / f"{tmp_path.stem}_detected.mp4"
            if out_vid_path.exists():
                st.video(str(out_vid_path))

        # 6. Bảng Chi Tiết Sự Kiện Vi Phạm
        if report.events:
            st.markdown("#### 📋 Danh Sách Sự Kiện Vi Phạm Chi Tiết")
            df = pd.DataFrame(
                [
                    {
                        "Track ID": e.track_id,
                        "Loại Vi Phạm": "Mũ bảo hộ (Helmet)"
                        if e.violation_type == "helmet"
                        else "Áo phản quang (Vest)",
                        "Frame": e.frame_id,
                        "Thời điểm (Giây)": e.time_seconds,
                        "Thời gian thực": e.detected_at,
                        "Ảnh Bằng Chứng": e.snapshot_path,
                    }
                    for e in report.events
                ]
            )
            st.dataframe(df, use_container_width=True)

    except (FileNotFoundError, OSError, ValueError) as err:
        st.error(f"❌ Xảy ra lỗi trong quá trình xử lý: {err}")
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
