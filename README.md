# PPE Safety Monitoring

[![CI](https://github.com/haminhthong/PPE-Safety-Monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/PPE-Safety-Monitoring/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-111F68)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Lint](https://img.shields.io/badge/lint-Ruff-D7FF64)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-3DA639)](LICENSE)

Hệ thống Computer Vision giám sát tuân thủ trang bị bảo hộ lao động (PPE - Personal Protective Equipment) trên video và camera giám sát công trường/nhà máy.

Điểm khác biệt cốt lõi: Kết quả đầu ra là **sự kiện vi phạm theo thời gian (Violation Event)** gắn với từng công nhân cụ thể, kèm **ảnh bằng chứng (Snapshot)** và **dấu thời gian (Timestamp)** — thay vì chỉ vẽ một bounding box đơn lẻ ở từng khung hình.

---

## 1. Bài toán Thực tế (The Problem)

Trong các công trường xây dựng hoặc xưởng sản xuất thực tế:
1. **False Alarm do nhấp nháy mô hình (Detector Flicker)**: Mô hình YOLO dự đoán theo từng frame độc lập. Nếu công nhân cúi đầu hoặc quay lưng trong 1-2 frame, detector có thể bỏ sót mũ và ngay lập tức báo vi phạm sai (False Alarm).
2. **Gán nhầm trang bị giữa người với người (Spatial Misassignment)**: Khi nhiều công nhân đứng sát nhau, việc chạy detection trực tiếp trên toàn khung hình dễ dẫn đến việc mũ của người này bị nhận nhầm là của người đứng cạnh.

---

## 2. Luồng Xử Lý 7 Bước (Pipeline Architecture)

Hệ thống kết hợp quy trình suy luận 2 giai đoạn (Two-Stage Detection), theo dõi vết (Tracking) và máy trạng thái thời gian (Temporal Confirmation FSM):

```text
VIDEO / CAMERA STREAM
         │
         ▼
 1. Person Detection (YOLOv8)
    Detect bounding box công nhân trên khung hình gốc
         │
         ▼
 2. Multi-Object Tracking (Two-Threshold IoU)
    Duy trì ID định danh ổn định & dự đoán vận tốc mượt mà
         │
         ▼
 3. Person Crop (PersonCropBuilder)
    Trích xuất vùng ảnh ROI của từng người với padding
         │
         ▼
 4. PPE Detection (YOLOv8)
    Nhận diện 4 lớp trang bị: helmet, no-helmet, vest, no-vest
         │
         ▼
 5. Body-Zone Spatial Filtering
    Lọc vị trí giải phẫu: Mũ (0 - 35% đầu), Áo (30 - 75% thân)
         │
         ▼
 6. Temporal Confirmation (FSM)
    Duy trì trạng thái PRESENT / ABSENT / UNKNOWN theo dwell time
         │
         ▼
 7. Violation Event & Snapshot
    Phát cảnh báo, chụp ảnh ROI bằng chứng và xuất báo cáo JSON/CSV
```

---

## 3. Các Điểm Kỹ Thuật Trọng Tâm (Technical Deep Dive)

### Tại sao cần Two-Stage Detection (Person → Crop → PPE)?
Thay vì cố phát hiện các vật thể nhỏ (mũ, áo) trực tiếp trên toàn bộ khung hình góc rộng (dễ bị nhiễu và gán sai chủ thể), hệ thống phát hiện người trước, cắt ROI từng worker rồi mới nhận diện PPE bên trong ROI đó. Cách tiếp cận này thu hẹp không gian tìm kiếm và đảm bảo trang bị luôn gắn chặt với đúng người.

### Body-Zone Filtering giải quyết vấn đề gì?
Khi hai công nhân đứng chồng lấn lên nhau, ROI cắt có thể chứa cả bộ phận của người bên cạnh. Bằng cách chia tỷ lệ giải phẫu:
- **Vùng đầu (Head Zone)**: Chiều cao $y \in [0.00, 0.35]$ của worker box.
- **Vùng thân (Torso Zone)**: Chiều cao $y \in [0.30, 0.75]$ của worker box.

Mọi phát hiện mũ ở vùng chân hoặc áo ở vùng đầu đều bị loại bỏ, ngăn ngừa việc "mũ người A bị xem là mũ người B".

### Tại sao cần Two-Threshold IoU Tracking?
- **Giai đoạn 1**: Ghép cặp các detection có độ tin cậy cao (`high_threshold >= 0.5`) với các track hiện có qua IoU.
- **Giai đoạn 2**: Dùng các detection có độ tin cậy thấp (`low_match_threshold >= 0.3`) để phục hồi các track bị che khuất một phần (occlusion) thay vì đánh mất ID hoặc sinh ID mới.
- **Velocity Prediction**: Khi khung hình không chạy detector để tiết kiệm tài nguyên, vị trí box được nội suy từ vector vận tốc mượt mà, chống hiện tượng đứng hình (box freeze).

### Tại sao cần Tri-State (PRESENT / ABSENT / UNKNOWN)?
Hệ thống không chỉ dùng True/False nhị phân:
- `PRESENT`: Nhìn thấy trang bị với độ tin cậy cao.
- `ABSENT`: Nhìn thấy rõ vùng đầu/thân nhưng không có trang bị.
- `UNKNOWN`: Góc nhìn bị che khuất, ánh sáng yếu hoặc mô hình không đủ bằng chứng.

`UNKNOWN` không làm tăng bằng chứng vi phạm nhưng cũng **không reset trạng thái vi phạm đang chờ xác nhận**, giúp hệ thống ổn định trước nhiễu camera.

### FSM theo Dwell Time giảm False Alert như thế nào?
Máy trạng thái hữu hạn hoạt động dựa trên thời gian tồn tại thực tế (`confirm_after_sec`):
```text
COMPLIANT (Tuân thủ)
   ↓ ABSENT liên tục >= 0.5s
VIOLATING → ALERTED (Phát cảnh báo & lưu snapshot)
   ↓ PRESENT liên tục >= 1.0s
RESOLVED (Khắc phục)
   ↓ ABSENT trở lại (sau cooldown 10s)
ALERTED (Báo động tái phạm - Recurrent Event)
```
Một frame đơn lẻ bị nhận diện sai không bao giờ kích hoạt cảnh báo giả.

### Tại sao Group Split theo Video/Session khi huấn luyện?
Trong dữ liệu video, các khung hình liên tiếp từ cùng một góc quay có độ tương đồng cực kỳ cao. Nếu chia ngẫu nhiên (random split), mô hình sẽ bị rò rỉ dữ liệu (data leakage) — frame 100 ở train và frame 101 ở test làm điểm test cao giả tạo nhưng trượt khi deploy thực tế. Dự án thực hiện **Group Split theo Video/Session**: toàn bộ frame của 1 video chỉ thuộc về 1 tập duy nhất.

---

## 4. Cấu trúc Dự án

```text
PPE-Safety-Monitoring/
├── configs/
│   └── config.yaml             # File cấu hình duy nhất: model, tracking, ppe, alert
├── ppe_detection/
│   ├── config.py               # Dataclass DetectionConfig
│   ├── crops.py                # PersonCropBuilder cắt ROI
│   ├── detector.py             # Two-stage DualModelDetector, body-zone, unicode I/O
│   ├── models.py               # Dataclasses: PPEDetection, PPEStatus, ViolationState
│   ├── pipeline.py             # Pipeline chính điều phối video/camera
│   ├── reporting.py            # SessionReport, ViolationEvent -> JSON, CSV
│   ├── service.py              # Quản lý phiên làm việc & thư mục đầu ra
│   ├── tracker.py              # TwoThresholdIoUTracker & IoUTracker + motion prediction
│   ├── violation_fsm.py        # TemporalViolationFSM theo dwell time
│   └── visualization.py        # Vẽ bounding box, nhãn và trạng thái an toàn
├── training/
│   ├── prepare_dataset.py      # Group split theo video + tạo person crops
│   ├── train.py                # Huấn luyện mô hình YOLO
│   ├── evaluate.py             # Đánh giá Detection (mAP) & Event (False alerts, recall)
│   └── data.yaml               # YOLO dataset config
├── scripts/
│   └── benchmark_video.py      # Đo FPS, độ trễ và số sự kiện vi phạm trên video
├── tests/
│   ├── conftest.py             # MockDetector độc lập weights
│   ├── test_tracker.py         # Kiểm tra tracking & recovery
│   ├── test_fsm.py             # Kiểm tra FSM dwell time, cooldown, tri-state
│   ├── test_spatial.py         # Kiểm tra body-zone head/torso, ROI
│   ├── test_pipeline.py        # Kiểm tra xử lý pipeline ảnh/frame
│   ├── test_config.py          # Kiểm tra nạp cấu hình YAML
│   └── test_reporting.py       # Kiểm tra xuất báo cáo JSON/CSV
├── app.py                      # Điểm vào CLI
├── web_app.py                  # Dashboard Web Streamlit
├── Dockerfile                  # Docker container chạy Streamlit
├── requirements.txt            # Phụ thuộc runtime
└── requirements-dev.txt        # Phụ thuộc dev/test (Ruff, Pytest)
```

---

## 5. Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt môi trường

```bash
git clone https://github.com/haminhthong/PPE-Safety-Monitoring.git
cd PPE-Safety-Monitoring

# Khởi tạo virtual environment (khuyến nghị Python 3.10 hoặc 3.11)
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Web Dashboard (Streamlit)

```bash
streamlit run web_app.py
```
Mở trình duyệt tại `http://localhost:8501` để:
- Tải lên video/ảnh giám sát công trường.
- Tùy chỉnh trực tiếp ngưỡng tin cậy, chu kỳ inspect và thời gian xác nhận FSM.
- Xem video/ảnh đã gắn nhãn bounding box, ID công nhân và ảnh chụp bằng chứng vi phạm.

### 3. Chạy qua giao diện dòng lệnh (CLI)

```bash
# Giám sát từ file Video
python app.py --source sample.mp4 --person-model models/yolov8n.pt --ppe-model models/best.pt --save

# Giám sát từ Webcam máy tính (Camera index 0)
python app.py --source 0 --person-model models/yolov8n.pt --ppe-model models/best.pt --save

# Chạy trên ảnh tĩnh
python app.py --source sample.jpg --person-model models/yolov8n.pt --ppe-model models/best.pt --save
```

### 4. Đo lường hiệu năng thực tế (Benchmark)

```bash
python scripts/benchmark_video.py --video sample.mp4 --config configs/config.yaml
```

### 5. Huấn luyện & Đánh giá

```bash
# Chuẩn bị dataset (Group split theo video & cắt crop)
python training/prepare_dataset.py --manifest data/manifest.csv --output-dir data/dataset_ppe

# Huấn luyện mô hình PPE
python training/train.py --config configs/train_yolov8n.yaml

# Đánh giá mAP phát hiện PPE
python training/evaluate.py --task detection --model models/best.pt --data training/data.yaml --split test

# Đánh giá sự kiện vi phạm (False alerts/hour, event precision/recall)
python training/evaluate.py --task event --gt-events gt.json --pred-events pred.json --duration-hours 1.5
```

---

## 6. Chạy với Docker

```bash
# Build Docker image
docker build -t ppe-monitoring .

# Khởi chạy container (truy cập tại http://localhost:8501)
docker run -p 8501:8501 ppe-monitoring
```

---

## 7. Kiểm thử Tự động (Testing)

Dự án bao gồm bộ unit test toàn diện cho các thành phần cốt lõi:
```bash
# Chạy toàn bộ unit tests
python -m pytest

# Kiểm tra quy chuẩn code với Ruff linter
python -m ruff check .
```

---

## 8. Giấy phép (License)

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
