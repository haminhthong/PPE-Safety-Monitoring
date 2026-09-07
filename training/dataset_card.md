# Dataset card

Dataset production của dự án phải được xây từ file metadata do người thu thập
cung cấp bằng `training/build_manifest.py`.

Repo không chứa dataset ảnh/video thật. Vì vậy các con số 5.200 mẫu, 14 phiên
ghi hình, phân phối lớp hoặc unseen-camera trong tài liệu cũ không còn được xem
là benchmark. `training/split_manifest.csv` cũ đã được loại khỏi production
story vì được sinh từ metadata tổng hợp.

Manifest hợp lệ phải có:

- provenance: `source_id`, `parent_video_id`, `camera_id`, `site_id`,
  `recording_session`;
- annotation: `image_path`, `label_path`, `class_labels`, `annotation_version`;
- thời gian: `frame_id`, `timestamp_ms`;
- integrity: `file_sha256` tính trên bytes ảnh và `image_phash` tính trên ảnh đã
  decode;
- `split` được gán theo recording session trước khi tạo person crop.

Chỉ report được tạo từ manifest thật, model weights thật và locked test mới được
đưa vào README hoặc release bundle.
