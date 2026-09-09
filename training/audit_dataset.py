"""Kiểm toán manifest ảnh thật và giao thức chống rò rỉ.

Kiểm tra:
1. Kiểm tra chia nhóm; không có session xuất hiện ở nhiều split.
2. Kiểm tra test có camera độc lập nếu protocol yêu cầu.
3. Kiểm tra mã băm SHA-256, không có ảnh trùng lặp.
4. Thống kê phân phối nhãn chi tiết (helmet, no-helmet, vest, no-vest).
5. Ghi trạng thái evidence; không tạo benchmark nếu audit thất bại.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .build_manifest import image_phash, sha256_file
except ImportError:
    from build_manifest import image_phash, sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("audit_dataset")


def audit_dataset(manifest_path: Path, output_report_path: Path | None = None) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        required = {
            "sample_id",
            "recording_session",
            "camera_id",
            "site_id",
            "split",
            "image_path",
            "label_path",
            "class_labels",
            "file_sha256",
            "image_phash",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest thiếu cột bắt buộc: {sorted(missing)}")

    total_samples = len(rows)
    if total_samples == 0:
        raise ValueError("Manifest không có mẫu nào.")
    LOGGER.info("Auditing manifest: %s (%d total samples)", manifest_path, total_samples)

    # 1. Group check: recording_session per split
    session_splits: dict[str, set[str]] = defaultdict(set)
    camera_splits: dict[str, set[str]] = defaultdict(set)
    sha256_map: dict[str, str] = {}
    phash_map: dict[str, str] = {}
    duplicate_sha256: list[str] = []
    duplicate_phash: list[str] = []
    invalid_files: list[str] = []
    invalid_hashes: list[str] = []

    split_counts: Counter[str] = Counter()
    class_distribution: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
        "total": Counter(),
    }

    for row in rows:
        sample_id = row["sample_id"]
        sess = row["recording_session"]
        cam = row["camera_id"]
        split = row["split"].lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split không hợp lệ: {split}")
        sha = row["file_sha256"]
        phash = row["image_phash"]
        labels = [lbl.strip() for lbl in row["class_labels"].split(";") if lbl.strip()]

        split_counts[split] += 1
        session_splits[sess].add(split)
        camera_splits[cam].add(split)

        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        if not label_path.is_absolute():
            label_path = (manifest_path.parent / label_path).resolve()
        if not image_path.is_file() or not label_path.is_file():
            invalid_files.append(sample_id)
        else:
            actual_sha = sha256_file(image_path)
            actual_phash = image_phash(image_path)
            if actual_sha != sha or actual_phash != phash:
                invalid_hashes.append(sample_id)

        if sha in sha256_map:
            duplicate_sha256.append(f"{sample_id} duplicates {sha256_map[sha]}")
        else:
            sha256_map[sha] = sample_id
        if phash in phash_map:
            duplicate_phash.append(f"{sample_id} gần/trùng {phash_map[phash]}")
        else:
            phash_map[phash] = sample_id

        for lbl in labels:
            class_distribution[split][lbl] += 1
            class_distribution["total"][lbl] += 1

    # Kiểm tra rò rỉ session
    leaked_sessions = {s: sp for s, sp in session_splits.items() if len(sp) > 1}

    # Kiểm tra unseen camera cho Test
    train_cams = {cam for cam, sp in camera_splits.items() if "train" in sp}
    test_cams = {cam for cam, sp in camera_splits.items() if "test" in sp}
    unseen_test_cameras = sorted(test_cams - train_cams)

    is_leak_free = (
        len(leaked_sessions) == 0
        and not invalid_files
        and not invalid_hashes
        and not duplicate_sha256
    )
    has_unseen_cameras = bool(unseen_test_cameras)

    report = {
        "manifest_path": str(manifest_path),
        "total_samples": total_samples,
        "split_summary": dict(split_counts),
        "class_distribution": {k: dict(v) for k, v in class_distribution.items()},
        "group_anti_leakage": {
            "is_leak_free": is_leak_free,
            "total_recording_sessions": len(session_splits),
            "leaked_sessions_count": len(leaked_sessions),
            "leaked_sessions": leaked_sessions,
        },
        "domain_shift_protocol": {
            "has_unseen_test_cameras": has_unseen_cameras,
            "train_cameras": sorted(train_cams),
            "test_cameras": sorted(test_cams),
            "unseen_test_cameras": unseen_test_cameras,
        },
        "exact_duplicates_count": len(duplicate_sha256),
        "duplicate_phash_count": len(duplicate_phash),
        "invalid_files": invalid_files,
        "invalid_hashes": invalid_hashes,
        "evidence_status": "measured" if is_leak_free else "invalid",
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with output_report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved dataset audit report to: %s", output_report_path)

    LOGGER.info("Audit Summary:")
    LOGGER.info("  - Total Samples: %d", total_samples)
    LOGGER.info("  - Leak Free: %s", is_leak_free)
    LOGGER.info("  - Unseen Test Cameras: %s", unseen_test_cameras)
    LOGGER.info("  - Classes: %s", dict(class_distribution["total"]))

    if not is_leak_free:
        LOGGER.error(
            "Manifest không hợp lệ: leaked_sessions=%s, invalid_files=%s, invalid_hashes=%s",
            leaked_sessions,
            invalid_files,
            invalid_hashes,
        )
        raise ValueError("Dataset audit thất bại; không được phát hành report benchmark.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset split manifest for leakage and invariants"
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/dataset.csv",
        help="Manifest được build từ metadata thật",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_audit.json",
        help="Nơi ghi report audit",
    )
    args = parser.parse_args()
    audit_dataset(Path(args.manifest), Path(args.output))


if __name__ == "__main__":
    main()
