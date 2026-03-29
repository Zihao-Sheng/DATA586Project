#!/usr/bin/env python3
"""
Retrieve and repair the Food-101 dataset in the local data directory.

Default behavior:
1) Check integrity of data/food-101
2) Download archive if needed
3) Repair missing/corrupt files when possible
4) Re-check integrity and exit non-zero if still incomplete
"""

from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DATASET_NAME = "food-101"
ARCHIVE_NAME = f"{DATASET_NAME}.tar.gz"
ARCHIVE_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EXPECTED_SPLIT_COUNTS = {"train": 75750, "test": 25250}
META_MEMBERS = {
    "food-101/meta/train.txt",
    "food-101/meta/test.txt",
    "food-101/meta/classes.txt",
}
CHUNK_SIZE = 1024 * 1024  # 1 MB


@dataclass
class IntegrityReport:
    is_complete: bool
    issues: list[str] = field(default_factory=list)
    restore_meta: bool = False
    missing_image_members: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Food-101 into data/ and repair missing files if needed."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Destination folder for dataset files (default: data).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    return parser.parse_args()


def load_split_entries(split_file: Path) -> list[str]:
    content = split_file.read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def check_dataset_integrity(dataset_root: Path) -> IntegrityReport:
    issues: list[str] = []
    missing_image_members: set[str] = set()
    restore_meta = False

    if not dataset_root.exists():
        issues.append(f"Dataset folder is missing: {dataset_root}")
        return IntegrityReport(
            is_complete=False,
            issues=issues,
            restore_meta=True,
            missing_image_members=missing_image_members,
        )

    meta_dir = dataset_root / "meta"
    images_dir = dataset_root / "images"

    if not meta_dir.is_dir():
        issues.append(f"Meta directory is missing: {meta_dir}")
        restore_meta = True
    if not images_dir.is_dir():
        issues.append(f"Images directory is missing: {images_dir}")

    split_entries: dict[str, list[str]] = {}
    if meta_dir.is_dir():
        for split_name, expected_count in EXPECTED_SPLIT_COUNTS.items():
            split_file = meta_dir / f"{split_name}.txt"
            if not split_file.is_file():
                issues.append(f"Missing split file: {split_file}")
                restore_meta = True
                continue

            entries = load_split_entries(split_file)
            split_entries[split_name] = entries
            if len(entries) != expected_count:
                issues.append(
                    f"{split_name}.txt has {len(entries)} entries, expected {expected_count}."
                )
                restore_meta = True

        classes_file = meta_dir / "classes.txt"
        if not classes_file.is_file():
            issues.append(f"Missing classes file: {classes_file}")
            restore_meta = True

    if images_dir.is_dir():
        for split_name, entries in split_entries.items():
            split_missing = 0
            for rel_path in entries:
                image_path = images_dir / f"{rel_path}.jpg"
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    missing_image_members.add(f"food-101/images/{rel_path}.jpg")
                    split_missing += 1

            if split_missing:
                issues.append(
                    f"{split_name} split has missing/corrupt images: {split_missing} file(s)."
                )

    is_complete = not issues and not restore_meta and not missing_image_members
    return IntegrityReport(
        is_complete=is_complete,
        issues=issues,
        restore_meta=restore_meta,
        missing_image_members=missing_image_members,
    )


def is_safe_member_path(base_dir: Path, member_name: str) -> bool:
    base_resolved = base_dir.resolve()
    target_resolved = (base_dir / member_name).resolve(strict=False)
    try:
        common = os.path.commonpath([str(base_resolved), str(target_resolved)])
    except ValueError:
        return False
    return common == str(base_resolved)


def download_archive(archive_path: Path, force_redownload: bool = False) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")

    if archive_path.exists() and not force_redownload:
        print(f"Using existing archive: {archive_path}")
        return

    if tmp_path.exists():
        tmp_path.unlink()

    request = urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "data-retrieval-script/1.0"}
    )
    print(f"Downloading {ARCHIVE_URL}")
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as file_obj:
        content_length = response.headers.get("Content-Length", "0")
        total_size = int(content_length) if content_length.isdigit() else 0
        downloaded = 0

        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            file_obj.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(
                    f"\rDownloaded {downloaded / (1024**2):.1f} MB / "
                    f"{total_size / (1024**2):.1f} MB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )
    print()

    if archive_path.exists():
        archive_path.unlink()
    tmp_path.replace(archive_path)
    print(f"Archive saved to: {archive_path}")


def validate_archive(archive_path: Path) -> bool:
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        return False
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            tar.getmember("food-101/meta/train.txt")
            tar.getmember("food-101/meta/test.txt")
            tar.getmember("food-101/meta/classes.txt")
        return True
    except (tarfile.TarError, KeyError):
        return False


def ensure_archive(archive_path: Path, force_redownload: bool = False) -> None:
    if archive_path.exists() and not force_redownload:
        if validate_archive(archive_path):
            print(f"Archive is valid: {archive_path}")
            return
        print(f"Archive exists but is invalid, re-downloading: {archive_path}")
        archive_path.unlink()

    download_archive(archive_path, force_redownload=force_redownload)
    if not validate_archive(archive_path):
        raise RuntimeError(f"Downloaded archive failed validation: {archive_path}")


def extract_all_members(archive_path: Path, destination_dir: Path) -> None:
    print("Extracting full dataset archive...")
    with tarfile.open(archive_path, mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            if not is_safe_member_path(destination_dir, member.name):
                raise RuntimeError(f"Unsafe archive path detected: {member.name}")
        tar.extractall(path=destination_dir, members=members)
    print("Full extraction finished.")


def extract_selected_members(
    archive_path: Path, destination_dir: Path, member_names: Iterable[str]
) -> None:
    requested = set(member_names)
    if not requested:
        return

    extracted = 0
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar:
            if member.name in requested:
                if not is_safe_member_path(destination_dir, member.name):
                    raise RuntimeError(f"Unsafe archive path detected: {member.name}")
                tar.extract(member, path=destination_dir)
                extracted += 1
                requested.remove(member.name)
                if not requested:
                    break

    if requested:
        sample = sorted(requested)[:5]
        raise RuntimeError(
            f"{len(requested)} requested file(s) not found in archive. Sample: {sample}"
        )

    print(f"Patched {extracted} file(s) from archive.")


def repair_dataset(
    dataset_root: Path, data_dir: Path, archive_path: Path, initial_report: IntegrityReport
) -> None:
    if not dataset_root.exists():
        extract_all_members(archive_path, data_dir)
        return

    if initial_report.restore_meta:
        print("Repairing metadata files...")
        extract_selected_members(archive_path, data_dir, META_MEMBERS)

    follow_up_report = check_dataset_integrity(dataset_root)
    if follow_up_report.missing_image_members:
        print(
            f"Repairing missing/corrupt images: {len(follow_up_report.missing_image_members)} file(s)..."
        )
        extract_selected_members(
            archive_path, data_dir, follow_up_report.missing_image_members
        )


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    dataset_root = data_dir / DATASET_NAME
    archive_path = data_dir / ARCHIVE_NAME

    data_dir.mkdir(parents=True, exist_ok=True)

    if args.force_redownload:
        print("Force redownload mode enabled.")
        ensure_archive(archive_path, force_redownload=True)
        extract_all_members(archive_path, data_dir)
    else:
        report = check_dataset_integrity(dataset_root)
        if report.is_complete:
            print(f"Dataset is complete: {dataset_root}")
            return

        print("Dataset integrity check failed:")
        for issue in report.issues:
            print(f"- {issue}")

        ensure_archive(archive_path, force_redownload=False)
        repair_dataset(dataset_root, data_dir, archive_path, report)

    final_report = check_dataset_integrity(dataset_root)
    if not final_report.is_complete:
        print("Dataset is still incomplete after repair.")
        for issue in final_report.issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print(f"Dataset is ready at: {dataset_root}")


if __name__ == "__main__":
    main()
