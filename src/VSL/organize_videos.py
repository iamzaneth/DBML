import csv
import os
import shutil
import sys
import re
from pathlib import Path
from collections import defaultdict

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ORGANIZE VSL VIDEOS BY LABEL
# ==========================================

# Paths
csv_path = "data/raw/VSL/Dataset/Labels/label.csv"
videos_src_dir = "data/raw/VSL/Dataset/Videos"
output_dir = "data/interim/VSL"


def sanitize_folder_name(label):
    """
    Sanitize label so it can be safely used as a folder name on Windows/Linux.
    Vietnamese characters are kept. Only invalid path characters are replaced.
    """
    label = str(label).strip()

    # Windows invalid characters: < > : " / \ | ? *
    label = re.sub(r'[<>:"/\\|?*]', "_", label)

    # Remove trailing spaces/dots because Windows does not allow them
    label = label.strip(" .")

    # Avoid empty folder name
    return label if label else "unknown_label"


def load_label_csv(csv_file):
    """Load VSL label mapping CSV file"""
    rows = []

    # utf-8-sig handles CSV files saved with BOM
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"VIDEO", "LABEL"}
        missing_cols = required_cols - set(reader.fieldnames or [])

        if missing_cols:
            raise ValueError(
                f"CSV file is missing required columns: {sorted(missing_cols)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            video_name = (row.get("VIDEO") or "").strip()
            label = (row.get("LABEL") or "").strip()

            if video_name and label:
                rows.append({
                    "id": (row.get("ID") or "").strip(),
                    "video": video_name,
                    "label": label
                })

    return rows


def create_label_to_videos_mapping(rows):
    """Create a mapping of label -> list of video filenames"""
    label_videos = defaultdict(list)

    for row in rows:
        label = row["label"]
        video_name = row["video"]

        if label and video_name:
            label_videos[label].append(video_name)

    return label_videos


def organize_videos(label_videos, src_dir, dst_dir):
    """
    Organize VSL videos into folders by label
    """
    # Create output directory if it doesn't exist
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    total_videos = 0
    copied_videos = 0
    skipped_existing = 0
    missing_videos = []
    error_videos = []

    # For each VSL label
    for label, video_names in sorted(label_videos.items()):
        safe_label_folder = sanitize_folder_name(label)
        label_folder = os.path.join(dst_dir, safe_label_folder)
        Path(label_folder).mkdir(parents=True, exist_ok=True)

        print(f"\n[Processing] Label: '{label}'")
        if safe_label_folder != label:
            print(f"   Folder name sanitized to: '{safe_label_folder}'")

        print(f"   Found {len(video_names)} video(s)")

        # For each video in this label
        for video_name in video_names:
            total_videos += 1

            # Ensure filename is only filename, not accidentally a path
            video_file = os.path.basename(video_name)
            src_path = os.path.join(src_dir, video_file)
            dst_path = os.path.join(label_folder, video_file)

            # Check if source video exists
            if os.path.exists(src_path):
                try:
                    # Skip copy if destination already exists
                    if os.path.exists(dst_path):
                        skipped_existing += 1
                        print(f"   [SKIP] Already exists: {video_file}")
                        continue

                    # Copy video to label folder
                    shutil.copy2(src_path, dst_path)
                    copied_videos += 1
                    print(f"   [OK] {video_file}")

                except Exception as e:
                    print(f"   [ERROR] Error copying {video_file}: {e}")
                    error_videos.append(video_file)
            else:
                print(f"   [MISSING] {video_file}")
                missing_videos.append(video_file)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total labels:       {len(label_videos)}")
    print(f"Total video refs:   {total_videos}")
    print(f"Videos copied:      {copied_videos}")
    print(f"Skipped existing:   {skipped_existing}")
    print(f"Videos missing:     {len(missing_videos)}")
    print(f"Copy errors:        {len(error_videos)}")
    print(f"Output directory:   {os.path.abspath(dst_dir)}")

    if missing_videos:
        print(f"\nMissing videos: {missing_videos[:10]}")
        if len(missing_videos) > 10:
            print(f"   ... and {len(missing_videos) - 10} more")

    if error_videos:
        print(f"\nError videos: {error_videos[:10]}")
        if len(error_videos) > 10:
            print(f"   ... and {len(error_videos) - 10} more")


def main():
    print("=== Starting VSL video organization process ===\n")

    # Check if source files exist
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return

    if not os.path.exists(videos_src_dir):
        print(f"ERROR: Videos directory not found at {videos_src_dir}")
        return

    # Load label CSV
    print(f"[1] Loading label CSV from {csv_path}...")
    rows = load_label_csv(csv_path)
    print(f"    Loaded {len(rows)} valid video-label row(s)")

    # Create mapping
    print(f"\n[2] Creating label -> videos mapping...")
    label_videos = create_label_to_videos_mapping(rows)
    print(f"    Found {len(label_videos)} unique label(s)")

    # Organize videos
    print(f"\n[3] Organizing videos by label into {output_dir}/...")
    organize_videos(label_videos, videos_src_dir, output_dir)

    print("\n=== Process completed ===\n")


if __name__ == "__main__":
    main()
