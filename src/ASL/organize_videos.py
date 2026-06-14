import json
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ORGANIZE VIDEOS BY GLOSS LABEL
# ==========================================

# Paths
json_path = "data/raw/ASL/WLASL_v0.3.json"
videos_src_dir = "data/raw/ASL/videos"
output_dir = "data/interim/ASL"

def load_metadata(json_file):
    """Load metadata from WLASL JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_gloss_to_videos_mapping(metadata):
    """Create a mapping of gloss -> list of video_ids"""
    gloss_videos = defaultdict(list)
    
    for entry in metadata:
        gloss = entry.get("gloss")
        for instance in entry.get("instances", []):
            video_id = instance.get("video_id")
            if gloss and video_id:
                gloss_videos[gloss].append(video_id)
    
    return gloss_videos

def organize_videos(gloss_videos, src_dir, dst_dir):
    """
    Organize videos into folders by gloss label
    """
    # Create output directory if it doesn't exist
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
    total_videos = 0
    copied_videos = 0
    missing_videos = []
    
    # For each gloss label
    for gloss, video_ids in sorted(gloss_videos.items()):
        gloss_folder = os.path.join(dst_dir, gloss)
        Path(gloss_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Processing] Gloss: '{gloss}'")
        print(f"   Found {len(video_ids)} video(s)")
        
        # For each video in this gloss
        for video_id in video_ids:
            total_videos += 1
            video_file = f"{video_id}.mp4"
            src_path = os.path.join(src_dir, video_file)
            dst_path = os.path.join(gloss_folder, video_file)
            
            # Check if source video exists
            if os.path.exists(src_path):
                try:
                    # Copy video to gloss folder
                    shutil.copy2(src_path, dst_path)
                    copied_videos += 1
                    print(f"   [OK] {video_file}")
                except Exception as e:
                    print(f"   [ERROR] Error copying {video_file}: {e}")
                    missing_videos.append(video_id)
            else:
                print(f"   [MISSING] {video_file}")
                missing_videos.append(video_id)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total glosses:     {len(gloss_videos)}")
    print(f"Total video refs:  {total_videos}")
    print(f"Videos copied:     {copied_videos}")
    print(f"Videos missing:    {len(missing_videos)}")
    print(f"Output directory:  {os.path.abspath(dst_dir)}")
    
    if missing_videos:
        print(f"\nMissing video IDs: {missing_videos[:10]}")
        if len(missing_videos) > 10:
            print(f"   ... and {len(missing_videos) - 10} more")

def main():
    print("=== Starting video organization process ===\n")
    
    # Check if source files exist
    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found at {json_path}")
        return
    
    if not os.path.exists(videos_src_dir):
        print(f"ERROR: Videos directory not found at {videos_src_dir}")
        return
    
    # Load metadata
    print(f"[1] Loading metadata from {json_path}...")
    metadata = load_metadata(json_path)
    print(f"    Loaded {len(metadata)} gloss entries")
    
    # Create mapping
    print(f"\n[2] Creating gloss -> videos mapping...")
    gloss_videos = create_gloss_to_videos_mapping(metadata)
    print(f"    Found {len(gloss_videos)} unique glosses")
    
    # Organize videos
    print(f"\n[3] Organizing videos by gloss into {output_dir}/...")
    organize_videos(gloss_videos, videos_src_dir, output_dir)
    
    print("\n=== Process completed ===\n")

if __name__ == "__main__":
    main()
