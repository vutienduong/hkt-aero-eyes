#!/usr/bin/env python
"""
Extract frames from videos for preprocessing.

Usage:
    python scripts/extract_frames.py \
        --video data/train/samples/Backpack_0/drone_video.mp4 \
        --output data/train/processed/Backpack_0 \
        --stride 1
"""

import argparse
import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, stride=1, max_frames=None):
    """
    Extract frames from video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        stride: Extract every Nth frame (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Stride: {stride}")

    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted += 1

            if extracted % 100 == 0:
                print(f"Extracted {extracted} frames...")

            if max_frames and extracted >= max_frames:
                break

        frame_idx += 1

    cap.release()

    print(f"\nExtracted {extracted} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--stride", type=int, default=1, help="Extract every Nth frame (default: 1)")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to extract")

    args = parser.parse_args()

    extract_frames(args.video, args.output, args.stride, args.max_frames)


if __name__ == "__main__":
    main()
