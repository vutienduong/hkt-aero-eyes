#!/usr/bin/env python
"""
Dataset preparation utilities.

Usage:
    # Print dataset statistics
    python scripts/prepare_dataset.py \
        --annotations data/train/annotations/annotations.json \
        --stats

    # Create train/val split
    python scripts/prepare_dataset.py \
        --annotations data/train/annotations/annotations.json \
        --create_splits \
        --train_ratio 0.8 \
        --output_dir data/splits

    # Analyze annotations
    python scripts/prepare_dataset.py \
        --annotations data/train/annotations/annotations.json \
        --analyze
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def load_annotations(annotations_path):
    """Load annotations from JSON file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)


def print_statistics(annotations):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    total_videos = len(annotations)
    total_intervals = 0
    total_frames = 0
    interval_lengths = []
    videos_by_object = defaultdict(list)

    for item in annotations:
        video_id = item["video_id"]
        intervals = item.get("annotations", [])
        total_intervals += len(intervals)

        # Extract object type from video_id (e.g., "Backpack_0" -> "Backpack")
        object_type = video_id.rsplit('_', 1)[0]
        videos_by_object[object_type].append(video_id)

        for interval in intervals:
            bboxes = interval["bboxes"]
            interval_length = len(bboxes)
            total_frames += interval_length
            interval_lengths.append(interval_length)

    print(f"\nOverall:")
    print(f"  Total videos: {total_videos}")
    print(f"  Total intervals: {total_intervals}")
    print(f"  Total annotated frames: {total_frames}")
    print(f"  Avg frames per interval: {total_frames / total_intervals:.1f}")

    print(f"\nInterval lengths:")
    print(f"  Min: {min(interval_lengths)}")
    print(f"  Max: {max(interval_lengths)}")
    print(f"  Mean: {sum(interval_lengths) / len(interval_lengths):.1f}")

    print(f"\nVideos by object type:")
    for object_type, video_ids in sorted(videos_by_object.items()):
        print(f"  {object_type}: {len(video_ids)} videos")

    print("\n" + "="*60 + "\n")


def analyze_annotations(annotations):
    """Detailed analysis of annotations."""
    print("\n" + "="*60)
    print("DETAILED ANNOTATION ANALYSIS")
    print("="*60)

    for item in annotations:
        video_id = item["video_id"]
        intervals = item.get("annotations", [])

        print(f"\n{video_id}:")
        print(f"  Number of intervals: {len(intervals)}")

        for i, interval in enumerate(intervals):
            bboxes = interval["bboxes"]
            frames = [b["frame"] for b in bboxes]

            print(f"  Interval {i}:")
            print(f"    Frames: {min(frames)} - {max(frames)} ({len(frames)} frames)")
            print(f"    Frame coverage: {min(frames)}, ..., {max(frames)}")

            # Check for gaps
            if len(frames) > 1:
                sorted_frames = sorted(frames)
                gaps = []
                for j in range(len(sorted_frames) - 1):
                    gap = sorted_frames[j+1] - sorted_frames[j]
                    if gap > 1:
                        gaps.append((sorted_frames[j], sorted_frames[j+1], gap-1))

                if gaps:
                    print(f"    Gaps detected:")
                    for start, end, gap_size in gaps:
                        print(f"      Frame {start} -> {end}: {gap_size} frame gap")

            # Bbox size statistics
            widths = [b["x2"] - b["x1"] for b in bboxes]
            heights = [b["y2"] - b["y1"] for b in bboxes]

            print(f"    BBox sizes:")
            print(f"      Width:  {min(widths):.0f} - {max(widths):.0f} (avg: {sum(widths)/len(widths):.0f})")
            print(f"      Height: {min(heights):.0f} - {max(heights):.0f} (avg: {sum(heights)/len(heights):.0f})")

    print("\n" + "="*60 + "\n")


def create_splits(annotations, train_ratio=0.8, output_dir="data/splits", seed=42):
    """Create train/val splits."""
    random.seed(seed)

    video_ids = [item["video_id"] for item in annotations]
    random.shuffle(video_ids)

    split_idx = int(len(video_ids) * train_ratio)
    train_ids = video_ids[:split_idx]
    val_ids = video_ids[split_idx:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write train split
    train_path = output_dir / "train.txt"
    with open(train_path, 'w') as f:
        for video_id in sorted(train_ids):
            f.write(f"{video_id}\n")

    # Write val split
    val_path = output_dir / "val.txt"
    with open(val_path, 'w') as f:
        for video_id in sorted(val_ids):
            f.write(f"{video_id}\n")

    print(f"\nCreated splits:")
    print(f"  Train: {len(train_ids)} videos -> {train_path}")
    print(f"  Val:   {len(val_ids)} videos -> {val_path}")

    # Print object distribution
    train_objects = defaultdict(int)
    val_objects = defaultdict(int)

    for video_id in train_ids:
        object_type = video_id.rsplit('_', 1)[0]
        train_objects[object_type] += 1

    for video_id in val_ids:
        object_type = video_id.rsplit('_', 1)[0]
        val_objects[object_type] += 1

    print(f"\nObject distribution:")
    print(f"  Train: {dict(train_objects)}")
    print(f"  Val:   {dict(val_objects)}")


def main():
    parser = argparse.ArgumentParser(description="Dataset preparation utilities")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed analysis"
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create train/val splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)"
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits",
        help="Output directory for splits (default: data/splits)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)"
    )

    args = parser.parse_args()

    # Load annotations
    annotations = load_annotations(args.annotations)
    print(f"Loaded annotations for {len(annotations)} videos")

    # Run requested operations
    if args.stats:
        print_statistics(annotations)

    if args.analyze:
        analyze_annotations(annotations)

    if args.create_splits:
        create_splits(
            annotations,
            train_ratio=args.train_ratio,
            output_dir=args.output_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
