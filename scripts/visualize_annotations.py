#!/usr/bin/env python
"""
Visualization script for annotations and predictions.

Usage:
    # Visualize ground truth annotations
    python scripts/visualize_annotations.py \
        --video data/train/samples/Backpack_0/drone_video.mp4 \
        --annotations data/train/annotations/annotations.json \
        --video_id Backpack_0 \
        --output outputs/debug_viz/Backpack_0_gt.mp4

    # Visualize predictions
    python scripts/visualize_annotations.py \
        --video data/train/samples/Backpack_0/drone_video.mp4 \
        --predictions outputs/predictions/predictions.json \
        --video_id Backpack_0 \
        --output outputs/debug_viz/Backpack_0_pred.mp4

    # Visualize both (side by side comparison)
    python scripts/visualize_annotations.py \
        --video data/train/samples/Backpack_0/drone_video.mp4 \
        --annotations data/train/annotations/annotations.json \
        --predictions outputs/predictions/predictions.json \
        --video_id Backpack_0 \
        --output outputs/debug_viz/Backpack_0_comparison.mp4 \
        --compare
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def load_annotations(annotations_path, video_id):
    """Load annotations for a specific video."""
    with open(annotations_path, 'r') as f:
        annotations_list = json.load(f)

    for item in annotations_list:
        if item["video_id"] == video_id:
            return item.get("annotations", [])

    return []


def load_predictions(predictions_path, video_id):
    """Load predictions for a specific video."""
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    if video_id in predictions:
        return predictions[video_id].get("detections", [])

    return []


def create_frame_bbox_map(intervals):
    """Create a mapping from frame number to bboxes."""
    frame_map = {}

    for interval in intervals:
        for bbox in interval["bboxes"]:
            frame_num = bbox["frame"]
            if frame_num not in frame_map:
                frame_map[frame_num] = []
            frame_map[frame_num].append(bbox)

    return frame_map


def draw_bbox(frame, bbox, color, label, thickness=2):
    """Draw a bounding box on the frame."""
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(
        frame,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


def visualize_video(video_path, gt_intervals=None, pred_intervals=None, output_path=None, compare=False):
    """
    Visualize annotations and/or predictions on video.

    Args:
        video_path: Path to video file
        gt_intervals: Ground truth annotation intervals
        pred_intervals: Prediction intervals
        output_path: Path to save output video
        compare: If True, create side-by-side comparison
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create frame maps
    gt_map = create_frame_bbox_map(gt_intervals) if gt_intervals else {}
    pred_map = create_frame_bbox_map(pred_intervals) if pred_intervals else {}

    # Setup output video writer
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compare and gt_intervals and pred_intervals:
            # Side by side: double width
            out_width = width * 2
            out_height = height
        else:
            out_width = width
            out_height = height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    else:
        out = None

    print(f"Processing video: {total_frames} frames at {fps} FPS")
    print(f"GT intervals: {len(gt_intervals) if gt_intervals else 0}")
    print(f"Pred intervals: {len(pred_intervals) if pred_intervals else 0}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if compare and gt_intervals and pred_intervals:
            # Create side-by-side comparison
            frame_gt = frame.copy()
            frame_pred = frame.copy()

            # Draw GT bboxes (green)
            if frame_idx in gt_map:
                for bbox in gt_map[frame_idx]:
                    draw_bbox(frame_gt, bbox, (0, 255, 0), "GT")

            # Draw predicted bboxes (blue)
            if frame_idx in pred_map:
                for bbox in pred_map[frame_idx]:
                    draw_bbox(frame_pred, bbox, (255, 0, 0), "Pred")

            # Add labels
            cv2.putText(frame_gt, "Ground Truth", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_pred, "Prediction", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Concatenate horizontally
            combined_frame = np.hstack([frame_gt, frame_pred])

        else:
            # Single view
            combined_frame = frame.copy()

            # Draw GT bboxes (green)
            if gt_intervals and frame_idx in gt_map:
                for bbox in gt_map[frame_idx]:
                    draw_bbox(combined_frame, bbox, (0, 255, 0), "GT")

            # Draw predicted bboxes (blue)
            if pred_intervals and frame_idx in pred_map:
                for bbox in pred_map[frame_idx]:
                    draw_bbox(combined_frame, bbox, (255, 0, 0), "Pred")

        # Add frame number
        cv2.putText(combined_frame, f"Frame: {frame_idx}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write or display
        if out:
            out.write(combined_frame)
        else:
            cv2.imshow("Visualization", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    if out:
        out.release()
        print(f"\nOutput saved to: {output_path}")
    else:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize annotations and predictions")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--video_id", required=True, help="Video ID")
    parser.add_argument("--annotations", help="Path to annotations JSON")
    parser.add_argument("--predictions", help="Path to predictions JSON")
    parser.add_argument("--output", help="Path to save output video (if not provided, display only)")
    parser.add_argument("--compare", action="store_true", help="Create side-by-side comparison")

    args = parser.parse_args()

    if not args.annotations and not args.predictions:
        parser.error("Must provide at least one of --annotations or --predictions")

    # Load data
    gt_intervals = None
    pred_intervals = None

    if args.annotations:
        gt_intervals = load_annotations(args.annotations, args.video_id)
        print(f"Loaded {len(gt_intervals)} ground truth intervals")

    if args.predictions:
        pred_intervals = load_predictions(args.predictions, args.video_id)
        print(f"Loaded {len(pred_intervals)} prediction intervals")

    # Visualize
    visualize_video(
        args.video,
        gt_intervals=gt_intervals,
        pred_intervals=pred_intervals,
        output_path=args.output,
        compare=args.compare
    )


if __name__ == "__main__":
    main()
