import numpy as np
import json
from pathlib import Path


def compute_iou_2d(box1, box2):
    """
    Compute 2D IoU between two bounding boxes.

    Args:
        box1: dict with keys {x1, y1, x2, y2}
        box2: dict with keys {x1, y1, x2, y2}

    Returns:
        iou: float, IoU value between 0 and 1
    """
    x1_max = max(box1["x1"], box2["x1"])
    y1_max = max(box1["y1"], box2["y1"])
    x2_min = min(box1["x2"], box2["x2"])
    y2_min = min(box1["y2"], box2["y2"])

    if x2_min < x1_max or y2_min < y1_max:
        return 0.0

    intersection = (x2_min - x1_max) * (y2_min - y1_max)

    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_st_iou(pred_interval, gt_interval):
    """
    Compute Spatio-Temporal IoU between predicted and ground truth intervals.

    ST-IoU measures the overlap of 3D volumes (x, y, time).

    Args:
        pred_interval: dict with key "bboxes" containing list of frame-bbox dicts
        gt_interval: dict with key "bboxes" containing list of frame-bbox dicts

    Returns:
        st_iou: float, ST-IoU value between 0 and 1
    """
    pred_bboxes = {bbox["frame"]: bbox for bbox in pred_interval["bboxes"]}
    gt_bboxes = {bbox["frame"]: bbox for bbox in gt_interval["bboxes"]}

    # Get all frames that appear in either prediction or GT
    all_frames = set(pred_bboxes.keys()) | set(gt_bboxes.keys())

    if len(all_frames) == 0:
        return 0.0

    # Compute frame-by-frame intersection and union
    total_intersection = 0.0
    total_union = 0.0

    for frame in all_frames:
        pred_box = pred_bboxes.get(frame)
        gt_box = gt_bboxes.get(frame)

        if pred_box is not None and gt_box is not None:
            # Both have detection at this frame
            iou_2d = compute_iou_2d(pred_box, gt_box)

            # Area in 3D: spatial_area * 1 (time duration = 1 frame)
            pred_area = (pred_box["x2"] - pred_box["x1"]) * (pred_box["y2"] - pred_box["y1"])
            gt_area = (gt_box["x2"] - gt_box["x1"]) * (gt_box["y2"] - gt_box["y1"])

            intersection_2d = iou_2d * min(pred_area, gt_area)
            union_2d = pred_area + gt_area - intersection_2d

            total_intersection += intersection_2d
            total_union += union_2d

        elif pred_box is not None:
            # Only prediction at this frame (false positive in time)
            pred_area = (pred_box["x2"] - pred_box["x1"]) * (pred_box["y2"] - pred_box["y1"])
            total_union += pred_area

        elif gt_box is not None:
            # Only GT at this frame (false negative in time)
            gt_area = (gt_box["x2"] - gt_box["x1"]) * (gt_box["y2"] - gt_box["y1"])
            total_union += gt_area

    if total_union == 0:
        return 0.0

    return total_intersection / total_union


def compute_video_st_iou(pred_detections, gt_annotations):
    """
    Compute ST-IoU for a single video by matching intervals.

    Uses greedy matching: for each GT interval, find the best matching predicted interval.

    Args:
        pred_detections: list of detection intervals (each with "bboxes" key)
        gt_annotations: list of annotation intervals (each with "bboxes" key)

    Returns:
        avg_st_iou: float, average ST-IoU across all GT intervals
        matched_ious: list of (gt_idx, pred_idx, st_iou) tuples
    """
    if len(gt_annotations) == 0:
        return 1.0 if len(pred_detections) == 0 else 0.0, []

    matched_ious = []
    used_preds = set()

    # For each GT interval, find best matching prediction
    for gt_idx, gt_interval in enumerate(gt_annotations):
        best_iou = 0.0
        best_pred_idx = -1

        for pred_idx, pred_interval in enumerate(pred_detections):
            if pred_idx in used_preds:
                continue

            st_iou = compute_st_iou(pred_interval, gt_interval)

            if st_iou > best_iou:
                best_iou = st_iou
                best_pred_idx = pred_idx

        if best_pred_idx >= 0:
            used_preds.add(best_pred_idx)

        matched_ious.append((gt_idx, best_pred_idx, best_iou))

    # Average ST-IoU across all GT intervals
    avg_st_iou = sum(iou for _, _, iou in matched_ious) / len(gt_annotations)

    return avg_st_iou, matched_ious


def evaluate_predictions(predictions_path, annotations_path):
    """
    Evaluate predictions against ground truth annotations.

    Args:
        predictions_path: Path to predictions JSON file
        annotations_path: Path to ground truth annotations JSON file

    Returns:
        results: dict with evaluation metrics
    """
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    # Load ground truth
    with open(annotations_path, 'r') as f:
        annotations_list = json.load(f)

    # Convert annotations list to dict
    annotations = {item["video_id"]: item for item in annotations_list}

    video_scores = {}
    all_ious = []

    for video_id, pred_data in predictions.items():
        if video_id not in annotations:
            print(f"Warning: {video_id} not in ground truth annotations")
            continue

        gt_data = annotations[video_id]

        pred_detections = pred_data.get("detections", [])
        gt_annotations = gt_data.get("annotations", [])

        avg_iou, matched_ious = compute_video_st_iou(pred_detections, gt_annotations)

        video_scores[video_id] = {
            "avg_st_iou": avg_iou,
            "num_gt_intervals": len(gt_annotations),
            "num_pred_intervals": len(pred_detections),
            "matched_ious": matched_ious
        }

        all_ious.append(avg_iou)

    # Compute overall metrics
    results = {
        "mean_st_iou": np.mean(all_ious) if all_ious else 0.0,
        "median_st_iou": np.median(all_ious) if all_ious else 0.0,
        "min_st_iou": np.min(all_ious) if all_ious else 0.0,
        "max_st_iou": np.max(all_ious) if all_ious else 0.0,
        "num_videos": len(all_ious),
        "video_scores": video_scores
    }

    return results


def print_evaluation_results(results):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("SPATIO-TEMPORAL IOU EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Mean ST-IoU:     {results['mean_st_iou']:.4f}")
    print(f"  Median ST-IoU:   {results['median_st_iou']:.4f}")
    print(f"  Min ST-IoU:      {results['min_st_iou']:.4f}")
    print(f"  Max ST-IoU:      {results['max_st_iou']:.4f}")
    print(f"  Number of videos: {results['num_videos']}")

    print(f"\nPer-Video Results:")
    print(f"{'Video ID':<20} {'ST-IoU':>8} {'GT Intervals':>14} {'Pred Intervals':>16}")
    print("-" * 60)

    for video_id, scores in sorted(results['video_scores'].items()):
        print(f"{video_id:<20} {scores['avg_st_iou']:>8.4f} "
              f"{scores['num_gt_intervals']:>14} {scores['num_pred_intervals']:>16}")

    print("="*60 + "\n")
