#!/usr/bin/env python
"""
Evaluation script for computing ST-IoU between predictions and ground truth.

Usage:
    python scripts/eval_st_iou.py \
        --predictions outputs/predictions/predictions.json \
        --annotations data/train/annotations/annotations.json \
        --output outputs/evaluation_results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import evaluate_predictions, print_evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-IoU metrics")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to ground truth annotations JSON file"
    )
    parser.add_argument(
        "--output",
        help="Optional path to save evaluation results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-video results"
    )

    args = parser.parse_args()

    print(f"Evaluating predictions from: {args.predictions}")
    print(f"Against ground truth: {args.annotations}")

    # Run evaluation
    results = evaluate_predictions(args.predictions, args.annotations)

    # Print results
    print_evaluation_results(results)

    # Print detailed results if requested
    if args.verbose:
        print("\nDetailed Per-Interval Matches:")
        print("="*80)
        for video_id, scores in sorted(results['video_scores'].items()):
            print(f"\n{video_id}:")
            for gt_idx, pred_idx, iou in scores['matched_ious']:
                if pred_idx >= 0:
                    print(f"  GT interval {gt_idx} -> Pred interval {pred_idx}: ST-IoU = {iou:.4f}")
                else:
                    print(f"  GT interval {gt_idx} -> No match (ST-IoU = {iou:.4f})")

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
