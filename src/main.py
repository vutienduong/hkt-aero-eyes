import argparse
from .config import load_config
from . import train as train_module
from . import infer as infer_module

def build_parser():
    parser = argparse.ArgumentParser(prog="aeroeyes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train model")
    p_train.add_argument("--config", required=True, help="Path to YAML config")

    # Infer
    p_infer = subparsers.add_parser("infer", help="Run inference on samples")
    p_infer.add_argument("--config", required=True)
    p_infer.add_argument("--data_dir", required=True)
    p_infer.add_argument("--output", required=True)
    p_infer.add_argument("--checkpoint", required=True)

    # Eval
    p_eval = subparsers.add_parser("eval", help="Evaluate predictions vs GT")
    p_eval.add_argument("--predictions", required=True, help="Path to predictions JSON")
    p_eval.add_argument("--annotations", required=True, help="Path to annotations JSON")
    p_eval.add_argument("--output", help="Optional path to save results JSON")
    p_eval.add_argument("--verbose", action="store_true", help="Print detailed results")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = load_config(args.config)
        train_module.run_training(cfg)
    elif args.command == "infer":
        cfg = load_config(args.config)
        infer_module.run_inference(cfg, args)
    elif args.command == "eval":
        from .utils.metrics import evaluate_predictions, print_evaluation_results
        import json
        from pathlib import Path

        results = evaluate_predictions(args.predictions, args.annotations)
        print_evaluation_results(results)

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

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
