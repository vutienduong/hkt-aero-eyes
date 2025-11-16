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
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--annotations", required=True)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "train":
        train_module.run_training(cfg)
    elif args.command == "infer":
        infer_module.run_inference(cfg, args)
    elif args.command == "eval":
        from .utils.metrics import evaluate_predictions
        evaluate_predictions(args.predictions, args.annotations)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
