import json
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from .models.siam_tracker import SiamTracker
from .utils.video_io import iter_video_frames
from .utils.tracking import postprocess_track

def run_inference(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SiamTracker(
        backbone_name=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Get data directory from config or args
    if hasattr(args, 'data_dir') and args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(cfg["data"]["root"]) / cfg["data"]["samples_dir"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = {}

    # Get template image names from config
    template_names = cfg["data"]["template_images"]
    target_size = tuple(cfg["data"]["frame_size"])  # (width, height)

    # Loop over each video folder
    for video_dir in sorted(data_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name
        video_path = video_dir / "drone_video.mp4"
        template_dir = video_dir / "object_images"

        # Load and encode template (3 images: img_1.jpg, img_2.jpg, img_3.jpg)
        template_imgs = []
        for img_name in template_names:
            img_path = template_dir / img_name
            img = Image.open(img_path).convert("RGB")
            img = img.resize(cfg["model"]["template_size"])  # Resize to template size
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            template_imgs.append(img_tensor)

        # Stack and average the 3 template images
        template_tensor = torch.stack(template_imgs).mean(dim=0).unsqueeze(0).to(device)

        # Set template in model
        with torch.no_grad():
            model.set_template(template_tensor)

        detections = []

        frame_idx = 0
        for frame in iter_video_frames(str(video_path)):
            # Preprocess frame to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, target_size)
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                cls_logits, bbox_pred = model(frame_tensor)

            # Decode bbox and confidence from model outputs
            # cls_logits: (1, 1, H, W) - objectness heatmap
            # bbox_pred: (1, 4, H, W) - bbox predictions at each location

            # Apply sigmoid to get confidence scores
            cls_scores = torch.sigmoid(cls_logits[0, 0])  # (H, W)

            # Find location with maximum confidence
            max_score = cls_scores.max().item()
            max_loc = (cls_scores == max_score).nonzero(as_tuple=False)[0]  # (y, x)
            max_y, max_x = max_loc[0].item(), max_loc[1].item()

            # Get bbox prediction at max location
            bbox_at_max = bbox_pred[0, :, max_y, max_x]  # (4,)

            # Scale bbox to image coordinates
            # bbox_at_max is in normalized form, scale to target_size
            x1 = bbox_at_max[0].item() * target_size[0]
            y1 = bbox_at_max[1].item() * target_size[1]
            x2 = bbox_at_max[2].item() * target_size[0]
            y2 = bbox_at_max[3].item() * target_size[1]

            # Clamp to image bounds
            x1 = max(0, min(x1, target_size[0]))
            y1 = max(0, min(y1, target_size[1]))
            x2 = max(0, min(x2, target_size[0]))
            y2 = max(0, min(y2, target_size[1]))

            # Add detection if score is reasonable
            if max_score > 0.01:  # Very low threshold to catch all potential detections
                detections.append({
                    "frame": frame_idx,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "score": max_score
                })

            frame_idx += 1

        # Post-process into intervals & JSON structure
        predictions[video_id] = postprocess_track(
            raw_dets=detections,
            confidence_threshold=cfg["infer"]["confidence_threshold"],
            max_gap=cfg["infer"]["max_gap"],
            min_len=cfg["infer"]["min_interval_len"],
        )

    with output_path.open("w") as f:
        json.dump(predictions, f, indent=2)
