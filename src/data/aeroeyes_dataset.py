from torch.utils.data import Dataset
from pathlib import Path
import cv2
import json
import numpy as np
import torch

class AeroEyesDataset(Dataset):
    """
    Returns (template_image, search_image, target_bbox)
    built from challenge annotations.

    Args:
        root: Path to data root (e.g., "data/train")
        annotations_file: Relative path to annotations JSON (e.g., "annotations/annotations.json")
        split_file: Path to train/val split file
        transforms: Optional transforms to apply
    """

    def __init__(self, root, annotations_file, split_file, transforms=None):
        self.root = Path(root)
        self.transforms = transforms

        # Load single annotations.json file
        ann_path = self.root / annotations_file
        with open(ann_path, "r") as f:
            all_annotations = json.load(f)

        # Index annotations by video_id
        self.annotations_map = {item["video_id"]: item for item in all_annotations}

        # Load split (video IDs to use)
        with open(split_file, "r") as f:
            self.video_ids = [line.strip() for line in f if line.strip()]

        # Build training samples from annotations
        self.samples = self._build_samples()

    def _build_samples(self):
        """
        Parse annotations into training pairs.
        Each sample: (video_id, template_frame_idx, search_frame_idx, template_bbox, search_bbox)
        """
        samples = []

        for video_id in self.video_ids:
            if video_id not in self.annotations_map:
                continue

            video_ann = self.annotations_map[video_id]

            # Each video has multiple annotation intervals
            for interval in video_ann["annotations"]:
                bboxes = interval["bboxes"]
                if len(bboxes) < 2:
                    continue

                # For each interval, create training pairs
                # Use first frame as template, sample nearby frames as search
                template_idx = 0
                template_bbox = bboxes[template_idx]

                # Sample search frames from the same interval
                for search_idx in range(1, len(bboxes)):
                    search_bbox = bboxes[search_idx]

                    samples.append({
                        "video_id": video_id,
                        "template_frame": template_bbox["frame"],
                        "search_frame": search_bbox["frame"],
                        "template_bbox": template_bbox,
                        "search_bbox": search_bbox,
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]

        # Load video
        video_path = self.root / "samples" / video_id / "drone_video.mp4"
        cap = cv2.VideoCapture(str(video_path))

        # Read template frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["template_frame"])
        ret, template_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read template frame {sample['template_frame']} from {video_id}")

        # Crop template bbox
        tb = sample["template_bbox"]
        template_img = template_frame[tb["y1"]:tb["y2"], tb["x1"]:tb["x2"]]
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

        # Read search frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["search_frame"])
        ret, search_frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read search frame {sample['search_frame']} from {video_id}")

        search_img = cv2.cvtColor(search_frame, cv2.COLOR_BGR2RGB)
        cap.release()

        # Target bbox in search frame
        sb = sample["search_bbox"]
        target_bbox = np.array([sb["x1"], sb["y1"], sb["x2"], sb["y2"]], dtype=np.float32)

        # Apply transforms if provided
        if self.transforms:
            template_img, search_img, target_bbox = self.transforms(
                template_img, search_img, target_bbox
            )
        else:
            # Basic conversion to tensor
            template_img = torch.from_numpy(template_img).permute(2, 0, 1).float() / 255.0
            search_img = torch.from_numpy(search_img).permute(2, 0, 1).float() / 255.0
            target_bbox = torch.from_numpy(target_bbox)

        return template_img, search_img, target_bbox
