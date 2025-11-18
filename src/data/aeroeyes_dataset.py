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

        # Cache video frame counts for validation
        print("Validating video frame counts...")
        self.video_frame_counts = self._get_video_frame_counts()

        # Build training samples from annotations
        self.samples = self._build_samples()
        print(f"Dataset initialized with {len(self.samples)} valid samples")

    def _get_video_frame_counts(self):
        """Get frame count for each video to validate annotations."""
        frame_counts = {}
        for video_id in self.video_ids:
            video_path = self.root / "samples" / video_id / "drone_video.mp4"
            if not video_path.exists():
                print(f"Warning: Video not found: {video_id}")
                frame_counts[video_id] = 0
                continue

            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_counts[video_id] = frame_count

        return frame_counts

    def _build_samples(self):
        """
        Parse annotations into training pairs.
        Each sample: (video_id, template_frame_idx, search_frame_idx, template_bbox, search_bbox)
        """
        samples = []
        skipped_count = 0

        for video_id in self.video_ids:
            if video_id not in self.annotations_map:
                continue

            video_ann = self.annotations_map[video_id]
            max_frame = self.video_frame_counts.get(video_id, 0)

            if max_frame == 0:
                print(f"Skipping {video_id}: no valid frames")
                continue

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

                    # Validate frame numbers
                    template_frame = template_bbox["frame"]
                    search_frame = search_bbox["frame"]

                    if template_frame >= max_frame or search_frame >= max_frame:
                        skipped_count += 1
                        continue

                    samples.append({
                        "video_id": video_id,
                        "template_frame": template_frame,
                        "search_frame": search_frame,
                        "template_bbox": template_bbox,
                        "search_bbox": search_bbox,
                    })

        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples due to invalid frame numbers")

        return samples

    def __len__(self):
        return len(self.samples)

    def _read_frame(self, cap, frame_idx, video_id, frame_type="frame"):
        """
        Safely read a frame from video with fallback strategies.

        Args:
            cap: cv2.VideoCapture object
            frame_idx: Frame index to read
            video_id: Video ID for error messages
            frame_type: "template" or "search" for error messages

        Returns:
            frame: BGR frame as numpy array

        Raises:
            RuntimeError: If frame cannot be read after all attempts
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if frame is within bounds (should be caught during sample building)
        if frame_idx >= total_frames:
            raise RuntimeError(
                f"Frame {frame_idx} out of bounds for {video_id} "
                f"(total: {total_frames}). This should have been filtered during dataset construction."
            )

        # Try seeking method first (faster)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret and frame is not None:
            return frame

        # Fallback: sequential read (slower but more reliable)
        print(f"Warning: Seeking failed for frame {frame_idx} in {video_id}, trying sequential read...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(frame_idx + 1):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(
                    f"Failed to read {frame_type} frame {frame_idx} from {video_id} "
                    f"(stopped at frame {i}/{total_frames})"
                )

        return frame

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]

        # Load video
        video_path = self.root / "samples" / video_id / "drone_video.mp4"
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video {video_id}")

        try:
            # Read template frame
            template_frame = self._read_frame(
                cap, sample["template_frame"], video_id, "template"
            )

            # Crop template bbox
            tb = sample["template_bbox"]
            template_img = template_frame[tb["y1"]:tb["y2"], tb["x1"]:tb["x2"]]
            template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)

            # Read search frame
            search_frame = self._read_frame(
                cap, sample["search_frame"], video_id, "search"
            )

            search_img = cv2.cvtColor(search_frame, cv2.COLOR_BGR2RGB)
        finally:
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
