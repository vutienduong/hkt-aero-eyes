# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AeroEyes is a few-shot drone object localizer for the AeroEyes hackathon. The system uses a Siamese tracker architecture to localize target objects in drone videos given only 3 reference images. The model predicts spatio-temporal bounding boxes (x, y, t volumes) and outputs detections in the hackathon's JSON format.

**Core Problem**: Given 3 RGB reference images of a target object and a 3-5 minute drone video, predict time intervals where the object appears with frame-by-frame bounding boxes. Performance measured by Spatio-Temporal IoU (ST-IoU).

## Common Commands

### Configuration Files Convention

- **`configs/baseline.yaml`**: For local development and training
- **`configs/jetson_infer.yaml`**: For hackathon submission, Docker deployment, and Jetson inference

```bash
# Environment setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Using Makefile.py (Python-based Makefile)
python Makefile.py create-env  # Create virtual environment
python Makefile.py install      # Install dependencies
python Makefile.py train        # Train model (uses baseline.yaml)
python Makefile.py infer        # Run inference (uses jetson_infer.yaml)

# Direct commands - Local development
python -m src.main train --config configs/baseline.yaml
python -m src.main infer --config configs/baseline.yaml --data_dir data/train/samples --output outputs/predictions/predictions.json --checkpoint checkpoints/baseline.ckpt
python -m src.main eval --predictions outputs/predictions/predictions.json --annotations data/train/annotations/annotations.json

# Direct commands - Hackathon submission (public_test)
python -m src.main infer --config configs/jetson_infer.yaml --data_dir data/public_test/samples --output outputs/predictions/predictions.json --checkpoint checkpoints/baseline.ckpt

# Docker (for deployment)
docker build -f docker/Dockerfile.infer.md -t aeroeyes-infer .
```

## Architecture

### High-Level Pipeline

1. **Template Encoding**: Encode 3 reference images into a template embedding using a CNN backbone
2. **Frame Processing**: For each video frame, compute feature map and match against template
3. **Detection**: Predict bounding box and confidence score per frame
4. **Temporal Tracking**: Filter, smooth, and group detections into continuous intervals
5. **Output**: Convert intervals to hackathon JSON format

### Model Structure (Siamese Tracker)

- **Backbone** (`src/models/backbone.py`): ResNet18/MobileNet for feature extraction (pretrained on ImageNet)
- **Head** (`src/models/head.py`): SiamRPN-style head for objectness classification and bbox regression
- **SiamTracker** (`src/models/siam_tracker.py`): Wrapper combining backbone + head
  - Training mode: processes both template and search images
  - Inference mode: pre-encodes template via `set_template()`, reuses for all frames

### Data Pipeline

- **AeroEyesDataset** (`src/data/aeroeyes_dataset.py`): Constructs training pairs from annotations
  - Each sample: `(template_image, search_image, target_bbox)`
  - Template extracted from GT bbox within annotated interval
  - Search images sampled from nearby frames
- **Actual data structure**:
  ```
  data/
    train/
      samples/
        Backpack_0/
          drone_video.mp4
          object_images/
            img_1.jpg, img_2.jpg, img_3.jpg
        ... (14 videos total)
      annotations/
        annotations.json  # Single file with all annotations
    public_test/
      samples/
        BlackBox_0/
          drone_video.mp4
          object_images/
            img_1.jpg, img_2.jpg, img_3.jpg
        ... (6 test videos, no annotations)
    splits/
      train.txt  # 11 videos
      val.txt    # 3 videos
  ```

### Inference Pipeline (`src/infer.py`)

1. Load trained model checkpoint
2. For each video folder:
   - Load 3 template images and encode via `model.set_template()`
   - Iterate frames, extract features, predict bbox + confidence
   - Collect raw detections: `{frame, x1, y1, x2, y2, score}`
3. Post-process via `postprocess_track()` (`src/utils/tracking.py`):
   - Filter by confidence threshold
   - Group consecutive frames into intervals (merge if gap ≤ max_gap frames)
   - Drop intervals shorter than min_interval_len
   - Output as JSON: `{"detections": [{"bboxes": [...]}]}`

### Training Pipeline (`src/train.py`)

- Constructs `AeroEyesDataset` from train split
- Optimizer: AdamW with configurable lr/weight_decay
- Loss functions (TODO in code): classification (BCE/focal) + bbox regression (L1/GIoU)
- Checkpoints saved to `checkpoints/`

### Configuration (`configs/baseline.yaml`)

All hyperparameters in YAML:
- Data: paths, frame_size, batch_size, splits
- Model: backbone type, feature_dim, head_type, template_size
- Training: epochs, lr, weight_decay, intervals
- Inference: confidence_threshold, max_gap, min_interval_len

## Key Implementation Details

### Template Encoding (Multiple Reference Images)

The 3 reference images are averaged into a single template embedding. Alternative approaches in extensions: attention-based weighting across the 3 images.

### Temporal Post-Processing Strategy

Critical for ST-IoU performance:
- **Confidence filtering**: Only keep frames above threshold
- **Gap merging**: Small gaps (1-3 frames) between detections can be interpolated to maintain continuity
- **Smoothing**: Use moving average or Kalman filter on bbox coordinates to reduce jitter
- **Interval length**: Discard very short intervals (likely false positives)

### Output Format

JSON structure per video:
```json
{
  "video_id": {
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
          {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
        ]
      }
    ]
  }
}
```

Multiple detection intervals per video are supported (object may appear/disappear/reappear).

## Development Status

This is an early-stage implementation with TODOs remaining:

- **Dataset construction**: `_build_samples()` in `aeroeyes_dataset.py` needs annotation parsing
- **Training loop**: Loss functions and backward pass in `train.py`
- **Model components**: `backbone.py` and `head.py` are referenced but not implemented
- **Transforms**: Data augmentation pipeline (`src/data/transforms.py`)
- **Inference preprocessing**: Template and frame preprocessing in `infer.py`
- **Metrics**: ST-IoU evaluation implementation

## Deployment (Jetson/Docker)

For final on-drone deployment:
- Model must run real-time on NVIDIA Jetson
- Use smaller backbone (MobileNetV3) for speed
- Consider mixed precision (FP16) and lower input resolution
- Docker entrypoint: `python -m src.main infer` with organizer-provided args
- ONNX export planned in `src/export_onnx.py` for TensorRT optimization

## Project Context Documents

- **PLAN.md**: Detailed technical strategy for few-shot video object tracking
- **SOLUTION.md**: Step-by-step implementation phases and testing strategy
- **README.md**: Repository structure and quickstart commands
- **PRD**: Original problem requirements from hackathon organizers
