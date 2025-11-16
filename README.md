# AeroEyes – Few-Shot Drone Object Localizer

A Siamese few-shot video tracker for the AeroEyes hackathon. This system localizes target objects in drone videos using only 3 reference images, leveraging deep learning and spatio-temporal tracking.

## 🎯 Problem Overview

Given:
- **3 RGB reference images** of a target object (backpack, person, phone, etc.)
- **1 drone video** (3-5 minutes, 25 FPS)

Output:
- **Temporal intervals** where the object appears
- **Frame-by-frame bounding boxes** for each interval
- **JSON predictions** in the hackathon submission format

**Evaluation Metric:** Spatio-Temporal IoU (ST-IoU) - measures 3D volume overlap (x, y, time)

## ✨ Features

### Implemented (Phases 0-3)
- ✅ **Siamese Tracker Architecture** - ResNet18/MobileNetV3 backbone + detection head
- ✅ **Complete Training Pipeline** - GIoU loss, data augmentation, validation
- ✅ **Inference Pipeline** - Template encoding, frame-by-frame detection, temporal post-processing
- ✅ **ST-IoU Evaluation** - Official metric implementation with detailed analysis
- ✅ **Visualization Tools** - GT/prediction overlays, side-by-side comparisons
- ✅ **Dataset Utilities** - Statistics, analysis, automated split creation
- ✅ **14 Training Videos** - 11 train / 3 val split, multiple object types
- ✅ **6 Test Videos** - Public test set for final evaluation

### Architecture Highlights
- **Template Encoding**: Averages 3 reference images into single embedding
- **Feature Matching**: Cross-correlation between template and search features
- **Detection Head**: Predicts objectness heatmap + bounding box at each location
- **Temporal Tracking**: Confidence filtering, interval grouping, smoothing

## 🚀 Quick Start

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd hkt-aero-eyes

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Training

```bash
# Train the model (uses configs/baseline.yaml)
python -m src.main train --config configs/baseline.yaml

# Outputs:
# - checkpoints/best_model.ckpt (best validation model)
# - checkpoints/epoch_N.ckpt (per-epoch checkpoints)
# - Training logs with loss metrics
```

**Training Configuration** (`configs/baseline.yaml`):
- Backbone: ResNet18 (pretrained on ImageNet)
- Batch size: 8
- Learning rate: 0.0001
- Epochs: 20
- Data: 16,779 training samples, 3,319 validation samples

### Inference

```bash
# Run inference on public test data
python -m src.main infer \
  --config configs/jetson_infer.yaml \
  --data_dir data/public_test/samples \
  --output outputs/predictions/predictions.json \
  --checkpoint checkpoints/best_model.ckpt

# Run inference on training data (for debugging)
python -m src.main infer \
  --config configs/baseline.yaml \
  --data_dir data/train/samples \
  --output outputs/predictions/train_predictions.json \
  --checkpoint checkpoints/best_model.ckpt
```

### Evaluation

```bash
# Evaluate predictions against ground truth
python -m src.main eval \
  --predictions outputs/predictions/predictions.json \
  --annotations data/train/annotations/annotations.json \
  --verbose \
  --output outputs/evaluation_results.json

# Output: Mean/median/min/max ST-IoU, per-video breakdown
```

### Visualization

```bash
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

# Side-by-side comparison (GT vs Pred)
python scripts/visualize_annotations.py \
  --video data/train/samples/Backpack_0/drone_video.mp4 \
  --annotations data/train/annotations/annotations.json \
  --predictions outputs/predictions/predictions.json \
  --video_id Backpack_0 \
  --output outputs/debug_viz/Backpack_0_comparison.mp4 \
  --compare
```

## 📁 Repository Structure

```
hkt-aero-eyes/
├── configs/
│   ├── baseline.yaml          # Training/inference config for local dev
│   └── jetson_infer.yaml      # Inference config for public_test/Jetson
│
├── data/
│   ├── train/
│   │   ├── samples/           # 14 training videos
│   │   │   ├── Backpack_0/
│   │   │   │   ├── drone_video.mp4
│   │   │   │   └── object_images/
│   │   │   │       ├── img_1.jpg
│   │   │   │       ├── img_2.jpg
│   │   │   │       └── img_3.jpg
│   │   │   └── ...
│   │   └── annotations/
│   │       └── annotations.json  # All training annotations
│   │
│   ├── public_test/
│   │   └── samples/           # 6 test videos (no annotations)
│   │
│   └── splits/
│       ├── train.txt          # 11 training video IDs
│       └── val.txt            # 3 validation video IDs
│
├── src/
│   ├── main.py                # CLI entry point (train/infer/eval)
│   ├── config.py              # YAML config loader
│   ├── train.py               # Training loop
│   ├── infer.py               # Inference pipeline
│   │
│   ├── data/
│   │   ├── aeroeyes_dataset.py  # Dataset loader
│   │   └── transforms.py        # Data augmentation
│   │
│   ├── models/
│   │   ├── backbone.py        # ResNet/MobileNet feature extraction
│   │   ├── head.py            # Siamese detection head
│   │   └── siam_tracker.py    # Full model wrapper
│   │
│   └── utils/
│       ├── logging_utils.py   # Loguru logger
│       ├── video_io.py        # Video frame iteration
│       ├── tracking.py        # Temporal post-processing
│       └── metrics.py         # ST-IoU calculation
│
├── scripts/
│   ├── eval_st_iou.py         # Evaluation script
│   ├── visualize_annotations.py  # Visualization tool
│   ├── prepare_dataset.py     # Dataset analysis utilities
│   └── extract_frames.py      # Frame extraction tool
│
├── checkpoints/               # Saved model checkpoints
├── outputs/
│   ├── predictions/           # Inference outputs
│   ├── logs/                  # Training logs
│   └── debug_viz/             # Visualization videos
│
├── requirements.txt           # Python dependencies
├── QUICKSTART.md             # Detailed usage guide
├── CLAUDE.md                 # Claude Code guidance
└── README.md                 # This file
```

## 🔧 Dataset Structure

### Training Data (14 videos)
- **Object types**: Backpack, Jacket, Laptop, Lifering, MobilePhone, Person, WaterBottle
- **Annotations**: Single `annotations.json` file with all ground truth
- **Format**: List of videos, each with multiple temporal intervals
- **Each interval**: Frame-by-frame bounding boxes

### Test Data (6 videos)
- **No annotations** provided (for final evaluation)
- **Object types**: BlackBox, CardboardBox, LifeJacket, etc.

## 📊 Model Architecture

```
Template Images (3x)         Search Frame
      ↓                            ↓
   Backbone (ResNet18)         Backbone (shared)
      ↓                            ↓
Template Features           Search Features
      ↓                            ↓
      └─────── Cross-correlation ──┘
                     ↓
           Siamese Detection Head
                     ↓
      ┌──────────────┴──────────────┐
      ↓                              ↓
Objectness Heatmap          Bounding Box Predictions
  (1, H, W)                      (4, H, W)
```

### Training
- **Loss**: GIoU (bbox regression) + BCE (classification with Gaussian heatmap)
- **Optimizer**: AdamW with learning rate scheduling
- **Augmentation**: Color jitter, horizontal flip, Gaussian blur, ImageNet normalization

### Inference
1. Load 3 template images → encode to single embedding
2. For each video frame:
   - Extract features
   - Cross-correlate with template
   - Find max confidence location
   - Extract bbox prediction
3. Filter by confidence threshold
4. Group into temporal intervals
5. Output JSON

## 📈 Evaluation Metrics

### Spatio-Temporal IoU (ST-IoU)
Measures 3D volume overlap between predicted and ground truth detections:
- **Spatial**: 2D bounding box IoU per frame
- **Temporal**: Frame-level matching across intervals
- **Final score**: Average ST-IoU across all GT intervals

### Example Results
```
===========================================================
SPATIO-TEMPORAL IOU EVALUATION RESULTS
===========================================================

Overall Metrics:
  Mean ST-IoU:     0.6543
  Median ST-IoU:   0.6821
  Min ST-IoU:      0.2134
  Max ST-IoU:      0.9456
  Number of videos: 14

Per-Video Results:
Video ID             ST-IoU   GT Intervals   Pred Intervals
-----------------------------------------------------------
Backpack_0            0.7234             10               12
Jacket_1              0.6892              8                9
...
```

## 🛠️ Dataset Utilities

### Statistics
```bash
python scripts/prepare_dataset.py \
  --annotations data/train/annotations/annotations.json \
  --stats
```

### Detailed Analysis
```bash
python scripts/prepare_dataset.py \
  --annotations data/train/annotations/annotations.json \
  --analyze
```

### Create Custom Splits
```bash
python scripts/prepare_dataset.py \
  --annotations data/train/annotations/annotations.json \
  --create_splits \
  --train_ratio 0.8 \
  --output_dir data/splits
```

## 🐛 Debugging

### Common Issues

**Out of Memory:**
- Reduce `batch_size` in configs/baseline.yaml
- Reduce `frame_size` (e.g., from [640, 360] to [320, 180])
- Use smaller backbone: `mobilenet_v3_small`

**Training Too Slow:**
- Reduce `num_workers` if CPU bottleneck
- Use GPU if available
- Reduce video resolution

**Poor Performance:**
- Train for more epochs
- Visualize predictions to understand failure modes
- Check data quality and annotations
- Adjust confidence threshold in config

### Visualization for Debugging
Always visualize predictions vs ground truth to understand model behavior:
```bash
python scripts/visualize_annotations.py \
  --video <path> \
  --annotations <path> \
  --predictions <path> \
  --video_id <id> \
  --compare
```

## 📝 Output Format

Predictions JSON structure:
```json
{
  "Backpack_0": {
    "detections": [
      {
        "bboxes": [
          {"frame": 3483, "x1": 321, "y1": 0, "x2": 381, "y2": 12},
          {"frame": 3484, "x1": 302, "y1": 0, "x2": 387, "y2": 21},
          ...
        ]
      },
      {
        "bboxes": [...]  // Second interval
      }
    ]
  },
  "Jacket_1": {
    "detections": [...]
  }
}
```

## 🚢 Deployment (Jetson/Docker)

For on-device deployment:
- Use `configs/jetson_infer.yaml`
- Smaller backbone: MobileNetV3
- Lower resolution for speed
- FP16 mixed precision (optional)

Docker inference:
```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "src.main", "infer"]
```

## 📚 Documentation

- **QUICKSTART.md**: Detailed usage examples and testing guide
- **CLAUDE.md**: Guidance for Claude Code when working with this repository
- **PLAN.md**: Technical strategy and model design
- **SOLUTION.md**: Implementation phases and architecture details

## 🎓 References

- **SiamFC**: Fully-Convolutional Siamese Networks for Object Tracking
- **SiamRPN**: High Performance Visual Tracking with Siamese Region Proposal Network
- **GIoU Loss**: Generalized Intersection over Union

## 📄 License

[Add your license here]

## 🤝 Contributing

This is a hackathon project for AeroEyes competition.

## 📧 Contact

[Add your contact information here]
