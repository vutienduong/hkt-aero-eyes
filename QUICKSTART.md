# AeroEyes Quick Start Guide

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Structure

Ensure your data is organized as:
```
data/
├── train/
│   ├── samples/          # 14 training videos
│   └── annotations/
│       └── annotations.json
├── public_test/
│   └── samples/          # 6 test videos
└── splits/
    ├── train.txt         # 11 videos
    └── val.txt           # 3 videos
```

## Training

```bash
# Train the model
python -m src.main train --config configs/baseline.yaml

# Training will:
# - Load 11 training videos and 3 validation videos
# - Train for 20 epochs (default)
# - Save checkpoints to checkpoints/
# - Save best model as checkpoints/best_model.ckpt
```

### Monitor Training

Training logs will show:
- Loss breakdown (classification + bbox)
- Training and validation metrics
- Progress bars with live updates

### Adjust Hyperparameters

Edit `configs/baseline.yaml`:
```yaml
train:
  epochs: 20           # Number of epochs
  lr: 1e-4             # Learning rate
  batch_size: 8        # Batch size
  log_interval: 50     # Log every N steps
  val_interval: 1      # Validate every N epochs
```

## Inference

### On Training Data (for debugging)

```bash
python -m src.main infer \
  --config configs/baseline.yaml \
  --data_dir data/train/samples \
  --output outputs/predictions/train_predictions.json \
  --checkpoint checkpoints/best_model.ckpt
```

### On Public Test Data (for submission)

```bash
python -m src.main infer \
  --config configs/jetson_infer.yaml \
  --data_dir data/public_test/samples \
  --output outputs/predictions/predictions.json \
  --checkpoint checkpoints/best_model.ckpt
```

## Quick Tests

### Test Dataset Loading

```python
from src.data.aeroeyes_dataset import AeroEyesDataset
from src.data.transforms import AeroEyesTransform

# Create transform
transform = AeroEyesTransform(
    template_size=(128, 128),
    search_size=(640, 360),
)

# Load dataset
ds = AeroEyesDataset(
    root='data/train',
    annotations_file='annotations/annotations.json',
    split_file='data/splits/train.txt',
    transforms=transform
)

print(f'Dataset size: {len(ds)}')

# Test loading a sample
template, search, bbox = ds[0]
print(f'Template: {template.shape}')
print(f'Search: {search.shape}')
print(f'BBox: {bbox}')
```

### Test Model Forward Pass

```python
import torch
from src.models.siam_tracker import SiamTracker

# Create model
model = SiamTracker(backbone_name='resnet18', feature_dim=256)

# Test forward pass
template = torch.randn(2, 3, 128, 128)
search = torch.randn(2, 3, 640, 360)

cls_logits, bbox_pred = model(search, template_img=template)
print(f'Classification: {cls_logits.shape}')  # (2, 1, H, W)
print(f'BBox: {bbox_pred.shape}')             # (2, 4, H, W)
```

## Common Issues

### Out of Memory
- Reduce `batch_size` in config
- Reduce `frame_size` in config
- Use smaller backbone: `mobilenet_v3_small`

### Training Too Slow
- Reduce `num_workers` if CPU is bottleneck
- Use GPU if available
- Reduce video frame size

### Poor Performance
- Train for more epochs
- Adjust learning rate
- Check data quality and annotations
- Visualize predictions to debug

## Output Format

Predictions JSON format:
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

## Evaluation

### Compute ST-IoU Metrics

```bash
# Using main CLI
python -m src.main eval \
  --predictions outputs/predictions/predictions.json \
  --annotations data/train/annotations/annotations.json \
  --verbose \
  --output outputs/evaluation_results.json

# Or use dedicated script
python scripts/eval_st_iou.py \
  --predictions outputs/predictions/predictions.json \
  --annotations data/train/annotations/annotations.json \
  --verbose
```

## Visualization

### Visualize Ground Truth Annotations

```bash
python scripts/visualize_annotations.py \
  --video data/train/samples/Backpack_0/drone_video.mp4 \
  --annotations data/train/annotations/annotations.json \
  --video_id Backpack_0 \
  --output outputs/debug_viz/Backpack_0_gt.mp4
```

### Visualize Predictions

```bash
python scripts/visualize_annotations.py \
  --video data/train/samples/Backpack_0/drone_video.mp4 \
  --predictions outputs/predictions/predictions.json \
  --video_id Backpack_0 \
  --output outputs/debug_viz/Backpack_0_pred.mp4
```

### Side-by-Side Comparison

```bash
python scripts/visualize_annotations.py \
  --video data/train/samples/Backpack_0/drone_video.mp4 \
  --annotations data/train/annotations/annotations.json \
  --predictions outputs/predictions/predictions.json \
  --video_id Backpack_0 \
  --output outputs/debug_viz/Backpack_0_comparison.mp4 \
  --compare
```

## Dataset Analysis

### Print Statistics

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

## Next Steps

1. **Optimization**: Tune hyperparameters on validation set
2. **Jetson Deployment**: Optimize for on-device inference
3. **Advanced Features**: Add temporal smoothing, multi-scale detection
