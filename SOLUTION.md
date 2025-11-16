
# AeroEyes – Few‑Shot Drone Object Localizer

This document describes the planned solution for the AeroEyes hackathon: the model approach, implementation phases, potential extensions, and how to run and test the repository locally.

---

## 1. Problem Overview

Given:
- A long drone video (`drone_video.mp4`) ~3–5 minutes.
- Three reference RGB images of the target object (`object_images/`).
- Ground truth annotations (for training/validation) describing when and where the target object appears.

Goal:
- For each video, predict one or more time intervals where the object is visible.
- For each interval, output frame‑by‑frame bounding boxes of the target object.
- Format: JSON with a list of detections per video.
- Metric: Spatio‑Temporal IoU (ST‑IoU) between predicted 3D volume (x, y, t) and ground truth.

---

## 2. High‑Level Solution

We treat this as a **few‑shot video object tracking/localization** problem.

Core ideas:

1. Encode the **target object** using the 3 reference images.
2. For each video frame:
   - Extract a **feature map** using a CNN backbone.
   - Compare (match) the template embedding with the frame features.
   - Predict:
     - A confidence score (object present / not present).
     - A bounding box (x1, y1, x2, y2) for the object.
3. Over time:
   - Filter low‑confidence detections.
   - Group consecutive detections into continuous intervals.
   - Smooth bounding boxes to reduce jitter.
   - Optionally interpolate short gaps to improve temporal consistency.

Model family: **Siamese tracker** (SiamFC / SiamRPN‑style).

---

## 3. Implementation Plan – Step by Step

### Phase 0 – Repo Skeleton

- Basic folder structure under `src/`, `configs/`, `data/`, `scripts/`.
- Minimal `main.py` CLI placeholder.
- `requirements.txt` with core dependencies.

This phase is meant to get a clean starting point that is easy to extend.

---

### Phase 1 – Minimal End‑to‑End Baseline (No Training)

Objective: have something that runs end‑to‑end on sample data and produces a valid `predictions.json`.

Tasks:

1. **Template encoding (naive):**
   - Load the 3 reference images.
   - Resize to a fixed size (e.g., 128×128).
   - Extract features using an ImageNet‑pretrained backbone (e.g., ResNet18 or MobileNet).
   - Average‑pool spatial dimensions + average over the 3 images → a single template embedding vector.

2. **Frame processing:**
   - For each frame in the video:
     - Extract a feature map from the backbone.
     - Compute cosine similarity between template embedding and each spatial location.
     - Find the max similarity location and construct a **fixed‑size** bbox around it (e.g., fixed width/height proportional to frame size).

3. **Post‑processing:**
   - Filter out frames with similarity below a threshold.
   - Group consecutive frames into intervals and discard very short intervals.
   - Return `{"detections": [{"bboxes": [ ... ]}]}` for each video.

4. **Output:**
   - Implement `src/infer.py` to run this pipeline and dump a valid JSON to `outputs/predictions/predictions.json`.

Result:
- Very naive, but satisfies the hackathon I/O format.
- Useful to verify Docker, data layout, and evaluation tools.

---

### Phase 2 – Proper Siamese Tracker with Training

Objective: learn a better model from the challenge training set.

Tasks:

1. **Dataset builder (`AeroEyesDataset`):**
   - Parse annotation files (e.g., JSON).
   - For each annotated interval:
     - Select a **template frame** and crop the ground truth bbox → template image.
     - Sample multiple **search frames** around it (same interval, maybe ±k frames).
     - For each pair, store: (template_image, search_image, gt_bbox_in_search).

2. **Transforms:**
   - Implement basic augmentations in `src/data/transforms.py`:
     - Random scale/crop around object.
     - Horizontal flip (if valid).
     - Color jitter, blur, noise.
   - Normalize images with ImageNet stats.

3. **Model implementation:**
   - `src/models/backbone.py`:
     - Build a small ResNet/MobileNet backbone that outputs a feature map.
   - `src/models/head.py`:
     - Implement a Siamese head:
       - Cross‑correlation or similarity between template feature map and search feature map.
       - Conv layers to predict objectness and bbox offsets per spatial location.
   - `src/models/siam_tracker.py`:
     - Wrap backbone + head into a single module.
     - During training: use both template and search images.
     - During inference: pre‑encode template and reuse it.

4. **Training loop (`src/train.py`):**
   - Use `DataLoader` to sample batches of (template, search, gt_bbox).
   - Compute:
     - Classification loss (object vs background).
     - Bounding box regression loss (e.g., L1 or GIoU).
   - Support:
     - Checkpoint saving.
     - Basic logging (loss per step/epoch).
     - Optional validation loop.

5. **Configuration:**
   - Update `configs/baseline.yaml` with:
     - Train/val split paths.
     - Model hyperparameters (feature_dim, etc.).
     - Training hyperparameters (lr, batch size, epochs).

Result:
- You now have a trained model that should outperform the naive baseline.

---

### Phase 3 – Better Temporal Post‑Processing & Metric Alignment

Objective: improve ST‑IoU by making predictions more continuous and stable.

Tasks:

1. **Smoothing:**
   - Implement simple moving average or Kalman filter over box coordinates in `src/utils/tracking.py`.

2. **Gap handling:**
   - Identify small gaps between detections (e.g., missing 1–3 frames).
   - Interpolate bounding boxes to fill these gaps.
   - Keep track of interval lengths and discard very short ones.

3. **Threshold tuning:**
   - Use a validation split to tune:
     - Confidence threshold.
     - Max allowed gap.
     - Minimum interval length.

4. **Evaluation script:**
   - Implement `scripts/eval_st_iou.py` using `src/utils/metrics.py`.
   - Given predictions and ground truth, compute approximate ST‑IoU for validation.

Result:
- Higher leaderboard score.
- More robust predictions.

---

### Phase 4 – Jetson / On‑Device Optimization (Optional for Later)

Objective: make the model run in real‑time on an NVIDIA Jetson board.

Tasks:

1. **Model export:**
   - Add `src/export_onnx.py` to export the trained model to ONNX.
   - Use TensorRT or ONNX Runtime on Jetson for faster inference.

2. **Lightweight backbone:**
   - Replace ResNet18 with MobileNetV3 or EfficientNet‑Lite.
   - Reduce input resolution while keeping accuracy acceptable.

3. **Runtime script on Jetson (`jetson/run_infer.py`):**
   - Use GStreamer/OpenCV to capture frames from drone or video file.
   - Run model inference on each frame.
   - Visualize detections or send them to the flight controller.

---

## 4. Potential Extensions

Ideas for improving the system beyond the initial baseline:

1. **Multi‑scale detection:**
   - Allow different anchor scales or FPN‑style features to handle varying object size.

2. **Attention over the 3 reference images:**
   - Instead of naive averaging, use attention to focus on the most informative reference image or region.

3. **Background suppression:**
   - Learn a background prototype for each video to reduce false positives.

4. **Temporal modeling:**
   - Add a lightweight temporal model (e.g., 1D conv or GRU over frame‑level features) to better capture motion.

5. **Hard negative mining:**
   - During training, sample difficult frames with similar objects/background to improve robustness.

6. **Semi‑supervised fine‑tuning:**
   - Use confident predictions on unlabeled videos as pseudo‑labels and fine‑tune the model further.

---

## 5. How to Run

### Configuration Files

The project uses two main configuration files:

- **`configs/baseline.yaml`**: For local development and training
- **`configs/jetson_infer.yaml`**: For hackathon submission, Docker deployment, and Jetson inference

Use `baseline.yaml` when developing locally, and `jetson_infer.yaml` for the final submission/deployment pipeline.

### 5.1. Setup

```bash
# 1. Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place the official dataset under `data/`:

```text
data/
  train/
    samples/
      Backpack_0/
        drone_video.mp4
        object_images/
          img_1.jpg
          img_2.jpg
          img_3.jpg
      ...
    annotations/
      annotations.json  # Single file with all training annotations
  public_test/
    samples/
      BlackBox_0/
        drone_video.mp4
        object_images/
          img_1.jpg
          img_2.jpg
          img_3.jpg
      ...
  splits/
    train.txt  # 11 training videos
    val.txt    # 3 validation videos
```

### 5.2. Training (planned)

Once `train.py` and `AeroEyesDataset` are implemented:

```bash
python -m src.main train --config configs/baseline.yaml
```

Checkpoints will be saved to `checkpoints/`.

### 5.3. Inference (planned)

After training a model (or using a baseline checkpoint):

**For local testing (on training data):**
```bash
python -m src.main infer \
  --config configs/baseline.yaml \
  --data_dir data/train/samples \
  --output outputs/predictions/predictions.json \
  --checkpoint checkpoints/baseline.ckpt
```

**For hackathon submission / Docker (on public_test):**
```bash
python -m src.main infer \
  --config configs/jetson_infer.yaml \
  --data_dir data/public_test/samples \
  --output outputs/predictions/predictions.json \
  --checkpoint checkpoints/baseline.ckpt
```

The `predictions.json` should match the hackathon submission format.

---

## 6. How to Test

We plan three levels of testing:

1. **Unit tests (lightweight, optional later):**
   - Test data loading and basic dataset operations.
   - Test that the model forward pass works for random tensors.
   - Test that post‑processing converts raw detections into valid JSON.

2. **Integration tests:**
   - Use a tiny toy dataset (e.g., 1–2 short videos).
   - Run:
     - `train` for a few iterations.
     - `infer` to generate predictions.
   - Confirm:
     - No crashes.
     - JSON is valid and contains expected keys.

3. **Evaluation sanity checks:**
   - Manually craft a simple example where ground truth and predictions are identical to ensure ST‑IoU ≈ 1.0.
   - Manually shift boxes or timestamps and confirm that ST‑IoU decreases as expected.

---

## 7. Next Steps

1. Implement:
   - `AeroEyesDataset`
   - Backbone + Siamese head
   - Minimal training loop
   - Baseline inference pipeline

2. Add:
   - Post‑processing (tracking + smoothing)
   - Evaluation script and metric helpers.

3. Optimize:
   - Thresholds and hyperparameters on validation split.
   - Model architecture for speed/accuracy trade‑off.

This document can evolve as the project matures; treat it as the living design doc for the solution.
