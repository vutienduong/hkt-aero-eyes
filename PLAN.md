> **Note**: This is an internal design plan and technical strategy document. For the exact folder structure and commands to run the project, see README.md and SOLUTION.md.

---

## 1. Understand the task and metric

You’re given:

* 3 RGB images of the target object (backpack, person, bike, etc.)
* 1 long drone video (3–5 mins @ 25 fps)
* Annotations: for some frames, bounding boxes where the object appears, possibly multiple visible intervals per video. 

You must output, for each video, **one or more intervals**:

```json
{
  "video_id": "drone_video_001",
  "detections": [
    {
      "bboxes": [
        {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
        {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
      ]
    }
  ]
}
```

The score is **Spatio-Temporal IoU (ST-IoU)**: overlap between your predicted 3D volume (x, y, t) and the ground-truth volume. If you’re off in time *or* bbox, you lose points. 

So the model must:

* Detect **when** the object appears (temporal localization)
* Predict **where** it is in each frame (spatial bbox)
* Keep detections **continuous** (avoid gaps and jitter)

---

## 2. Overall solution architecture

The problem is **few-shot video object localization / tracking**. A practical high-level pipeline:

1. **Extract frames** from each video (maybe every 1–2 frames to save compute).
2. **Encode the target object** (from the 3 reference images) into a template embedding.
3. For each frame:

   * Compute a **feature map** with a CNN backbone.
   * Do **template vs. frame matching** to get a similarity heatmap.
   * Use a small head to regress a bounding box around the peak response.
4. **Temporal tracking + smoothing**:

   * Filter out low-confidence frames.
   * Link consecutive frames into “visible intervals”.
   * Smooth bboxes over time (Kalman filter / simple averaging).
5. Convert each interval into the **JSON “detections” format**.

This is basically a **Siamese tracker** adapted for this dataset.

---

## 3. Model design options

### 3.1. Backbone and template encoder

You want something **fast** on Jetson:

* Backbone: `MobileNetV3`, `EfficientNet-Lite`, or `ResNet18/34`.
* Pretrained on ImageNet; fine-tune on the challenge training set.

Pipeline:

```text
reference image (3x)
   ↓ backbone (shared)
template embedding (C-dim vector or small feature map)

video frame
   ↓ backbone
frame feature map (C x H x W)
```

Aggregate the 3 template embeddings (e.g. mean or attention).

### 3.2. Matching + bounding box head

Two common approaches:

1. **Siamese correlation** (like SiamFC / SiamRPN):

   * Cross-correlate template features with frame features → similarity map.
   * Small conv head predicts:

     * Objectness score per location.
     * Box offsets (dx, dy, dw, dh).

2. **Feature similarity map**:

   * Normalize features.
   * Compute cosine similarity between template embedding and each spatial location in the frame feature map.
   * Take the largest connected high-similarity region and fit a bbox.
   * Refinement head to regress more precise box.

Option 1 is more “tracker-style” and usually gives better performance if you implement it well.

### 3.3. Temporal tracking

You don’t want frame-by-frame noisy boxes. Add:

* **Score threshold**: only keep frames where the objectness score > τ.
* **Temporal grouping**:

  * Group consecutive frames with detections into one interval.
  * If there’s a gap ≤ k frames, you can interpolate boxes to keep continuity (helps ST-IoU).
* **Smoothing**:

  * For each track, smooth `(x1, y1, x2, y2)` with a moving average or a Kalman filter.

Result: cleaner 3D tube, better ST-IoU.

---

## 4. How to train on the provided data

You only get the organizers’ dataset (plus any allowed open-source data). A practical training scheme:

### 4.1. Dataset construction

From each video with annotations:

For each annotated interval:

1. Pick one or more **template frames** within the interval.
2. Crop the GT bbox from the template frame → “template image”.
3. For multiple other frames around it (before/after in time), take:

   * **Search image**: full frame or a slightly enlarged crop around the GT.
   * **Target label**: bounding box in the search image.

Thus each training sample has:

* Template image → template features.
* Search image → frame features.
* Ground-truth bbox in search image.

You’re basically reconstructing a tracking dataset from their annotations.

### 4.2. Loss functions

* **Classification loss**: object vs background at each location (BCE / focal loss).
* **Regression loss**: L1 or IoU / GIoU loss for bounding box.
* Optional **center-ness** loss if you follow FCOS-like head.

### 4.3. Data augmentation

Given drones + outdoor scenes, use:

* Random scale & aspect ratio.
* Random crop around object.
* Color jitter, brightness/contrast.
* Random horizontal flip (if consistent).
* Slight rotation; blur; noise (simulating motion blur).

Goal: robust to different scales, angles, lighting.

---

## 5. Inference pipeline (qualification round)

You need a deterministic `infer.py` that the Docker will run.

### 5.1. Preprocessing

* Decode `drone_video.mp4` with OpenCV / decord.
* Optionally resize frames to a fixed size (e.g. 640×360) for speed.
* Preprocess the 3 `object_images/` once and compute the template embedding.

### 5.2. Frame processing

For each video:

1. Loop over frames (maybe step = 1 or 2 to reduce compute).
2. For each frame:

   * Get feature map.
   * Cross-correlate with template.
   * Predict bbox + confidence.
3. Record detections for frames where confidence > τ.

### 5.3. Post-processing to match submission format

* Group detections into intervals (track segments).
* For each interval:

  * Create an object in `"detections"` with a `bboxes` list.
  * Each bbox entry: `{"frame": frame_idx, "x1": ..., "y1": ..., "x2": ..., "y2": ...}`.
* If no frames above threshold: `"detections": []`.

Finally, dump JSON for all videos in one file.

---

## 6. Optimizing for ST-IoU

Some specific tricks:

1. **Avoid fragmented intervals**:

   * If you only miss 1–2 frames in the middle, interpolate instead of breaking into two detections.
   * This improves temporal overlap with GT volume.

2. **Tune thresholds on a validation split**:

   * Confidence threshold τ.
   * Minimum interval length (ignore very short intervals).
   * Maximum allowed gap length for merging intervals.

3. **Box stability**:

   * Jittery boxes lower spatial IoU.
   * Smoothing trajectories will increase average IoU.

4. **Frame sampling rate**:

   * If you process every 2nd frame, you can interpolate boxes for skipped frames when writing JSON.

---

## 7. Jetson deployment (final round)

For the final on-drone phase, constraints:

* Model must run **real-time** on Jetson (no cloud). 
* Prefer PyTorch; you can still optimize with:

  * Mixed precision (FP16).
  * Smaller backbone (e.g., MobileNetV3 small).
  * Lower input resolution, but still enough to see the object.

Architecture for on-drone:

```text
Drone camera → Jetson:
  - Video capture (GStreamer/OpenCV)
  - Preprocess frame
  - Model inference (PyTorch)
  - Post-process: bbox, score
  - (Optional) send bbox/score to flight controller to steer drone
```

For the hackathon qualification round, you can skip the autopilot part, but for final round you’ll likely:

* Define a **search pattern** (lawnmower pattern) until score > threshold.
* Once high-confidence detection appears, hover / center object in frame.

---

## 8. Practical project structure

> **Note**: This section describes the final repository structure. See README.md for the canonical layout and commands.

```text
aeroeyes/
├─ configs/
│  ├─ baseline.yaml          # for local dev/training
│  └─ jetson_infer.yaml      # for hackathon submission/Jetson
├─ data/
│  ├─ samples/               # provided by organizers (videos + images)
│  ├─ annotations/           # train/val annotations from challenge
│  └─ splits/
│     ├─ train.txt
│     ├─ val.txt
│     └─ test.txt
├─ src/
│  ├─ main.py                # CLI dispatcher (train / infer / eval)
│  ├─ config.py              # load/parse YAML configs
│  ├─ train.py               # training loop
│  ├─ infer.py               # offline inference for competition
│  ├─ data/
│  │  └─ aeroeyes_dataset.py
│  ├─ models/
│  │  ├─ backbone.py
│  │  ├─ head.py
│  │  └─ siam_tracker.py     # full model wrapper
│  └─ utils/
│     ├─ video_io.py
│     ├─ tracking.py         # smoothing, interval grouping
│     └─ metrics.py          # ST-IoU etc
├─ docker/
│  └─ Dockerfile.infer.md
└─ README.md
```

* `src/main.py`: CLI entry point with subcommands (train, infer, eval)
* `src/train.py`: build dataset from annotations, train Siamese detector
* `src/infer.py`: load weights, process all videos in `samples/`, save `predictions.json`
* `docker/Dockerfile.infer.md`: PyTorch + CUDA image for hackathon submission

---

## 9. Step-by-step action plan

If you want something concrete to start coding:

1. **Baseline (no training yet)**

   * Use a pretrained backbone (e.g., ResNet18) to extract embeddings.
   * Implement naive template matching:

     * Extract template embedding from 3 images.
     * For each frame, compute cosine similarity heatmap and choose max location.
     * Build naive bbox around it (fixed size).
   * Generate predictions JSON and check if the pipeline runs end-to-end.

2. **Turn baseline into trainable Siamese model**

   * Implement dataset builder from annotations.
   * Add a learnable head to regress bounding boxes.
   * Train on training videos; validate on a held-out subset.

3. **Add temporal smoothing + interval grouping**

   * Simple Kalman filter or just moving average over boxes.
   * Interval merging and interpolation.

4. **Optimize + tune for leaderboard**

   * Thresholds, sampling rate, input resolution.
   * Lightweight backbone for speed.

5. **Wrap in Docker + test**

   * Ensure `docker run` can do:

     * `python -m src.main infer --config configs/jetson_infer.yaml --data_dir /data --output /output/predictions.json --checkpoint /model.ckpt`