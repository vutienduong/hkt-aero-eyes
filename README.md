1. Repository structure
aeroeyes/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ setup.cfg
├─ pyproject.toml          # optional, if you like modern packaging
├─ Makefile
├─ configs/
│  ├─ baseline.yaml
│  ├─ model_small.yaml
│  └─ jetson_infer.yaml
├─ data/
│  ├─ train/
│  │  ├─ samples/          # training videos (14 videos)
│  │  └─ annotations/
│  │     └─ annotations.json  # single file with all annotations
│  ├─ public_test/
│  │  └─ samples/          # test videos (6 videos, no annotations)
│  └─ splits/
│     ├─ train.txt         # training video IDs (11 videos)
│     └─ val.txt           # validation video IDs (3 videos)
├─ checkpoints/
│  └─ .gitkeep
├─ outputs/
│  ├─ logs/
│  ├─ predictions/
│  └─ debug_viz/
├─ scripts/
│  ├─ extract_frames.py
│  ├─ prepare_dataset.py
│  ├─ visualize_annotations.py
│  └─ eval_st_iou.py
├─ src/
│  ├─ __init__.py
│  ├─ main.py              # CLI dispatcher (train / infer / eval)
│  ├─ config.py            # load/parse YAML configs
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ logging_utils.py
│  │  ├─ video_io.py
│  │  ├─ json_io.py
│  │  ├─ metrics.py        # ST-IoU etc
│  │  └─ tracking.py       # smoothing, interval grouping
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ aeroeyes_dataset.py
│  │  └─ transforms.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ backbone.py       # MobileNet/ResNet
│  │  ├─ head.py           # bbox + classification head
│  │  └─ siam_tracker.py   # full model wrapper
│  ├─ train.py             # training loop
│  ├─ infer.py             # offline inference for competition
│  └─ export_onnx.py       # for Jetson / TensorRT
├─ docker/
│  ├─ Dockerfile.train
│  └─ Dockerfile.infer
└─ jetson/
   ├─ README.md
   ├─ run_infer.py         # lightweight Jetson runtime script
   └─ install_requirements.sh

2. Core files – minimal contents

Just enough so reviewers & organizers know how to run:

# AeroEyes – Few-Shot Drone Object Localizer

This repo implements a Siamese few-shot video tracker for the AeroEyes hackathon.

## Quickstart

```bash
# 1. Create and activate venv
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2. Install deps
pip install --upgrade pip
pip install -r requirements.txt

Training
python -m src.main train --config configs/baseline.yaml

Inference (competition format on public_test)
python -m src.main infer \
  --config configs/jetson_infer.yaml \
  --data_dir data/public_test/samples \
  --output outputs/predictions/predictions.json \
  --checkpoint checkpoints/baseline.ckpt


---

### 2.2. `requirements.txt`

Minimal starting point:

```txt
torch
torchvision
torchaudio
opencv-python
PyYAML
tqdm
numpy
scipy
matplotlib
loguru