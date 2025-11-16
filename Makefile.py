PYTHON = python

create-env:
	python -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.main train --config configs/baseline.yaml

infer:
	$(PYTHON) -m src.main infer \
		--config configs/jetson_infer.yaml \
		--data_dir data/samples \
		--output outputs/predictions/predictions.json \
		--checkpoint checkpoints/baseline.ckpt
