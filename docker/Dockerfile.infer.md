FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.main", "infer"]
# Then organizer will add args like:
#   --config configs/jetson_infer.yaml --data_dir /data --output /output/predictions.json --checkpoint /model.ckpt
