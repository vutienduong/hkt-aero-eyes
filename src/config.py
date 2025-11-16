import yaml
from pathlib import Path

def load_config(path):
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg
