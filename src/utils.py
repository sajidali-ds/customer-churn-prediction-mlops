"""Small shared utilities: config loading and reproducibility helpers."""
import json
import os
import random

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load the YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across numpy / python / tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
