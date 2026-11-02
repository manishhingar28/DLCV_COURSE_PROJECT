import json
import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class Logger:
    """Prints and appends to a log file (logs are part of the required deliverables)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        try:
            print(s, flush=True)
        except UnicodeEncodeError:
            print(s.encode("ascii", "replace").decode(), flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def count_params(model):
    return sum(p.numel() for p in model.parameters())
