import os
import re
import random
from typing import List

import wandb
import torch
import numpy as np

def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)

def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")

def get_latest_run(project_name, experiment_name):
    runs = get_runs(project_name)

    filtered = [
        run for run in runs
        if run.config.get("experiment_name") == experiment_name
    ]

    if not filtered:
        default_name = f"{experiment_name.replace('_', '-')}-000"
        return default_name
    
    return filtered[0].name

def make_error_run_dir(
    root: str = "error_logs",
    prefix: str = "version"
) -> str:
    os.makedirs(root, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})$")
    existing = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and pattern.match(d)
    ]

    next_idx = (
        max(int(pattern.match(d).group(1)) for d in existing) + 1
        if existing else 0
    )

    dir_name = f"{prefix}_{next_idx:03d}"
    run_dir  = os.path.join(root, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[make_error_run_dir] Created → {run_dir}")
    return run_dir