import os
import sys
import numpy as np
import json
import shutil
import datetime
import subprocess
import socket 
import getpass
from pathlib import Path
from typing import Dict, Any

from config import VERSION_CODENAME

from time import perf_counter  # Runtime measurement
from contextlib import contextmanager

import warnings           # Suppress non-critical runtime warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

#For verbose functionality
def suppress_prints():
    """Redirects stdout to null to suppress print statements."""
    sys.stdout = open(os.devnull, 'w')

def restore_prints():
    """Restores normal stdout printing."""
    sys.stdout = sys.__stdout__

#The following method was written by ChatGPT 4o
#This helps convert python dictionaries to json-compatible objects
def convert_json_compat(obj):
    """
    Recursively converts numpy datatypes to native Python types
    so that they can be safely serialized to JSON.
    """
    if isinstance(obj, dict):
        return {k: convert_json_compat(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_compat(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_json_compat(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, set):
        return [convert_json_compat(v) for v in sorted(obj)]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

@contextmanager
def timer(verbose=True):
    start = perf_counter()
    try:
        yield
    finally:
        if verbose:
            print(f"Time: {perf_counter() - start:2f}s")

class RunManager:
    def __init__(self, root="runs", version="0.0", codename=None, run_name=None, mode=""):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        git = self._git_short()
        codename = f"{VERSION_CODENAME}_" if VERSION_CODENAME is not "" or None else codename
        rid = run_name or f"{codename}{mode}_{ts}_v{version}{('_'+git) if git else ''}"
        self.root = Path(root)
        self.run_dir = self.root / rid
        # create structure
        for sub in ["config", "checkpoints", "metrics", "figures", "coco", "logs"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=False)
        # latest symlink
        try:
            latest = self.root / "latest"
            if latest.exists() or latest.is_symlink(): latest.unlink()
            latest.symlink_to(self.run_dir.name)
        except Exception:
            pass

    # dirs
    @property
    def ckpt_dir(self): return str(self.run_dir / "checkpoints")
    @property
    def metrics_dir(self): return str(self.run_dir / "metrics")
    @property
    def figures_dir(self): return str(self.run_dir / "figures")
    @property
    def coco_dir(self): return str(self.run_dir / "coco")
    @property
    def config_dir(self): return str(self.run_dir / "config")
    @property
    def logs_dir(self): return str(self.run_dir / "logs")
    def path(self, *parts): return str(self.run_dir.joinpath(*parts))

    def snapshot_config(self, config_module, extras: Dict[str, Any] = None):
        # Save a verbatim copy of config.py and a resolved JSON of the values actually used
        cfg_py_src = Path(config_module.__file__)
        shutil.copy2(cfg_py_src, self.path("config", "config_snapshot.py"))
        values = {k: getattr(config_module, k) for k in dir(config_module) if k.isupper()}
        if extras: values.update(extras)
        with open(self.path("config", "config_values.json"), "w") as f:
            json.dump(values, f, indent=2, default=str)

    def save_json(self, relpath, obj):
        with open(self.path(relpath), "w") as f:
            json.dump(obj, f, indent=2)

    def copy_in(self, src: str | Path, rel_dest: str) -> Path:
        dst = self.run_dir / rel_dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def append_manifest(self, **kv):
        mpath = Path(self.path("manifest.json"))
        data = {}
        if mpath.exists():
            try: data = json.load(open(mpath))
            except Exception: data = {}
        data.update(kv)
        with open(mpath, "w") as f:
            json.dump(convert_json_compat(data), f, indent=2)

    def _git_short(self):
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            return ""

    def env_fingerprint(self):
        import torch, torchvision
        return {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": getattr(__import__("torch"), "__version__", "unknown"),
            "torchvision": getattr(__import__("torchvision"), "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }