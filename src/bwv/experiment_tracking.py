"""
Lightweight experiment tracking for buying-winners-vietnam.

Features:
- start_run(config, tracking_cfg=None, base_dir="results") -> (run_id, run_dir)
- log_params_metrics(run_id, params: dict, metrics: dict, step: Optional[int]=None, base_dir="results")
- log_artifact(run_id, src_path: str, artifact_name: Optional[str]=None, base_dir="results")
- end_run(run_id, status="FINISHED", base_dir="results")
The module writes a run directory with manifest.json, config.json, environment.json and stores artifacts under artifacts/.
"""

from __future__ import annotations
import json
import os
import shutil
import sys
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4
import csv

DEFAULT_BASE_DIR = "results"


def _now_ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _short_id():
    return uuid4().hex[:8]


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _hash_of_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def start_run(config: Dict[str, Any], tracking_cfg: Optional[Dict[str, Any]] = None, base_dir: Optional[str] = None):
    """
    Create a run directory and manifest. Returns (run_id, run_dir_path).
    base_dir can be overridden for tests.
    """
    base = Path(base_dir or tracking_cfg.get("base_dir") if tracking_cfg else DEFAULT_BASE_DIR)
    ts = _now_ts()
    run_id = f"{ts}_{_short_id()}"
    run_dir = base / "runs" / run_id
    _safe_mkdir(run_dir)

    # Write config
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    # Environment
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "created_at": ts,
    }
    env_path = run_dir / "environment.json"
    with env_path.open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    # Git
    commit = _git_commit_hash()
    git_info = {"commit": commit}
    git_path = run_dir / "git.json"
    with git_path.open("w", encoding="utf-8") as f:
        json.dump(git_info, f, indent=2)

    # Seeds if present in config
    seeds = {"seed": config.get("seed")} if config and isinstance(config, dict) and "seed" in config else {}
    seeds_path = run_dir / "seeds.json"
    with seeds_path.open("w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2)

    # Manifest
    manifest = {
        "run_id": run_id,
        "created_at": ts,
        "status": "RUNNING",
        "commit": commit,
        "config_hash": _hash_of_obj(config),
        "artifacts": [],
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Create artifacts dir and metrics file placeholder
    artifacts_dir = run_dir / "artifacts"
    _safe_mkdir(artifacts_dir)
    metrics_csv = run_dir / "metrics.csv"
    # Create head if not exists
    if not metrics_csv.exists():
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "step", "key", "value"])

    return run_id, str(run_dir)


def _get_run_dir(run_id: str, base_dir: Optional[str] = None) -> Path:
    base = Path(base_dir or DEFAULT_BASE_DIR)
    run_dir = base / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    return run_dir


def log_params_metrics(run_id: str, params: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, Any]] = None, step: Optional[int] = None, base_dir: Optional[str] = None):
    """
    Append params and metrics. Params written/merged to params.json. Metrics appended to metrics.csv.
    """
    run_dir = _get_run_dir(run_id, base_dir)
    ts = _now_ts()

    # Params
    if params:
        params_path = run_dir / "params.json"
        # Merge if exists
        existing = {}
        if params_path.exists():
            try:
                existing = json.loads(params_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        merged = {**existing, **params}
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, default=str)

    # Metrics
    if metrics:
        metrics_csv = run_dir / "metrics.csv"
        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                writer.writerow([ts, step if step is not None else "", k, v])


def log_artifact(run_id: str, src_path: str, artifact_name: Optional[str] = None, base_dir: Optional[str] = None):
    """
    Copy a file into run's artifacts/ and update manifest.
    """
    run_dir = _get_run_dir(run_id, base_dir)
    artifacts_dir = run_dir / "artifacts"
    _safe_mkdir(artifacts_dir)

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"artifact source does not exist: {src_path}")

    name = artifact_name or src.name
    dest = artifacts_dir / name
    # If destination exists, append a suffix
    if dest.exists():
        dest = artifacts_dir / f"{dest.stem}_{_short_id()}{dest.suffix}"
    shutil.copy2(str(src), str(dest))

    # Update manifest
    manifest_path = run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifacts", []).append(str(dest.relative_to(run_dir)))
    manifest["last_artifact_at"] = _now_ts()
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def end_run(run_id: str, status: str = "FINISHED", base_dir: Optional[str] = None):
    run_dir = _get_run_dir(run_id, base_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json missing for run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["ended_at"] = _now_ts()
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
