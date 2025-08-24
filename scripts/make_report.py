#!/usr/bin/env python
"""
Simple CLI to render a report from a results object or a run directory.

Usage:
  python scripts/make_report.py --results-file /path/to/results.json --out results/output_dir
  python scripts/make_report.py --run-dir results/runs/20250101T000000_abcdef12 --out results/runs/20250101T000000_abcdef12
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from bwv import report


def load_results_from_file(path: Path):
    # Support JSON or pickle via suffix
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in [".json", ".ndjson"]:
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        # attempt to use pandas to read pickles or fallback to json
        try:
            import pandas as pd  # type: ignore
            return pd.read_pickle(path)
        except Exception:
            return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-file", type=str, help="Path to a JSON or pickle results file")
    p.add_argument("--run-dir", type=str, help="Path to a run directory containing artifacts/results.json")
    p.add_argument("--out", type=str, required=True, help="Output directory to write report artifacts")
    args = p.parse_args()

    results = None
    if args.results_file:
        results = load_results_from_file(Path(args.results_file))
    elif args.run_dir:
        # look for results.json or results.pkl
        rdir = Path(args.run_dir)
        if (rdir / "results.json").exists():
            results = json.loads((rdir / "results.json").read_text(encoding="utf-8"))
        elif (rdir / "results.pkl").exists():
            import pandas as pd  # type: ignore
            results = pd.read_pickle(rdir / "results.pkl")
        else:
            # try manifest or metrics as simple fallback
            raise FileNotFoundError("No results file found in run dir")
    else:
        print("Provide --results-file or --run-dir", file=sys.stderr)
        sys.exit(2)

    cfg = {}
    out_dir = args.out
    created = report.generate_report(results, cfg, out_dir)
    print("Created artifacts:", created)


if __name__ == "__main__":
    main()
