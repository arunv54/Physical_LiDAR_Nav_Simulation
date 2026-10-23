"""Numerical commands used to generate the report and simulation videos.

Presentation runs separately through tools/ and never loads this module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError
from .paths import DATA_DIR, DEFAULT_TRAJECTORY_CONFIG


def evaluate(args):
    from .experiments.runner import run_evaluation
    result = run_evaluation(args.data_dir, args.part)
    print(f"Saved controlled evaluation to {Path(args.data_dir).resolve()}")
    for name, study in result["studies"].items():
        print(f"  {name}: {study['elapsed_s']:.2f} s")
    return 0


def evaluate_navigation_dependencies(args):
    from .experiments.navigation_dependencies import run_study
    run_study(Path(args.data_dir).resolve(), args.config, seeds=args.seeds,
              case_ids=args.cases)
    print(f"Saved navigation dependency comparisons to {Path(args.data_dir).resolve()}")
    return 0


def evaluate_navigation_threshold(args):
    from .experiments.navigation_threshold import run_study
    run_study(args.data_dir, source_dir=args.source_dir, trials_per_setting=args.trials)
    print(f"Saved navigation threshold trials to {Path(args.data_dir).resolve()}")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate", help="Controlled geometry, response and uncertainty checks")
    p.set_defaults(func=evaluate)
    p.add_argument("--part", choices=["all", "geometry", "response"], default="all")
    p.add_argument("--data-dir", default=str(DATA_DIR / "evaluation"))
    p = sub.add_parser("evaluate-navigation-dependencies", help="Fixed-scene report and sensor-state studies")
    p.set_defaults(func=evaluate_navigation_dependencies)
    p.add_argument("--config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cases", nargs="+", help="Named conditions; includes the shared baseline")
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_dependencies"))
    p = sub.add_parser("evaluate-navigation-threshold", help="Independent near-threshold scan trials")
    p.set_defaults(func=evaluate_navigation_threshold)
    p.add_argument("--source-dir", default=str(DATA_DIR / "navigation_dependencies"))
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_threshold"))
    p.add_argument("--trials", type=int, default=1000)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError, OSError, KeyError) as exc:
        ap.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
