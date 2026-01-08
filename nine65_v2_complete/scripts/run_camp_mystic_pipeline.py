#!/usr/bin/env python3
"""
Run the Camp Mystic end-to-end pipeline (data -> training -> validation).
"""

import argparse
import os
import subprocess
import sys


def run_step(label, cmd, env, cwd=None):
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run Camp Mystic pipeline")
    parser.add_argument("--offline", action="store_true", help="Force synthetic data")
    parser.add_argument("--with-rust", action="store_true", help="Run Rust validation binary")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.offline:
        env["MYSTIC_OFFLINE"] = "1"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    run_step(
        "Fetch Camp Mystic data",
        [python, os.path.join(script_dir, "fetch_camp_mystic_2007.py")],
        env,
    )

    run_step(
        "Build unified dataset",
        [python, os.path.join(script_dir, "build_camp_mystic_unified.py")],
        env,
    )

    run_step(
        "Train basin attractor model",
        [python, os.path.join(script_dir, "train_basin_attractor.py")],
        env,
    )

    run_step(
        "Validate with trained model",
        [python, os.path.join(script_dir, "validate_with_training.py")],
        env,
    )

    if args.with_rust:
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))
        run_step(
            "Rust Camp Mystic validation",
            ["cargo", "run", "--bin", "test_camp_mystic_2007", "--features", "v2"],
            env,
            cwd=repo_root,
        )

    print("\n✓ Camp Mystic pipeline complete")


if __name__ == "__main__":
    main()
