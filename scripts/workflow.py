#!/usr/bin/env python3
"""
Project workflow runner.

Current pipeline:
1) data_retrieval

Add future steps by appending new Step entries in WORKFLOW_STEPS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON_BIN = sys.executable


@dataclass(frozen=True)
class Step:
    key: str
    description: str
    command_builder: Callable[[argparse.Namespace], list[str]]


def build_data_retrieval_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "data_retrieval.py"),
        "--data-dir",
        str(args.data_dir),
    ]
    if args.force_redownload:
        cmd.append("--force-redownload")
    return cmd


WORKFLOW_STEPS: tuple[Step, ...] = (
    Step(
        key="data_retrieval",
        description="Download and repair dataset files in data/.",
        command_builder=build_data_retrieval_command,
    ),
)


def get_step_keys() -> list[str]:
    return [step.key for step in WORKFLOW_STEPS]


def parse_args() -> argparse.Namespace:
    keys = get_step_keys()
    parser = argparse.ArgumentParser(
        description="Run the project workflow in a fixed, reproducible order."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Data directory passed to data retrieval step (default: PROJECT_ROOT/data).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download during data retrieval step.",
    )
    parser.add_argument(
        "--from-step",
        choices=keys,
        default=keys[0],
        help="Start execution from this step (default: first step).",
    )
    parser.add_argument(
        "--only-step",
        choices=keys,
        help="Run only one step and skip all others.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List available workflow steps and exit.",
    )
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> list[Step]:
    if args.only_step:
        return [step for step in WORKFLOW_STEPS if step.key == args.only_step]

    start_index = get_step_keys().index(args.from_step)
    return list(WORKFLOW_STEPS[start_index:])


def run_step(step: Step, args: argparse.Namespace) -> None:
    command = step.command_builder(args)
    printable_command = " ".join(f'"{part}"' if " " in part else part for part in command)

    print(f"\n=== Running step: {step.key} ===")
    print(f"Description: {step.description}")
    print(f"Command: {printable_command}\n")

    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise SystemExit(
            f"Workflow failed at step '{step.key}' with exit code {completed.returncode}."
        )

    print(f"=== Step completed: {step.key} ===")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()

    if args.list_steps:
        print("Workflow steps:")
        for step in WORKFLOW_STEPS:
            print(f"- {step.key}: {step.description}")
        return

    steps_to_run = resolve_steps(args)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {args.data_dir}")
    print(f"Selected steps: {', '.join(step.key for step in steps_to_run)}")

    for step in steps_to_run:
        run_step(step, args)

    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
