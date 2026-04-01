#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether Python packages are installed and install missing ones."
    )
    parser.add_argument(
        "packages",
        nargs="*",
        help="Package specifiers to verify, for example PySide6 or torch==2.7.0.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Requirements file to read package specifiers from (default: PROJECT_ROOT/requirements.txt).",
    )
    parser.add_argument(
        "--index-url",
        help="Optional pip index URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without running pip.",
    )
    return parser.parse_args()


def normalize_requirement_lines(lines: list[str]) -> list[str]:
    specs: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)
    return specs


def read_requirement_file(path: Path) -> list[str]:
    return normalize_requirement_lines(path.read_text(encoding="utf-8").splitlines())


def distribution_name(specifier: str) -> str:
    for separator in ("==", ">=", "<=", "!=", "~=", ">", "<", "[", ";"):
        if separator in specifier:
            return specifier.split(separator, 1)[0].strip()
    return specifier.strip()


def is_installed(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def gather_specs(args: argparse.Namespace) -> list[str]:
    specs: list[str] = []
    requirements_path = args.requirements.expanduser().resolve() if args.requirements is not None else None
    if requirements_path is not None and requirements_path.is_file():
        specs.extend(read_requirement_file(requirements_path))
    specs.extend(args.packages)
    deduped: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec not in seen:
            deduped.append(spec)
            seen.add(spec)
    return deduped


def install_missing_packages(missing_specs: list[str], index_url: str | None, dry_run: bool) -> int:
    if not missing_specs:
        print("All requested packages are already installed.")
        return 0

    print("Missing packages:")
    for spec in missing_specs:
        print(f"- {spec}")

    command = [sys.executable, "-m", "pip", "install", *missing_specs]
    if index_url:
        command.extend(["--index-url", index_url])

    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"Install command: {printable}")

    if dry_run:
        print("Dry run enabled, skipping installation.")
        return 0

    completed = subprocess.run(command)
    return completed.returncode


def main() -> None:
    args = parse_args()
    specs = gather_specs(args)
    if not specs:
        raise SystemExit(
            f"No packages specified and no readable requirements file found at {args.requirements}."
        )

    missing_specs: list[str] = []
    for spec in specs:
        package_name = distribution_name(spec)
        if is_installed(package_name):
            print(f"[ok] {package_name}")
        else:
            print(f"[missing] {package_name}")
            missing_specs.append(spec)

    raise SystemExit(install_missing_packages(missing_specs, args.index_url, args.dry_run))


if __name__ == "__main__":
    main()
