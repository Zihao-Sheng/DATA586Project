#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
TORCH_FAMILY = {"torch", "torchvision", "torchaudio"}
TORCH_INDEX_URLS = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu128": "https://download.pytorch.org/whl/cu128",
}


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
    parser.add_argument(
        "--torch-variant",
        choices=["auto", "cpu", "cu128"],
        default="auto",
        help="PyTorch wheel variant to install for torch/torchvision/torchaudio (default: auto).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically accept install prompts.",
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


def run_install_command(command: list[str], dry_run: bool) -> int:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"Install command: {printable}")
    if dry_run:
        print("Dry run enabled, skipping installation.")
        return 0
    completed = subprocess.run(command)
    return completed.returncode


def detect_installed_torch_variant() -> str | None:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    if "+cu128" in version:
        return "cu128"
    if "+cpu" in version:
        return "cpu"
    return None


def has_nvidia_gpu() -> bool:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def choose_torch_variant(requested_variant: str) -> str:
    if requested_variant != "auto":
        return requested_variant
    installed_variant = detect_installed_torch_variant()
    if installed_variant is not None:
        return installed_variant
    return "cu128" if has_nvidia_gpu() else "cpu"


def should_prompt_for_cuda_choice(*, interactive: bool, assume_yes: bool) -> bool:
    return interactive and not assume_yes


def ask_yes_no(question: str, default_yes: bool = True) -> bool:
    prompt = " [Y/n] " if default_yes else " [y/N] "
    while True:
        answer = input(question + prompt).strip().lower()
        if not answer:
            return default_yes
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def resolve_torch_variant(requested_variant: str, assume_yes: bool) -> str:
    if requested_variant != "auto":
        return requested_variant

    installed_variant = detect_installed_torch_variant()
    gpu_available = has_nvidia_gpu()
    interactive = sys.stdin.isatty()

    if gpu_available and installed_variant == "cpu" and should_prompt_for_cuda_choice(interactive=interactive, assume_yes=assume_yes):
        if ask_yes_no("NVIDIA GPU detected and CPU-only PyTorch is installed. Install the CUDA-enabled cu128 build instead?", default_yes=True):
            return "cu128"
        return "cpu"

    if gpu_available and installed_variant is None and should_prompt_for_cuda_choice(interactive=interactive, assume_yes=assume_yes):
        if ask_yes_no("NVIDIA GPU detected. Install the CUDA-enabled cu128 build of PyTorch?", default_yes=True):
            return "cu128"
        return "cpu"

    if installed_variant is not None:
        return installed_variant
    return "cu128" if gpu_available else "cpu"


def install_missing_packages(
    missing_specs: list[str],
    *,
    index_url: str | None,
    dry_run: bool,
    torch_variant: str,
) -> int:
    if not missing_specs:
        print("All requested packages are already installed.")
        return 0

    torch_specs: list[str] = []
    other_specs: list[str] = []
    for spec in missing_specs:
        if distribution_name(spec) in TORCH_FAMILY:
            torch_specs.append(spec)
        else:
            other_specs.append(spec)

    if torch_specs:
        torch_index_url = TORCH_INDEX_URLS[torch_variant]
        print(f"PyTorch install variant: {torch_variant}")
        if torch_variant == "cu128":
            print("Detected or selected NVIDIA/CUDA environment. Installing GPU wheels.")
        else:
            print("No NVIDIA/CUDA environment detected. Installing CPU wheels.")
        print("Missing PyTorch packages:")
        for spec in torch_specs:
            print(f"- {spec}")
        result = run_install_command(
            [sys.executable, "-m", "pip", "install", *torch_specs, "--index-url", torch_index_url],
            dry_run,
        )
        if result != 0:
            return result

    if other_specs:
        print("Missing packages:")
        for spec in other_specs:
            print(f"- {spec}")
        command = [sys.executable, "-m", "pip", "install", *other_specs]
        if index_url:
            command.extend(["--index-url", index_url])
        result = run_install_command(command, dry_run)
        if result != 0:
            return result

    return 0


def main() -> None:
    args = parse_args()
    specs = gather_specs(args)
    if not specs:
        raise SystemExit(
            f"No packages specified and no readable requirements file found at {args.requirements}."
        )

    missing_specs: list[str] = []
    requested_torch_specs: list[str] = []
    for spec in specs:
        package_name = distribution_name(spec)
        if package_name in TORCH_FAMILY:
            requested_torch_specs.append(spec)
        if is_installed(package_name):
            print(f"[ok] {package_name}")
        else:
            print(f"[missing] {package_name}")
            missing_specs.append(spec)

    selected_torch_variant = resolve_torch_variant(args.torch_variant, args.yes)
    installed_torch_variant = detect_installed_torch_variant()
    if requested_torch_specs and installed_torch_variant == "cpu" and selected_torch_variant == "cu128":
        print("[upgrade] CPU-only PyTorch detected; CUDA-enabled cu128 build selected.")
        for spec in requested_torch_specs:
            if spec not in missing_specs:
                missing_specs.append(spec)

    raise SystemExit(
        install_missing_packages(
            missing_specs,
            index_url=args.index_url,
            dry_run=args.dry_run,
            torch_variant=selected_torch_variant,
        )
    )


if __name__ == "__main__":
    main()
