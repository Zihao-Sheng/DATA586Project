from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import traceback
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import torch

from core import runtime_paths
from core import custom_model_generator
from core.gradcam import render_gradcam_overlay_image_with_diagnostics
from core.model_registry import discover_model_names, discover_model_names_generated_first, load_model_module, resolve_preferred_model_name


@dataclass
class CaseResult:
    model_name: str
    source: str
    method: str
    base_model: str
    ok: bool
    details: dict[str, Any]


def build_tiny_food101_dataset(root: Path, *, image_size: int = 64) -> Path:
    data_root = root / "tiny_food101"
    meta_dir = data_root / "meta"
    image_dir = data_root / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    classes = ["alpha", "beta", "gamma"]
    (meta_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    rng = random.Random(42)
    train_map: dict[str, list[str]] = {name: [] for name in classes}
    test_map: dict[str, list[str]] = {name: [] for name in classes}

    for class_index, class_name in enumerate(classes):
        for i in range(6):
            stem = f"{class_name}_train_{i:02d}"
            path = image_dir / f"{stem}.jpg"
            image = Image.new("RGB", (image_size, image_size), (32 + class_index * 50, 24 + i * 6, 90 + i * 5))
            pixels = image.load()
            for y in range(image_size):
                for x in range(image_size):
                    if (x + y + i) % 11 == 0:
                        pixels[x, y] = (min(255, pixels[x, y][0] + rng.randint(10, 60)), pixels[x, y][1], pixels[x, y][2])
            image.save(path, format="JPEG")
            train_map[class_name].append(stem)
        for i in range(3):
            stem = f"{class_name}_test_{i:02d}"
            path = image_dir / f"{stem}.jpg"
            image = Image.new("RGB", (image_size, image_size), (80 + class_index * 30, 30 + i * 20, 40 + i * 15))
            image.save(path, format="JPEG")
            test_map[class_name].append(stem)

    (meta_dir / "train.json").write_text(json.dumps(train_map, indent=2), encoding="utf-8")
    (meta_dir / "test.json").write_text(json.dumps(test_map, indent=2), encoding="utf-8")
    return data_root


def choose_existing_model_name(candidates: list[str], available: set[str]) -> str | None:
    normalized = {name.lower(): name for name in available}
    for candidate in candidates:
        resolved = normalized.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None


def generate_custom_matrix_models(prefix: str = "reldbg") -> list[dict[str, str]]:
    required = [
        ("resnet18", "baseline"),
        ("resnet50", "bn_tuning"),
        ("efficientnet_v2_s", "bn_last1"),
        ("efficientnet_v2_s", "bn_last2"),
        ("efficientnet_v2_s", "dora"),
        ("convnext_tiny", "lora"),
        ("mobilenet_v3_large", "tsa"),
        ("densenet121", "full_finetune"),
    ]
    extras = [
        ("resnet50", "adapter"),
        ("mobilenet_v3_large", "bitfit"),
        ("convnext_tiny", "ssf"),
        ("densenet121", "norm_tuning"),
    ]

    generated: list[dict[str, str]] = []
    for base_model, method_type in [*required, *extras]:
        model_name = f"{prefix}_{base_model}_{method_type}".replace(".", "_")
        spec = custom_model_generator.build_preset_spec(
            model_name=model_name,
            base_model=base_model,
            method_type=method_type,
        )
        payload = custom_model_generator.spec_to_dict(spec)
        payload["pretrained"] = False
        payload.setdefault("generator_version", "reliability_matrix_v1")
        patched = custom_model_generator.spec_from_dict(payload)
        custom_model_generator.generate_custom_model(patched, overwrite=True)
        generated.append(
            {
                "model_name": model_name,
                "source": "fresh_custom_generated",
                "method": method_type,
                "base_model": base_model,
            }
        )
    return generated


def run_training_case(
    *,
    model_name: str,
    data_root: Path,
    checkpoint_dir: Path,
    epochs: int,
    image_size: int,
    resume_path: Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pipeline" / "training.py"),
        "--model",
        model_name,
        "--data-root",
        str(data_root),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--image-size",
        str(image_size),
        "--device",
        "cpu",
        "--optimizer",
        "adam",
        "--scheduler",
        "none",
        "--train-transforms-preset",
        "baseline",
        "--use-validation-split",
        "--validation-proportion",
        "0.2",
    ]
    if resume_path is not None:
        command.extend(["--resume", str(resume_path)])

    result = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    run_logs = sorted(
        (checkpoint_dir / "_run_logs").glob("*.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    latest_log = run_logs[0] if run_logs else None
    payload: dict[str, Any] = {
        "exit_code": int(result.returncode),
        "checkpoint_dir": str(checkpoint_dir),
        "latest_log": str(latest_log) if latest_log is not None else None,
        "stderr_tail": result.stderr[-2000:],
        "stdout_tail": result.stdout[-2000:],
    }

    if latest_log is None:
        payload["error"] = "No run log generated"
        return False, payload

    try:
        run_data = json.loads(latest_log.read_text(encoding="utf-8"))
    except Exception as exc:
        payload["error"] = f"Could not parse run log: {exc}"
        return False, payload

    status = str(run_data.get("status", ""))
    epochs_data = run_data.get("epochs") if isinstance(run_data.get("epochs"), list) else []
    summary = run_data.get("summary") if isinstance(run_data.get("summary"), dict) else {}
    args = run_data.get("args") if isinstance(run_data.get("args"), dict) else {}
    eval_name = "val" if bool(args.get("use_validation_split")) else "test"
    invalid_epochs = [
        idx
        for idx, item in enumerate(epochs_data, 1)
        if not isinstance(item, dict) or "train" not in item or eval_name not in item
    ]

    payload.update(
        {
            "status": status,
            "epoch_count": len(epochs_data),
            "planned_epochs_this_run": args.get("planned_epochs_this_run"),
            "summary_last_completed_epoch": summary.get("last_completed_epoch"),
            "warnings": run_data.get("warnings") if isinstance(run_data.get("warnings"), list) else [],
            "invalid_epoch_records": invalid_epochs,
        }
    )

    ok = (
        result.returncode == 0
        and status == "completed"
        and len(epochs_data) > 0
        and not invalid_epochs
    )
    return ok, payload


def ensure_synthetic_checkpoint(
    *,
    model_name: str,
    checkpoint_root: Path,
    num_classes: int = 3,
) -> Path:
    out_dir = checkpoint_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pth"
    if checkpoint_path.exists():
        return checkpoint_path

    model_module = load_model_module(model_name)
    build_fn = getattr(model_module, "build_model")
    signature = inspect.signature(build_fn)
    kwargs: dict[str, Any] = {"num_classes": num_classes}
    if "freeze_backbone" in signature.parameters:
        kwargs["freeze_backbone"] = True
    if "device" in signature.parameters:
        kwargs["device"] = "cpu"
    if "pretrained" in signature.parameters:
        kwargs["pretrained"] = False

    generated_spec_backup = None
    if hasattr(model_module, "GENERATED_SPEC") and isinstance(getattr(model_module, "GENERATED_SPEC"), dict):
        generated_spec_backup = dict(getattr(model_module, "GENERATED_SPEC"))
        generated_spec = dict(generated_spec_backup)
        generated_spec["pretrained"] = False
        setattr(model_module, "GENERATED_SPEC", generated_spec)

    try:
        model = build_fn(**kwargs)
    except Exception:
        if generated_spec_backup is not None:
            setattr(model_module, "GENERATED_SPEC", generated_spec_backup)
        raise
    if generated_spec_backup is not None:
        setattr(model_module, "GENERATED_SPEC", generated_spec_backup)

    class_to_idx = {"alpha": 0, "beta": 1, "gamma": 2}
    checkpoint = {
        "epoch": 0,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "best_acc": 0.0,
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "use_validation_split": False,
        "validation_proportion": 0.0,
        "optimizer": "adam",
        "scheduler": "none",
        "amp": False,
        "seed": 42,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def run_gradcam_case(*, model_name: str, image_path: Path, checkpoint_path: Path, image_size: int) -> tuple[bool, dict[str, Any]]:
    try:
        _overlay, diagnostic = render_gradcam_overlay_image_with_diagnostics(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            image_size=image_size,
            device="cpu",
        )
    except Exception as exc:
        return False, {"error": f"Exception: {exc}", "traceback": traceback.format_exc()}

    ok = diagnostic is None
    return ok, {"diagnostic": diagnostic}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reliability matrix checks for training logs and Grad-CAM.")
    parser.add_argument("--work-root", type=Path, default=runtime_paths.logs_dir() / "reliability_debug")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--max-training-cases", type=int, default=12)
    args = parser.parse_args()

    work_root = args.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str((work_root / "torch_cache").resolve()))

    tiny_data_root = build_tiny_food101_dataset(work_root, image_size=args.image_size)
    available_models = set(discover_model_names_generated_first(include_legacy_fallback=True))

    fresh_models = generate_custom_matrix_models(prefix="reldbg")
    for item in fresh_models:
        available_models.add(item["model_name"])

    legacy_candidates = [
        "efficientnet_baseline",
        "efficientnet_bn_last2",
        "resnet18_baseline",
        "resnet50_baseline",
        "convnext_tiny_baseline",
        "mobilenet_v3_large_baseline",
        "densenet121_baseline",
    ]
    generated_candidates = [
        "efficientnet_baseline_gen",
        "resnet18_baseline_gen",
        "resnet50_bn_tuning_gen",
        "convnext_tiny_lora_gen",
        "mobilenet_v3_large_tsa_gen",
        "densenet121_full_finetune_gen",
    ]

    legacy_name = choose_existing_model_name(legacy_candidates, available_models)
    generated_name = choose_existing_model_name(generated_candidates, available_models)

    training_cases: list[dict[str, str]] = []
    if legacy_name:
        training_cases.append({"model_name": legacy_name, "source": "legacy", "method": "unknown", "base_model": "unknown"})
    if generated_name:
        training_cases.append({"model_name": generated_name, "source": "existing_generated", "method": "unknown", "base_model": "unknown"})
    training_cases.extend(fresh_models[: max(0, args.max_training_cases - len(training_cases))])

    training_results: list[CaseResult] = []
    checkpoints_for_gradcam: dict[str, Path] = {}

    if not args.skip_training:
        for case in training_cases:
            model_name = resolve_preferred_model_name(case["model_name"]) or case["model_name"]
            checkpoint_dir = work_root / "training_ckpts" / model_name
            ok, details = run_training_case(
                model_name=model_name,
                data_root=tiny_data_root,
                checkpoint_dir=checkpoint_dir,
                epochs=1,
                image_size=args.image_size,
            )
            best_path = checkpoint_dir / "best.pth"
            if best_path.exists():
                checkpoints_for_gradcam[model_name] = best_path
            training_results.append(
                CaseResult(
                    model_name=model_name,
                    source=case["source"],
                    method=case["method"],
                    base_model=case["base_model"],
                    ok=ok,
                    details=details,
                )
            )

        # Resume-path reliability check on one fresh generated model.
        if fresh_models:
            resume_model = resolve_preferred_model_name(fresh_models[0]["model_name"]) or fresh_models[0]["model_name"]
            resume_ckpt_dir = work_root / "training_ckpts" / f"{resume_model}_resume"
            first_ok, first_details = run_training_case(
                model_name=resume_model,
                data_root=tiny_data_root,
                checkpoint_dir=resume_ckpt_dir,
                epochs=1,
                image_size=args.image_size,
            )
            resume_path = resume_ckpt_dir / "last.pth"
            second_ok, second_details = run_training_case(
                model_name=resume_model,
                data_root=tiny_data_root,
                checkpoint_dir=resume_ckpt_dir,
                epochs=2,
                image_size=args.image_size,
                resume_path=resume_path if resume_path.exists() else None,
            )
            training_results.append(
                CaseResult(
                    model_name=resume_model,
                    source="resume_check",
                    method=fresh_models[0]["method"],
                    base_model=fresh_models[0]["base_model"],
                    ok=bool(first_ok and second_ok),
                    details={"first": first_details, "second": second_details},
                )
            )

    gradcam_cases: list[dict[str, str]] = []
    gradcam_cases.extend(fresh_models)
    if legacy_name:
        gradcam_cases.append({"model_name": legacy_name, "source": "legacy", "method": "unknown", "base_model": "unknown"})
    if generated_name:
        gradcam_cases.append({"model_name": generated_name, "source": "existing_generated", "method": "unknown", "base_model": "unknown"})

    # add additional discoverable baseline names for broad family sweep when available
    family_sweep = ["resnet18", "resnet50", "efficientnet_v2_s", "convnext_tiny", "mobilenet_v3_large", "densenet121"]
    all_discovered = set(discover_model_names())
    for family in family_sweep:
        candidate = choose_existing_model_name([f"{family}_baseline_gen", f"{family}_baseline", family], all_discovered)
        if candidate:
            gradcam_cases.append({"model_name": candidate, "source": "family_sweep", "method": "baseline", "base_model": family})

    # de-duplicate by preferred name
    deduped_gradcam_cases: list[dict[str, str]] = []
    seen_gradcam: set[str] = set()
    for case in gradcam_cases:
        preferred = resolve_preferred_model_name(case["model_name"]) or case["model_name"]
        if preferred in seen_gradcam:
            continue
        seen_gradcam.add(preferred)
        deduped_gradcam_cases.append({**case, "model_name": preferred})

    sample_image = next((tiny_data_root / "images").glob("*.jpg"), None)
    if sample_image is None:
        raise RuntimeError("Tiny dataset image generation failed.")

    gradcam_results: list[CaseResult] = []
    for case in deduped_gradcam_cases:
        model_name = case["model_name"]
        checkpoint_path = checkpoints_for_gradcam.get(model_name)
        if checkpoint_path is None:
            checkpoint_path = ensure_synthetic_checkpoint(
                model_name=model_name,
                checkpoint_root=work_root / "gradcam_ckpts",
                num_classes=3,
            )
        ok, details = run_gradcam_case(
            model_name=model_name,
            image_path=sample_image,
            checkpoint_path=checkpoint_path,
            image_size=args.image_size,
        )
        details["checkpoint"] = str(checkpoint_path)
        gradcam_results.append(
            CaseResult(
                model_name=model_name,
                source=case["source"],
                method=case["method"],
                base_model=case["base_model"],
                ok=ok,
                details=details,
            )
        )

    output = {
        "work_root": str(work_root),
        "tiny_data_root": str(tiny_data_root),
        "training": {
            "total": len(training_results),
            "passed": sum(1 for item in training_results if item.ok),
            "failed": sum(1 for item in training_results if not item.ok),
            "results": [item.__dict__ for item in training_results],
        },
        "gradcam": {
            "total": len(gradcam_results),
            "passed": sum(1 for item in gradcam_results if item.ok),
            "failed": sum(1 for item in gradcam_results if not item.ok),
            "results": [item.__dict__ for item in gradcam_results],
        },
    }

    report_path = work_root / "reliability_report.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Reliability report: {report_path}")
    print(f"Training passed {output['training']['passed']}/{output['training']['total']}")
    print(f"Grad-CAM passed {output['gradcam']['passed']}/{output['gradcam']['total']}")

    return 0 if output["training"]["failed"] == 0 and output["gradcam"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
