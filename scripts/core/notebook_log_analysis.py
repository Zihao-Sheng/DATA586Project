from __future__ import annotations

import json
from pathlib import Path


def load_runs(log_paths: list[Path]) -> list[dict]:
    runs: list[dict] = []
    for path in log_paths:
        try:
            data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data["_log_path"] = str(path.expanduser().resolve())
            runs.append(data)
    return runs


def plot_efficiency_tradeoff(log_paths: list[Path], x_metric: str = "Train Wall Time"):
    import matplotlib.pyplot as plt

    runs = load_runs(log_paths)
    points: list[tuple[float, float, float, str]] = []
    for run in runs:
        x_value = efficiency_x_value(run, x_metric)
        y_value = final_accuracy(run)
        size = float(((run.get("model") or {}).get("trainable_params", 1)) or 1)
        if x_value is None or y_value is None:
            continue
        points.append((x_value, y_value, size, run_label(run)))

    if not points:
        print("No efficiency data available.")
        return

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    sizes = [40 + 220 * (point[2] / max(max(point[2] for point in points), 1.0)) for point in points]
    labels = [point[3] for point in points]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=sizes, alpha=0.65)
    for x, y, label in zip(x_values, y_values, labels):
        plt.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    plt.xlabel(efficiency_x_label(x_metric))
    plt.ylabel("Accuracy")
    plt.title("Performance vs Efficiency")
    plt.grid(alpha=0.25)
    plt.show()


def display_confusion_matrix(log_paths: list[Path], view: str = "summary", top_k: int = 10):
    import matplotlib.pyplot as plt

    runs = load_runs(log_paths)
    if len(runs) != 1:
        print("Select exactly one run for confusion matrix export.")
        return

    labels, matrix = confusion_matrix_from_run(runs[0], view=view, top_k=top_k)
    if not labels:
        print("No confusion data available for this run.")
        return

    plt.figure(figsize=(max(6, top_k * 0.65), max(5, top_k * 0.6)))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Top-{len(labels)} Confusion Matrix")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            plt.text(col_index, row_index, str(value), ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout()
    plt.show()


def efficiency_x_value(run: dict, x_metric: str) -> float | None:
    timing_summary = run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {}
    stage_totals = timing_summary.get("stage_totals") if isinstance(timing_summary.get("stage_totals"), dict) else {}
    model_info = run.get("model") if isinstance(run.get("model"), dict) else {}
    final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else {}

    if x_metric == "Train Wall Time":
        return safe_float((stage_totals.get("train") or {}).get("total_seconds"))
    if x_metric == "Train Pure Time":
        return safe_float((stage_totals.get("train") or {}).get("pure_seconds"))
    if x_metric == "Test Avg Pure / Batch":
        timing = final_test.get("timing") if isinstance(final_test.get("timing"), dict) else None
        if isinstance(timing, dict):
            pure = safe_float(timing.get("pure_seconds"))
            batches = safe_float(timing.get("batches"))
            if pure is not None and batches is not None and batches > 0:
                return pure / batches
        return None
    if x_metric == "Trainable Params":
        return safe_float(model_info.get("trainable_params"))
    return None


def efficiency_x_label(x_metric: str) -> str:
    labels = {
        "Train Wall Time": "Train Wall Time (s)",
        "Train Pure Time": "Train Pure Time (s)",
        "Test Avg Pure / Batch": "Test Avg Pure / Batch (s)",
        "Trainable Params": "Trainable Params",
    }
    return labels.get(x_metric, x_metric)


def final_accuracy(run: dict) -> float | None:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    value = safe_float(summary.get("final_test_acc"))
    if value is not None:
        return value
    return safe_float(summary.get("best_eval_acc"))


def run_label(run: dict) -> str:
    args = run.get("args") if isinstance(run.get("args"), dict) else {}
    started = str(run.get("start_time_utc", "-"))[:10]
    model = str(args.get("model", "run"))
    checkpoint_name = Path(str(args.get("checkpoint_dir", "-"))).name
    return f"{started} {model} ({checkpoint_name})"


def confusion_matrix_from_run(run: dict, *, view: str, top_k: int) -> tuple[list[str], list[list[int]]]:
    analysis_root = run.get("analysis") if isinstance(run.get("analysis"), dict) else {}
    normalized_view = view.strip().lower()
    if normalized_view == "summary":
        analysis = analysis_root.get("final_test") if isinstance(analysis_root.get("final_test"), dict) else analysis_root.get("last_eval")
    elif normalized_view in {"val", "test"}:
        last_stage = analysis_root.get("last_eval_stage")
        if last_stage == normalized_view:
            analysis = analysis_root.get("last_eval")
        elif normalized_view == "test":
            analysis = analysis_root.get("final_test")
        else:
            analysis = None
    else:
        analysis = None

    if not isinstance(analysis, dict):
        return [], []

    class_names = analysis.get("class_names") if isinstance(analysis.get("class_names"), list) else []
    pair_entries = analysis.get("confusion_pairs") if isinstance(analysis.get("confusion_pairs"), list) else []
    involvement: dict[int, int] = {}
    for entry in pair_entries:
        if not isinstance(entry, dict):
            continue
        true_idx = safe_int(entry.get("true_idx"))
        pred_idx = safe_int(entry.get("pred_idx"))
        count = safe_int(entry.get("count")) or 0
        if true_idx is None or pred_idx is None:
            continue
        if true_idx != pred_idx:
            involvement[true_idx] = involvement.get(true_idx, 0) + count
            involvement[pred_idx] = involvement.get(pred_idx, 0) + count

    if not involvement:
        return [], []

    selected_indices = [idx for idx, _ in sorted(involvement.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
    lookup = {idx: position for position, idx in enumerate(selected_indices)}
    matrix = [[0 for _ in selected_indices] for _ in selected_indices]
    for entry in pair_entries:
        if not isinstance(entry, dict):
            continue
        true_idx = safe_int(entry.get("true_idx"))
        pred_idx = safe_int(entry.get("pred_idx"))
        count = safe_int(entry.get("count")) or 0
        if true_idx in lookup and pred_idx in lookup:
            matrix[lookup[true_idx]][lookup[pred_idx]] = count
    labels = [str(class_names[idx]) if 0 <= idx < len(class_names) else str(idx) for idx in selected_indices]
    return labels, matrix


def safe_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def safe_int(value) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None
