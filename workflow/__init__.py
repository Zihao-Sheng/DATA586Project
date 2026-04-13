from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "WorkflowConfig",
    "build_model_family_path_groups",
    "build_model_specs_from_latest_df",
    "build_split_analysis_from_paths",
    "compare_test_split_results",
    "get_latest_workflow_runs",
    "build_split_analysis_from_latest",
    "list_workflow_runs",
    "load_workflow_summary",
    "show_model_confusions_paginated_interactive",
    "build_model_specs_from_checkpoints",
    "show_gradcam_compare_all_models",
    "print_rubric_summary",
    "show_gradcam_compare",
    "show_prediction_compare",
    "sample_test_split_images",
    "show_workflow_summary_table",
    "plot_test_split_comparison_interactive",
    "plot_efficiency_compare_interactive",
    "show_model_epoch_dynamics_paginated_interactive",
    "plot_val_accuracy_all_models_interactive",
    "plot_final_test_accuracy_model_comparison_interactive",
    "plot_training_history",
    "run_experiment_workflow",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module("workflow.experiment_workflow")
    return getattr(module, name)

