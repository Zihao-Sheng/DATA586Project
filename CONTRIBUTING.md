# Contributing Guide

## Working Rules

1. Work from repository root.
2. Reuse shared modules under `scripts/` instead of one-off local scripts.
3. Keep paths relative so the repo is portable across machines.

## Dependencies

- Canonical dependency file: `requirements.txt`
- Dependency check tool: `scripts/maintenance/ensure_packages.py`

When dependencies change:
1. Update `requirements.txt`.
2. Verify `python scripts/maintenance/ensure_packages.py` still works.
3. Mention dependency changes in your PR/handoff note.

## Entry Points

- GUI: `scripts/app/training_gui.py`
- Train CLI: `scripts/entry/training.py`
- Predict CLI: `scripts/entry/predicting.py`
- Data retrieval: `scripts/entry/data_retrieval.py`
- Requirements checker: `scripts/entry/ensure_packages.py`

## Shortcut Policy

Keep these two root shortcuts committed:

- `Check Requirements.lnk`
- `Launch Training GUI.lnk`

If target bindings drift, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\maintenance\rebuild_gui_shortcuts.ps1
```

## Model Integration

To add a model:

1. Add a file under `scripts/model/`.
2. Implement `build_model(...)` and `build_optimizer(...)`.
3. Keep data loading outside model definition files.

Model discovery is automatic through `scripts/core/model_registry.py`.

## Verification Before Merge

Validate changed behavior with at least:

- dependency check still runs;
- GUI still opens;
- changed CLI commands still support `--help`;
- affected training/prediction/data workflows still run.

## Documentation Update Rule

If behavior changes, update:

- `README.md`
- `CONTRIBUTING.md`
