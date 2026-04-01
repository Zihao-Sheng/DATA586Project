# Contributing Guide

This repository is meant to stay easy to run for teammates without hidden local setup.

## 1. Working Style

1. Use the project root as the working directory.
2. Prefer updating shared scripts under `scripts/` instead of creating one-off local helpers.
3. Keep new code rerunnable and explicit about paths.

## 2. Environment Setup

The canonical dependency list lives in [requirements.txt](c:/Users/18447/DATA586Project/requirements.txt).

Why it stays in the repo root:
- it is the standard Python location;
- it is easy for teammates to find;
- the root launchers and README already point to it;
- `scripts/ensure_packages.py` uses it as the default source of packages.

When dependencies change:
1. Update [requirements.txt](c:/Users/18447/DATA586Project/requirements.txt).
2. Make sure [ensure_packages.py](c:/Users/18447/DATA586Project/scripts/ensure_packages.py) still works with the new list.
3. Mention the dependency change in your PR or handoff note.

## 3. Main Entry Points

Current shared entry points are:
- [training_gui.py](c:/Users/18447/DATA586Project/scripts/training_gui.py)
- [training.py](c:/Users/18447/DATA586Project/scripts/training.py)
- [predicting.py](c:/Users/18447/DATA586Project/scripts/predicting.py)
- [data_retrieval.py](c:/Users/18447/DATA586Project/scripts/data_retrieval.py)
- [ensure_packages.py](c:/Users/18447/DATA586Project/scripts/ensure_packages.py)

Root-level shortcuts exist for non-terminal usage:
- [Check Requirements.lnk](c:/Users/18447/DATA586Project/Check%20Requirements.lnk)
- [Launch Training GUI.lnk](c:/Users/18447/DATA586Project/Launch%20Training%20GUI.lnk)

## 4. Model Integration Rule

To add a new trainable model:
1. Add a new Python file under [scripts/model](c:/Users/18447/DATA586Project/scripts/model).
2. Implement two functions with the exact names:
   `build_model(...)`
   `build_optimizer(...)`
3. Keep dataset loading outside the model file.

Model discovery is automatic through [model_registry.py](c:/Users/18447/DATA586Project/scripts/model_registry.py), so new model files should appear in the CLI and GUI without extra registration work.

## 5. Data and Artifacts

- `data/` must not be committed.
- `checkpoints/` must not be committed.
- Large generated outputs should stay out of git unless explicitly requested.

## 6. GUI Expectations

If you change the desktop app:
1. Keep `Training`, `Predicting`, and `Data` tabs usable without terminal commands.
2. Avoid adding features that only work from CLI unless they are also exposed in the GUI when appropriate.
3. Keep launch behavior friendly for double-click usage on Windows.

## 7. Verification Before Merging

At minimum, verify the parts you touched:
- package check still runs;
- GUI still opens;
- data preparation still works if you changed dataset logic;
- training/prediction scripts still show `--help` successfully if you changed CLI behavior.

## 8. Documentation Rule

If behavior changes, update:
- [README.md](c:/Users/18447/DATA586Project/README.md)
- [CONTRIBUTING.md](c:/Users/18447/DATA586Project/CONTRIBUTING.md)

The goal is that a teammate can open the repo and understand how to run it without guessing.
