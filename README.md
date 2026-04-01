# DATA 586 Course Project (W2026 T2)

## Quick Start

Double-click [Check Requirements.lnk](c:/Users/18447/DATA586Project/Check%20Requirements.lnk) to open the package checker without a terminal window.

Double-click [Launch Training GUI.lnk](c:/Users/18447/DATA586Project/Launch%20Training%20GUI.lnk) to open the desktop GUI without a terminal window.

The GUI includes tabs for dataset preparation, training, and prediction, so you can download/repair data from inside the app.

`Check Requirements.lnk` now runs a resilient launcher and repairs shortcut targets when they are stale, including `Launch Training GUI.lnk`.

If shortcut targets drift after changing environments, rebuild them with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\maintenance\rebuild_gui_shortcuts.ps1
```

## Run in Jupyter (.ipynb)

If teammates cannot open the GUI, run training directly from notebook cells:

```python
%cd c:/Users/18447/DATA586Project
```

```python
!python scripts/entry/data_retrieval.py
```

```python
!python scripts/entry/training.py --model resnet18 --epochs 3 --batch-size 32 --num-workers 4 --image-size 224 --checkpoint-dir checkpoints/resnet18_ipynb
```

```python
!python scripts/entry/training.py --help
```

## Run in Terminal

If GUI cannot be launched, run from PowerShell or CMD:

```powershell
cd c:\Users\18447\DATA586Project
```

```powershell
python scripts/entry/data_retrieval.py
```

```powershell
python scripts/entry/training.py --model resnet18 --epochs 3 --batch-size 32 --num-workers 4 --image-size 224 --checkpoint-dir checkpoints\resnet18_cli
```

```powershell
python scripts/entry/training.py --help
```

## Basic Information

- Course: `DATA 586 - Advanced Machine Learning`
- Instructor: `Dr. Shan Du`
- Team size: `3 students per group`
- Total project score: `60 points`
- Due date: **April 24, 2026 (Friday), 11:59 PM**

## Project Goal

This project studies how different fine-tuning strategies affect image classification performance and robustness on **Food-101**.

You will compare required pretrained backbones and adaptation methods, then analyze trade-offs between:

- accuracy,
- robustness under distribution shift,
- training/inference cost,
- and model efficiency.

## Required Models and Training Strategies

Use ImageNet-1K pretrained weights for both backbones:

- `ResNet18`
- `EfficientNetV2-S`

Implement and compare:

1. `Linear Probing` (required baseline)
2. `PEFT methods`:
1. `LoRA`
2. `Task-Specific Adapters (TSA)`
3. `BatchNorm Tuning`
3. `Full Fine-Tuning` (optional, but explain if omitted)

## Data and Evaluation Rules

- Train using **only the training split**.
- Use test sets only for evaluation.
- Evaluate on:
- clean Food-101 test set,
- provided transformed/corrupted test variants (robustness).
- Report:
- clean test performance,
- per-variant robustness scores,
- optional aggregate robustness summary.

Robustness test variants (~4.07 GB):
`https://drive.google.com/file/d/1MvkUSd2ESqBzXgObUYBBxOH6kMFLDP5B/view?usp=sharing`

## Suggested Project Workflow

1. **Environment Setup**
   - Prepare dependencies and reproducible training/evaluation scripts.
2. **Data Preparation**
   - Load Food-101 train/test and robustness variants with consistent preprocessing.
3. **Baseline Training**
   - Train linear probes on ResNet18 and EfficientNetV2-S.
4. **PEFT Experiments**
   - Add LoRA, TSA, and BatchNorm tuning variants.
5. **Custom Improvement**
   - Build at least one improved model/pipeline (augmentation, regularization, or alternate adaptation).
6. **Optional Full Fine-Tuning**
   - Fine-tune all parameters if resources allow; otherwise document attempts.
7. **Evaluation**
   - Run all models on clean + transformed sets.
8. **Efficiency Analysis**
   - Compare performance vs FLOPs/wall-clock time, with bubble size proportional to trainable parameters.
   - Also compare inference speed vs performance.
9. **Explainability and Failure Analysis**
   - Use Grad-CAM (or similar XAI) to explain where models focus and why they fail on specific variants.
10. **Deliverable Packaging**
   - Prepare report PDF, reproducible notebook, and 8-10 minute presentation in one ZIP (without dataset).

## Training GUI

Launch the PySide6 desktop trainer:

```bash
python scripts/app/training_gui.py
```

The GUI wraps `scripts/entry/training.py`, lets you adjust the currently exposed training arguments, and streams terminal output into the window while training runs.

Script layout:
- `scripts/pipeline/`: training, prediction, and dataset retrieval implementations.
- `scripts/core/`: shared core modules (for example model discovery).
- `scripts/maintenance/`: dependency checks and Windows shortcut launch/repair logic.
- `scripts/build/`: build scripts for distributable launchers.

To build a Windows `.exe` after installing `PySide6` and `PyInstaller`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build\build_training_gui.ps1
```

## Deliverables

Submit one ZIP file containing:

1. **Project Report (PDF)**
   - Suggested sections: Summary, Introduction, Methodology, Experiments, Conclusion/Future Work, References, Group Contributions.
2. **Notebook (`.ipynb`)**
   - Includes runnable code and output for replication.
3. **Project Presentation**
   - Slides for an **8-10 minute** talk.

## Bonus (Optional)

You may explore newer methods (2024+) such as:

- Vision Transformers,
- newer PEFT techniques,
- alternative classifier designs.

## Recommended References

- LoRA: `https://github.com/microsoft/LoRA/tree/main`
- TSA example: `https://github.com/VICO-UoE/URL/blob/master/models/tsa.py`
- Grad-CAM: `https://github.com/jacobgil/pytorch-grad-cam`
- FLOPs tools: `https://github.com/MrYxJ/calculate-flops.pytorch`
