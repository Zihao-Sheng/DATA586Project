# DATA 586 Course Project (W2026 T2)

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

## Unified Workflow Runner

Use one command to run the project pipeline in a fixed order:

```bash
uv run python scripts/workflow.py
```

Current registered step:

- `data_retrieval`

Useful options:

```bash
uv run python scripts/workflow.py --list-steps
uv run python scripts/workflow.py --only-step data_retrieval
uv run python scripts/workflow.py --force-redownload
```

When new pipeline steps are added (e.g., training/evaluation), register them in `scripts/workflow.py` to keep team runs consistent and reproducible.

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
