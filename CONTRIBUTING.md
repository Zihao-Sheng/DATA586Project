# Contributing Guide

This repository is intended to be reproducible across teammates.  
Please follow this guide when adding code, data steps, and experiments.

## 1. Environment Setup

1. Use the project root as your working directory.
2. Use `uv` to run scripts so everyone uses the same project environment.
3. Keep local Python/package changes minimal and document any required dependency changes in your PR.

## 2. Canonical Run Path

Always use the workflow entrypoint first:

```bash
uv run python scripts/workflow.py
```

Do not bypass workflow for shared steps.  
If a step is intended for team usage, it must be registered in `scripts/workflow.py`.

Useful commands:

```bash
uv run python scripts/workflow.py --list-steps
uv run python scripts/workflow.py --only-step data_retrieval
uv run python scripts/workflow.py --force-redownload
```

## 3. Workflow Pitfalls and How to Avoid Them

### Pitfall A: Team members run scripts in different order
- Problem: Results differ because prerequisites were not prepared consistently.
- Fix: Run `scripts/workflow.py` as the default entrypoint and keep step order centralized in `WORKFLOW_STEPS`.

### Pitfall B: Running scripts directly with custom local flags
- Problem: Hidden local differences lead to non-reproducible outputs.
- Fix: Expose shared flags through `workflow.py`, then pass them into step scripts in a controlled way.

### Pitfall C: Partial dataset or corrupted files
- Problem: Training fails or metrics drift because data files are missing.
- Fix: Use `data_retrieval` via workflow; it checks integrity and repairs missing metadata/images.

### Pitfall D: Data path inconsistency
- Problem: One user uses `./data`, another uses a different folder; results or failures diverge.
- Fix: Keep `--data-dir` explicit when needed and document non-default paths in PR descriptions.

### Pitfall E: "Works on my machine" workflow changes
- Problem: A new step works locally but not for others.
- Fix: New steps must be idempotent, use clear CLI args, and avoid hidden assumptions about local files.

## 4. Adding a New Workflow Step

When adding training/evaluation or any shared stage:

1. Create a dedicated script under `scripts/` with `argparse` CLI.
2. Make the script safe to re-run (no destructive side effects by default).
3. Register the step in `WORKFLOW_STEPS` inside `scripts/workflow.py`.
4. If step options are needed, add them to `workflow.py` and wire them into the step command builder.
5. Update `README.md` and this file if behavior or run instructions change.

## 5. Data and Artifact Policy

- `data/` must never be committed (already ignored in `.gitignore`).
- Do not commit large generated artifacts, checkpoints, or temporary outputs unless explicitly requested.
- Keep notebooks/scripts reproducible from repository code and documented commands.

## 6. Commit and PR Expectations

1. Keep PRs focused and scoped.
2. Explain what changed, why, and how to run/verify.
3. Mention workflow impact explicitly:
   - new step added?
   - step order changed?
   - new required arguments?
4. Include at least one runnable command example in the PR description.

## 7. Quick Verification Before PR

Run at least:

```bash
uv run python scripts/workflow.py --list-steps
uv run python scripts/workflow.py --only-step data_retrieval
```

If your PR adds a new step, include one end-to-end command showing that step execution path.
