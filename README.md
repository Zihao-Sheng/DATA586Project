# FoodVision Transfer Learning Studio

FoodVision Transfer Learning Studio is a **GUI-first, experiment-oriented transfer learning platform** for image classification research. It combines a desktop application, reproducible CLI entry points, structured model generation, checkpoint management, logging, and visualization into one workspace. The current repository grew out of Food-101 experiments, but the application is designed as a broader **model training and analysis studio** rather than a single-purpose training script collection.

This project is especially useful for:
- researchers who want a **visual workflow** for training, evaluation, and comparison;
- developers who still need **scriptable entry points** and structured model definitions;
- iterative experimentation across **baseline, PEFT, staged fine-tuning, custom model variants, and robustness evaluation**.

---

## What the Application Does

At a high level, the app gives you one place to:

- prepare and manage datasets;
- launch classification training runs with checkpointed outputs;
- inspect run logs visually instead of only reading JSON or console text;
- evaluate checkpoints on curated robustness/test splits;
- run single-model or multi-model prediction workflows;
- generate new structured models from a **Custom Models / Canvas** interface;
- compare model behavior quantitatively and qualitatively, including **Grad-CAM**;
- manage queued jobs from a persistent side panel.

The repository also keeps a CLI path for reproducibility and automation, but the GUI is the main operating surface.

---

## Application Workflow at a Glance

A typical workflow looks like this:

1. **Data Tab** – prepare or validate the dataset you want to train on.
2. **Training Tab** – choose a model, configure training, and launch a run.
3. **Predicting Tab** – inspect checkpoints on images or folders and optionally compare models.
4. **Test Splits Tab** – evaluate a checkpoint on curated robustness splits.
5. **Logs Tab** – compare runs through plots, metrics, confusion matrices, and efficiency summaries.
6. **Custom Models Tab** – define or modify model configurations through the structured canvas.
7. **Global Queue** – manage queued jobs that can be launched from multiple tabs.

---

## GUI Tabs, Explained

## 1. Training Tab

The **Training** tab is the main launch surface for supervised classification training. It exposes the core configuration needed to start a run while keeping advanced options available without forcing the user into command-line use.

![Training Tab](./training_tab.png)

### What this tab is for
Use this tab to:
- choose the model to train;
- inspect where that model came from (generated, legacy, provider/family/variant);
- set the core training hyperparameters;
- configure checkpoints, run naming, and data root;
- optionally resume from an existing checkpoint;
- preview or export the generated command;
- enqueue jobs instead of running immediately.

### Main sections
**Core Training Config**
- model selection and model source display;
- epochs, batch size, optimizer, image size, learning rate;
- AMP / precision options;
- transform preset selection;
- common workflow toggles such as freeze-backbone, validation split, resume.

**Data / Output**
- run name;
- dataset root;
- checkpoint directory;
- summarized advanced settings.

**Command Preview**
- keeps the GUI reproducible by exposing the effective command that would be run.

### Why it matters
This tab is not just a thin launcher. It ties together:
- the structured model registry;
- generated-first / legacy-fallback awareness;
- checkpoint naming and output layout;
- training configuration visibility for reproducibility.

---

## 2. Predicting Tab

The **Predicting** tab is the inference and visual inspection workspace. It is designed for both quick single-model prediction and richer checkpoint comparison workflows.

![Predicting Tab](./predicting_tab.png)

### What this tab is for
Use this tab to:
- browse available checkpoints;
- choose one or more models for prediction;
- load individual images or folders;
- review predictions visually;
- switch between single-model and multi-model comparison workflows;
- trigger Grad-CAM or related qualitative inspection where supported;
- enqueue prediction jobs into the global queue.

### Main UI regions
**Checkpoint Selector**
- lists discovered checkpoints;
- supports generated-first presentation and source-aware model naming;
- can be used for single-model prediction or multi-model comparison.

**Predict Config**
- device;
- image size;
- compare mode;
- queue/export controls;
- prediction progress and navigation across images.

**Image Browser**
- thumbnail-driven browsing for selected images or folders;
- helps move through large batches without leaving the app.

**Prediction Preview**
- large central display of the current image;
- intended for quick visual interpretation and model review.

**Prediction Result**
- textual result summary for the currently focused image, including label and related metadata.

### Example prediction output
The repository also includes a captured example showing a rendered prediction view with a Grad-CAM overlay.

![Prediction Example](./predicting_example.png)

This kind of output is useful not only for debugging but also for qualitative comparison between tuning strategies.

---

## 3. Test Splits Tab

The **Test Splits** tab is the robustness-evaluation surface. It is designed for evaluating a checkpoint against curated dataset variants such as blur, downsample, masking, or other controlled perturbation sets.

![Test Splits Tab](./test_split_tab.png)

### What this tab is for
Use this tab to:
- select a trained checkpoint;
- point the app to a test-split root directory;
- configure evaluation settings such as device, image size, batch size, and AMP;
- evaluate one checkpoint against multiple structured test subsets;
- record per-split metrics that can later be visualized in the Logs tab.

### Why this tab matters
A standard train/val/test workflow does not tell the whole story for transfer learning. This tab makes it easy to turn robustness from an informal afterthought into a repeatable, logged evaluation step.

### Example visualization from test-split outputs
The repository includes a line-chart style comparison of per-split performance across several runs:

![Test Split Comparison](./test_split_comparison.png)

This makes it easy to see where one strategy wins or fails—for example on downsampled or blur-heavy variants.

---

## 4. Data Tab

The **Data** tab is the dataset-entry and dataset-preparation layer of the application.

### What this tab is for
Use this tab to:
- prepare or retrieve required data assets;
- validate whether the expected dataset structure is present;
- support dataset-specific setup workflows;
- act as the front door for future broader dataset support beyond Food-101.

### Current role in the repository
Historically, the repository has centered on Food-101, so this tab plays an important role in data readiness and local dataset management. In the current architecture, it is the natural place for:
- dataset discovery;
- validation of directory layout;
- built-in preparation logic for supported datasets.

### Why it matters
Moving data preparation into a dedicated tab keeps the rest of the application cleaner:
- training does not need to duplicate data checks;
- prediction and evaluation can rely on consistent dataset roots;
- later multi-dataset support can plug into a single UI surface instead of spreading logic across the app.

---

## 5. Logs Tab

The **Logs** tab is one of the strongest parts of the application. It converts raw run logs into an analysis workspace for comparing experiments visually.

![Logs Tab](./log_tab.png)

### What this tab is for
Use this tab to:
- browse available runs;
- compare curves across runs;
- inspect a single run in detail;
- summarize model behavior without leaving the GUI;
- connect training outcomes with later reporting/export workflows.

### Main capabilities
**Run list management**
- view discovered runs;
- add, refresh, delete, and export selections;
- prepare multiple runs for side-by-side plotting.

**Plot controls**
- choose detail or summary views;
- choose plot metric;
- choose stage selection (auto/all/specific stage).

**Training run details**
- inspect stored metadata such as run ID, timestamps, completion state, and configuration.

### Included visualization examples

#### Accuracy-over-epoch comparison
This plot compares validation accuracy across runs to show convergence behavior and relative performance trends.

![Accuracy Comparison](./acc_comparison.png)

#### Confusion matrix view
The confusion matrix panel helps inspect class-specific mistakes rather than only top-line accuracy.

![Confusion Matrix](./confusion_matrix.png)

#### Performance vs efficiency
This bubble chart captures the trade-off between train wall time, accuracy, and trainable parameter count.

![Performance vs Efficiency](./eff_comparison.png)

### Why this tab matters
Instead of treating logs as passive files, the app turns them into a first-class analysis surface. This is particularly valuable for transfer learning studies where:
- multiple tuning strategies are compared;
- robustness and efficiency matter alongside accuracy;
- qualitative interpretation should stay close to the numerical results.

---

## 6. Custom Models Tab

The **Custom Models** tab is the structured model-authoring and editing workspace. This is where the repository goes beyond standard training GUIs and starts behaving like a configurable model studio.

![Custom Models Tab](./coustom_model.png)

### What this tab is for
Use this tab to:
- create a new custom model configuration from a blank start;
- load an existing spec;
- open an existing generated model;
- inspect and modify family-aware stage or sub-stage structure;
- apply stage-level methods such as freeze, unfreeze, BN tuning, norm tuning, LoRA, DoRA, TSA, Adapter, BitFit, and SSF;
- assign custom learning rates to stages/substages where supported;
- view structured static information about the selected stage;
- save, version, and generate new model definitions.

### Main UI regions
**Top control bar**
- New Blank;
- Load Spec;
- Open Existing Model;
- legacy fallback visibility;
- model name;
- base model family/variant;
- pretrained toggle;
- global mode.

**Left method toolbox**
- direct method actions available to the selected node;
- intentionally separated from parameter editing.

**Center hierarchy canvas**
- stage/sub-stage representation of the current architecture;
- expandable, family-aware structure instead of raw module dumps;
- intended to stay readable while still supporting finer-grained customization.

**Right Stage Details panel**
- allowed operations for the selected node;
- stage method selection;
- custom LR control;
- static stage info;
- Grad-CAM hint;
- method summary.

**Spec view**
- structured JSON-like model specification shown directly in the workspace;
- keeps the model system inspectable instead of hidden behind opaque UI state.

### Why this tab matters
This tab is the app’s strongest differentiator. It makes the model system:
- structured;
- generator-backed;
- inspectable;
- extensible;
- safer than ad hoc hand-editing of Python model files.

It is the bridge between GUI convenience and research-grade experimentation.

---

## 7. Global Queue

The **Global Queue** is the persistent job-management panel shown on the right side of the application across tabs.

### What this panel is for
Use it to:
- queue training, prediction, or test-split jobs;
- reorder pending jobs;
- duplicate or remove queue items;
- launch queued work sequentially;
- pause/stop current jobs where supported;
- keep multi-step workflows organized without manually babysitting every action.

### Why this matters
Because the application supports multiple workflows from multiple tabs, a shared queue makes the entire app feel more like an experimentation environment and less like a collection of isolated forms.

---

## Why the App Feels Different from a Simple Training GUI

This repository is not just a launcher for one training script. The application is built around a few higher-level ideas:

### 1. Generated-first model management
The model system distinguishes between:
- generated/spec-backed models;
- legacy models;
- mapped fallback paths.

That makes model provenance visible instead of implicit.

### 2. Structured customization instead of ad hoc patching
The Custom Models system provides a controlled path for experimenting with:
- baseline;
- BN tuning;
- staged fine-tuning;
- LoRA / DoRA / TSA;
- stage-specific or sub-stage-specific method assignment;
- custom LR overrides.

### 3. Training, evaluation, and interpretation stay connected
A trained checkpoint is not the end of the workflow. The app keeps:
- training configuration;
- checkpoint outputs;
- robustness evaluation;
- prediction;
- Grad-CAM;
- plotting and comparison

inside the same environment.

---

## Repository Layout

Below is the logical structure of the repository based on the current workflow design. This builds on the original draft, but reflects the application more clearly as a studio rather than a loose script collection.

```text
.
├─ checkpoints/                  # Saved checkpoints and run outputs
├─ data/                         # Local datasets (Food-101 and future datasets)
├─ docs/                         # Project documentation
├─ exports/                      # Exported plots / artifacts / images
├─ logs/                         # Runtime logs and evaluation summaries
├─ model/                        # Model implementations (legacy + generated)
├─ model_specs/                  # Structured JSON model specs
├─ notebooks/                    # Experiment notebooks
├─ reports/                      # Generated report outputs
├─ scripts/
│  ├─ app/                       # GUI layer
│  ├─ core/                      # Shared runtime/model helpers
│  ├─ entry/                     # Stable CLI entry points
│  ├─ maintenance/               # Build, validation, environment, regression tools
│  └─ pipeline/                  # Training / predicting / data / evaluation pipelines
├─ workflow/                     # Workflow utilities
├─ Check Requirements.lnk        # Windows setup helper
├─ Launch Training GUI.lnk       # Main GUI launcher
└─ requirements.txt
```

---

## Quick Start

### GUI-first path
For most Windows users, the intended onboarding path is still:

1. Double-click **Check Requirements.lnk**
2. Wait for dependency checks/install to complete
3. Double-click **Launch Training GUI.lnk**

This keeps the repo accessible to teammates who do not want to use the terminal.

### CLI fallback
For scripted or reproducible runs:

```powershell
python -m pip install -r requirements.txt
python scripts/entry/data_retrieval.py
python scripts/entry/training.py --help
```

Example:

```powershell
python scripts/entry/training.py `
  --model resnet18 `
  --epochs 3 `
  --batch-size 32 `
  --num-workers 4 `
  --image-size 224 `
  --checkpoint-dir checkpoints\resnet18_cli
```

---

## What Makes This Repository Valuable

This repository is valuable not only because it trains image classifiers, but because it brings together:

- a GUI workflow that is actually usable for iterative experimentation;
- a structured custom-model system;
- generated/spec-backed model management;
- robustness evaluation beyond plain test accuracy;
- prediction and Grad-CAM inspection;
- log-driven comparison views for reporting and decision-making.

It is best thought of as a **transfer learning studio** rather than only a Food-101 training repo.

---

## Notes

- `data/` should remain local and generally should not be committed.
- The linked GUI launch flow is important for teammate onboarding on Windows.
- If image assets live in another folder in your repo, update the Markdown image paths accordingly.
