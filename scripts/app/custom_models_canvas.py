from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core import custom_model_generator
from core.model_registry import (
    discover_model_names_generated_first,
    model_catalog_entry,
    model_display_label,
    resolve_model_spec_path,
    resolve_model_structure_for_canvas,
    sort_model_names_for_ui,
)


class StrategyPaletteList(QListWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        mime = QMimeData()
        mime.setText(item.text())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class CanvasStageNode(QFrame):
    def __init__(
        self,
        stage_key: str,
        title: str,
        *,
        editable: bool,
        allowed_strategies: set[str],
        on_selected: Callable[[str], None],
        on_strategy_applied: Callable[[str, str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self.editable = editable
        self.allowed_strategies = allowed_strategies
        self._on_selected = on_selected
        self._on_strategy_applied = on_strategy_applied
        self._selected = False
        self.setAcceptDrops(editable)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("CanvasStageNode")
        self.setProperty("editable", editable)
        self.setProperty("selected", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setProperty("sectionTitle", True)
        self.state_label = QLabel("")
        self.state_label.setProperty("muted", True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.state_label)
        self._refresh_style()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._refresh_style()

    def set_state_text(self, text: str) -> None:
        self.state_label.setText(text)

    def _refresh_style(self) -> None:
        self.setProperty("selected", self._selected)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.update()

    def mousePressEvent(self, event) -> None:
        self._on_selected(self.stage_key)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:
        strategy = event.mimeData().text().strip()
        if self.editable and strategy in self.allowed_strategies:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        strategy = event.mimeData().text().strip()
        if self.editable and strategy in self.allowed_strategies:
            self._on_strategy_applied(self.stage_key, strategy)
            event.acceptProposedAction()
            return
        event.ignore()


class CustomModelCanvasWidget(QWidget):
    STRATEGIES = ["Freeze", "Unfreeze", "BN Tuning", "Norm Tuning", "LoRA", "DoRA", "TSA", "Adapter", "BitFit", "SSF"]
    GLOBAL_MODES = ["Manual", "Linear Probe", "Full Finetune"]

    FLOWS: dict[str, list[tuple[str, str]]] = {
        "efficientnet_v2_s": [
            ("input", "Input"),
            ("stem", "Stem"),
            *[(f"features.{i}", f"Features[{i}]") for i in range(8)],
            ("classifier", "Classifier"),
            ("output", "Output"),
        ],
        "resnet18": [
            ("input", "Input"),
            ("stem", "Stem / Conv1"),
            ("layer1", "Layer1"),
            ("layer2", "Layer2"),
            ("layer3", "Layer3"),
            ("layer4", "Layer4"),
            ("classifier", "FC / Classifier"),
            ("output", "Output"),
        ],
        "resnet50": [
            ("input", "Input"),
            ("stem", "Stem / Conv1"),
            ("layer1", "Layer1"),
            ("layer2", "Layer2"),
            ("layer3", "Layer3"),
            ("layer4", "Layer4"),
            ("classifier", "FC / Classifier"),
            ("output", "Output"),
        ],
        "convnext_tiny": [
            ("input", "Input"),
            ("stem", "Stem"),
            ("stage1", "Stage1"),
            ("stage2", "Stage2"),
            ("stage3", "Stage3"),
            ("stage4", "Stage4"),
            ("classifier", "Classifier"),
            ("output", "Output"),
        ],
        "mobilenet_v3_large": [
            ("input", "Input"),
            ("stem", "Stem"),
            ("stage1", "Feature Stage1"),
            ("stage2", "Feature Stage2"),
            ("stage3", "Feature Stage3"),
            ("stage4", "Feature Stage4"),
            ("classifier", "Classifier"),
            ("output", "Output"),
        ],
        "densenet121": [
            ("input", "Input"),
            ("stem", "Stem"),
            ("denseblock1", "DenseBlock1"),
            ("denseblock2", "DenseBlock2"),
            ("denseblock3", "DenseBlock3"),
            ("denseblock4", "DenseBlock4"),
            ("classifier", "Classifier"),
            ("output", "Output"),
        ],
    }
    STAGE_STATIC_INFO: dict[str, dict[str, dict[str, str]]] = {
        "efficientnet_v2_s": {
            "stem": {"module": "Conv+BN+SiLU", "repeat": "1", "kernel": "3", "stride": "2", "channels": "3 -> 24"},
            "features.0": {"module": "FusedMBConv", "repeat": "2", "kernel": "3", "stride": "1", "channels": "24 -> 24"},
            "features.1": {"module": "FusedMBConv", "repeat": "4", "kernel": "3", "stride": "2/1", "channels": "24 -> 48"},
            "features.2": {"module": "FusedMBConv", "repeat": "4", "kernel": "3", "stride": "2/1", "channels": "48 -> 64"},
            "features.3": {"module": "MBConv", "repeat": "6", "kernel": "3", "stride": "2/1", "channels": "64 -> 128"},
            "features.4": {"module": "MBConv", "repeat": "9", "kernel": "3", "stride": "1", "channels": "128 -> 160"},
            "features.5": {"module": "MBConv", "repeat": "15", "kernel": "3", "stride": "2/1", "channels": "160 -> 256"},
            "features.6": {"module": "Head Conv+BN+SiLU", "repeat": "1", "kernel": "1", "stride": "1", "channels": "256 -> 1280"},
            "features.7": {"module": "AvgPool", "repeat": "1", "kernel": "-", "stride": "-", "channels": "1280 -> 1280"},
            "classifier": {"module": "Dropout + Linear", "repeat": "1", "kernel": "-", "stride": "-", "channels": "1280 -> num_classes"},
        },
        "resnet18": {
            "stem": {"module": "Conv7x7+BN+ReLU+MaxPool", "repeat": "1", "kernel": "7", "stride": "2", "channels": "3 -> 64"},
            "layer1": {"module": "BasicBlock", "repeat": "2", "kernel": "3", "stride": "1", "channels": "64 -> 64"},
            "layer2": {"module": "BasicBlock", "repeat": "2", "kernel": "3", "stride": "2", "channels": "64 -> 128"},
            "layer3": {"module": "BasicBlock", "repeat": "2", "kernel": "3", "stride": "2", "channels": "128 -> 256"},
            "layer4": {"module": "BasicBlock", "repeat": "2", "kernel": "3", "stride": "2", "channels": "256 -> 512"},
            "classifier": {"module": "GlobalAvgPool + Linear", "repeat": "1", "kernel": "-", "stride": "-", "channels": "512 -> num_classes"},
        },
        "resnet50": {
            "stem": {"module": "Conv7x7+BN+ReLU+MaxPool", "repeat": "1", "kernel": "7", "stride": "2", "channels": "3 -> 64"},
            "layer1": {"module": "Bottleneck", "repeat": "3", "kernel": "1/3/1", "stride": "1", "channels": "64 -> 256"},
            "layer2": {"module": "Bottleneck", "repeat": "4", "kernel": "1/3/1", "stride": "2", "channels": "256 -> 512"},
            "layer3": {"module": "Bottleneck", "repeat": "6", "kernel": "1/3/1", "stride": "2", "channels": "512 -> 1024"},
            "layer4": {"module": "Bottleneck", "repeat": "3", "kernel": "1/3/1", "stride": "2", "channels": "1024 -> 2048"},
            "classifier": {"module": "GlobalAvgPool + Linear", "repeat": "1", "kernel": "-", "stride": "-", "channels": "2048 -> num_classes"},
        },
        "convnext_tiny": {
            "stem": {"module": "Conv4x4 stem", "repeat": "1", "kernel": "4", "stride": "4", "channels": "3 -> 96"},
            "stage1": {"module": "ConvNeXt Block", "repeat": "3", "kernel": "7(dw)", "stride": "1", "channels": "96 -> 96"},
            "stage2": {"module": "ConvNeXt Block", "repeat": "3", "kernel": "7(dw)", "stride": "2/1", "channels": "96 -> 192"},
            "stage3": {"module": "ConvNeXt Block", "repeat": "9", "kernel": "7(dw)", "stride": "2/1", "channels": "192 -> 384"},
            "stage4": {"module": "ConvNeXt Block", "repeat": "3", "kernel": "7(dw)", "stride": "2/1", "channels": "384 -> 768"},
            "classifier": {"module": "LayerNorm + Linear", "repeat": "1", "kernel": "-", "stride": "-", "channels": "768 -> num_classes"},
        },
        "mobilenet_v3_large": {
            "stem": {"module": "Conv+BN+HSwish", "repeat": "1", "kernel": "3", "stride": "2", "channels": "3 -> 16"},
            "stage1": {"module": "InvertedResidual", "repeat": "3", "kernel": "3/5", "stride": "1/2", "channels": "16 -> 40"},
            "stage2": {"module": "InvertedResidual", "repeat": "4", "kernel": "3", "stride": "1/2", "channels": "40 -> 80"},
            "stage3": {"module": "InvertedResidual", "repeat": "3", "kernel": "3", "stride": "1", "channels": "80 -> 112"},
            "stage4": {"module": "InvertedResidual", "repeat": "5", "kernel": "5", "stride": "1/2", "channels": "112 -> 160"},
            "classifier": {"module": "Pool + Linear", "repeat": "1", "kernel": "-", "stride": "-", "channels": "960 -> num_classes"},
        },
        "densenet121": {
            "stem": {"module": "Conv7x7+BN+ReLU+MaxPool", "repeat": "1", "kernel": "7", "stride": "2", "channels": "3 -> 64"},
            "denseblock1": {"module": "Dense Layer", "repeat": "6", "kernel": "3", "stride": "1", "channels": "64 -> 256"},
            "denseblock2": {"module": "Dense Layer", "repeat": "12", "kernel": "3", "stride": "1", "channels": "128 -> 512"},
            "denseblock3": {"module": "Dense Layer", "repeat": "24", "kernel": "3", "stride": "1", "channels": "256 -> 1024"},
            "denseblock4": {"module": "Dense Layer", "repeat": "16", "kernel": "3", "stride": "1", "channels": "512 -> 1024"},
            "classifier": {"module": "GlobalAvgPool + Linear", "repeat": "1", "kernel": "-", "stride": "-", "channels": "1024 -> num_classes"},
        },
    }

    def __init__(self, on_model_generated: Callable[[str], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec_path: Path | None = None
        self._updating = False
        self._on_model_generated = on_model_generated
        self._structure_source = "explicit"
        self._structure_confidence = "high"
        self._read_only_introspection = False
        self._selected_stage = "classifier"
        self._base_model = "efficientnet_v2_s"
        self._flow_nodes: list[tuple[str, str]] = []
        self._stage_strategy_overrides: dict[str, set[str]] = {}
        self._introspection_payload: dict[str, object] | None = None
        self._global_mode = "Manual"

        self.model_name_edit = QLineEdit("custom_canvas_model")
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems(custom_model_generator.list_supported_base_models())
        self.pretrained_checkbox = QCheckBox("Use pretrained weights")
        self.pretrained_checkbox.setChecked(True)
        self.gradcam_edit = QLineEdit()

        self.stage_frozen = QCheckBox("Frozen")
        self.stage_train_bn = QCheckBox("Train BN")
        self.stage_train_norm = QCheckBox("Norm Tuning")
        self.stage_method = QComboBox()
        self.stage_method.addItems(["none", "lora", "dora", "tsa", "adapter", "bitfit", "ssf"])
        self.stage_rank = QSpinBox()
        self.stage_rank.setRange(1, 128)
        self.stage_rank.setValue(8)
        self.stage_alpha = QDoubleSpinBox()
        self.stage_alpha.setRange(0.1, 512.0)
        self.stage_alpha.setDecimals(2)
        self.stage_alpha.setValue(16.0)
        self.stage_adapter_dim = QSpinBox()
        self.stage_adapter_dim.setRange(1, 1024)
        self.stage_adapter_dim.setValue(32)
        self.stage_bitfit_scope = QComboBox()
        self.stage_bitfit_scope.addItems(["all_bias", "norm_and_classifier_bias"])
        self.stage_ssf_scale = QDoubleSpinBox()
        self.stage_ssf_scale.setRange(-16.0, 16.0)
        self.stage_ssf_scale.setDecimals(3)
        self.stage_ssf_scale.setValue(1.0)
        self.stage_ssf_shift = QDoubleSpinBox()
        self.stage_ssf_shift.setRange(-16.0, 16.0)
        self.stage_ssf_shift.setDecimals(3)
        self.stage_ssf_shift.setValue(0.0)
        self.stage_rank_label = QLabel("LoRA/DoRA Rank")
        self.stage_alpha_label = QLabel("LoRA/DoRA Alpha")
        self.stage_adapter_dim_label = QLabel("Adapter Dim")
        self.stage_bitfit_scope_label = QLabel("BitFit Scope")
        self.stage_ssf_scale_label = QLabel("SSF Init Scale")
        self.stage_ssf_shift_label = QLabel("SSF Init Shift")
        self.stage_static_info_label = QLabel("Layer Static Info")
        self.stage_static_info_value = QLabel("-")
        self.stage_static_info_value.setWordWrap(True)
        self.stage_static_info_value.setProperty("muted", True)
        self.stage_use_custom_lr = QCheckBox("Use custom LR")
        self.stage_custom_lr = QDoubleSpinBox()
        self.stage_custom_lr.setRange(1e-7, 10.0)
        self.stage_custom_lr.setDecimals(7)
        self.stage_custom_lr.setSingleStep(1e-4)
        self.stage_custom_lr.setValue(1e-3)
        self.method_preview = QLabel("Method: baseline")
        self.method_preview.setProperty("muted", True)
        self.spec_path_label = QLabel("Spec File: (new unsaved spec)")
        self.structure_origin_label = QLabel("Structure Source: explicit (high confidence)")
        self.structure_origin_label.setProperty("muted", True)
        self.model_info_label = QLabel("Model Info: torchvision/efficientnet/efficientnet_v2_s | baseline | pretrained | generated")
        self.model_info_label.setProperty("muted", True)
        self.status_label = QLabel("Canvas edits structured spec only.")
        self.status_label.setProperty("muted", True)
        self.spec_summary = QPlainTextEdit()
        self.spec_summary.setReadOnly(True)
        self.spec_summary.setMaximumHeight(180)

        self.stage_frozen.toggled.connect(self._on_detail_changed)
        self.stage_train_bn.toggled.connect(self._on_detail_changed)
        self.stage_train_norm.toggled.connect(self._on_detail_changed)
        self.stage_method.currentTextChanged.connect(self._on_detail_changed)
        self.stage_rank.valueChanged.connect(self._on_detail_changed)
        self.stage_alpha.valueChanged.connect(self._on_detail_changed)
        self.stage_adapter_dim.valueChanged.connect(self._on_detail_changed)
        self.stage_bitfit_scope.currentTextChanged.connect(self._on_detail_changed)
        self.stage_ssf_scale.valueChanged.connect(self._on_detail_changed)
        self.stage_ssf_shift.valueChanged.connect(self._on_detail_changed)
        self.stage_use_custom_lr.toggled.connect(self._on_detail_changed)
        self.stage_custom_lr.valueChanged.connect(self._on_detail_changed)
        self.model_name_edit.textChanged.connect(self._on_state_changed)
        self.gradcam_edit.textChanged.connect(self._on_state_changed)
        self.pretrained_checkbox.toggled.connect(self._on_state_changed)
        self.base_model_combo.currentTextChanged.connect(self._on_base_model_changed)

        self.new_button = QPushButton("New Blank")
        self.open_model_button = QPushButton("Open Existing Model")
        self.show_legacy_checkbox = QCheckBox("Show Legacy Fallback")
        self.show_legacy_checkbox.setChecked(False)
        self.global_mode_combo = QComboBox()
        self.global_mode_combo.addItems(self.GLOBAL_MODES)
        self.load_button = QPushButton("Load Spec")
        self.save_button = QPushButton("Save")
        self.save_as_button = QPushButton("Save As")
        self.generate_button = QPushButton("Generate Model")
        self.new_button.clicked.connect(self.new_spec)
        self.open_model_button.clicked.connect(self.open_existing_model)
        self.load_button.clicked.connect(self.load_spec)
        self.save_button.clicked.connect(self.save_spec)
        self.save_as_button.clicked.connect(self.save_spec_as)
        self.generate_button.clicked.connect(self.generate_model)
        self.global_mode_combo.currentTextChanged.connect(self._on_global_mode_changed)

        self.palette = StrategyPaletteList()
        self.palette.setObjectName("StrategyPaletteList")
        self.palette.addItems(self.STRATEGIES)
        self.palette.setMaximumWidth(150)

        self.canvas_container = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(4)
        self.nodes: dict[str, CanvasStageNode] = {}
        self.editable_stages: set[str] = set()
        self.state: dict[str, dict[str, object]] = {}

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setWidget(self.canvas_container)
        canvas_scroll.setFrameShape(QScrollArea.NoFrame)

        left_panel = QWidget()
        left_layout = QHBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.palette)
        left_layout.addWidget(canvas_scroll, stretch=1)

        detail_group = QGroupBox("Stage Details")
        detail_form = QFormLayout(detail_group)
        self.stage_title = QLabel("-")
        self.stage_ops_label = QLabel("-")
        self.stage_ops_label.setProperty("muted", True)
        detail_form.addRow("Selected Stage", self.stage_title)
        detail_form.addRow("Allowed Ops", self.stage_ops_label)
        detail_form.addRow("", self.stage_frozen)
        detail_form.addRow("", self.stage_train_bn)
        detail_form.addRow("", self.stage_train_norm)
        detail_form.addRow("Stage Method", self.stage_method)
        detail_form.addRow(self.stage_rank_label, self.stage_rank)
        detail_form.addRow(self.stage_alpha_label, self.stage_alpha)
        detail_form.addRow(self.stage_adapter_dim_label, self.stage_adapter_dim)
        detail_form.addRow(self.stage_bitfit_scope_label, self.stage_bitfit_scope)
        detail_form.addRow(self.stage_ssf_scale_label, self.stage_ssf_scale)
        detail_form.addRow(self.stage_ssf_shift_label, self.stage_ssf_shift)
        detail_form.addRow("", self.stage_use_custom_lr)
        detail_form.addRow("Custom LR", self.stage_custom_lr)
        detail_form.addRow(self.stage_static_info_label, self.stage_static_info_value)
        detail_form.addRow("Grad-CAM", self.gradcam_edit)
        detail_form.addRow("Method", self.method_preview)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(detail_group)
        splitter.setSizes([760, 330])

        workflow_row = QHBoxLayout()
        workflow_row.setContentsMargins(0, 0, 0, 0)
        workflow_row.setSpacing(8)
        workflow_row.addWidget(self.new_button)
        workflow_row.addWidget(self.load_button)
        workflow_row.addWidget(self.open_model_button)
        workflow_row.addWidget(self.show_legacy_checkbox)
        workflow_row.addStretch(1)

        settings_row = QHBoxLayout()
        settings_row.setContentsMargins(0, 0, 0, 0)
        settings_row.setSpacing(8)
        settings_row.addWidget(QLabel("Model Name"))
        settings_row.addWidget(self.model_name_edit, stretch=1)
        settings_row.addWidget(QLabel("Base"))
        settings_row.addWidget(self.base_model_combo)
        settings_row.addWidget(self.pretrained_checkbox)
        settings_row.addWidget(QLabel("Global Mode"))
        settings_row.addWidget(self.global_mode_combo)

        top_controls = QVBoxLayout()
        top_controls.setContentsMargins(0, 0, 0, 0)
        top_controls.setSpacing(8)
        top_controls.addLayout(workflow_row)
        top_controls.addLayout(settings_row)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        actions_row.addWidget(self.save_button)
        actions_row.addWidget(self.save_as_button)
        actions_row.addWidget(self.generate_button)

        meta_left_column = QVBoxLayout()
        meta_left_column.setContentsMargins(0, 0, 0, 0)
        meta_left_column.setSpacing(4)
        meta_left_column.addWidget(self.spec_path_label)
        meta_left_column.addWidget(self.structure_origin_label)
        meta_left_column.addWidget(self.model_info_label)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(12)
        meta_row.addLayout(meta_left_column, stretch=1)
        meta_row.addLayout(actions_row, stretch=0)

        layout = QVBoxLayout(self)
        layout.addLayout(top_controls)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(meta_row)
        layout.addWidget(self.spec_summary)
        layout.addWidget(self.status_label)

        self.new_spec()

    def _existing_model_choices(self) -> tuple[list[str], dict[str, str]]:
        models = sort_model_names_for_ui(
            discover_model_names_generated_first(include_legacy_fallback=self.show_legacy_checkbox.isChecked())
        )
        display_to_model: dict[str, str] = {}
        for model_name in models:
            display = model_display_label(model_name, include_name=True)
            display_to_model[display] = model_name
        return list(display_to_model.keys()), display_to_model

    def _stage_to_feature_index(self, stage_key: str) -> int | None:
        if stage_key.startswith("features."):
            try:
                return int(stage_key.split(".")[1])
            except Exception:
                return None
        return None

    def _set_structure_origin(self, source: str, confidence: str) -> None:
        self._structure_source = source
        self._structure_confidence = confidence
        self.structure_origin_label.setText(f"Structure Source: {source} ({confidence} confidence)")

    def _set_editing_enabled(self, enabled: bool) -> None:
        self._read_only_introspection = not enabled
        self.palette.setEnabled(enabled)
        self.stage_frozen.setEnabled(enabled)
        self.stage_train_bn.setEnabled(enabled)
        self.stage_train_norm.setEnabled(enabled)
        self.stage_method.setEnabled(enabled)
        self.stage_rank.setEnabled(enabled)
        self.stage_alpha.setEnabled(enabled)
        self.stage_adapter_dim.setEnabled(enabled)
        self.stage_bitfit_scope.setEnabled(enabled)
        self.stage_ssf_scale.setEnabled(enabled)
        self.stage_ssf_shift.setEnabled(enabled)
        self.stage_use_custom_lr.setEnabled(enabled)
        self.stage_custom_lr.setEnabled(enabled and self.stage_use_custom_lr.isChecked())
        self.model_name_edit.setEnabled(enabled)
        self.base_model_combo.setEnabled(enabled)
        self.pretrained_checkbox.setEnabled(enabled)
        self.global_mode_combo.setEnabled(enabled)
        self.gradcam_edit.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_as_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)

    def _refresh_model_info_label(self, *, model_name_hint: str | None = None, method_hint: str | None = None) -> None:
        provider = "torchvision"
        if self._base_model.startswith("resnet"):
            family = "resnet"
        elif self._base_model.startswith("efficientnet"):
            family = "efficientnet"
        elif self._base_model.startswith("convnext"):
            family = "convnext"
        elif self._base_model.startswith("mobilenet"):
            family = "mobilenet_v3"
        elif self._base_model.startswith("densenet"):
            family = "densenet"
        else:
            family = self._base_model
        variant = self._base_model
        method = method_hint or "baseline"
        pretrained = "pretrained" if self.pretrained_checkbox.isChecked() else "scratch"
        source = self._structure_source
        if isinstance(model_name_hint, str) and model_name_hint.strip():
            info = model_catalog_entry(model_name_hint.strip())
            provider = str(info.get("provider", provider))
            family = str(info.get("family", family))
            variant = str(info.get("variant", variant))
            method = str(info.get("method_type", method))
            pre = info.get("pretrained")
            pretrained = "pretrained" if pre is True else ("scratch" if pre is False else pretrained)
            source = str(info.get("source", source))
        self.model_info_label.setText(
            f"Model Info: {provider}/{family}/{variant} | {method} | {pretrained} | source={source}"
        )

    def _supported_methods(self) -> set[str]:
        try:
            return set(custom_model_generator.supported_methods_for_base(self._base_model))
        except Exception:
            return {"baseline"}

    def _strategy_label_from_safe_op(self, op: str) -> str | None:
        normalized = str(op).strip().lower()
        mapping = {
            "freeze": "Freeze",
            "unfreeze": "Unfreeze",
            "bn_tuning": "BN Tuning",
            "norm_tuning": "Norm Tuning",
            "lora": "LoRA",
            "dora": "DoRA",
            "tsa": "TSA",
            "adapter": "Adapter",
            "bitfit": "BitFit",
            "ssf": "SSF",
        }
        return mapping.get(normalized)

    def _method_from_strategy_label(self, label: str) -> str | None:
        mapping = {
            "LoRA": "lora",
            "DoRA": "dora",
            "TSA": "tsa",
            "Adapter": "adapter",
            "BitFit": "bitfit",
            "SSF": "ssf",
        }
        return mapping.get(label)

    def _configure_default_flow_for_base(self, base_model: str) -> None:
        self._flow_nodes = list(self.FLOWS[base_model])
        if base_model == "efficientnet_v2_s":
            self.editable_stages = {"stem", *[f"features.{i}" for i in range(8)], "classifier"}
        elif base_model in {"resnet18", "resnet50"}:
            self.editable_stages = {"stem", "layer1", "layer2", "layer3", "layer4", "classifier"}
        elif base_model in {"convnext_tiny", "mobilenet_v3_large"}:
            self.editable_stages = {"stem", "stage1", "stage2", "stage3", "stage4", "classifier"}
        elif base_model == "densenet121":
            self.editable_stages = {"stem", "denseblock1", "denseblock2", "denseblock3", "denseblock4", "classifier"}
        else:
            self.editable_stages = {key for key, _ in self._flow_nodes if key not in {"input", "output"}}
        self._stage_strategy_overrides = {}
        self._introspection_payload = None

    def _configure_introspected_flow(self, structure_payload: dict[str, object]) -> None:
        raw_stages = structure_payload.get("stages")
        if not isinstance(raw_stages, list):
            raise ValueError("Invalid introspected structure payload: missing stages list.")
        flow_nodes: list[tuple[str, str]] = []
        editable_stages: set[str] = set()
        strategy_overrides: dict[str, set[str]] = {}
        for item in raw_stages:
            if not isinstance(item, dict):
                continue
            stage_key = str(item.get("key", "")).strip()
            title = str(item.get("title", stage_key)).strip() or stage_key
            if not stage_key:
                continue
            flow_nodes.append((stage_key, title))
            if bool(item.get("editable", False)):
                editable_stages.add(stage_key)
            safe_ops_raw = item.get("safe_operations", [])
            safe_ops = safe_ops_raw if isinstance(safe_ops_raw, list) else []
            strategy_labels = {label for op in safe_ops if (label := self._strategy_label_from_safe_op(str(op))) is not None}
            strategy_overrides[stage_key] = strategy_labels
        if not flow_nodes:
            raise ValueError("Introspection returned no stage nodes.")
        self._flow_nodes = flow_nodes
        self.editable_stages = editable_stages
        self._stage_strategy_overrides = strategy_overrides
        self._introspection_payload = structure_payload

    def _allowed_strategies_for_stage(self, stage_key: str) -> set[str]:
        if stage_key not in self.editable_stages:
            return set()
        override = self._stage_strategy_overrides.get(stage_key)
        if isinstance(override, set):
            return set(override)
        supported_methods = self._supported_methods()
        allowed = {"Freeze", "Unfreeze"}
        if "bn_tuning" in supported_methods and stage_key != "classifier":
            allowed.add("BN Tuning")
        if "norm_tuning" in supported_methods and stage_key != "classifier":
            allowed.add("Norm Tuning")

        peft_targetable: set[str]
        if self._base_model == "efficientnet_v2_s":
            peft_targetable = {"features.5", "features.6", "features.7", "classifier"}
        elif self._base_model in {"resnet18", "resnet50"}:
            peft_targetable = {"layer3", "layer4", "classifier"}
        elif self._base_model in {"convnext_tiny", "mobilenet_v3_large"}:
            peft_targetable = {"stage3", "stage4", "classifier"}
        elif self._base_model == "densenet121":
            peft_targetable = {"denseblock3", "denseblock4", "classifier"}
        else:
            peft_targetable = {"classifier"}

        if stage_key in peft_targetable:
            if "lora" in supported_methods:
                allowed.add("LoRA")
            if "tsa" in supported_methods:
                allowed.add("TSA")
            if "dora" in supported_methods:
                allowed.add("DoRA")
            if "adapter" in supported_methods:
                allowed.add("Adapter")
            if "bitfit" in supported_methods:
                allowed.add("BitFit")
            if "ssf" in supported_methods:
                allowed.add("SSF")
        elif stage_key == "classifier" and "bitfit" in supported_methods:
            allowed.add("BitFit")
        return allowed

    def _build_flow(self) -> None:
        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.nodes.clear()

        if not self._flow_nodes:
            self._configure_default_flow_for_base(self._base_model)

        for stage_key, _ in self._flow_nodes:
            if stage_key not in self.state:
                self.state[stage_key] = {
                    "frozen": True,
                    "train_bn": False,
                    "train_norm": False,
                    "stage_method": "none",
                    "rank": 8,
                    "alpha": 16.0,
                    "adapter_dim": 32,
                    "bitfit_scope": "all_bias",
                    "ssf_scale": 1.0,
                    "ssf_shift": 0.0,
                    "lr_override_enabled": False,
                    "lr_override": 1e-3,
                }
        for stage in list(self.state.keys()):
            if stage not in {key for key, _ in self._flow_nodes}:
                self.state.pop(stage, None)

        if "classifier" in self.state and self.state["classifier"]["frozen"] is True:
            self.state["classifier"]["frozen"] = False

        flow = self._flow_nodes
        for idx, (stage_key, title) in enumerate(flow):
            node = CanvasStageNode(
                stage_key,
                title,
                editable=stage_key in self.editable_stages,
                allowed_strategies=self._allowed_strategies_for_stage(stage_key),
                on_selected=self._select_stage,
                on_strategy_applied=self._apply_strategy,
            )
            self.nodes[stage_key] = node
            self.canvas_layout.addWidget(node)
            if idx < len(flow) - 1:
                arrow = QLabel("v")
                arrow.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                arrow.setProperty("muted", True)
                self.canvas_layout.addWidget(arrow)
        self.canvas_layout.addStretch(1)

        if self._selected_stage not in self.editable_stages:
            self._selected_stage = "classifier" if "classifier" in self.editable_stages else next(iter(self.editable_stages), "")

    def _select_stage(self, stage_key: str) -> None:
        self._selected_stage = stage_key
        for key, node in self.nodes.items():
            node.set_selected(key == stage_key)
        state = self.state.get(stage_key, {})
        self._updating = True
        try:
            self.stage_title.setText(stage_key)
            self._refresh_stage_static_info(stage_key)
            editable = stage_key in self.editable_stages and not self._read_only_introspection
            allowed_labels = sorted(self._allowed_strategies_for_stage(stage_key))
            self.stage_ops_label.setText(", ".join(allowed_labels) if allowed_labels else "None")
            allowed_stage_methods = ["none"] + [
                method for method in (self._method_from_strategy_label(label) for label in allowed_labels) if method is not None
            ]
            existing_method = str(state.get("stage_method", "none")).strip().lower()
            self.stage_method.blockSignals(True)
            self.stage_method.clear()
            self.stage_method.addItems(allowed_stage_methods)
            self.stage_method.blockSignals(False)
            self.stage_frozen.setEnabled(editable)
            self.stage_train_bn.setEnabled(editable)
            self.stage_train_norm.setEnabled(editable)
            self.stage_method.setEnabled(editable)
            self.stage_rank.setEnabled(editable)
            self.stage_alpha.setEnabled(editable)
            self.stage_adapter_dim.setEnabled(editable)
            self.stage_bitfit_scope.setEnabled(editable)
            self.stage_ssf_scale.setEnabled(editable)
            self.stage_ssf_shift.setEnabled(editable)
            self.stage_use_custom_lr.setEnabled(editable)
            self.stage_frozen.setChecked(bool(state.get("frozen", True)))
            self.stage_train_bn.setChecked(bool(state.get("train_bn", False)))
            self.stage_train_norm.setChecked(bool(state.get("train_norm", False)))
            stage_method = existing_method if existing_method in set(allowed_stage_methods) else "none"
            self.stage_method.setCurrentText(stage_method)
            self.stage_rank.setValue(int(state.get("rank", 8)))
            self.stage_alpha.setValue(float(state.get("alpha", 16.0)))
            self.stage_adapter_dim.setValue(int(state.get("adapter_dim", 32)))
            self.stage_bitfit_scope.setCurrentText(str(state.get("bitfit_scope", "all_bias")))
            self.stage_ssf_scale.setValue(float(state.get("ssf_scale", 1.0)))
            self.stage_ssf_shift.setValue(float(state.get("ssf_shift", 0.0)))
            self.stage_use_custom_lr.setChecked(bool(state.get("lr_override_enabled", False)))
            self.stage_custom_lr.setValue(float(state.get("lr_override", 1e-3)))
            self.stage_custom_lr.setEnabled(editable and bool(state.get("lr_override_enabled", False)))
            self._refresh_method_parameter_visibility(stage_method)
        finally:
            self._updating = False

    def _refresh_method_parameter_visibility(self, stage_method: str) -> None:
        method = str(stage_method).strip().lower()
        show_rank_alpha = method in {"lora", "dora"}
        show_adapter = method == "adapter"
        show_bitfit = method == "bitfit"
        show_ssf = method == "ssf"
        self.stage_rank_label.setVisible(show_rank_alpha)
        self.stage_alpha_label.setVisible(show_rank_alpha)
        self.stage_adapter_dim_label.setVisible(show_adapter)
        self.stage_bitfit_scope_label.setVisible(show_bitfit)
        self.stage_ssf_scale_label.setVisible(show_ssf)
        self.stage_ssf_shift_label.setVisible(show_ssf)
        self.stage_rank.setVisible(show_rank_alpha)
        self.stage_alpha.setVisible(show_rank_alpha)
        self.stage_adapter_dim.setVisible(show_adapter)
        self.stage_bitfit_scope.setVisible(show_bitfit)
        self.stage_ssf_scale.setVisible(show_ssf)
        self.stage_ssf_shift.setVisible(show_ssf)
        self.stage_custom_lr.setEnabled(self.stage_use_custom_lr.isChecked() and self.stage_use_custom_lr.isEnabled())

    def _refresh_stage_static_info(self, stage_key: str) -> None:
        base = str(self._base_model).strip().lower()
        stage_info = self.STAGE_STATIC_INFO.get(base, {}).get(stage_key, {})
        if isinstance(stage_info, dict) and stage_info:
            self.stage_static_info_value.setText(
                " | ".join(
                    [
                        f"Module: {stage_info.get('module', '-')}",
                        f"Repeat: {stage_info.get('repeat', '-')}",
                        f"Kernel: {stage_info.get('kernel', '-')}",
                        f"Stride: {stage_info.get('stride', '-')}",
                        f"Channels: {stage_info.get('channels', '-')}",
                    ]
                )
            )
            return
        self.stage_static_info_value.setText("No static profile available for this stage.")

    def _refresh_gradcam_default_for_base(self) -> None:
        try:
            method = self._derive_method()
        except Exception:
            method = "baseline"
        targets = custom_model_generator._default_gradcam_targets(self._base_model, method)
        self.gradcam_edit.setText(",".join(targets))

    def _resolve_spec_output_path(
        self,
        model_name: str,
        *,
        selected_path: Path | str | None = None,
        prefer_current: bool = False,
    ) -> tuple[Path, str | None]:
        if selected_path is not None:
            target = custom_model_generator.canonicalize_spec_path_for_model_name(model_name, selected_path)
            try:
                requested = Path(selected_path).expanduser().resolve()
            except Exception:
                requested = Path(selected_path)
            note = None if target == requested else f"Spec filename was aligned to model name '{model_name}'."
            return target, note
        if prefer_current and self._spec_path is not None and custom_model_generator.spec_path_matches_model_name(self._spec_path, model_name):
            return self._spec_path.expanduser().resolve(), None
        target = custom_model_generator.default_spec_path_for_model_name(model_name).resolve()
        if prefer_current and self._spec_path is not None:
            return target, "Model name changed, so the spec was redirected to a matching file."
        return target, None

    def _on_global_mode_changed(self, value: str) -> None:
        if self._updating or self._read_only_introspection:
            return
        mode = str(value).strip()
        self._global_mode = mode if mode in set(self.GLOBAL_MODES) else "Manual"
        if self._global_mode == "Manual":
            return
        if self._global_mode == "Linear Probe":
            for key in self.editable_stages:
                self.state[key]["frozen"] = key != "classifier"
                self.state[key]["train_bn"] = False
                self.state[key]["train_norm"] = False
                self.state[key]["stage_method"] = "none"
                self.state[key]["lr_override_enabled"] = False
        elif self._global_mode == "Full Finetune":
            for key in self.editable_stages:
                self.state[key]["frozen"] = False
                self.state[key]["train_bn"] = False
                self.state[key]["train_norm"] = False
                self.state[key]["stage_method"] = "none"
                self.state[key]["lr_override_enabled"] = False
        self._select_stage(self._selected_stage if self._selected_stage in self.editable_stages else "classifier")
        self._on_state_changed()

    def _apply_strategy(self, stage_key: str, strategy: str) -> None:
        if self._read_only_introspection:
            return
        if stage_key not in self.editable_stages:
            return
        strategy = strategy.strip()
        if strategy == "Freeze":
            self.state[stage_key]["frozen"] = True
        elif strategy == "Unfreeze":
            self.state[stage_key]["frozen"] = False
        elif strategy == "BN Tuning":
            self.state[stage_key]["train_bn"] = True
        elif strategy == "Norm Tuning":
            self.state[stage_key]["train_norm"] = True
        elif strategy in {"LoRA", "DoRA", "TSA", "Adapter", "BitFit", "SSF"}:
            method = strategy.lower()
            if method not in self._supported_methods():
                self.status_label.setText(f"{strategy} is not supported for base model '{self._base_model}'.")
                return
            for key in self.editable_stages:
                if self.state[key]["stage_method"] != "none" and self.state[key]["stage_method"] != method:
                    self.state[key]["stage_method"] = "none"
            self.state[stage_key]["stage_method"] = method
        self._select_stage(stage_key)
        self._global_mode = "Manual"
        self.global_mode_combo.setCurrentText("Manual")
        self._on_state_changed()

    def _on_detail_changed(self) -> None:
        if self._read_only_introspection:
            return
        if self._updating or self._selected_stage not in self.editable_stages:
            return
        stage_method = self.stage_method.currentText().strip().lower()
        if stage_method not in {"none", *self._supported_methods()}:
            stage_method = "none"
        if stage_method in {"lora", "dora", "tsa", "adapter", "bitfit", "ssf"}:
            for key in self.editable_stages:
                if self.state[key]["stage_method"] != "none" and self.state[key]["stage_method"] != stage_method:
                    self.state[key]["stage_method"] = "none"
        self.state[self._selected_stage].update(
            {
                "frozen": self.stage_frozen.isChecked(),
                "train_bn": self.stage_train_bn.isChecked(),
                "train_norm": self.stage_train_norm.isChecked(),
                "stage_method": stage_method,
                "rank": int(self.stage_rank.value()),
                "alpha": float(self.stage_alpha.value()),
                "adapter_dim": int(self.stage_adapter_dim.value()),
                "bitfit_scope": self.stage_bitfit_scope.currentText().strip().lower(),
                "ssf_scale": float(self.stage_ssf_scale.value()),
                "ssf_shift": float(self.stage_ssf_shift.value()),
                "lr_override_enabled": bool(self.stage_use_custom_lr.isChecked()),
                "lr_override": float(self.stage_custom_lr.value()),
            }
        )
        self._refresh_method_parameter_visibility(stage_method)
        self._global_mode = "Manual"
        self.global_mode_combo.setCurrentText("Manual")
        self._on_state_changed()

    def _derive_method(self) -> str:
        supported_methods = self._supported_methods()
        if self._global_mode == "Linear Probe":
            return "baseline"
        if self._global_mode == "Full Finetune" and "full_finetune" in supported_methods:
            return "full_finetune"

        method_stages = [key for key in self.editable_stages if self.state[key]["stage_method"] != "none"]
        stage_method = next((self.state[key]["stage_method"] for key in method_stages), None)
        if stage_method and stage_method in supported_methods:
            return str(stage_method)

        all_unfrozen = all(not bool(self.state[key]["frozen"]) for key in self.editable_stages)
        train_bn_any = any(bool(self.state[key]["train_bn"]) for key in self.editable_stages)
        train_norm_any = any(bool(self.state[key].get("train_norm", False)) for key in self.editable_stages)
        if all_unfrozen and "full_finetune" in supported_methods:
            return "full_finetune"
        if self._base_model == "efficientnet_v2_s" and train_bn_any and "bn_tuning" in supported_methods:
            unfrozen_feature = sorted(
                idx
                for stage in self.editable_stages
                for idx in ([self._stage_to_feature_index(stage)] if self._stage_to_feature_index(stage) is not None else [])
                if not bool(self.state[stage]["frozen"])
            )
            if unfrozen_feature == [7] and "bn_last1" in supported_methods:
                return "bn_last1"
            if unfrozen_feature == [6, 7] and "bn_last2" in supported_methods:
                return "bn_last2"
            return "bn_tuning"
        if train_bn_any and "bn_tuning" in supported_methods:
            return "bn_tuning"
        if train_norm_any and "norm_tuning" in supported_methods:
            return "norm_tuning"
        return "baseline"

    def _derive_spec(self):
        method = self._derive_method()
        spec = custom_model_generator.build_preset_spec(
            model_name=self.model_name_edit.text().strip() or "custom_canvas_model",
            base_model=self._base_model,
            method_type=method,
        )
        payload = custom_model_generator.spec_to_dict(spec)
        payload["pretrained"] = bool(self.pretrained_checkbox.isChecked())
        train_bn_any = any(bool(self.state[key]["train_bn"]) for key in self.editable_stages)
        train_norm_any = any(bool(self.state[key].get("train_norm", False)) for key in self.editable_stages)
        payload["train_bn"] = train_bn_any
        payload["train_norm"] = train_norm_any
        payload["stage_lr_overrides"] = {
            key: float(self.state[key].get("lr_override", 0.0))
            for key in sorted(self.editable_stages)
            if bool(self.state[key].get("lr_override_enabled", False)) and float(self.state[key].get("lr_override", 0.0)) > 0.0
        }
        unfrozen_feature_stages = sorted(
            idx
            for key in self.editable_stages
            for idx in ([self._stage_to_feature_index(key)] if self._stage_to_feature_index(key) is not None else [])
            if not bool(self.state[key]["frozen"])
        )

        if self._base_model == "efficientnet_v2_s":
            payload["unfreeze_stages"] = list(unfrozen_feature_stages)
            if train_bn_any:
                if unfrozen_feature_stages in ([7], [6, 7]):
                    payload["freeze_strategy"] = "bn_tuning_with_last_stages"
                else:
                    payload["freeze_strategy"] = "bn_tuning"
            elif train_norm_any:
                payload["freeze_strategy"] = "norm_tuning"
            elif method in {"lora", "dora", "tsa", "adapter", "ssf"}:
                payload["freeze_strategy"] = "frozen_backbone_peft"
            elif method == "bitfit":
                payload["freeze_strategy"] = "bias_tuning"
            elif all(not bool(self.state[key]["frozen"]) for key in self.editable_stages):
                payload["freeze_strategy"] = "full_finetune"

        peft_stages = [key for key in self.editable_stages if self.state[key]["stage_method"] != "none"]
        if method in {"lora", "dora", "tsa", "adapter", "bitfit", "ssf"}:
            if self._base_model == "efficientnet_v2_s":
                payload["peft_targets"] = {
                    "feature_stages": sorted(
                        idx for key in peft_stages for idx in ([self._stage_to_feature_index(key)] if self._stage_to_feature_index(key) is not None else [])
                    ),
                    "layer_keys": [],
                    "classifier": "classifier" in peft_stages,
                }
            else:
                payload["peft_targets"] = {
                    "feature_stages": [],
                    "layer_keys": sorted(key for key in peft_stages if key != "classifier"),
                    "classifier": "classifier" in peft_stages,
                }
            if method in {"lora", "dora"} and peft_stages:
                ref = self.state[peft_stages[0]]
                payload["peft_params"] = {"rank": int(ref["rank"]), "alpha": float(ref["alpha"])}
            elif method in {"lora", "dora"}:
                payload["peft_params"] = {"rank": int(self.stage_rank.value()), "alpha": float(self.stage_alpha.value())}
            elif method == "adapter":
                ref = self.state[peft_stages[0]] if peft_stages else self.state.get("classifier", {})
                payload["peft_params"] = {"bottleneck_dim": int(ref.get("adapter_dim", self.stage_adapter_dim.value()))}
            elif method == "bitfit":
                ref = self.state[peft_stages[0]] if peft_stages else self.state.get("classifier", {})
                payload["peft_params"] = {"scope": str(ref.get("bitfit_scope", self.stage_bitfit_scope.currentText())).strip().lower()}
            elif method == "ssf":
                ref = self.state[peft_stages[0]] if peft_stages else self.state.get("classifier", {})
                payload["peft_params"] = {
                    "init_scale": float(ref.get("ssf_scale", self.stage_ssf_scale.value())),
                    "init_shift": float(ref.get("ssf_shift", self.stage_ssf_shift.value())),
                }

        hints = [item.strip() for item in self.gradcam_edit.text().split(",") if item.strip()]
        if hints:
            payload["gradcam_target_hint"] = hints
        return custom_model_generator.spec_from_dict(payload)

    def _apply_spec(self, spec) -> None:
        self._updating = True
        try:
            self.model_name_edit.setText(spec.model_name)
            self.pretrained_checkbox.setChecked(bool(getattr(spec, "pretrained", True)))
            self._base_model = spec.base_model
            self.base_model_combo.setCurrentText(spec.base_model)
            self._configure_default_flow_for_base(self._base_model)
            self._build_flow()
            self.gradcam_edit.setText(",".join(spec.gradcam_target_hint))
            self._global_mode = "Manual"
            self.global_mode_combo.setCurrentText("Manual")

            for key in self.editable_stages:
                self.state[key] = {
                    "frozen": True,
                    "train_bn": False,
                    "train_norm": False,
                    "stage_method": "none",
                    "rank": 8,
                    "alpha": 16.0,
                    "adapter_dim": 32,
                    "bitfit_scope": "all_bias",
                    "ssf_scale": 1.0,
                    "ssf_shift": 0.0,
                    "lr_override_enabled": False,
                    "lr_override": 1e-3,
                }
            self.state["classifier"]["frozen"] = False
            stage_lr_overrides = getattr(spec, "stage_lr_overrides", {})
            if not isinstance(stage_lr_overrides, dict):
                stage_lr_overrides = {}
            for key, value in stage_lr_overrides.items():
                if key not in self.state:
                    continue
                try:
                    lr_value = float(value)
                except Exception:
                    continue
                if lr_value <= 0:
                    continue
                self.state[key]["lr_override_enabled"] = True
                self.state[key]["lr_override"] = lr_value

            method = spec.method_type
            train_bn_enabled = bool(getattr(spec, "train_bn", False))
            train_norm_enabled = bool(getattr(spec, "train_norm", False))
            freeze_strategy = str(getattr(spec, "freeze_strategy", "manual")).strip().lower()
            if freeze_strategy == "full_finetune":
                for key in self.editable_stages:
                    self.state[key]["frozen"] = False
                self._global_mode = "Full Finetune"
                self.global_mode_combo.setCurrentText("Full Finetune")
            elif freeze_strategy == "linear_probe" and method == "baseline":
                self._global_mode = "Linear Probe"
                self.global_mode_combo.setCurrentText("Linear Probe")
            if train_bn_enabled:
                for key in self.editable_stages:
                    self.state[key]["train_bn"] = train_bn_enabled
            if train_norm_enabled:
                for key in self.editable_stages:
                    self.state[key]["train_norm"] = train_norm_enabled
            for stage_idx in getattr(spec, "unfreeze_stages", []):
                key = f"features.{int(stage_idx)}"
                if key in self.state:
                    self.state[key]["frozen"] = False
            if method in {"lora", "dora", "tsa", "adapter", "bitfit", "ssf"}:
                targets = spec.peft_targets if isinstance(spec.peft_targets, dict) else {}
                if self._base_model == "efficientnet_v2_s":
                    for idx in targets.get("feature_stages", []):
                        key = f"features.{int(idx)}"
                        if key in self.state:
                            self.state[key]["stage_method"] = method
                else:
                    for layer_key in targets.get("layer_keys", []):
                        key = str(layer_key)
                        if key in self.state:
                            self.state[key]["stage_method"] = method
                if bool(targets.get("classifier", False)):
                    self.state["classifier"]["stage_method"] = method
                params = spec.peft_params if isinstance(spec.peft_params, dict) else {}
                for key in self.editable_stages:
                    if self.state[key]["stage_method"] != "none":
                        self.state[key]["rank"] = int(params.get("rank", 8))
                        self.state[key]["alpha"] = float(params.get("alpha", 16.0))
                        self.state[key]["adapter_dim"] = int(params.get("bottleneck_dim", 32))
                        self.state[key]["bitfit_scope"] = str(params.get("scope", "all_bias")).strip().lower() or "all_bias"
                        self.state[key]["ssf_scale"] = float(params.get("init_scale", 1.0))
                        self.state[key]["ssf_shift"] = float(params.get("init_shift", 0.0))
        finally:
            self._updating = False
        self._select_stage(self._selected_stage if self._selected_stage in self.editable_stages else "classifier")
        self._on_state_changed()

    def _on_state_changed(self) -> None:
        if self._updating:
            return
        for key, node in self.nodes.items():
            if key not in self.editable_stages:
                node.set_state_text("Fixed")
                continue
            stage = self.state[key]
            tags = ["Frozen" if stage["frozen"] else "Unfrozen"]
            if stage["train_bn"]:
                tags.append("Train BN")
            if stage.get("train_norm", False):
                tags.append("Norm")
            if stage.get("stage_method", "none") != "none":
                tags.append(str(stage.get("stage_method", "none")).upper())
            node.set_state_text(" | ".join(tags))
        if self._read_only_introspection:
            self.method_preview.setText("Method: inspection-only")
            summary_payload = self._introspection_payload if isinstance(self._introspection_payload, dict) else {}
            self.spec_summary.setPlainText(json.dumps(summary_payload, indent=2, sort_keys=True))
            self._refresh_model_info_label(method_hint="inspection")
            return
        try:
            spec = self._derive_spec()
            self.method_preview.setText(f"Method: {spec.method_type}")
            self.spec_summary.setPlainText(json.dumps(custom_model_generator.spec_to_dict(spec), indent=2, sort_keys=True))
            self._refresh_model_info_label(method_hint=spec.method_type)
        except Exception as exc:
            self.method_preview.setText("Method: invalid")
            self.spec_summary.setPlainText(f"Invalid spec: {exc}")
            self._refresh_model_info_label(method_hint="invalid")

    def _on_base_model_changed(self, value: str) -> None:
        if self._updating:
            return
        if self._read_only_introspection:
            return
        self._base_model = value
        self._configure_default_flow_for_base(self._base_model)
        self.state = {}
        self._build_flow()
        self._select_stage("classifier")
        self._global_mode = "Manual"
        self.global_mode_combo.setCurrentText("Manual")
        self._refresh_gradcam_default_for_base()
        self._on_state_changed()

    def new_spec(self) -> None:
        spec = custom_model_generator.build_preset_spec(
            model_name="custom_canvas_model",
            base_model="efficientnet_v2_s",
            method_type="baseline",
        )
        self._spec_path = None
        self.spec_path_label.setText("Spec File: (new unsaved spec)")
        self._set_structure_origin("explicit", "high")
        self._set_editing_enabled(True)
        self.status_label.setText("New canvas spec initialized.")
        self._apply_spec(spec)

    def load_spec(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Custom Model Spec", str(custom_model_generator.SPEC_DIR), "Spec JSON (*.json)")
        if not selected_path:
            return
        try:
            spec = custom_model_generator.load_spec_file(Path(selected_path))
        except Exception as exc:
            QMessageBox.warning(self, "Load Spec Failed", str(exc))
            return
        self._spec_path = Path(selected_path).expanduser().resolve()
        self.spec_path_label.setText(f"Spec File: {self._spec_path}")
        self._set_structure_origin("explicit", "high")
        self._set_editing_enabled(True)
        self.status_label.setText("Spec loaded.")
        self._apply_spec(spec)

    def _open_introspected_model(self, model_name: str, structure_payload: dict[str, object]) -> None:
        base_family = str(structure_payload.get("base_family", "unknown")).strip().lower()
        source = str(structure_payload.get("structure_source", "fallback")).strip().lower()
        confidence = str(structure_payload.get("confidence", "low")).strip().lower()

        self._set_structure_origin(source, confidence)
        self._configure_introspected_flow(structure_payload)
        self._spec_path = None
        self.spec_path_label.setText("Spec File: (none - introspected structure)")

        supported_base = base_family in set(custom_model_generator.list_supported_base_models())
        if supported_base:
            self._set_editing_enabled(True)
            self._base_model = base_family
            self._updating = True
            try:
                self.base_model_combo.setCurrentText(base_family)
                self.model_name_edit.setText(f"{model_name}_custom")
                self.pretrained_checkbox.setChecked(True)
            finally:
                self._updating = False
            self._build_flow()
            for stage_key in self.editable_stages:
                stage_type = next(
                    (
                        str(item.get("stage_type", ""))
                        for item in structure_payload.get("stages", [])
                        if isinstance(item, dict) and str(item.get("key", "")).strip() == stage_key
                    ),
                    "",
                )
                self.state.setdefault(
                    stage_key,
                    {
                        "frozen": True,
                        "train_bn": False,
                        "train_norm": False,
                        "stage_method": "none",
                        "rank": 8,
                        "alpha": 16.0,
                        "adapter_dim": 32,
                        "bitfit_scope": "all_bias",
                        "ssf_scale": 1.0,
                        "ssf_shift": 0.0,
                        "lr_override_enabled": False,
                        "lr_override": 1e-3,
                    },
                )
                self.state[stage_key]["frozen"] = stage_type not in {"head", "classifier"}
                self.state[stage_key]["train_bn"] = False
                self.state[stage_key]["train_norm"] = False
                self.state[stage_key]["stage_method"] = "none"
            first_stage = next(iter(self.editable_stages), "classifier")
            self._selected_stage = first_stage if first_stage in self.nodes else "classifier"
            self.status_label.setText(f"Loaded '{model_name}' using {source} structure parsing. Editing is enabled with conservative stage constraints.")
        else:
            self._set_editing_enabled(False)
            self._updating = True
            try:
                self.model_name_edit.setText(model_name)
            finally:
                self._updating = False
            self._build_flow()
            self.status_label.setText(
                f"Loaded '{model_name}' using {source} structure parsing. Family is unsupported for generator edits, so canvas is inspection-only."
            )

        self._select_stage(self._selected_stage if self._selected_stage in self.nodes else next(iter(self.nodes), ""))
        self._on_state_changed()
        self._refresh_model_info_label(model_name_hint=model_name)

    def open_existing_model(self) -> None:
        display_items, display_to_model = self._existing_model_choices()
        if not display_items:
            QMessageBox.information(
                self,
                "No Models Available",
                "No models are available under the current filter.",
            )
            return
        selected_display, accepted = QInputDialog.getItem(
            self,
            "Open Existing Model",
            "Select a model to reopen from spec:",
            display_items,
            0,
            False,
        )
        if not accepted or not selected_display:
            return
        model_name = display_to_model.get(str(selected_display).strip())
        if not model_name:
            return
        spec_path = resolve_model_spec_path(model_name, allow_legacy_mapping=True)
        if spec_path is not None:
            try:
                spec = custom_model_generator.load_spec_file(spec_path)
            except Exception as exc:
                QMessageBox.warning(self, "Open Model Failed", str(exc))
                return
            self._spec_path = spec_path
            self.spec_path_label.setText(f"Spec File: {self._spec_path}")
            self._set_structure_origin("explicit", "high")
            self._set_editing_enabled(True)
            self.status_label.setText(f"Loaded model '{model_name}' from spec.")
            self._apply_spec(spec)
            self._refresh_model_info_label(model_name_hint=model_name)
            return

        try:
            structure_payload = resolve_model_structure_for_canvas(model_name)
        except Exception as exc:
            QMessageBox.warning(self, "Open Model Failed", f"Could not derive structure for '{model_name}'.\n{exc}")
            return
        self._open_introspected_model(model_name, structure_payload)

    def save_spec(self) -> None:
        try:
            spec = self._derive_spec()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        path, note = self._resolve_spec_output_path(spec.model_name, prefer_current=True)
        try:
            saved = custom_model_generator.save_spec_file(spec, path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Spec Failed", str(exc))
            return
        self._spec_path = saved
        self.spec_path_label.setText(f"Spec File: {saved}")
        self.status_label.setText(note or "Spec saved.")

    def save_spec_as(self) -> None:
        try:
            spec = self._derive_spec()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        default_path = custom_model_generator.default_spec_path_for_model_name(spec.model_name)
        selected_path, _ = QFileDialog.getSaveFileName(self, "Save Spec As", str(default_path), "Spec JSON (*.json)")
        if not selected_path:
            return
        target_path, note = self._resolve_spec_output_path(spec.model_name, selected_path=selected_path)
        try:
            saved = custom_model_generator.save_spec_file(spec, target_path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Spec Failed", str(exc))
            return
        self._spec_path = saved
        self.spec_path_label.setText(f"Spec File: {saved}")
        self.status_label.setText(note or "Spec saved as new file.")

    def generate_model(self) -> None:
        try:
            spec = self._derive_spec()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        model_path = custom_model_generator.MODEL_DIR / f"{spec.model_name}.py"
        spec_path, path_note = self._resolve_spec_output_path(spec.model_name, prefer_current=True)
        overwrite = False
        existing_outputs: list[str] = []
        if model_path.exists():
            existing_outputs.append(str(model_path))
        if spec_path.exists():
            existing_outputs.append(str(spec_path))
        if existing_outputs:
            answer = QMessageBox.question(
                self,
                "Regenerate Existing Outputs",
                "The following output files already exist:\n"
                + "\n".join(existing_outputs)
                + "\n\nRegenerate and overwrite them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            overwrite = True
        try:
            artifacts = custom_model_generator.generate_custom_model(spec, overwrite=overwrite)
            saved = custom_model_generator.save_spec_file(spec, spec_path)
        except Exception as exc:
            QMessageBox.critical(self, "Generate Model Failed", str(exc))
            return
        self._spec_path = saved
        self.spec_path_label.setText(f"Spec File: {saved}")
        self.status_label.setText(path_note or "Model generated from canvas.")
        if self._on_model_generated is not None:
            self._on_model_generated(artifacts.model_name)
        QMessageBox.information(self, "Model Generated", f"Model file: {artifacts.model_file_path}\nSpec file: {artifacts.spec_file_path}")
