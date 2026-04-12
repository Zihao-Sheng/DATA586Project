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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
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
        border = "#4e8cff" if self._selected else "#314154"
        bg = "#0f1722" if self.editable else "#0b1119"
        self.setStyleSheet(f"QFrame{{border:1px solid {border}; border-radius:8px; background:{bg};}}")

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
    STRATEGIES = ["Freeze", "Unfreeze", "BN Tuning", "LoRA", "DoRA", "TSA"]

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
    }

    def __init__(self, on_model_generated: Callable[[str], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec_path: Path | None = None
        self._updating = False
        self._on_model_generated = on_model_generated
        self._selected_stage = "classifier"
        self._base_model = "efficientnet_v2_s"

        self.model_name_edit = QLineEdit("custom_canvas_model")
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems(custom_model_generator.list_supported_base_models())
        self.pretrained_checkbox = QCheckBox("Use pretrained weights")
        self.pretrained_checkbox.setChecked(True)
        self.gradcam_edit = QLineEdit()

        self.stage_frozen = QCheckBox("Frozen")
        self.stage_train_bn = QCheckBox("Train BN")
        self.stage_peft = QComboBox()
        self.stage_peft.addItems(["none", "lora", "dora", "tsa"])
        self.stage_rank = QSpinBox()
        self.stage_rank.setRange(1, 128)
        self.stage_rank.setValue(8)
        self.stage_alpha = QDoubleSpinBox()
        self.stage_alpha.setRange(0.1, 512.0)
        self.stage_alpha.setDecimals(2)
        self.stage_alpha.setValue(16.0)
        self.method_preview = QLabel("Method: baseline")
        self.method_preview.setProperty("muted", True)
        self.spec_path_label = QLabel("Spec File: (new unsaved spec)")
        self.status_label = QLabel("Canvas edits structured spec only.")
        self.status_label.setProperty("muted", True)
        self.spec_summary = QPlainTextEdit()
        self.spec_summary.setReadOnly(True)
        self.spec_summary.setMaximumHeight(180)

        self.stage_frozen.toggled.connect(self._on_detail_changed)
        self.stage_train_bn.toggled.connect(self._on_detail_changed)
        self.stage_peft.currentTextChanged.connect(self._on_detail_changed)
        self.stage_rank.valueChanged.connect(self._on_detail_changed)
        self.stage_alpha.valueChanged.connect(self._on_detail_changed)
        self.model_name_edit.textChanged.connect(self._on_state_changed)
        self.gradcam_edit.textChanged.connect(self._on_state_changed)
        self.pretrained_checkbox.toggled.connect(self._on_state_changed)
        self.base_model_combo.currentTextChanged.connect(self._on_base_model_changed)

        self.new_button = QPushButton("New")
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        self.save_as_button = QPushButton("Save As")
        self.generate_button = QPushButton("Generate Model")
        self.new_button.clicked.connect(self.new_spec)
        self.load_button.clicked.connect(self.load_spec)
        self.save_button.clicked.connect(self.save_spec)
        self.save_as_button.clicked.connect(self.save_spec_as)
        self.generate_button.clicked.connect(self.generate_model)

        self.palette = StrategyPaletteList()
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
        detail_form.addRow("Selected Stage", self.stage_title)
        detail_form.addRow("", self.stage_frozen)
        detail_form.addRow("", self.stage_train_bn)
        detail_form.addRow("PEFT", self.stage_peft)
        detail_form.addRow("Rank", self.stage_rank)
        detail_form.addRow("Alpha", self.stage_alpha)
        detail_form.addRow("Grad-CAM", self.gradcam_edit)
        detail_form.addRow("Method", self.method_preview)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(detail_group)
        splitter.setSizes([760, 330])

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Model Name"))
        toolbar.addWidget(self.model_name_edit, stretch=1)
        toolbar.addWidget(QLabel("Base"))
        toolbar.addWidget(self.base_model_combo)
        toolbar.addWidget(self.pretrained_checkbox)
        toolbar.addWidget(self.new_button)
        toolbar.addWidget(self.load_button)
        toolbar.addWidget(self.save_button)
        toolbar.addWidget(self.save_as_button)
        toolbar.addWidget(self.generate_button)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self.spec_path_label)
        layout.addWidget(self.spec_summary)
        layout.addWidget(self.status_label)

        self.new_spec()

    def _stage_to_feature_index(self, stage_key: str) -> int | None:
        if stage_key.startswith("features."):
            try:
                return int(stage_key.split(".")[1])
            except Exception:
                return None
        return None

    def _allowed_strategies_for_stage(self, stage_key: str) -> set[str]:
        if stage_key not in self.editable_stages:
            return set()
        allowed = set(self.STRATEGIES)
        if self._base_model == "efficientnet_v2_s":
            index = self._stage_to_feature_index(stage_key)
            if stage_key == "stem" or (index is not None and index <= 4):
                allowed -= {"LoRA", "DoRA", "TSA"}
        else:
            if stage_key not in {"layer1", "layer2", "layer3", "layer4", "classifier"}:
                allowed -= {"LoRA", "DoRA", "TSA"}
            allowed.discard("DoRA")
        return allowed

    def _build_flow(self) -> None:
        while self.canvas_layout.count():
            item = self.canvas_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.nodes.clear()

        if self._base_model == "efficientnet_v2_s":
            self.editable_stages = {"stem", *[f"features.{i}" for i in range(8)], "classifier"}
        else:
            self.editable_stages = {"stem", "layer1", "layer2", "layer3", "layer4", "classifier"}

        for stage_key, _ in self.FLOWS[self._base_model]:
            if stage_key not in self.state:
                self.state[stage_key] = {"frozen": True, "train_bn": False, "peft": "none", "rank": 8, "alpha": 16.0}
        for stage in list(self.state.keys()):
            if stage not in {key for key, _ in self.FLOWS[self._base_model]}:
                self.state.pop(stage, None)

        if "classifier" in self.state and self.state["classifier"]["frozen"] is True:
            self.state["classifier"]["frozen"] = False

        flow = self.FLOWS[self._base_model]
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
            self._selected_stage = "classifier"

    def _select_stage(self, stage_key: str) -> None:
        self._selected_stage = stage_key
        for key, node in self.nodes.items():
            node.set_selected(key == stage_key)
        state = self.state.get(stage_key, {})
        self._updating = True
        try:
            self.stage_title.setText(stage_key)
            editable = stage_key in self.editable_stages
            self.stage_frozen.setEnabled(editable)
            self.stage_train_bn.setEnabled(editable)
            self.stage_peft.setEnabled(editable)
            self.stage_rank.setEnabled(editable)
            self.stage_alpha.setEnabled(editable)
            self.stage_frozen.setChecked(bool(state.get("frozen", True)))
            self.stage_train_bn.setChecked(bool(state.get("train_bn", False)))
            peft = str(state.get("peft", "none"))
            if self._base_model != "efficientnet_v2_s" and peft == "dora":
                peft = "none"
            self.stage_peft.setCurrentText(peft)
            self.stage_rank.setValue(int(state.get("rank", 8)))
            self.stage_alpha.setValue(float(state.get("alpha", 16.0)))
        finally:
            self._updating = False

    def _apply_strategy(self, stage_key: str, strategy: str) -> None:
        if stage_key not in self.editable_stages:
            return
        strategy = strategy.strip()
        if strategy == "Freeze":
            self.state[stage_key]["frozen"] = True
        elif strategy == "Unfreeze":
            self.state[stage_key]["frozen"] = False
        elif strategy == "BN Tuning":
            self.state[stage_key]["train_bn"] = True
        elif strategy in {"LoRA", "DoRA", "TSA"}:
            method = strategy.lower()
            if self._base_model == "resnet18" and method == "dora":
                self.status_label.setText("DoRA is not supported for ResNet18 in current generator.")
                return
            for key in self.editable_stages:
                if self.state[key]["peft"] != "none" and self.state[key]["peft"] != method:
                    self.state[key]["peft"] = "none"
            self.state[stage_key]["peft"] = method
        self._select_stage(stage_key)
        self._on_state_changed()

    def _on_detail_changed(self) -> None:
        if self._updating or self._selected_stage not in self.editable_stages:
            return
        peft = self.stage_peft.currentText().strip().lower()
        if self._base_model == "resnet18" and peft == "dora":
            peft = "none"
        if peft in {"lora", "dora", "tsa"}:
            for key in self.editable_stages:
                if self.state[key]["peft"] != "none" and self.state[key]["peft"] != peft:
                    self.state[key]["peft"] = "none"
        self.state[self._selected_stage].update(
            {
                "frozen": self.stage_frozen.isChecked(),
                "train_bn": self.stage_train_bn.isChecked(),
                "peft": peft,
                "rank": int(self.stage_rank.value()),
                "alpha": float(self.stage_alpha.value()),
            }
        )
        self._on_state_changed()

    def _derive_method(self) -> str:
        peft_stages = [key for key in self.editable_stages if self.state[key]["peft"] != "none"]
        peft_method = next((self.state[key]["peft"] for key in peft_stages), None)
        if peft_method:
            return str(peft_method)

        all_unfrozen = all(not bool(self.state[key]["frozen"]) for key in self.editable_stages)
        train_bn_any = any(bool(self.state[key]["train_bn"]) for key in self.editable_stages)
        if all_unfrozen:
            return "full_finetune"
        if self._base_model == "efficientnet_v2_s" and train_bn_any:
            unfrozen_feature = sorted(
                idx
                for stage in self.editable_stages
                for idx in ([self._stage_to_feature_index(stage)] if self._stage_to_feature_index(stage) is not None else [])
                if not bool(self.state[stage]["frozen"])
            )
            if unfrozen_feature == [7]:
                return "bn_last1"
            if unfrozen_feature == [6, 7]:
                return "bn_last2"
            return "bn_tuning"
        if train_bn_any:
            return "bn_tuning"
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

        peft_stages = [key for key in self.editable_stages if self.state[key]["peft"] != "none"]
        if method in {"lora", "dora", "tsa"}:
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
                    "layer_keys": sorted(key for key in peft_stages if key.startswith("layer")),
                    "classifier": "classifier" in peft_stages,
                }
            if method in {"lora", "dora"} and peft_stages:
                ref = self.state[peft_stages[0]]
                payload["peft_params"] = {"rank": int(ref["rank"]), "alpha": float(ref["alpha"])}
            elif method in {"lora", "dora"}:
                payload["peft_params"] = {"rank": int(self.stage_rank.value()), "alpha": float(self.stage_alpha.value())}

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
            self._build_flow()
            self.gradcam_edit.setText(",".join(spec.gradcam_target_hint))

            for key in self.editable_stages:
                self.state[key] = {"frozen": True, "train_bn": False, "peft": "none", "rank": 8, "alpha": 16.0}
            self.state["classifier"]["frozen"] = False

            method = spec.method_type
            if method == "bn_tuning":
                for key in self.editable_stages:
                    self.state[key]["train_bn"] = True
            elif method == "bn_last1":
                for key in self.editable_stages:
                    self.state[key]["train_bn"] = True
                if "features.7" in self.state:
                    self.state["features.7"]["frozen"] = False
            elif method == "bn_last2":
                for key in self.editable_stages:
                    self.state[key]["train_bn"] = True
                if "features.6" in self.state:
                    self.state["features.6"]["frozen"] = False
                if "features.7" in self.state:
                    self.state["features.7"]["frozen"] = False
            elif method == "full_finetune":
                for key in self.editable_stages:
                    self.state[key]["frozen"] = False
            elif method in {"lora", "dora", "tsa"}:
                targets = spec.peft_targets if isinstance(spec.peft_targets, dict) else {}
                if self._base_model == "efficientnet_v2_s":
                    for idx in targets.get("feature_stages", []):
                        key = f"features.{int(idx)}"
                        if key in self.state:
                            self.state[key]["peft"] = method
                else:
                    for layer_key in targets.get("layer_keys", []):
                        key = str(layer_key)
                        if key in self.state:
                            self.state[key]["peft"] = method
                if bool(targets.get("classifier", False)):
                    self.state["classifier"]["peft"] = method
                params = spec.peft_params if isinstance(spec.peft_params, dict) else {}
                for key in self.editable_stages:
                    if self.state[key]["peft"] != "none":
                        self.state[key]["rank"] = int(params.get("rank", 8))
                        self.state[key]["alpha"] = float(params.get("alpha", 16.0))
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
            if stage["peft"] != "none":
                tags.append(str(stage["peft"]).upper())
            node.set_state_text(" | ".join(tags))
        try:
            spec = self._derive_spec()
            self.method_preview.setText(f"Method: {spec.method_type}")
            self.spec_summary.setPlainText(json.dumps(custom_model_generator.spec_to_dict(spec), indent=2, sort_keys=True))
        except Exception as exc:
            self.method_preview.setText("Method: invalid")
            self.spec_summary.setPlainText(f"Invalid spec: {exc}")

    def _on_base_model_changed(self, value: str) -> None:
        if self._updating:
            return
        self._base_model = value
        self._build_flow()
        self._select_stage("classifier")
        self._on_state_changed()

    def new_spec(self) -> None:
        spec = custom_model_generator.build_preset_spec(
            model_name="custom_canvas_model",
            base_model="efficientnet_v2_s",
            method_type="baseline",
        )
        self._spec_path = None
        self.spec_path_label.setText("Spec File: (new unsaved spec)")
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
        self.status_label.setText("Spec loaded.")
        self._apply_spec(spec)

    def save_spec(self) -> None:
        try:
            spec = self._derive_spec()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        path = self._spec_path if self._spec_path is not None else custom_model_generator.default_spec_path_for_model_name(spec.model_name)
        try:
            saved = custom_model_generator.save_spec_file(spec, path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Spec Failed", str(exc))
            return
        self._spec_path = saved
        self.spec_path_label.setText(f"Spec File: {saved}")
        self.status_label.setText("Spec saved.")

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
        try:
            saved = custom_model_generator.save_spec_file(spec, Path(selected_path))
        except Exception as exc:
            QMessageBox.warning(self, "Save Spec Failed", str(exc))
            return
        self._spec_path = saved
        self.spec_path_label.setText(f"Spec File: {saved}")
        self.status_label.setText("Spec saved as new file.")

    def generate_model(self) -> None:
        try:
            spec = self._derive_spec()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        model_path = custom_model_generator.MODEL_DIR / f"{spec.model_name}.py"
        overwrite = False
        if model_path.exists():
            answer = QMessageBox.question(
                self,
                "Regenerate Existing Model",
                f"Model file already exists:\n{model_path}\n\nRegenerate and overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            overwrite = True
        try:
            artifacts = custom_model_generator.generate_custom_model(spec, overwrite=overwrite)
            saved = custom_model_generator.save_spec_file(spec, self._spec_path)
        except Exception as exc:
            QMessageBox.critical(self, "Generate Model Failed", str(exc))
            return
        self._spec_path = saved
        self.spec_path_label.setText(f"Spec File: {saved}")
        self.status_label.setText("Model generated from canvas.")
        if self._on_model_generated is not None:
            self._on_model_generated(artifacts.model_name)
        QMessageBox.information(self, "Model Generated", f"Model file: {artifacts.model_file_path}\nSpec file: {artifacts.spec_file_path}")

