from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QObject, QPointF, QProcess, QRect, QRectF, QSize, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFontMetrics, QIcon, QLinearGradient, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QListView,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplashScreen,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QToolTip,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core import custom_model_generator, run_log_compat
from core import model_registry as model_registry_module
from core import runtime_paths
from app import app_themes, global_job_queue
from app.custom_models_canvas import CustomModelCanvasWidget
from core.model_registry import (
    discover_model_names_generated_first,
    model_catalog_entry,
    model_detailed_tooltip,
    model_display_label,
    resolve_preferred_model_name,
    sort_model_names_for_ui,
)

PROJECT_ROOT = runtime_paths.project_root()
SCRIPTS_ENTRY_ROOT = PROJECT_ROOT / "scripts" / "entry"
DATA_RETRIEVAL_SCRIPT = SCRIPTS_ENTRY_ROOT / "data_retrieval.py"
TRAINING_SCRIPT = SCRIPTS_ENTRY_ROOT / "training.py"
PREDICTING_SCRIPT = PROJECT_ROOT / "scripts" / "pipeline" / "predicting.py"
TRAINING_WORKER_EXE = PROJECT_ROOT / "DATA586TrainingWorker.exe"
DATA_WORKER_EXE = PROJECT_ROOT / "DATA586DataWorker.exe"
DEFAULT_DATA_DIR = runtime_paths.data_dir()
DEFAULT_DATA_ROOT = DEFAULT_DATA_DIR / "food-101"
DEFAULT_TEST_SPLITS_ROOT = DEFAULT_DATA_DIR / "test_splits"
DEFAULT_CHECKPOINT_DIR = runtime_paths.checkpoints_dir()
APP_ICON_PATH = PROJECT_ROOT / "scripts" / "assets" / "training_launcher_icon.ico"
APP_ID = "MLWorkbench.TrainingLauncher"
WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_DEFAULTSIZE = 0x00000040
NEW_CHECKPOINT_NAME_LABEL = "New checkpoint name..."
RUN_LOG_DIRNAME = "_run_logs"
SETTINGS_ORG = "DATA586Project"
SETTINGS_APP = "TrainingLauncher"
TRASH_ROOT = PROJECT_ROOT / ".trash"
LOG_TRASH_DIR = TRASH_ROOT / "logs"
MODEL_TRASH_DIR = TRASH_ROOT / "models"
LOG_TRASH_LIMIT = 10
MODEL_TRASH_LIMIT = 5
PREDICT_THUMBNAIL_CACHE_LIMIT = 192
PREDICT_DISPLAY_CACHE_LIMIT = 48
PREDICT_GRADCAM_CACHE_LIMIT = 24
PREDICT_COMPARE_DISPLAY_CACHE_LIMIT = 12
CHECKPOINT_SELECTOR_MODEL_TEXT_MAX_CHARS = 56
CHECKPOINT_SELECTOR_MODEL_COLUMN_MAX_WIDTH = 420


class ComboDeleteItemDelegate(QStyledItemDelegate):
    DELETE_WIDTH = 54

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text_rect = QRect(opt.rect)
        text_rect.setRight(text_rect.right() - self.DELETE_WIDTH - 10)
        opt.rect = text_rect
        opt.text = painter.fontMetrics().elidedText(opt.text, Qt.ElideRight, max(32, text_rect.width() - 8))
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        delete_rect = self.delete_rect(option.rect)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        border = QColor(180, 74, 74)
        fill = QColor(90, 28, 28, 38) if option.state & QStyle.State_MouseOver else QColor(90, 28, 28, 20)
        painter.setPen(border)
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(delete_rect), 6, 6)
        painter.setPen(QColor(198, 84, 84))
        painter.drawText(delete_rect, Qt.AlignCenter, "Delete")
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        size = super().sizeHint(option, index)
        return QSize(max(size.width(), 220), max(size.height(), 28))

    @classmethod
    def delete_rect(cls, item_rect: QRect) -> QRect:
        return QRect(item_rect.right() - cls.DELETE_WIDTH - 6, item_rect.top() + 4, cls.DELETE_WIDTH, max(20, item_rect.height() - 8))


class DeletableComboBox(QComboBox):
    deleteRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        list_view = QListView(self)
        list_view.setMouseTracking(True)
        list_view.viewport().installEventFilter(self)
        list_view.setItemDelegate(ComboDeleteItemDelegate(list_view))
        self.setView(list_view)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        view = self.view()
        if watched is view.viewport() and event.type() == QEvent.MouseButtonPress:
            index = view.indexAt(event.pos())
            if index.isValid():
                item_rect = view.visualRect(index)
                if ComboDeleteItemDelegate.delete_rect(item_rect).contains(event.pos()):
                    model_name = index.data(Qt.UserRole)
                    if isinstance(model_name, str) and model_name.strip():
                        self.hidePopup()
                        self.deleteRequested.emit(model_name.strip())
                        return True
        return super().eventFilter(watched, event)


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def build_startup_splash(theme_key: str | None) -> QSplashScreen:
    theme = app_themes.get_theme(theme_key)
    width, height = 680, 300
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    outer = QRectF(0, 0, float(width), float(height))
    panel_rect = outer.adjusted(10, 10, -10, -10)
    painter.setPen(QPen(QColor(theme["border_strong"]), 1))
    shell_gradient = QLinearGradient(panel_rect.topLeft(), panel_rect.bottomLeft())
    shell_gradient.setColorAt(0.0, QColor(theme["panel_bg"]))
    shell_gradient.setColorAt(0.55, QColor(theme["panel_alt_bg"]))
    shell_gradient.setColorAt(1.0, QColor(theme["base_bg"]))
    painter.setBrush(QBrush(shell_gradient))
    painter.drawRoundedRect(panel_rect, 16, 16)
    inner_rect = panel_rect.adjusted(1, 1, -1, -1)
    painter.setPen(QPen(QColor(theme["border"]), 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(inner_rect, 15, 15)
    for i in range(12):
        y = panel_rect.top() + 18 + i * 18
        painter.setPen(QPen(QColor(255, 255, 255, 7), 1))
        painter.drawLine(int(panel_rect.left() + 14), int(y), int(panel_rect.right() - 14), int(y))
    accent_rect = QRectF(panel_rect.left(), panel_rect.top(), panel_rect.width(), 8)
    painter.setPen(Qt.NoPen)
    accent_gradient = QLinearGradient(accent_rect.topLeft(), accent_rect.bottomLeft())
    accent_gradient.setColorAt(0.0, QColor(theme["accent_hover"]))
    accent_gradient.setColorAt(1.0, QColor(theme["accent"]))
    painter.setBrush(QBrush(accent_gradient))
    painter.drawRoundedRect(accent_rect, 16, 16)
    title_rect = QRectF(panel_rect.left() + 24, panel_rect.top() + 42, panel_rect.width() - 48, 46)
    subtitle_rect = QRectF(panel_rect.left() + 24, panel_rect.top() + 96, panel_rect.width() - 48, 30)
    hint_rect = QRectF(panel_rect.left() + 24, panel_rect.bottom() - 52, panel_rect.width() - 48, 24)
    title_font = painter.font()
    title_font.setFamily(theme["font_family"])
    title_font.setPointSize(17)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QColor(theme["text"]))
    painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, "DATA586 Training Launcher")
    subtitle_font = painter.font()
    subtitle_font.setPointSize(10)
    subtitle_font.setBold(False)
    painter.setFont(subtitle_font)
    painter.setPen(QColor(theme["text_muted"]))
    painter.drawText(subtitle_rect, Qt.AlignLeft | Qt.AlignVCenter, "Unified Workspace: Training, Predicting, Queue, and Custom Models")
    hint_font = painter.font()
    hint_font.setPointSize(9)
    painter.setFont(hint_font)
    painter.drawText(hint_rect, Qt.AlignLeft | Qt.AlignVCenter, "Initializing application modules...")
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.FramelessWindowHint, True)
    splash.setEnabled(False)
    return splash


def apply_windows_taskbar_icon(window: QMainWindow) -> None:
    if sys.platform != "win32" or not APP_ICON_PATH.is_file():
        return
    try:
        hwnd = int(window.winId())
        hicon = ctypes.windll.user32.LoadImageW(
            None,
            str(APP_ICON_PATH),
            IMAGE_ICON,
            0,
            0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        )
        if hicon:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
    except Exception:
        pass


def validate_predict_image_paths(image_paths: list[Path], sample_limit: int = 12) -> tuple[list[Path], list[str]]:
    readable: list[Path] = []
    errors: list[str] = []
    try:
        from PIL import Image
    except Exception as exc:
        return [], [f"PIL unavailable: {exc}"]

    for image_path in image_paths:
        if len(readable) >= sample_limit and not errors:
            break
        try:
            resolved = image_path.expanduser().resolve(strict=False)
            os.stat(resolved)
            with Image.open(resolved) as image:
                image.verify()
            readable.append(resolved)
        except Exception as exc:
            errors.append(f"{image_path}: {exc}")
            if len(errors) >= 5:
                break
    return readable, errors


class CustomModelsWorkspaceWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec_path: Path | None = None
        self._updating_form = False

        self.model_name_edit = QLineEdit("efficientnet_custom_baseline")
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems(custom_model_generator.list_supported_base_models())
        self.base_model_combo.currentTextChanged.connect(self._on_base_model_changed)
        self.method_combo = QComboBox()
        self.method_combo.currentTextChanged.connect(self._on_method_changed)

        self.freeze_strategy_combo = QComboBox()
        self.freeze_strategy_combo.addItems(sorted(custom_model_generator.SUPPORTED_FREEZE_STRATEGIES))

        self.train_bn_checkbox = QCheckBox("Train BatchNorm layers")
        self.unfreeze_profile_combo = QComboBox()
        self.unfreeze_profile_combo.addItems(["none", "last1", "last2", "custom"])
        self.unfreeze_profile_combo.currentTextChanged.connect(self._on_unfreeze_profile_changed)
        self.unfreeze_stage_checks: list[QCheckBox] = []
        unfreeze_stages_layout = QHBoxLayout()
        unfreeze_stages_layout.setContentsMargins(0, 0, 0, 0)
        unfreeze_stages_layout.setSpacing(6)
        for stage in range(8):
            box = QCheckBox(str(stage))
            self.unfreeze_stage_checks.append(box)
            unfreeze_stages_layout.addWidget(box)
        unfreeze_stages_layout.addStretch(1)
        self.unfreeze_stages_widget = QWidget()
        self.unfreeze_stages_widget.setLayout(unfreeze_stages_layout)

        self.peft_method_combo = QComboBox()
        self.peft_method_combo.currentTextChanged.connect(self._on_peft_method_changed)

        self.dora_stage_checks: list[QCheckBox] = []
        dora_stages_layout = QHBoxLayout()
        dora_stages_layout.setContentsMargins(0, 0, 0, 0)
        dora_stages_layout.setSpacing(6)
        for stage in range(8):
            box = QCheckBox(str(stage))
            self.dora_stage_checks.append(box)
            dora_stages_layout.addWidget(box)
        dora_stages_layout.addStretch(1)
        self.dora_stages_widget = QWidget()
        self.dora_stages_widget.setLayout(dora_stages_layout)
        self.dora_classifier_checkbox = QCheckBox("Apply DoRA to classifier")
        self.dora_classifier_checkbox.setChecked(True)
        self.dora_rank_spin = QSpinBox()
        self.dora_rank_spin.setRange(1, 128)
        self.dora_rank_spin.setValue(8)
        self.dora_alpha_spin = QDoubleSpinBox()
        self.dora_alpha_spin.setRange(0.1, 512.0)
        self.dora_alpha_spin.setDecimals(2)
        self.dora_alpha_spin.setSingleStep(0.5)
        self.dora_alpha_spin.setValue(16.0)
        self.peft_layer_keys_edit = QLineEdit("layer4")
        self.peft_layer_keys_edit.setPlaceholderText("Comma-separated module keys, e.g. layer4")

        self.gradcam_hint_edit = QLineEdit("features.7")
        self.gradcam_hint_edit.setPlaceholderText("Comma-separated layer names, e.g. features.7")

        basic_group = QGroupBox("Basic Info")
        basic_form = QFormLayout(basic_group)
        basic_form.addRow("Model Name", self.model_name_edit)
        basic_form.addRow("Base Model", self.base_model_combo)
        basic_form.addRow("Method Type", self.method_combo)

        strategy_group = QGroupBox("Training Strategy")
        strategy_form = QFormLayout(strategy_group)
        strategy_form.addRow("Freeze Strategy", self.freeze_strategy_combo)
        strategy_form.addRow("", self.train_bn_checkbox)
        strategy_form.addRow("Unfreeze Profile", self.unfreeze_profile_combo)
        strategy_form.addRow("Unfreeze Stages", self.unfreeze_stages_widget)

        peft_group = QGroupBox("PEFT Strategy")
        peft_form = QFormLayout(peft_group)
        peft_form.addRow("PEFT Method", self.peft_method_combo)
        peft_form.addRow("DoRA Target Stages", self.dora_stages_widget)
        peft_form.addRow("PEFT Layer Keys", self.peft_layer_keys_edit)
        peft_form.addRow("", self.dora_classifier_checkbox)
        peft_form.addRow("DoRA Rank", self.dora_rank_spin)
        peft_form.addRow("DoRA Alpha", self.dora_alpha_spin)

        gradcam_group = QGroupBox("Grad-CAM Hints")
        gradcam_form = QFormLayout(gradcam_group)
        gradcam_form.addRow("Default Targets", self.gradcam_hint_edit)

        self.spec_path_label = QLabel("Spec File: (new unsaved spec)")
        self.spec_path_label.setWordWrap(True)
        self.model_output_label = QLabel("Generated Model Name: (pending)")
        self.model_output_label.setWordWrap(True)
        self.status_label = QLabel("Ready. Structured spec/template generation for EfficientNet and ResNet18.")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("muted", True)

        actions_layout = QHBoxLayout()
        self.new_spec_button = QPushButton("New Spec")
        self.load_spec_button = QPushButton("Load Spec")
        self.save_spec_button = QPushButton("Save Spec")
        self.save_as_spec_button = QPushButton("Save As")
        self.generate_model_button = QPushButton("Generate Model")
        actions_layout.addWidget(self.new_spec_button)
        actions_layout.addWidget(self.load_spec_button)
        actions_layout.addWidget(self.save_spec_button)
        actions_layout.addWidget(self.save_as_spec_button)
        actions_layout.addStretch(1)
        actions_layout.addWidget(self.generate_model_button)

        self.new_spec_button.clicked.connect(self.new_spec)
        self.load_spec_button.clicked.connect(self.load_spec)
        self.save_spec_button.clicked.connect(self.save_spec)
        self.save_as_spec_button.clicked.connect(self.save_spec_as)
        self.generate_model_button.clicked.connect(self.generate_model)
        self.model_name_edit.textChanged.connect(self._refresh_model_output_label)

        layout = QVBoxLayout(self)
        layout.addWidget(basic_group)
        layout.addWidget(strategy_group)
        layout.addWidget(peft_group)
        layout.addWidget(gradcam_group)
        layout.addLayout(actions_layout)
        layout.addWidget(self.spec_path_label)
        layout.addWidget(self.model_output_label)
        layout.addWidget(self.status_label)

        self.new_spec()

    @staticmethod
    def _set_combo_items(combo: QComboBox, items: list[str], selected: str | None = None) -> None:
        keep = selected if selected in items else (items[0] if items else "")
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if keep:
            combo.setCurrentText(keep)
        combo.blockSignals(False)

    def _selected_stage_indices(self, checks: list[QCheckBox]) -> list[int]:
        return [index for index, box in enumerate(checks) if box.isChecked()]

    def _set_stage_indices(self, checks: list[QCheckBox], indices: list[int]) -> None:
        selected = set(indices)
        for index, box in enumerate(checks):
            box.setChecked(index in selected)

    def _refresh_model_output_label(self) -> None:
        self.model_output_label.setText(f"Generated Model Name: {self.model_name_edit.text().strip() or '(pending)'}")

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

    def _on_unfreeze_profile_changed(self, profile: str) -> None:
        if self._updating_form:
            return
        normalized = profile.strip().lower()
        if normalized == "none":
            self._set_stage_indices(self.unfreeze_stage_checks, [])
        elif normalized == "last1":
            self._set_stage_indices(self.unfreeze_stage_checks, [7])
        elif normalized == "last2":
            self._set_stage_indices(self.unfreeze_stage_checks, [6, 7])

    def _on_base_model_changed(self, base_model: str) -> None:
        if self._updating_form:
            return
        methods = custom_model_generator.supported_methods_for_base(base_model)
        current_method = self.method_combo.currentText().strip().lower()
        selected = current_method if current_method in methods else methods[0]
        self._set_combo_items(self.method_combo, methods, selected)
        self._on_method_changed(selected)

    def _on_peft_method_changed(self, _: str) -> None:
        if self._updating_form:
            return
        self._on_method_changed(self.method_combo.currentText())

    def _refresh_method_controls(self, base_model: str, method: str) -> None:
        is_efficientnet = base_model == "efficientnet_v2_s"
        is_dora = method == "dora"
        show_unfreeze_profile = is_efficientnet and method in {"bn_last1", "bn_last2"}
        allow_unfreeze_stages = is_efficientnet and method in {"bn_tuning", "dora", "manual"}

        self.unfreeze_profile_combo.setEnabled(show_unfreeze_profile)
        self.unfreeze_stages_widget.setEnabled(show_unfreeze_profile or allow_unfreeze_stages)
        self.dora_stages_widget.setEnabled(is_dora)
        self.dora_classifier_checkbox.setEnabled(is_dora)
        self.dora_rank_spin.setEnabled(is_dora)
        self.dora_alpha_spin.setEnabled(is_dora)
        self.peft_layer_keys_edit.setEnabled(method in {"lora", "dora", "tsa"} and not is_efficientnet)

    def _refresh_gradcam_placeholder(self, targets: list[str]) -> None:
        example = ",".join(targets) if targets else "layer4"
        self.gradcam_hint_edit.setPlaceholderText(f"Comma-separated layer names, e.g. {example}")

    def _on_method_changed(self, _: str) -> None:
        if self._updating_form:
            return
        self._updating_form = True
        try:
            base_model = self.base_model_combo.currentText().strip().lower() or "efficientnet_v2_s"
            method = self.method_combo.currentText().strip().lower() or "baseline"
            preset = custom_model_generator.build_preset_spec(
                model_name=self.model_name_edit.text().strip() or "temp_model",
                base_model=base_model,
                method_type=method,
            )

            self.freeze_strategy_combo.setCurrentText(preset.freeze_strategy)
            self.train_bn_checkbox.setChecked(bool(preset.train_bn))
            self._set_stage_indices(self.unfreeze_stage_checks, list(preset.unfreeze_stages))
            if preset.unfreeze_stages == [7]:
                self.unfreeze_profile_combo.setCurrentText("last1")
            elif preset.unfreeze_stages == [6, 7]:
                self.unfreeze_profile_combo.setCurrentText("last2")
            elif not preset.unfreeze_stages:
                self.unfreeze_profile_combo.setCurrentText("none")
            else:
                self.unfreeze_profile_combo.setCurrentText("custom")

            expected_peft = preset.peft_method or "none"
            self._set_combo_items(self.peft_method_combo, [expected_peft], expected_peft)
            targets = preset.peft_targets if isinstance(preset.peft_targets, dict) else {}
            self._set_stage_indices(self.dora_stage_checks, list(targets.get("feature_stages", [])))
            self.peft_layer_keys_edit.setText(",".join(str(value) for value in targets.get("layer_keys", [])))
            self.dora_classifier_checkbox.setChecked(bool(targets.get("classifier", False)))
            params = preset.peft_params if isinstance(preset.peft_params, dict) else {}
            self.dora_rank_spin.setValue(int(params.get("rank", 8)))
            self.dora_alpha_spin.setValue(float(params.get("alpha", 16.0)))
            self.gradcam_hint_edit.setText(",".join(preset.gradcam_target_hint))
            self._refresh_gradcam_placeholder(list(preset.gradcam_target_hint))
            self._refresh_method_controls(base_model, method)
        finally:
            self._updating_form = False

    def _collect_gradcam_hints(self) -> list[str]:
        return [part.strip() for part in self.gradcam_hint_edit.text().split(",") if part.strip()]

    def _collect_layer_keys(self) -> list[str]:
        return [part.strip() for part in self.peft_layer_keys_edit.text().split(",") if part.strip()]

    def _collect_spec_from_form(self) -> custom_model_generator.CustomModelSpec:
        base_model = self.base_model_combo.currentText().strip().lower()
        method = self.method_combo.currentText().strip().lower()
        peft_method = self.peft_method_combo.currentText().strip().lower()
        payload = {
            "model_name": self.model_name_edit.text().strip(),
            "base_model": base_model,
            "task_type": "classification",
            "method_type": method,
            "freeze_strategy": self.freeze_strategy_combo.currentText().strip().lower(),
            "train_bn": self.train_bn_checkbox.isChecked(),
            "unfreeze_stages": self._selected_stage_indices(self.unfreeze_stage_checks),
            "peft_method": None if peft_method == "none" else peft_method,
            "peft_targets": {
                "feature_stages": self._selected_stage_indices(self.dora_stage_checks),
                "layer_keys": self._collect_layer_keys(),
                "classifier": self.dora_classifier_checkbox.isChecked(),
            },
            "peft_params": {},
            "gradcam_target_hint": self._collect_gradcam_hints(),
            "metadata_version": custom_model_generator.SPEC_VERSION,
            "generator_version": custom_model_generator.GENERATOR_VERSION,
        }
        if method in {"lora", "dora"}:
            payload["peft_params"] = {
                "rank": int(self.dora_rank_spin.value()),
                "alpha": float(self.dora_alpha_spin.value()),
            }
        return custom_model_generator.spec_from_dict(payload)

    def _apply_spec_to_form(self, spec: custom_model_generator.CustomModelSpec) -> None:
        self._updating_form = True
        try:
            self.model_name_edit.setText(spec.model_name)
            self.base_model_combo.setCurrentText(spec.base_model)
            methods = custom_model_generator.supported_methods_for_base(spec.base_model)
            self._set_combo_items(self.method_combo, methods, spec.method_type)
            self.freeze_strategy_combo.setCurrentText(spec.freeze_strategy)
            self.train_bn_checkbox.setChecked(bool(spec.train_bn))
            self._set_stage_indices(self.unfreeze_stage_checks, list(spec.unfreeze_stages))
            if spec.unfreeze_stages == [7]:
                self.unfreeze_profile_combo.setCurrentText("last1")
            elif spec.unfreeze_stages == [6, 7]:
                self.unfreeze_profile_combo.setCurrentText("last2")
            elif not spec.unfreeze_stages:
                self.unfreeze_profile_combo.setCurrentText("none")
            else:
                self.unfreeze_profile_combo.setCurrentText("custom")

            peft_method = spec.peft_method or "none"
            self._set_combo_items(self.peft_method_combo, [peft_method], peft_method)
            targets = spec.peft_targets if isinstance(spec.peft_targets, dict) else {}
            self._set_stage_indices(self.dora_stage_checks, list(targets.get("feature_stages", [])))
            self.peft_layer_keys_edit.setText(",".join(str(value) for value in targets.get("layer_keys", [])))
            self.dora_classifier_checkbox.setChecked(bool(targets.get("classifier", False)))
            params = spec.peft_params if isinstance(spec.peft_params, dict) else {}
            self.dora_rank_spin.setValue(int(params.get("rank", 8)))
            self.dora_alpha_spin.setValue(float(params.get("alpha", 16.0)))
            self.gradcam_hint_edit.setText(",".join(spec.gradcam_target_hint))
            self._refresh_gradcam_placeholder(list(spec.gradcam_target_hint))
            self._refresh_model_output_label()
            self._refresh_method_controls(spec.base_model, spec.method_type)
        finally:
            self._updating_form = False

    def new_spec(self) -> None:
        spec = custom_model_generator.build_preset_spec(
            model_name="efficientnet_custom_baseline",
            base_model="efficientnet_v2_s",
            method_type="baseline",
        )
        self._spec_path = None
        self._apply_spec_to_form(spec)
        self.spec_path_label.setText("Spec File: (new unsaved spec)")
        self.status_label.setText("New spec initialized.")

    def load_spec(self) -> None:
        start_dir = custom_model_generator.SPEC_DIR
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Custom Model Spec", str(start_dir), "Spec JSON (*.json)")
        if not selected_path:
            return
        try:
            spec = custom_model_generator.load_spec_file(Path(selected_path))
        except Exception as exc:
            QMessageBox.warning(self, "Load Spec Failed", str(exc))
            return
        self._spec_path = Path(selected_path).expanduser().resolve()
        self._apply_spec_to_form(spec)
        self.spec_path_label.setText(f"Spec File: {self._spec_path}")
        self.status_label.setText("Spec loaded.")

    def save_spec(self) -> None:
        try:
            spec = self._collect_spec_from_form()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        path, note = self._resolve_spec_output_path(spec.model_name, prefer_current=True)
        try:
            saved_path = custom_model_generator.save_spec_file(spec, path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Spec Failed", str(exc))
            return
        self._spec_path = saved_path
        self.spec_path_label.setText(f"Spec File: {saved_path}")
        self.status_label.setText(note or "Spec saved.")

    def save_spec_as(self) -> None:
        try:
            spec = self._collect_spec_from_form()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        default_path = custom_model_generator.default_spec_path_for_model_name(spec.model_name)
        selected_path, _ = QFileDialog.getSaveFileName(self, "Save Spec As", str(default_path), "Spec JSON (*.json)")
        if not selected_path:
            return
        target_path, note = self._resolve_spec_output_path(spec.model_name, selected_path=selected_path)
        try:
            saved_path = custom_model_generator.save_spec_file(spec, target_path)
        except Exception as exc:
            QMessageBox.warning(self, "Save Spec Failed", str(exc))
            return
        self._spec_path = saved_path
        self.spec_path_label.setText(f"Spec File: {saved_path}")
        self.status_label.setText(note or "Spec saved as new file.")

    def generate_model(self) -> None:
        try:
            spec = self._collect_spec_from_form()
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
            saved_path = custom_model_generator.save_spec_file(spec, spec_path)
        except Exception as exc:
            QMessageBox.critical(self, "Generate Model Failed", str(exc))
            return

        self._spec_path = saved_path
        self.spec_path_label.setText(f"Spec File: {saved_path}")
        self.status_label.setText(path_note or "Model generated successfully.")

        parent = self.parent()
        if isinstance(parent, TrainingLauncher):
            parent.refresh_available_models(preferred_model=artifacts.model_name)
            parent.refresh_checkpoint_output_options(preserve_text=artifacts.model_name)
        QMessageBox.information(
            self,
            "Model Generated",
            f"Model file: {artifacts.model_file_path}\nSpec file: {artifacts.spec_file_path}",
        )


class LogPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMouseTracking(True)
        self.plot_title = "Run Plot"
        self.x_label = "Epoch"
        self.y_label = "Value"
        self.x_tick_labels: dict[float, str] = {}
        self.note = ""
        self.series: list[dict] = []
        self._point_hits: list[tuple[QPointF, str]] = []

    def set_plot(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        series: list[dict],
        note: str = "",
        x_tick_labels: dict[float, str] | None = None,
    ) -> None:
        self.plot_title = title
        self.x_label = x_label
        self.y_label = y_label
        self.series = series
        self.note = note
        self.x_tick_labels = x_tick_labels or {}
        self.update()

    def _build_x_ticks(self, x_min: float, x_max: float, max_ticks: int = 6) -> list[float]:
        if x_max <= x_min:
            return [x_min]
        if float(x_min).is_integer() and float(x_max).is_integer():
            start = int(round(x_min))
            end = int(round(x_max))
            span = max(end - start, 0)
            if span <= max_ticks - 1:
                return [float(value) for value in range(start, end + 1)]
            step = max(1, round(span / (max_ticks - 1)))
            ticks = [start]
            current = start
            while current + step < end:
                current += step
                ticks.append(current)
            if ticks[-1] != end:
                ticks.append(end)
            return [float(value) for value in ticks]
        return [x_min + (x_max - x_min) * (index / max(max_ticks - 1, 1)) for index in range(max_ticks)]

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:.6g}"

    def mouseMoveEvent(self, event) -> None:
        nearest_text = ""
        nearest_distance = 10.0
        cursor_pos = event.position()
        for point, text in self._point_hits:
            distance = ((point.x() - cursor_pos.x()) ** 2 + (point.y() - cursor_pos.y()) ** 2) ** 0.5
            if distance <= nearest_distance:
                nearest_distance = distance
                nearest_text = text
        if nearest_text:
            QToolTip.showText(event.globalPosition().toPoint(), nearest_text, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#11151a"))

        outer_rect = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(QPen(QColor("#313a47"), 1))
        painter.setBrush(QColor("#11151a"))
        painter.drawRoundedRect(outer_rect, 14, 14)

        title_rect = QRectF(outer_rect.left() + 16, outer_rect.top() + 12, outer_rect.width() - 32, 24)
        painter.setPen(QColor("#eef4fb"))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.plot_title)

        note_height = 18 if self.note else 0
        plot_rect = QRectF(
            outer_rect.left() + 58,
            outer_rect.top() + 56,
            max(outer_rect.width() - 84, 10),
            max(outer_rect.height() - 110 - note_height, 10),
        )

        if not self.series or not any(series.get("points") for series in self.series):
            self._point_hits = []
            painter.setPen(QColor("#93a0b2"))
            painter.drawText(plot_rect, Qt.AlignCenter, self.note or "No plot data available for this selection.")
            return

        x_values = [float(x) for series in self.series for x, _ in series.get("points", [])]
        y_values = [float(y) for series in self.series for _, y in series.get("points", [])]
        x_min = min(x_values) if x_values else 1.0
        x_max = max(x_values) if x_values else 1.0
        y_min = min(y_values) if y_values else 0.0
        y_max = max(y_values) if y_values else 1.0

        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            pad = 0.1 if y_max == 0 else abs(y_max) * 0.1
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min -= pad
            y_max += pad

        grid_pen = QPen(QColor("#28303b"), 1)
        axis_pen = QPen(QColor("#556070"), 1.3)
        label_pen = QPen(QColor("#aeb8c6"), 1)
        metrics = QFontMetrics(painter.font())

        for tick in range(5):
            fraction = tick / 4 if 4 > 0 else 0
            y = plot_rect.bottom() - fraction * plot_rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
            tick_value = y_min + fraction * (y_max - y_min)
            painter.setPen(label_pen)
            painter.drawText(QRectF(plot_rect.left() - 52, y - 10, 46, 20), Qt.AlignRight | Qt.AlignVCenter, f"{tick_value:.3g}")

        x_ticks = self._build_x_ticks(x_min, x_max)
        for tick_value in x_ticks:
            fraction = (tick_value - x_min) / (x_max - x_min) if x_max > x_min else 0.0
            x = plot_rect.left() + fraction * plot_rect.width()
            painter.setPen(grid_pen)
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
            painter.setPen(label_pen)
            tick_text = self.x_tick_labels.get(float(tick_value))
            if tick_text is None:
                tick_text = f"{tick_value:.0f}" if float(tick_value).is_integer() else f"{tick_value:.3g}"
            tick_width = max(40, min(110, metrics.horizontalAdvance(tick_text) + 8))
            painter.drawText(QRectF(x - tick_width / 2, plot_rect.bottom() + 6, tick_width, 18), Qt.AlignHCenter | Qt.AlignTop, tick_text)

        painter.setPen(axis_pen)
        painter.drawLine(plot_rect.left(), plot_rect.bottom(), plot_rect.right(), plot_rect.bottom())
        painter.drawLine(plot_rect.left(), plot_rect.top(), plot_rect.left(), plot_rect.bottom())

        def map_point(x_value: float, y_value: float) -> QPointF:
            x_ratio = (x_value - x_min) / (x_max - x_min)
            y_ratio = (y_value - y_min) / (y_max - y_min)
            return QPointF(
                plot_rect.left() + x_ratio * plot_rect.width(),
                plot_rect.bottom() - y_ratio * plot_rect.height(),
            )

        point_hits: list[tuple[QPointF, str]] = []
        for series in self.series:
            points = [(float(x), float(y)) for x, y in series.get("points", [])]
            if not points:
                continue
            label = str(series.get("label", "series"))
            color = QColor(series.get("color", "#4e8cff"))
            pen = QPen(color, 2.2)
            painter.setPen(pen)
            mapped_points = [map_point(x_value, y_value) for x_value, y_value in points]
            for point_index in range(len(mapped_points) - 1):
                painter.drawLine(mapped_points[point_index], mapped_points[point_index + 1])
            painter.setBrush(color)
            for point_index, point in enumerate(mapped_points):
                painter.drawEllipse(point, 3.2, 3.2)
                x_value, y_value = points[point_index]
                point_hits.append(
                    (
                        point,
                        f"{label}\n{self.x_label}: {self.x_tick_labels.get(float(x_value), self._format_value(x_value))}\n"
                        f"{self.y_label}: {self._format_value(y_value)}",
                    )
                )
        self._point_hits = point_hits

        painter.setPen(QColor("#aeb8c6"))
        painter.drawText(QRectF(plot_rect.left(), plot_rect.bottom() + 24, plot_rect.width(), 20), Qt.AlignCenter, self.x_label)

        painter.save()
        painter.translate(plot_rect.left() - 48, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_rect.height() / 2, -16, plot_rect.height(), 20), Qt.AlignCenter, self.y_label)
        painter.restore()

        legend_x = plot_rect.left()
        legend_y = outer_rect.top() + 32
        max_legend_width = plot_rect.width()
        row_height = 18
        current_x = legend_x
        current_y = legend_y
        for series in self.series:
            label = str(series.get("label", "series"))
            color = QColor(series.get("color", "#4e8cff"))
            item_width = 18 + metrics.horizontalAdvance(label) + 18
            if current_x + item_width > legend_x + max_legend_width:
                current_x = legend_x
                current_y += row_height
            painter.setPen(QPen(color, 5))
            painter.drawLine(current_x, current_y + 8, current_x + 12, current_y + 8)
            painter.setPen(QColor("#dfe7f3"))
            painter.drawText(QRectF(current_x + 16, current_y, item_width - 16, row_height), Qt.AlignLeft | Qt.AlignVCenter, label)
            current_x += item_width

        if self.note:
            painter.setPen(QColor("#93a0b2"))
            painter.drawText(
                QRectF(plot_rect.left(), outer_rect.bottom() - 28, plot_rect.width(), 20),
                Qt.AlignLeft | Qt.AlignVCenter,
                self.note,
            )


class ScatterPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMouseTracking(True)
        self.plot_title = "Efficiency Plot"
        self.x_label = "X"
        self.y_label = "Y"
        self.note = ""
        self.points: list[dict[str, object]] = []
        self._point_hits: list[tuple[QPointF, float, str]] = []

    def set_plot(self, *, title: str, x_label: str, y_label: str, points: list[dict[str, object]], note: str = "") -> None:
        self.plot_title = title
        self.x_label = x_label
        self.y_label = y_label
        self.points = points
        self.note = note
        self.update()

    def _build_x_ticks(self, x_min: float, x_max: float, max_ticks: int = 6) -> list[float]:
        if x_max <= x_min:
            return [x_min]
        return [x_min + (x_max - x_min) * (index / max(max_ticks - 1, 1)) for index in range(max_ticks)]

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:.6g}"

    def mouseMoveEvent(self, event) -> None:
        cursor_pos = event.position()
        for point, radius, text in self._point_hits:
            distance = ((point.x() - cursor_pos.x()) ** 2 + (point.y() - cursor_pos.y()) ** 2) ** 0.5
            if distance <= max(radius + 2.0, 8.0):
                QToolTip.showText(event.globalPosition().toPoint(), text, self)
                break
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#11151a"))

        outer_rect = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(QPen(QColor("#313a47"), 1))
        painter.setBrush(QColor("#11151a"))
        painter.drawRoundedRect(outer_rect, 14, 14)

        title_rect = QRectF(outer_rect.left() + 16, outer_rect.top() + 12, outer_rect.width() - 32, 24)
        painter.setPen(QColor("#eef4fb"))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.plot_title)

        note_height = 18 if self.note else 0
        plot_rect = QRectF(
            outer_rect.left() + 72,
            outer_rect.top() + 56,
            max(outer_rect.width() - 96, 10),
            max(outer_rect.height() - 110 - note_height, 10),
        )

        if not self.points:
            self._point_hits = []
            painter.setPen(QColor("#93a0b2"))
            painter.drawText(plot_rect, Qt.AlignCenter, self.note or "No efficiency data available for this selection.")
            return

        x_values = [float(point.get("x", 0.0)) for point in self.points]
        y_values = [float(point.get("y", 0.0)) for point in self.points]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)

        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        else:
            pad = (x_max - x_min) * 0.08
            x_min -= pad
            x_max += pad
        if y_min == y_max:
            pad = 0.1 if y_max == 0 else abs(y_max) * 0.1
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min -= pad
            y_max += pad

        grid_pen = QPen(QColor("#28303b"), 1)
        axis_pen = QPen(QColor("#556070"), 1.3)
        label_pen = QPen(QColor("#aeb8c6"), 1)

        for tick in range(5):
            fraction = tick / 4 if 4 > 0 else 0
            y = plot_rect.bottom() - fraction * plot_rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
            tick_value = y_min + fraction * (y_max - y_min)
            painter.setPen(label_pen)
            painter.drawText(QRectF(plot_rect.left() - 64, y - 10, 58, 20), Qt.AlignRight | Qt.AlignVCenter, f"{tick_value:.3g}")

        for tick_value in self._build_x_ticks(x_min, x_max, max_ticks=5):
            fraction = (tick_value - x_min) / (x_max - x_min) if x_max > x_min else 0.0
            x = plot_rect.left() + fraction * plot_rect.width()
            painter.setPen(grid_pen)
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
            painter.setPen(label_pen)
            painter.drawText(QRectF(x - 24, plot_rect.bottom() + 6, 48, 18), Qt.AlignHCenter | Qt.AlignTop, f"{tick_value:.3g}")

        painter.setPen(axis_pen)
        painter.drawLine(plot_rect.left(), plot_rect.bottom(), plot_rect.right(), plot_rect.bottom())
        painter.drawLine(plot_rect.left(), plot_rect.top(), plot_rect.left(), plot_rect.bottom())

        max_size = max(float(point.get("size", 1.0)) for point in self.points)

        def map_point(x_value: float, y_value: float) -> QPointF:
            x_ratio = (x_value - x_min) / (x_max - x_min)
            y_ratio = (y_value - y_min) / (y_max - y_min)
            return QPointF(
                plot_rect.left() + x_ratio * plot_rect.width(),
                plot_rect.bottom() - y_ratio * plot_rect.height(),
            )

        point_hits: list[tuple[QPointF, float, str]] = []
        for index, point in enumerate(self.points):
            mapped = map_point(float(point.get("x", 0.0)), float(point.get("y", 0.0)))
            color = QColor(point.get("color", "#4e8cff"))
            label = str(point.get("label", f"run-{index+1}"))
            x_value = float(point.get("x", 0.0))
            y_value = float(point.get("y", 0.0))
            size = float(point.get("size", 1.0))
            radius = 5.0 + (12.0 * (size / max(max_size, 1.0)))
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#dfe7f3"), 1))
            painter.drawEllipse(mapped, radius, radius)
            painter.setPen(QColor("#dfe7f3"))
            painter.drawText(QRectF(mapped.x() + radius + 4, mapped.y() - 10, 180, 20), Qt.AlignLeft | Qt.AlignVCenter, label)
            point_hits.append(
                (
                    mapped,
                    radius,
                    f"{label}\n{self.x_label}: {self._format_value(x_value)}\n{self.y_label}: {self._format_value(y_value)}",
                )
            )
        self._point_hits = point_hits

        painter.setPen(QColor("#aeb8c6"))
        painter.drawText(QRectF(plot_rect.left(), plot_rect.bottom() + 24, plot_rect.width(), 20), Qt.AlignCenter, self.x_label)

        painter.save()
        painter.translate(plot_rect.left() - 58, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_rect.height() / 2, -16, plot_rect.height(), 20), Qt.AlignCenter, self.y_label)
        painter.restore()

        if self.note:
            painter.setPen(QColor("#93a0b2"))
            painter.drawText(QRectF(plot_rect.left(), outer_rect.bottom() - 28, plot_rect.width(), 20), Qt.AlignLeft | Qt.AlignVCenter, self.note)


class ConfusionMatrixWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMouseTracking(True)
        self.title = "Confusion Matrix"
        self.labels: list[str] = []
        self.matrix: list[list[int]] = []
        self.note = ""
        self._cell_hits: list[tuple[QRect, str]] = []

    def set_matrix(self, *, title: str, labels: list[str], matrix: list[list[int]], note: str = "") -> None:
        self.title = title
        self.labels = labels
        self.matrix = matrix
        self.note = note
        self.update()

    def mouseMoveEvent(self, event) -> None:
        cursor_pos = event.position().toPoint()
        for rect, text in self._cell_hits:
            if rect.contains(cursor_pos):
                QToolTip.showText(event.globalPosition().toPoint(), text, self)
                break
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)

    def _elide_label(self, label: str, max_chars: int = 12) -> str:
        if len(label) <= max_chars:
            return label
        return label[: max_chars - 2] + ".."

    def _cell_color(self, value: int, max_value: int, diagonal: bool) -> QColor:
        import math

        normalized = 0.0
        if max_value > 0 and value > 0:
            normalized = math.log1p(float(value)) / math.log1p(float(max_value))
        normalized = max(0.0, min(normalized, 1.0))

        # Soft but readable heatmap: deep blue -> sky blue -> warm amber.
        stops = (
            ((12, 18, 28), 0.00),
            ((22, 44, 88), 0.16),
            ((41, 87, 148), 0.34),
            ((86, 145, 196), 0.56),
            ((163, 201, 226), 0.76),
            ((232, 196, 123), 0.90),
            ((243, 156, 74), 1.00),
        )

        for stop_index in range(len(stops) - 1):
            start_rgb, start_pos = stops[stop_index]
            end_rgb, end_pos = stops[stop_index + 1]
            if normalized <= end_pos:
                span = max(end_pos - start_pos, 1e-9)
                ratio = (normalized - start_pos) / span
                red = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
                green = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
                blue = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
                if diagonal and value > 0:
                    red = min(red + 8, 255)
                    green = min(green + 8, 255)
                    blue = min(blue + 8, 255)
                return QColor(red, green, blue)

        last_rgb = stops[-1][0]
        if diagonal and value > 0:
            return QColor(min(last_rgb[0] + 8, 255), min(last_rgb[1] + 8, 255), min(last_rgb[2] + 8, 255))
        return QColor(*last_rgb)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#11151a"))

        outer_rect = self.rect().adjusted(6, 6, -6, -6)
        painter.setPen(QPen(QColor("#313a47"), 1))
        painter.setBrush(QColor("#11151a"))
        painter.drawRoundedRect(outer_rect, 14, 14)

        title_rect = QRectF(outer_rect.left() + 16, outer_rect.top() + 12, outer_rect.width() - 32, 24)
        painter.setPen(QColor("#eef4fb"))
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, self.title)

        if not self.labels or not self.matrix:
            self._cell_hits = []
            painter.setPen(QColor("#93a0b2"))
            painter.drawText(outer_rect.adjusted(24, 48, -24, -24), Qt.AlignCenter, self.note or "No confusion matrix data available.")
            return

        top_label_band = 64
        left_label_band = 110
        right_legend_band = 64
        bottom_band = 64
        matrix_rect = QRectF(
            outer_rect.left() + left_label_band,
            outer_rect.top() + top_label_band,
            max(outer_rect.width() - left_label_band - right_legend_band - 24, 10),
            max(outer_rect.height() - top_label_band - bottom_band - 24, 10),
        )
        size = len(self.labels)
        cell_size = min(matrix_rect.width() / max(size, 1), matrix_rect.height() / max(size, 1))
        grid_width = cell_size * size
        grid_height = cell_size * size
        start_x = matrix_rect.left() + max((matrix_rect.width() - grid_width) / 2.0, 0.0)
        start_y = matrix_rect.top()
        max_value = max(max(row) for row in self.matrix) if self.matrix else 1

        panel_rect = QRectF(start_x - 10, start_y - 10, grid_width + 20, grid_height + 20)
        painter.setPen(QPen(QColor("#273244"), 1))
        painter.setBrush(QColor("#0f141c"))
        painter.drawRoundedRect(panel_rect, 10, 10)

        cell_hits: list[tuple[QRect, str]] = []
        # Draw the heatmap as pixel-aligned solid cells so Qt smoothing does not wash the colors out.
        painter.setRenderHint(QPainter.Antialiasing, False)
        for row_index, row in enumerate(self.matrix):
            for col_index, value in enumerate(row):
                diagonal = row_index == col_index
                color = self._cell_color(value, max_value, diagonal)
                x0 = round(start_x + col_index * cell_size)
                y0 = round(start_y + row_index * cell_size)
                x1 = round(start_x + (col_index + 1) * cell_size)
                y1 = round(start_y + (row_index + 1) * cell_size)
                cell_rect = QRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawRect(cell_rect)
                border_color = QColor("#50627a") if diagonal else QColor("#243041")
                painter.setPen(QPen(border_color, 0.8))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(cell_rect)
                true_label = self.labels[row_index] if row_index < len(self.labels) else str(row_index)
                pred_label = self.labels[col_index] if col_index < len(self.labels) else str(col_index)
                cell_hits.append((QRect(cell_rect), f"True: {true_label}\nPred: {pred_label}\nCount: {value}"))
                if cell_size >= 28:
                    import math
                    normalized = 0.0 if max_value <= 0 or value <= 0 else math.log1p(float(value)) / math.log1p(float(max_value))
                    text_color = QColor("#18212f") if normalized >= 0.72 else QColor("#f8fbff")
                    painter.setPen(text_color)
                    painter.drawText(cell_rect, Qt.AlignCenter, str(value))
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._cell_hits = cell_hits

        painter.setPen(QColor("#b6c4d6"))
        for index, label in enumerate(self.labels):
            short_label = self._elide_label(label)
            x_rect = QRectF(start_x + index * cell_size, start_y - 58, cell_size, 52)
            y_rect = QRectF(start_x - 106, start_y + index * cell_size, 100, cell_size)
            painter.save()
            painter.translate(x_rect.center().x(), x_rect.bottom())
            painter.rotate(-35)
            painter.drawText(QRectF(-cell_size * 0.45, -18, cell_size * 0.9, 18), Qt.AlignLeft | Qt.AlignVCenter, short_label)
            painter.restore()
            painter.drawText(y_rect, Qt.AlignRight | Qt.AlignVCenter, short_label)

        painter.setPen(QColor("#dfe7f3"))
        painter.drawText(QRectF(start_x, start_y + grid_height + 14, grid_width, 20), Qt.AlignCenter, "Predicted Label")
        painter.save()
        painter.translate(start_x - 94, start_y + grid_height / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-grid_height / 2, -16, grid_height, 20), Qt.AlignCenter, "True Label")
        painter.restore()

        legend_x = start_x + grid_width + 16
        legend_y = start_y + 2
        legend_height = min(180.0, grid_height)
        legend_rect = QRectF(legend_x, legend_y, 24, legend_height)
        if legend_rect.right() <= outer_rect.right() - 8:
            slot_rect = legend_rect.adjusted(-4, -4, 4, 4)
            painter.setPen(QPen(QColor("#425168"), 1))
            painter.setBrush(QColor("#101826"))
            painter.drawRoundedRect(slot_rect, 5, 5)
            painter.setRenderHint(QPainter.Antialiasing, False)
            steps = max(int(legend_height), 1)
            for step in range(steps):
                ratio = 1.0 - (step / max(steps - 1, 1))
                sample_color = self._cell_color(int(round(ratio * max_value)), max_value, diagonal=False)
                y0 = round(legend_rect.top() + step)
                y1 = round(legend_rect.top() + step + 1)
                painter.setPen(Qt.NoPen)
                painter.setBrush(sample_color)
                painter.drawRect(QRect(round(legend_rect.left()), y0, max(1, round(legend_rect.width())), max(1, y1 - y0)))
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor("#5d7391"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(legend_rect)
            painter.setPen(QColor("#9fb1c8"))
            painter.drawText(QRectF(legend_rect.right() + 8, legend_rect.top() - 8, 42, 18), Qt.AlignLeft | Qt.AlignVCenter, str(max_value))
            painter.drawText(QRectF(legend_rect.right() + 8, legend_rect.center().y() - 9, 42, 18), Qt.AlignLeft | Qt.AlignVCenter, str(max_value // 2))
            painter.drawText(QRectF(legend_rect.right() + 8, legend_rect.bottom() - 10, 42, 18), Qt.AlignLeft | Qt.AlignVCenter, "0")
            painter.drawText(QRectF(legend_rect.left() - 2, legend_rect.top() - 26, 64, 18), Qt.AlignLeft | Qt.AlignVCenter, "Count")

        if self.note:
            painter.setPen(QColor("#93a0b2"))
            painter.drawText(QRectF(start_x, outer_rect.bottom() - 28, min(grid_width + 120, outer_rect.width() - 32), 20), Qt.AlignLeft | Qt.AlignVCenter, self.note)


class TrainingLauncher(QMainWindow):
    def __init__(self, startup_progress_callback=None) -> None:
        super().__init__()
        self._startup_progress_callback = startup_progress_callback
        self.setWindowTitle("Training Launcher")
        self.resize(1080, 820)
        self.setMinimumSize(920, 680)
        if APP_ICON_PATH.is_file():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.started.connect(self.on_process_started)
        self.process.finished.connect(self.on_process_finished)
        self.process.errorOccurred.connect(self.on_process_error)

        self.data_process = QProcess(self)
        self.data_process.setWorkingDirectory(str(PROJECT_ROOT))
        self.data_process.setProcessChannelMode(QProcess.MergedChannels)
        self.data_process.readyReadStandardOutput.connect(self.handle_data_output)
        self.data_process.started.connect(self.on_data_process_started)
        self.data_process.finished.connect(self.on_data_process_finished)
        self.data_process.errorOccurred.connect(self.on_data_process_error)

        self.predict_process = QProcess(self)
        self.predict_process.setWorkingDirectory(str(PROJECT_ROOT))
        self.predict_process.setProcessChannelMode(QProcess.MergedChannels)
        self.predict_process.readyReadStandardOutput.connect(self.handle_predict_process_output)
        self.predict_process.started.connect(self.on_predict_process_started)
        self.predict_process.finished.connect(self.on_predict_process_finished)
        self.predict_process.errorOccurred.connect(self.on_predict_process_error)
        self._predict_process_output = ""
        self._predict_process_json_path: Path | None = None
        self._predict_process_input_list_path: Path | None = None
        self._predict_process_started_at: float | None = None

        self._committed_output = ""
        self._stream_buffer = ""
        self._data_committed_output = ""
        self._data_stream_buffer = ""
        self.predict_image_paths: list[Path] = []
        self.predict_results: list[dict[str, str | float | bool | None]] = []
        self.current_predict_index = -1
        self.predict_thread: QThread | None = None
        self.predict_worker: PredictionWorker | None = None
        self.predict_compact_built = False
        self.predict_compact_loading = False
        self.predict_compact_pending_indices: list[int] = []
        self.predict_browser_render_key: tuple[int, int, str] | None = None
        self.predict_thumbnail_cache: OrderedDict[str, QIcon] = OrderedDict()
        self.predict_display_cache: OrderedDict[tuple[str, int, int], QPixmap] = OrderedDict()
        self.predict_compare_items: list[dict[str, object]] = []
        self.predict_gradcam_cache: OrderedDict[tuple[str, str, str, int, str], QPixmap] = OrderedDict()
        self.predict_gradcam_diagnostics: OrderedDict[tuple[str, str, str, int, str], str] = OrderedDict()
        self.predict_compare_display_cache: OrderedDict[tuple[object, ...], QPixmap] = OrderedDict()
        self.predict_gradcam_thread: QThread | None = None
        self.predict_gradcam_worker: GradCamComparisonWorker | None = None
        self.predict_gradcam_request_key: tuple[object, ...] | None = None
        self.predict_gradcam_pending_request: dict[str, object] | None = None
        self._predict_checkpoint_selector_syncing = False
        self.predict_resize_timer = QTimer(self)
        self.predict_resize_timer.setSingleShot(True)
        self.predict_resize_timer.timeout.connect(self._refresh_predict_after_resize)
        self.predict_browser_thumbnail_timer = QTimer(self)
        self.predict_browser_thumbnail_timer.setSingleShot(True)
        self.predict_browser_thumbnail_timer.timeout.connect(self.process_predict_compact_thumbnail_batch)
        self.predict_detected_model_name: str | None = None
        self.test_split_thread: QThread | None = None
        self.test_split_worker: TestSplitEvaluationWorker | None = None
        self.test_split_detected_model_name: str | None = None
        self._last_export_notebook_path: Path | None = None
        self.available_models = sort_model_names_for_ui(
            discover_model_names_generated_first(include_legacy_fallback=True)
        )
        self._checkpoint_name_locked_to_model = True
        self._last_training_model_name = self.available_models[0] if self.available_models else ""
        self._last_predict_model_name = self.available_models[0] if self.available_models else ""
        self._stop_request_path: Path | None = None
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        saved_theme = str(self.settings.value("ui/theme", app_themes.DEFAULT_THEME_KEY))
        self.current_theme_key = saved_theme if saved_theme in app_themes.THEMES else app_themes.DEFAULT_THEME_KEY

        self._startup_progress("Preparing controls...")
        self._init_data_controls()
        self._init_training_controls()
        self._startup_progress("Preparing prediction workspace...")
        self._init_prediction_controls()
        self._init_test_split_controls()
        self._startup_progress("Preparing logs and queue...")
        self._init_log_controls()
        self._init_global_ui_controls()
        self._startup_progress("Building interface...")
        self._build_ui()
        self._install_wheel_guards()
        self.apply_visual_design()
        self._startup_progress("Loading model and checkpoint index...")
        self.refresh_predict_checkpoint_selector(select_default=True)
        self.refresh_training_settings_summary()
        self.refresh_command_preview()
        self.update_predict_detected_model()
        self.update_test_split_detected_model()
        self.refresh_predict_compare_summary()
        self.refresh_predict_page()
        self.on_predict_browser_mode_changed()
        self.on_predict_compact_toggled(self.predict_compact_checkbox.isChecked())
        self._startup_progress("Loading run logs...")
        self.refresh_training_log_runs()
        self._startup_progress("Ready.")

    def _startup_progress(self, message: str) -> None:
        callback = self._startup_progress_callback
        if callback is not None:
            try:
                callback(str(message))
            except Exception:
                pass
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _init_global_ui_controls(self) -> None:
        self.theme_label = QLabel("Theme")
        self.theme_label.setProperty("muted", True)
        self.theme_combo = QComboBox()
        for key, display_name in app_themes.theme_display_names():
            self.theme_combo.addItem(display_name, key)
        current_index = self.theme_combo.findData(self.current_theme_key)
        if current_index >= 0:
            self.theme_combo.setCurrentIndex(current_index)
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)

    def _init_training_controls(self) -> None:
        self.model_combo = DeletableComboBox()
        self._set_training_model_combo_items(self.available_models)
        self.model_combo.setToolTip("Choose which model architecture to train.")
        self.model_combo.setMinimumHeight(34)
        self.model_combo.deleteRequested.connect(self.delete_training_model)
        self.training_model_variant_label = QLabel("Generated models are preferred. Legacy models remain available as fallback.")
        self.training_model_variant_label.setWordWrap(True)
        self.training_model_variant_label.setProperty("muted", True)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10_000)
        self.epochs_spin.setValue(3)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 8_192)
        self.batch_size_spin.setValue(32)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 64)
        self.num_workers_spin.setValue(4)

        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(32, 2_048)
        self.image_size_spin.setValue(224)

        self.train_transforms_preset_combo = QComboBox()
        self.train_transforms_preset_combo.addItems(["baseline", "standard", "robust", "downsample_focus", "custom"])
        self.train_transforms_preset_combo.setCurrentText("baseline")
        self.train_transforms_preset_combo.setToolTip("Preset for training-time augmentation. Validation and test transforms stay deterministic.")

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0, 10.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["sgd", "adam", "adamw"])
        self.optimizer_combo.setCurrentText("adam")
        self.optimizer_combo.setToolTip("Optimizer used for trainable parameters.")

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["none", "cosine", "step", "plateau"])
        self.scheduler_combo.setCurrentText("none")

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)

        self.mild_blur_enabled = False
        self.mild_blur_prob = 0.10
        self.custom_downsample_enabled = True
        self.custom_downsample_prob = 0.65
        self.custom_downsample_min_scale = 0.18
        self.custom_downsample_max_scale = 0.55
        self.custom_mild_blur_enabled = False
        self.custom_mild_blur_prob = 0.10
        self.custom_random_erasing_enabled = True
        self.custom_random_erasing_prob = 0.08
        self.custom_color_jitter_enabled = True
        self.custom_horizontal_flip_enabled = True

        self.freeze_checkbox = QCheckBox("Freeze backbone")
        self.freeze_checkbox.setChecked(True)
        self.freeze_checkbox.setToolTip("Train the classifier head while keeping most pretrained backbone weights frozen.")

        self.amp_checkbox = QCheckBox("Use AMP")
        self.amp_checkbox.setChecked(False)
        self.amp_checkbox.setToolTip("Enable automatic mixed precision when supported by the selected device.")

        self.validation_checkbox = QCheckBox("Use validation split")
        self.validation_checkbox.setChecked(False)
        self.validation_checkbox.setToolTip("Reserve part of the training data for validation instead of training on the full set.")

        self.validation_proportion_spin = QDoubleSpinBox()
        self.validation_proportion_spin.setRange(0.01, 0.99)
        self.validation_proportion_spin.setDecimals(2)
        self.validation_proportion_spin.setSingleStep(0.01)
        self.validation_proportion_spin.setValue(0.10)
        self.validation_proportion_spin.setToolTip("Fraction of the training set to use as validation data when validation split is enabled.")

        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        self.resume_checkbox.setChecked(False)

        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText("Select a checkpoint file such as best.pth or last.pth")

        self.resume_browse_button = QPushButton("Browse...")
        self.resume_browse_button.clicked.connect(self.choose_resume_path)

        self.resume_clear_button = QPushButton("Clear")
        self.resume_clear_button.clicked.connect(self.clear_resume_path)

        self.checkpoint_output_combo = QComboBox()
        self.checkpoint_output_combo.setEditable(True)
        self.refresh_checkpoint_output_options()
        self.checkpoint_output_combo.setToolTip("Name of the checkpoint folder for this run. Existing names can be reused to resume or continue work.")
        self.checkpoint_output_combo.setMinimumHeight(34)
        checkpoint_line_edit = self.checkpoint_output_combo.lineEdit()
        if checkpoint_line_edit is not None:
            checkpoint_line_edit.setPlaceholderText("e.g. resnet18_baseline_trial1")

        self.data_root_label = QLabel(str(DEFAULT_DATA_ROOT))
        self.data_root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_root_label.setWordWrap(True)
        self.data_root_label.setProperty("muted", True)
        self.data_root_label.setProperty("readonlyDisplay", True)

        self.checkpoint_dir_label = QLabel(str(self.selected_checkpoint_dir()))
        self.checkpoint_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.checkpoint_dir_label.setWordWrap(True)
        self.checkpoint_dir_label.setProperty("muted", True)
        self.checkpoint_dir_label.setProperty("readonlyDisplay", True)

        self.command_preview = QLabel()
        self.command_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command_preview.setWordWrap(True)
        self.command_preview.setProperty("muted", True)
        self.command_preview.setProperty("codeblock", True)
        self.command_preview.setTextFormat(Qt.PlainText)

        self.command_preview_toggle = QCheckBox("Show command preview")
        self.command_preview_toggle.setChecked(False)
        self.command_preview_toggle.setToolTip("Expand to inspect the exact command that will be launched.")

        self.training_settings_button = QPushButton("⚙")
        self.training_settings_button.setText("Advanced")
        self.training_settings_button.setToolTip("Open advanced training settings")
        self.training_settings_button.setFixedHeight(32)
        self.training_settings_button.clicked.connect(self.open_training_settings_dialog)

        self.training_settings_summary = QLabel()
        self.training_settings_summary.setWordWrap(True)
        self.training_settings_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.training_settings_summary.setProperty("muted", True)
        self.training_settings_summary.setProperty("readonlyDisplay", True)

        self.export_include_paths_checkbox = QCheckBox("Include path setup")
        self.export_include_paths_checkbox.setChecked(True)
        self.export_include_paths_checkbox.setToolTip(
            "Enable for the first export into a notebook. Disable later exports to copy only the training command cell."
        )

        self.export_command_button = QPushButton("Export Command as Python Code")
        self.export_command_button.clicked.connect(self.export_command_as_python_code)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(120)
        self.output_text.setMaximumHeight(230)
        self.output_text.setPlaceholderText("Training logs and launch details will appear here.")

        self.progress_label = QLabel("Progress will appear here after training starts.")
        self.progress_label.setWordWrap(True)
        self.progress_label.setProperty("muted", True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setMinimumWidth(104)
        self.train_queue_button = QPushButton("Add to Queue")
        self.train_queue_button.clicked.connect(self.add_current_training_config_to_queue)
        self.train_queue_button.setMinimumWidth(118)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setMinimumWidth(88)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("SectionStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(132)
        self.global_queue_jobs: list[dict[str, object]] = []
        self.global_queue_list = QListWidget()
        self.global_queue_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.global_queue_list.setMinimumHeight(220)
        self.global_queue_list.itemSelectionChanged.connect(self.on_global_queue_selection_changed)
        self.queue_remove_button = QPushButton("Remove Selected")
        self.queue_remove_button.clicked.connect(self.remove_selected_global_queue_job)
        self.queue_duplicate_button = QPushButton("Duplicate Selected")
        self.queue_duplicate_button.clicked.connect(self.duplicate_selected_global_queue_job)
        self.queue_follow_on_test_split_button = QPushButton("Add Follow-on Test Split")
        self.queue_follow_on_test_split_button.clicked.connect(self.add_follow_on_test_split_for_selected_job)
        self.queue_follow_on_test_split_button.setEnabled(False)
        self.queue_move_up_button = QPushButton("Move Up")
        self.queue_move_up_button.clicked.connect(self.move_selected_global_queue_job_up)
        self.queue_move_down_button = QPushButton("Move Down")
        self.queue_move_down_button.clicked.connect(self.move_selected_global_queue_job_down)
        self.queue_run_button = QPushButton("Run Queue")
        self.queue_run_button.clicked.connect(self.run_global_queue)
        self.queue_stop_button = QPushButton("Stop Current / Pause")
        self.queue_stop_button.clicked.connect(self.stop_current_global_job)
        self.queue_stop_button.setEnabled(False)
        self.queue_clear_finished_button = QPushButton("Clear Finished")
        self.queue_clear_finished_button.clicked.connect(self.clear_finished_global_queue_jobs)
        self.global_queue_status_label = QLabel("Queue is empty.")
        self.global_queue_status_label.setWordWrap(True)
        self.global_queue_status_label.setProperty("muted", True)
        self.global_queue_running = False
        self.global_queue_stop_requested = False
        self.active_queue_job_id: str | None = None
        self.active_queue_job_type: str | None = None
        self.active_job_origin = "manual"
        self.active_job_config_snapshot: dict[str, object] | None = None
        self.training_stop_requested = False
        self.global_queue_button = QPushButton("Queue")
        self.global_queue_button.setCheckable(True)
        self.global_queue_button.setChecked(True)
        self.training_validation_proportion_label: QLabel | None = None
        self.training_resume_path_label: QLabel | None = None
        self.training_resume_path_widget: QWidget | None = None
        self.command_preview_body: QWidget | None = None
        self.training_run_name_label: QLabel | None = None
        self.training_data_root_title: QLabel | None = None
        self.training_checkpoint_dir_title: QLabel | None = None
        self.training_advanced_title: QLabel | None = None

    def _install_wheel_guards(self) -> None:
        for widget in self.findChildren(QWidget):
            if isinstance(widget, (QComboBox, QSpinBox, QDoubleSpinBox)):
                widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event) -> bool:
        if event.type() == QEvent.Wheel and isinstance(watched, (QComboBox, QSpinBox, QDoubleSpinBox)):
            # Disable wheel-driven value changes to prevent accidental edits while scrolling.
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _make_training_row_label(self, text: str, *, prominent: bool = False) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setMinimumWidth(148 if prominent else 136)
        if prominent:
            font = label.font()
            font.setBold(True)
            label.setFont(font)
        return label

    def _create_training_labeled_field(self, label_text: str, field: QWidget, *, prominent: bool = False) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(label_text)
        label.setProperty("sectionHint", True)
        if prominent:
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            field.setMinimumHeight(max(field.minimumHeight(), 34))
        layout.addWidget(label)
        layout.addWidget(field)
        return container

    def custom_augmentation_config(self) -> dict[str, object]:
        return {
            "downsample": {
                "enabled": bool(self.custom_downsample_enabled),
                "probability": float(self.custom_downsample_prob),
                "min_scale": float(self.custom_downsample_min_scale),
                "max_scale": float(self.custom_downsample_max_scale),
            },
            "mild_blur": {
                "enabled": bool(self.custom_mild_blur_enabled),
                "probability": float(self.custom_mild_blur_prob),
            },
            "random_erasing": {
                "enabled": bool(self.custom_random_erasing_enabled),
                "probability": float(self.custom_random_erasing_prob),
            },
            "color_jitter": {
                "enabled": bool(self.custom_color_jitter_enabled),
            },
            "horizontal_flip": {
                "enabled": bool(self.custom_horizontal_flip_enabled),
            },
        }

    def custom_augmentation_summary(self) -> str:
        config = self.custom_augmentation_config()
        parts: list[str] = []
        downsample = config["downsample"]
        assert isinstance(downsample, dict)
        if downsample.get("enabled"):
            parts.append(
                f"downsample p={float(downsample.get('probability', 0.0)):.2f} "
                f"scale={float(downsample.get('min_scale', 0.0)):.2f}-{float(downsample.get('max_scale', 0.0)):.2f}"
            )
        blur = config["mild_blur"]
        assert isinstance(blur, dict)
        if blur.get("enabled"):
            parts.append(f"blur p={float(blur.get('probability', 0.0)):.2f}")
        erasing = config["random_erasing"]
        assert isinstance(erasing, dict)
        if erasing.get("enabled"):
            parts.append(f"erase p={float(erasing.get('probability', 0.0)):.2f}")
        if bool(config["color_jitter"].get("enabled")):
            parts.append("jitter")
        if bool(config["horizontal_flip"].get("enabled")):
            parts.append("hflip")
        return ", ".join(parts) if parts else "custom minimal"

    def collect_training_config_snapshot(self) -> dict[str, object]:
        checkpoint_dir = self.selected_checkpoint_dir()
        return {
            "job_id": uuid.uuid4().hex[:8],
            "model": self.current_training_model_name(),
            "data_root": str(DEFAULT_DATA_ROOT),
            "checkpoint_name": self.checkpoint_output_name(),
            "checkpoint_dir": str(checkpoint_dir),
            "epochs": int(self.epochs_spin.value()),
            "batch_size": int(self.batch_size_spin.value()),
            "num_workers": int(self.num_workers_spin.value()),
            "image_size": int(self.image_size_spin.value()),
            "train_transforms_preset": self.train_transforms_preset_combo.currentText(),
            "lr": float(self.lr_spin.value()),
            "optimizer": self.optimizer_combo.currentText(),
            "scheduler": self.scheduler_combo.currentText(),
            "seed": int(self.seed_spin.value()),
            "mild_blur_enabled": bool(self.mild_blur_enabled),
            "mild_blur_prob": float(self.mild_blur_prob),
            "custom_augmentation": self.custom_augmentation_config(),
            "device": self.device_combo.currentText(),
            "amp": bool(self.amp_checkbox.isChecked()),
            "freeze_backbone": bool(self.freeze_checkbox.isChecked()),
            "use_validation_split": bool(self.validation_checkbox.isChecked()),
            "validation_proportion": float(self.validation_proportion_spin.value()),
            "resume_enabled": bool(self.resume_checkbox.isChecked()),
            "resume_path": self.resume_path_edit.text().strip(),
        }

    def training_config_summary(self, config: dict[str, object]) -> str:
        preset = str(config.get("train_transforms_preset", "baseline"))
        if preset == "custom":
            transform_text = f"custom ({self.describe_custom_augmentation_config(config.get('custom_augmentation'))})"
        elif bool(config.get("mild_blur_enabled", False)):
            transform_text = f"{preset} + blur {float(config.get('mild_blur_prob', 0.0)):.2f}"
        else:
            transform_text = preset
        return (
            f"{config.get('model', '-')}"
            f" | run={config.get('checkpoint_name', '-')}"
            f" | {transform_text}"
            f" | e={config.get('epochs', '-')}"
            f" bs={config.get('batch_size', '-')}"
            f" lr={float(config.get('lr', 0.0)):.4g}"
            f" opt={config.get('optimizer', '-')}"
        )

    def describe_custom_augmentation_config(self, config: object) -> str:
        if not isinstance(config, dict):
            return "custom"
        parts: list[str] = []
        downsample = config.get("downsample")
        if isinstance(downsample, dict) and downsample.get("enabled"):
            parts.append(
                f"downsample {float(downsample.get('probability', 0.0)):.2f} "
                f"[{float(downsample.get('min_scale', 0.0)):.2f}-{float(downsample.get('max_scale', 0.0)):.2f}]"
            )
        blur = config.get("mild_blur")
        if isinstance(blur, dict) and blur.get("enabled"):
            parts.append(f"blur {float(blur.get('probability', 0.0)):.2f}")
        erasing = config.get("random_erasing")
        if isinstance(erasing, dict) and erasing.get("enabled"):
            parts.append(f"erase {float(erasing.get('probability', 0.0)):.2f}")
        jitter = config.get("color_jitter")
        if isinstance(jitter, dict) and jitter.get("enabled"):
            parts.append("jitter")
        hflip = config.get("horizontal_flip")
        if isinstance(hflip, dict) and hflip.get("enabled"):
            parts.append("hflip")
        return ", ".join(parts) if parts else "custom"

    def _init_data_controls(self) -> None:
        self.data_dir_label = QLabel(str(DEFAULT_DATA_DIR))
        self.data_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_dir_label.setWordWrap(True)

        self.dataset_root_label = QLabel(str(DEFAULT_DATA_ROOT))
        self.dataset_root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dataset_root_label.setWordWrap(True)

        self.data_check_button = QPushButton("Check Data")
        self.data_check_button.clicked.connect(self.run_data_check)

        self.data_prepare_button = QPushButton("Prepare Data")
        self.data_prepare_button.clicked.connect(self.run_data_prepare)

        self.data_force_button = QPushButton("Force Redownload")
        self.data_force_button.clicked.connect(self.run_data_force_redownload)

        self.data_status_label = QLabel("Idle")
        self.data_status_label.setWordWrap(True)
        self.data_status_label.setObjectName("SectionStatus")

        self.data_task_value_label = QLabel("Idle")
        self.data_task_value_label.setWordWrap(True)

        self.data_state_value_label = QLabel("Ready")
        self.data_state_value_label.setWordWrap(True)

        self.data_target_value_label = QLabel(str(DEFAULT_DATA_ROOT))
        self.data_target_value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_target_value_label.setWordWrap(True)

        self.data_last_result_value_label = QLabel("No dataset task has been run yet.")
        self.data_last_result_value_label.setWordWrap(True)

        self.data_progress_label = QLabel("Dataset status will appear here.")
        self.data_progress_label.setWordWrap(True)
        self.data_progress_label.setProperty("muted", True)

        self.data_progress_bar = QProgressBar()
        self.data_progress_bar.setRange(0, 100)
        self.data_progress_bar.setValue(0)
        self.data_progress_bar.setFormat("%p%")

        self.data_output_text = QPlainTextEdit()
        self.data_output_text.setReadOnly(True)
        self.data_output_text.setMaximumHeight(220)
        self.data_output_text.setPlaceholderText("Dataset checks, downloads, and extraction details will appear here.")

    def _init_prediction_controls(self) -> None:
        self.predict_model_combo = QComboBox()
        self.predict_model_combo.addItems(self.available_models)
        self.predict_model_combo.hide()

        self.predict_checkpoint_tree = QTreeWidget()
        self.predict_checkpoint_tree.setColumnCount(3)
        self.predict_checkpoint_tree.setHeaderLabels(["Model", "Best", "Last"])
        self.predict_checkpoint_tree.setRootIsDecorated(False)
        self.predict_checkpoint_tree.setAlternatingRowColors(True)
        self.predict_checkpoint_tree.setIndentation(0)
        self.predict_checkpoint_tree.setTextElideMode(Qt.ElideRight)
        self.predict_checkpoint_tree.setMinimumHeight(176)
        self.predict_checkpoint_tree.setMaximumHeight(210)
        self.predict_checkpoint_tree.itemChanged.connect(self.on_predict_checkpoint_tree_item_changed)

        self.predict_select_all_best_button = QPushButton("Select All Best")
        self.predict_select_all_best_button.clicked.connect(self.select_all_predict_best_checkpoints)
        self.predict_select_all_best_button.setFixedHeight(28)

        self.predict_clear_selection_button = QPushButton("Clear")
        self.predict_clear_selection_button.clicked.connect(self.clear_predict_checkpoint_selection)
        self.predict_clear_selection_button.setFixedWidth(68)
        self.predict_clear_selection_button.setFixedHeight(28)

        self.predict_device_combo = QComboBox()
        self.predict_device_combo.addItems(["auto", "cpu", "cuda"])

        self.predict_image_size_spin = QSpinBox()
        self.predict_image_size_spin.setRange(32, 2048)
        self.predict_image_size_spin.setValue(224)

        self.predict_checkpoint_edit = QLineEdit(str(self.default_predict_checkpoint_path()))
        self.predict_checkpoint_edit.setReadOnly(True)
        self.predict_checkpoint_edit.editingFinished.connect(self.update_predict_detected_model)
        self.predict_checkpoint_browse_button = QPushButton("Browse...")
        self.predict_checkpoint_browse_button.clicked.connect(self.choose_predict_checkpoint)
        self.predict_checkpoint_browse_button.hide()
        self.predict_model_combo.currentTextChanged.connect(self.on_predict_model_changed)

        self.predict_detected_model_label = QLabel("Model will be auto-detected from the checkpoint.")
        self.predict_detected_model_label.setWordWrap(True)
        self.predict_detected_model_label.setProperty("muted", True)

        self.predict_select_images_button = QPushButton("Select Images")
        self.predict_select_images_button.clicked.connect(self.choose_predict_images)

        self.predict_select_folder_button = QPushButton("Select Folders")
        self.predict_select_folder_button.clicked.connect(self.choose_predict_folders)

        self.predict_run_button = QPushButton("Predict")
        self.predict_run_button.clicked.connect(self.run_predictions)
        self.predict_queue_button = QPushButton("Add to Queue")
        self.predict_queue_button.clicked.connect(self.add_current_predict_config_to_queue)

        self.predict_compact_checkbox = QCheckBox("Compact Mode")
        self.predict_compact_checkbox.toggled.connect(self.on_predict_compact_toggled)
        self.predict_compact_checkbox.hide()

        self.predict_compare_checkbox = QCheckBox("Model Comparison")
        self.predict_compare_checkbox.toggled.connect(self.on_predict_compare_toggled)
        self.predict_compare_checkbox.setEnabled(False)
        self.predict_compare_checkbox.setToolTip("Automatically enabled when two or more checkpoints are selected.")

        self.predict_compare_models_button = QPushButton("Add")
        self.predict_compare_models_button.clicked.connect(self.select_all_predict_best_checkpoints)
        self.predict_compare_models_button.setText("Select All Best")
        self.predict_compare_models_button.setFixedHeight(28)

        self.predict_compare_clear_button = QPushButton("Clear")
        self.predict_compare_clear_button.clicked.connect(self.clear_predict_checkpoint_selection)
        self.predict_compare_clear_button.setFixedWidth(68)
        self.predict_compare_clear_button.setFixedHeight(28)

        self.predict_compare_models_label = QLabel("No checkpoints selected.")
        self.predict_compare_models_label.setWordWrap(False)
        self.predict_compare_models_label.setProperty("muted", True)
        self.predict_compare_models_label.setMaximumHeight(28)
        self.predict_compare_models_label.setMinimumHeight(28)
        self.predict_compare_models_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.predict_export_include_paths_checkbox = QCheckBox("Include path setup")
        self.predict_export_include_paths_checkbox.setChecked(True)

        self.predict_export_button = QPushButton("Export Predicting as Python Code")
        self.predict_export_button.clicked.connect(self.export_predicting_as_python_code)

        self.predict_gradcam_button = QPushButton("Generate / Show Grad-CAM")
        self.predict_gradcam_button.clicked.connect(self.show_predict_gradcam_for_current_page)
        self.predict_gradcam_button.setEnabled(False)
        self.predict_gradcam_button.setToolTip("Generate Grad-CAM overlays for the current compare image, or show cached overlays if already available.")

        self.predict_prev_button = QPushButton("Previous")
        self.predict_prev_button.clicked.connect(self.show_previous_prediction)

        self.predict_next_button = QPushButton("Next")
        self.predict_next_button.clicked.connect(self.show_next_prediction)

        self.predict_selected_label = QLabel("No images selected.")
        self.predict_selected_label.setWordWrap(True)
        self.predict_selected_label.setProperty("muted", True)

        self.predict_browser_mode_combo = QComboBox()
        self.predict_browser_mode_combo.addItem("Thumbnails", "thumbnails")
        self.predict_browser_mode_combo.addItem("List", "list")
        self.predict_browser_mode_combo.currentIndexChanged.connect(self.on_predict_browser_mode_changed)

        self.predict_status_label = QLabel("Ready.")
        self.predict_status_label.setWordWrap(True)
        self.predict_status_label.setObjectName("SectionStatus")
        self.predict_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.predict_progress_bar = QProgressBar()
        self.predict_progress_bar.setRange(0, 100)
        self.predict_progress_bar.setValue(0)
        self.predict_progress_bar.setFormat("%p%")
        self.predict_progress_bar.setFixedHeight(20)
        self.predict_progress_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.predict_page_label = QLabel("0 / 0")
        self.predict_page_label.setProperty("muted", True)

        self.predict_image_label = QLabel("Select images and click Predict.")
        self.predict_image_label.setObjectName("ImagePreview")
        self.predict_image_label.setAlignment(Qt.AlignCenter)
        self.predict_image_label.setMinimumHeight(420)
        self.predict_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.predict_result_label = QLabel("Prediction result will appear here.")
        self.predict_result_label.setWordWrap(True)
        self.predict_result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.predict_result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.predict_compare_context_label = QLabel("Compare details will appear here.")
        self.predict_compare_context_label.setWordWrap(True)
        self.predict_compare_context_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.predict_compare_context_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.predict_compare_context_toggle = QToolButton()
        self.predict_compare_context_toggle.setText("Compare Context")
        self.predict_compare_context_toggle.setCheckable(True)
        self.predict_compare_context_toggle.setChecked(False)
        self.predict_compare_context_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.predict_compare_context_toggle.setArrowType(Qt.RightArrow)
        self.predict_compare_context_toggle.toggled.connect(self.on_predict_compare_context_toggled)

        self.predict_compare_context_summary_label = QLabel("No compare context available.")
        self.predict_compare_context_summary_label.setWordWrap(False)
        self.predict_compare_context_summary_label.setProperty("muted", True)
        self.predict_compare_context_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.predict_compare_shared_image_label = QLabel("Select images and click Predict.")
        self.predict_compare_shared_image_label.setObjectName("ImagePreview")
        self.predict_compare_shared_image_label.setAlignment(Qt.AlignCenter)
        self.predict_compare_shared_image_label.setMinimumHeight(220)
        self.predict_compare_shared_image_label.setMaximumHeight(280)
        self.predict_compare_shared_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.predict_compare_context_content = QWidget()
        self.predict_compare_context_content_layout = QVBoxLayout(self.predict_compare_context_content)
        self.predict_compare_context_content_layout.setContentsMargins(0, 0, 0, 0)
        self.predict_compare_context_content_layout.setSpacing(6)
        self.predict_compare_context_content_layout.addWidget(self.predict_compare_context_label)
        self.predict_compare_context_content_layout.addWidget(self.predict_compare_shared_image_label)
        self.predict_compare_context_content.setVisible(False)

        self.predict_compare_cards_widget = QWidget()
        self.predict_compare_cards_layout = QGridLayout(self.predict_compare_cards_widget)
        self.predict_compare_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.predict_compare_cards_layout.setHorizontalSpacing(10)
        self.predict_compare_cards_layout.setVerticalSpacing(10)
        self.predict_compare_cards_layout.setColumnStretch(0, 1)
        self.predict_compare_cards_layout.setColumnStretch(1, 1)

        self.predict_compare_cards_scroll = QScrollArea()
        self.predict_compare_cards_scroll.setWidgetResizable(True)
        self.predict_compare_cards_scroll.setWidget(self.predict_compare_cards_widget)
        self.predict_compare_cards_scroll.setFrameShape(QScrollArea.NoFrame)

        self.predict_compact_list = QListWidget()
        self.predict_compact_list.setViewMode(QListView.IconMode)
        self.predict_compact_list.setResizeMode(QListView.Adjust)
        self.predict_compact_list.setMovement(QListView.Static)
        self.predict_compact_list.setSpacing(10)
        self.predict_compact_list.setIconSize(QSize(160, 160))
        self.predict_compact_list.setGridSize(QSize(190, 250))
        self.predict_compact_list.setWordWrap(True)
        self.predict_compact_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.predict_compact_list.setUniformItemSizes(False)
        self.predict_compact_list.setMinimumHeight(280)
        self.predict_compact_list.itemClicked.connect(self.on_predict_compact_item_clicked)
        self.predict_compact_list.verticalScrollBar().valueChanged.connect(self.schedule_predict_visible_thumbnail_update)
        self.predict_compact_list.horizontalScrollBar().valueChanged.connect(self.schedule_predict_visible_thumbnail_update)

        self.predict_display_stack = QStackedWidget()

        single_predict_page = QWidget()
        single_predict_layout = QVBoxLayout(single_predict_page)
        single_predict_layout.setContentsMargins(4, 4, 4, 4)
        single_predict_layout.setSpacing(10)
        single_predict_layout.addWidget(self.predict_image_label, stretch=1)

        predict_result_group = QGroupBox("Prediction Result")
        predict_result_layout = QVBoxLayout(predict_result_group)
        predict_result_layout.setContentsMargins(10, 8, 10, 10)
        predict_result_layout.setSpacing(4)
        predict_result_layout.addWidget(self.predict_result_label)
        predict_result_group.setMaximumHeight(180)
        single_predict_layout.addWidget(predict_result_group)

        compare_predict_page = QWidget()
        compare_predict_layout = QVBoxLayout(compare_predict_page)
        compare_predict_layout.setContentsMargins(4, 4, 4, 4)
        compare_predict_layout.setSpacing(8)
        compare_context_section = QWidget()
        compare_context_layout = QVBoxLayout(compare_context_section)
        compare_context_layout.setContentsMargins(2, 0, 2, 0)
        compare_context_layout.setSpacing(4)
        compare_context_header = QHBoxLayout()
        compare_context_header.setContentsMargins(0, 0, 0, 0)
        compare_context_header.setSpacing(8)
        compare_context_header.addWidget(self.predict_compare_context_toggle)
        compare_context_header.addWidget(self.predict_compare_context_summary_label, stretch=1)
        compare_context_layout.addLayout(compare_context_header)
        compare_context_layout.addWidget(self.predict_compare_context_content)
        compare_predict_layout.addWidget(compare_context_section, stretch=0)
        compare_cards_section = QWidget()
        compare_cards_section_layout = QVBoxLayout(compare_cards_section)
        compare_cards_section_layout.setContentsMargins(0, 0, 0, 0)
        compare_cards_section_layout.setSpacing(6)
        compare_cards_label = QLabel("Model Compare")
        compare_cards_label.setProperty("muted", True)
        compare_cards_section_layout.addWidget(compare_cards_label)
        compare_cards_section_layout.addWidget(self.predict_compare_cards_scroll, stretch=1)
        compare_predict_layout.addWidget(compare_cards_section, stretch=1)

        self.predict_display_stack.addWidget(single_predict_page)
        self.predict_display_stack.addWidget(compare_predict_page)

    def _init_test_split_controls(self) -> None:
        self.test_split_device_combo = QComboBox()
        self.test_split_device_combo.addItems(["auto", "cpu", "cuda"])

        self.test_split_image_size_spin = QSpinBox()
        self.test_split_image_size_spin.setRange(32, 2048)
        self.test_split_image_size_spin.setValue(224)

        self.test_split_batch_size_spin = QSpinBox()
        self.test_split_batch_size_spin.setRange(1, 4096)
        self.test_split_batch_size_spin.setValue(32)

        self.test_split_amp_checkbox = QCheckBox("Use AMP for evaluation")
        self.test_split_amp_checkbox.setChecked(False)
        self.test_split_amp_checkbox.setToolTip("Use autocast during test-split inference on supported CUDA devices.")

        self.test_split_checkpoint_edit = QLineEdit(str(DEFAULT_CHECKPOINT_DIR / "efficientnet_baseline" / "best.pth"))
        self.test_split_checkpoint_edit.editingFinished.connect(self.update_test_split_detected_model)

        self.test_split_checkpoint_browse_button = QPushButton("Browse...")
        self.test_split_checkpoint_browse_button.clicked.connect(self.choose_test_split_checkpoint)

        self.test_split_detected_model_label = QLabel("Model will be auto-detected from the checkpoint.")
        self.test_split_detected_model_label.setWordWrap(True)
        self.test_split_detected_model_label.setProperty("muted", True)

        self.test_split_root_edit = QLineEdit(str(DEFAULT_TEST_SPLITS_ROOT))
        self.test_split_root_browse_button = QPushButton("Browse...")
        self.test_split_root_browse_button.clicked.connect(self.choose_test_splits_root)

        self.test_split_run_button = QPushButton("Evaluate Test Splits")
        self.test_split_run_button.clicked.connect(self.run_test_split_evaluation)
        self.test_split_queue_button = QPushButton("Add to Queue")
        self.test_split_queue_button.clicked.connect(self.add_current_test_split_config_to_queue)

        self.test_split_status_label = QLabel("Ready.")
        self.test_split_status_label.setWordWrap(True)
        self.test_split_status_label.setObjectName("SectionStatus")

        self.test_split_progress_bar = QProgressBar()
        self.test_split_progress_bar.setRange(0, 100)
        self.test_split_progress_bar.setValue(0)
        self.test_split_progress_bar.setFormat("%p%")

        self.test_split_result_label = QLabel("No evaluation has been run yet.")
        self.test_split_result_label.setWordWrap(True)
        self.test_split_result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.test_split_output_text = QPlainTextEdit()
        self.test_split_output_text.setReadOnly(True)
        self.test_split_output_text.setPlaceholderText("Per-split metrics and output file paths will appear here.")

    def _init_log_controls(self) -> None:
        self.training_log_runs: list[dict] = []
        self.training_log_available_list = QListWidget()
        self.training_log_available_list.setMaximumWidth(280)
        self.training_log_available_list.itemSelectionChanged.connect(self.on_available_log_selection_changed)

        self.training_log_selected_list = QListWidget()
        self.training_log_selected_list.setMaximumWidth(280)
        self.training_log_selected_list.itemSelectionChanged.connect(self.on_selected_log_selection_changed)

        self.training_log_add_button = QPushButton("+ Add")
        self.training_log_add_button.clicked.connect(self.add_selected_log_to_compare)
        self.training_log_add_button.setFixedWidth(72)

        self.training_log_remove_button = QPushButton("Remove")
        self.training_log_remove_button.clicked.connect(self.remove_selected_log_from_compare)
        self.training_log_remove_button.setFixedWidth(72)

        self.training_log_clear_button = QPushButton("Clear")
        self.training_log_clear_button.clicked.connect(self.clear_selected_logs)
        self.training_log_clear_button.setFixedWidth(64)

        self.training_log_stage_combo = QComboBox()
        self.training_log_stage_combo.addItems(["Summary", "Train", "Val", "Test"])
        self.training_log_stage_combo.currentIndexChanged.connect(self.refresh_training_log_view)

        self.training_log_refresh_button = QPushButton("Refresh")
        self.training_log_refresh_button.clicked.connect(self.refresh_training_log_runs)
        self.training_log_refresh_button.setFixedWidth(84)

        self.training_log_delete_button = QPushButton("Delete")
        self.training_log_delete_button.clicked.connect(self.delete_selected_log)
        self.training_log_delete_button.setFixedWidth(84)

        self.logs_export_include_paths_checkbox = QCheckBox("Include path setup")
        self.logs_export_include_paths_checkbox.setChecked(True)

        self.logs_export_button = QPushButton("Export")
        self.logs_export_button.clicked.connect(self.export_logs_as_python_code)

        self.training_log_status_label = QLabel("No training logs loaded.")
        self.training_log_status_label.setWordWrap(True)
        self.training_log_status_label.setObjectName("SectionStatus")

        self.training_plot_detail_label = QLabel("Detail View")
        self.training_plot_value_combo = QComboBox()
        self.training_plot_value_combo.addItems(["Accuracy", "Loss", "Timing", "Efficiency", "Confusion Matrix"])
        self.training_plot_value_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_metric_label = QLabel("Plot Metric")
        self.training_plot_stage_label = QLabel("Stage")
        self.training_plot_stage_combo = QComboBox()
        self.training_plot_stage_combo.addItems(["All / Auto", "Train", "Val", "Test"])
        self.training_plot_stage_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_timing_label = QLabel("Timing Metric")
        self.training_plot_timing_combo = QComboBox()
        self.training_plot_timing_combo.addItems(["Total Time", "Pure Time", "Avg Pure / Batch"])
        self.training_plot_timing_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_efficiency_label = QLabel("Efficiency X")
        self.training_plot_efficiency_combo = QComboBox()
        self.training_plot_efficiency_combo.addItems(["Train Wall Time", "Train Pure Time", "Test Avg Pure / Batch", "Trainable Params"])
        self.training_plot_efficiency_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_confusion_label = QLabel("Confusion Top-K")
        self.training_plot_confusion_spin = QSpinBox()
        self.training_plot_confusion_spin.setRange(3, 20)
        self.training_plot_confusion_spin.setValue(10)
        self.training_plot_confusion_spin.valueChanged.connect(self.refresh_training_log_plot)

        self.training_plot_widget = LogPlotWidget()
        self.training_efficiency_plot_widget = ScatterPlotWidget()
        self.training_confusion_widget = ConfusionMatrixWidget()
        self.training_plot_stack = QStackedWidget()
        self.training_plot_stack.addWidget(self.training_plot_widget)
        self.training_plot_stack.addWidget(self.training_efficiency_plot_widget)
        self.training_plot_stack.addWidget(self.training_confusion_widget)

        self.training_log_text = QPlainTextEdit()
        self.training_log_text.setReadOnly(True)
        self.training_log_text.setPlaceholderText("Training run summaries and stage details will appear here.")

    def _build_ui(self) -> None:
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(False)
        self.setCentralWidget(self.tabs)

        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        data_config_group = QGroupBox("Dataset Config")
        data_form = QFormLayout(data_config_group)
        data_form.addRow("Data Dir", self.data_dir_label)
        data_form.addRow("Dataset Root", self.dataset_root_label)
        data_layout.addWidget(data_config_group)

        data_controls = QHBoxLayout()
        data_controls.addWidget(self.data_check_button)
        data_controls.addWidget(self.data_prepare_button)
        data_controls.addWidget(self.data_force_button)
        data_controls.addWidget(self.data_status_label)
        data_controls.addStretch(1)
        data_layout.addLayout(data_controls)

        data_status_group = QGroupBox("Task Status")
        data_status_form = QFormLayout(data_status_group)
        data_status_form.addRow("Current Task", self.data_task_value_label)
        data_status_form.addRow("State", self.data_state_value_label)
        data_status_form.addRow("Target", self.data_target_value_label)
        data_status_form.addRow("Last Result", self.data_last_result_value_label)
        data_layout.addWidget(data_status_group)

        data_progress_group = QGroupBox("Data Progress")
        data_progress_layout = QVBoxLayout(data_progress_group)
        data_progress_layout.addWidget(self.data_progress_label)
        data_progress_layout.addWidget(self.data_progress_bar)
        data_layout.addWidget(data_progress_group)

        data_output_group = QGroupBox("Data Output")
        data_output_layout = QVBoxLayout(data_output_group)
        data_output_layout.addWidget(self.data_output_text)
        data_layout.addWidget(data_output_group)

        training_tab = QWidget()
        training_tab_layout = QVBoxLayout(training_tab)
        training_tab_layout.setContentsMargins(0, 0, 0, 0)
        training_scroll = QScrollArea()
        training_scroll.setWidgetResizable(True)
        training_scroll.setFrameShape(QScrollArea.NoFrame)
        training_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        training_scroll_content = QWidget()
        training_layout = QVBoxLayout(training_scroll_content)
        training_layout.setContentsMargins(8, 8, 8, 8)
        training_layout.setSpacing(16)

        core_group = QGroupBox("Core Training Config")
        core_layout = QVBoxLayout(core_group)
        core_layout.setSpacing(12)
        prominent_form = QFormLayout()
        prominent_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prominent_form.addRow(self._make_training_row_label("Model", prominent=True), self.model_combo)
        prominent_form.addRow(self._make_training_row_label("Model Source"), self.training_model_variant_label)
        core_layout.addLayout(prominent_form)
        compact_grid = QGridLayout()
        compact_grid.setContentsMargins(0, 0, 0, 0)
        compact_grid.setHorizontalSpacing(12)
        compact_grid.setVerticalSpacing(12)
        compact_grid.addWidget(self._create_training_labeled_field("Epochs", self.epochs_spin), 0, 0)
        compact_grid.addWidget(self._create_training_labeled_field("Batch Size", self.batch_size_spin), 0, 1)
        compact_grid.addWidget(self._create_training_labeled_field("Optimizer", self.optimizer_combo), 0, 2)
        compact_grid.addWidget(self._create_training_labeled_field("Image Size", self.image_size_spin), 1, 0)
        compact_grid.addWidget(self._create_training_labeled_field("Learning Rate", self.lr_spin), 1, 1)
        compact_grid.addWidget(self._create_training_labeled_field("Precision", self.amp_checkbox), 1, 2)
        compact_grid.addWidget(self._create_training_labeled_field("Train Transforms Preset", self.train_transforms_preset_combo), 2, 0, 1, 2)
        core_layout.addLayout(compact_grid)
        options_form = QFormLayout()
        options_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        options_form.setHorizontalSpacing(16)
        options_form.setVerticalSpacing(10)
        options_form.addRow(self._make_training_row_label("Options"), self.freeze_checkbox)
        options_form.addRow(self._make_training_row_label(""), self.validation_checkbox)
        self.training_validation_proportion_label = QLabel("Validation Proportion")
        self.training_validation_proportion_label.setMinimumWidth(136)
        self.training_validation_proportion_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        options_form.addRow(self.training_validation_proportion_label, self.validation_proportion_spin)
        options_form.addRow(self._make_training_row_label(""), self.resume_checkbox)
        resume_layout = QHBoxLayout()
        resume_layout.setContentsMargins(0, 0, 0, 0)
        resume_layout.setSpacing(8)
        resume_layout.addWidget(self.resume_path_edit, stretch=1)
        resume_layout.addWidget(self.resume_browse_button)
        resume_layout.addWidget(self.resume_clear_button)
        self.training_resume_path_widget = QWidget()
        self.training_resume_path_widget.setLayout(resume_layout)
        self.training_resume_path_label = QLabel("Resume Checkpoint")
        self.training_resume_path_label.setMinimumWidth(136)
        self.training_resume_path_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        options_form.addRow(self.training_resume_path_label, self.training_resume_path_widget)
        core_layout.addLayout(options_form)
        training_layout.addWidget(core_group)

        data_output_group = QGroupBox("Data & Output")
        data_output_layout = QVBoxLayout(data_output_group)
        data_output_layout.setSpacing(12)
        data_output_actions = QHBoxLayout()
        data_output_actions.setContentsMargins(0, 0, 0, 0)
        data_output_actions.addStretch(1)
        data_output_actions.addWidget(self.training_settings_button)
        data_output_layout.addLayout(data_output_actions)
        data_output_form = QFormLayout()
        data_output_form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        data_output_form.setHorizontalSpacing(16)
        data_output_form.setVerticalSpacing(10)
        self.training_run_name_label = self._make_training_row_label("Run Name", prominent=True)
        self.training_data_root_title = self._make_training_row_label("Data Root")
        self.training_checkpoint_dir_title = self._make_training_row_label("Checkpoint Dir")
        self.training_advanced_title = self._make_training_row_label("Advanced")
        data_output_form.addRow(self.training_run_name_label, self.checkpoint_output_combo)
        data_output_form.addRow(self.training_data_root_title, self.data_root_label)
        data_output_form.addRow(self.training_checkpoint_dir_title, self.checkpoint_dir_label)
        data_output_form.addRow(self.training_advanced_title, self.training_settings_summary)
        data_output_layout.addLayout(data_output_form)
        training_layout.addWidget(data_output_group)

        command_group = QGroupBox("Command Preview")
        command_group_layout = QVBoxLayout(command_group)
        command_group_layout.setSpacing(10)
        command_header_layout = QHBoxLayout()
        command_header_layout.setContentsMargins(0, 0, 0, 0)
        command_header_layout.addWidget(self.command_preview_toggle)
        command_header_layout.addStretch(1)
        command_header_layout.addWidget(self.export_include_paths_checkbox)
        command_header_layout.addWidget(self.export_command_button)
        command_group_layout.addLayout(command_header_layout)
        self.command_preview_body = QWidget()
        command_body_layout = QVBoxLayout(self.command_preview_body)
        command_body_layout.setContentsMargins(0, 0, 0, 0)
        command_body_layout.setSpacing(0)
        command_body_layout.addWidget(self.command_preview)
        command_group_layout.addWidget(self.command_preview_body)
        training_layout.addWidget(command_group)

        monitor_group = QGroupBox("Run Monitor")
        monitor_layout = QVBoxLayout(monitor_group)
        monitor_layout.setSpacing(14)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.train_queue_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)
        monitor_layout.addLayout(controls_layout)

        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        monitor_layout.addWidget(progress_group)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.output_text)
        monitor_layout.addWidget(log_group)
        training_layout.addWidget(monitor_group)
        training_layout.addStretch(1)
        training_scroll.setWidget(training_scroll_content)
        training_tab_layout.addWidget(training_scroll)

        predict_tab = QWidget()
        predict_layout = QVBoxLayout(predict_tab)
        predict_layout.setContentsMargins(6, 6, 6, 6)
        predict_layout.setSpacing(8)

        predict_selector_group = QGroupBox("Checkpoint Selector")
        predict_selector_group.setMinimumWidth(0)
        predict_selector_layout = QVBoxLayout(predict_selector_group)
        predict_selector_layout.setContentsMargins(8, 8, 8, 8)
        predict_selector_layout.setSpacing(5)
        predict_selector_layout.addWidget(self.predict_checkpoint_tree, stretch=1)
        predict_selector_actions = QHBoxLayout()
        predict_selector_actions.setContentsMargins(0, 0, 0, 0)
        predict_selector_actions.setSpacing(6)
        predict_selector_actions.addWidget(self.predict_select_all_best_button)
        predict_selector_actions.addWidget(self.predict_clear_selection_button)
        predict_selector_actions.addStretch(1)
        predict_selector_layout.addLayout(predict_selector_actions)

        predict_browser_group = QGroupBox("Image Browser")
        predict_browser_layout = QVBoxLayout(predict_browser_group)
        predict_browser_layout.setContentsMargins(8, 8, 8, 8)
        predict_browser_layout.setSpacing(5)
        predict_browser_header = QHBoxLayout()
        predict_browser_header.setContentsMargins(0, 0, 0, 0)
        predict_browser_header.setSpacing(6)
        predict_browser_view_label = QLabel("View")
        predict_browser_view_label.setProperty("muted", True)
        predict_browser_header.addWidget(predict_browser_view_label)
        self.predict_browser_mode_combo.setMaximumWidth(132)
        predict_browser_header.addWidget(self.predict_browser_mode_combo)
        predict_browser_header.addStretch(1)
        predict_browser_layout.addLayout(predict_browser_header)
        predict_browser_layout.addWidget(self.predict_compact_list, stretch=1)
        predict_browser_actions = QHBoxLayout()
        predict_browser_actions.setContentsMargins(0, 0, 0, 0)
        predict_browser_actions.setSpacing(6)
        predict_browser_actions.addWidget(self.predict_select_images_button)
        predict_browser_actions.addWidget(self.predict_select_folder_button)
        predict_browser_actions.addStretch(1)
        predict_browser_layout.addLayout(predict_browser_actions)

        predict_left_panel = QWidget()
        predict_left_layout = QVBoxLayout(predict_left_panel)
        predict_left_layout.setContentsMargins(0, 0, 0, 0)
        predict_left_layout.setSpacing(8)
        predict_left_layout.addWidget(predict_selector_group, stretch=0)
        predict_left_layout.addWidget(predict_browser_group, stretch=1)
        predict_left_panel.setMinimumWidth(280)

        predict_config_group = QGroupBox("Predict Config")
        predict_config_group.setMinimumWidth(460)
        predict_config_group.setMaximumHeight(330)
        predict_config_layout = QVBoxLayout(predict_config_group)
        predict_config_layout.setContentsMargins(8, 8, 8, 8)
        predict_config_layout.setSpacing(6)

        predict_info_grid = QGridLayout()
        predict_info_grid.setContentsMargins(0, 0, 0, 0)
        predict_info_grid.setHorizontalSpacing(8)
        predict_info_grid.setVerticalSpacing(4)
        predict_device_title = QLabel("Device")
        predict_device_title.setProperty("muted", True)
        predict_image_size_title = QLabel("Image Size")
        predict_image_size_title.setProperty("muted", True)
        predict_info_grid.addWidget(predict_device_title, 0, 0)
        predict_info_grid.addWidget(self.predict_device_combo, 0, 1)
        predict_info_grid.addWidget(predict_image_size_title, 0, 2)
        predict_info_grid.addWidget(self.predict_image_size_spin, 0, 3)
        predict_info_grid.addWidget(self.predict_compare_checkbox, 0, 4, 1, 1, Qt.AlignRight | Qt.AlignVCenter)
        predict_info_grid.setColumnStretch(1, 2)
        predict_info_grid.setColumnStretch(4, 1)
        predict_config_layout.addLayout(predict_info_grid)

        self.predict_compare_models_label.setProperty("muted", False)
        predict_config_layout.addWidget(self.predict_detected_model_label)
        predict_config_layout.addWidget(self.predict_compare_models_label)

        predict_action_row = QHBoxLayout()
        predict_action_row.setContentsMargins(0, 0, 0, 0)
        predict_action_row.setSpacing(6)
        predict_action_row.addWidget(self.predict_run_button)
        predict_action_row.addWidget(self.predict_queue_button)
        predict_action_row.addWidget(self.predict_gradcam_button)
        predict_action_row.addStretch(1)
        predict_action_row.addWidget(self.predict_export_include_paths_checkbox)
        predict_action_row.addWidget(self.predict_export_button)
        predict_config_layout.addLayout(predict_action_row)

        predict_footer_row = QHBoxLayout()
        predict_footer_row.setContentsMargins(0, 0, 0, 0)
        predict_footer_row.setSpacing(8)
        predict_footer_row.addWidget(self.predict_status_label, stretch=3)
        predict_footer_row.addWidget(self.predict_progress_bar, stretch=2)
        predict_nav_layout = QHBoxLayout()
        predict_nav_layout.setContentsMargins(0, 0, 0, 0)
        predict_nav_layout.setSpacing(6)
        predict_nav_layout.addWidget(self.predict_prev_button)
        predict_nav_layout.addWidget(self.predict_page_label)
        predict_nav_layout.addWidget(self.predict_next_button)
        predict_footer_row.addLayout(predict_nav_layout, stretch=0)
        predict_config_layout.addLayout(predict_footer_row)
        predict_config_layout.addStretch(0)

        predict_preview_group = QGroupBox("Prediction Preview")
        predict_preview_layout = QVBoxLayout(predict_preview_group)
        predict_preview_layout.setContentsMargins(8, 8, 8, 8)
        predict_preview_layout.setSpacing(8)
        predict_preview_layout.addWidget(self.predict_display_stack, stretch=1)

        predict_right_panel = QWidget()
        predict_right_layout = QVBoxLayout(predict_right_panel)
        predict_right_layout.setContentsMargins(0, 0, 0, 0)
        predict_right_layout.setSpacing(10)
        predict_right_layout.addWidget(predict_config_group, stretch=0)
        predict_right_layout.addWidget(predict_preview_group, stretch=1)

        self.predict_splitter = QSplitter(Qt.Horizontal)
        self.predict_splitter.addWidget(predict_left_panel)
        self.predict_splitter.addWidget(predict_right_panel)
        self.predict_splitter.setCollapsible(0, False)
        self.predict_splitter.setCollapsible(1, False)
        self.predict_splitter.setStretchFactor(0, 0)
        self.predict_splitter.setStretchFactor(1, 1)
        self.predict_splitter.setSizes([420, 980])
        predict_layout.addWidget(self.predict_splitter, stretch=1)

        test_split_tab = QWidget()
        test_split_layout = QVBoxLayout(test_split_tab)

        test_split_config_group = QGroupBox("Test Split Evaluation")
        test_split_form = QFormLayout(test_split_config_group)
        test_split_form.addRow("Detected Model", self.test_split_detected_model_label)
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.test_split_checkpoint_edit, stretch=1)
        checkpoint_layout.addWidget(self.test_split_checkpoint_browse_button)
        test_split_form.addRow("Checkpoint", checkpoint_layout)
        root_layout = QHBoxLayout()
        root_layout.addWidget(self.test_split_root_edit, stretch=1)
        root_layout.addWidget(self.test_split_root_browse_button)
        test_split_form.addRow("Test Splits Root", root_layout)
        test_split_form.addRow("Device", self.test_split_device_combo)
        test_split_form.addRow("Image Size", self.test_split_image_size_spin)
        test_split_form.addRow("Evaluation Batch Size", self.test_split_batch_size_spin)
        test_split_form.addRow("", self.test_split_amp_checkbox)
        test_split_layout.addWidget(test_split_config_group)

        test_split_controls = QHBoxLayout()
        test_split_controls.addWidget(self.test_split_run_button)
        test_split_controls.addWidget(self.test_split_queue_button)
        test_split_controls.addStretch(1)
        test_split_layout.addLayout(test_split_controls)
        test_split_layout.addWidget(self.test_split_status_label)
        test_split_layout.addWidget(self.test_split_progress_bar)

        test_split_result_group = QGroupBox("Evaluation Summary")
        test_split_result_layout = QVBoxLayout(test_split_result_group)
        test_split_result_layout.addWidget(self.test_split_result_label)
        test_split_layout.addWidget(test_split_result_group)

        test_split_output_group = QGroupBox("Per-Split Results")
        test_split_output_layout = QVBoxLayout(test_split_output_group)
        test_split_output_layout.addWidget(self.test_split_output_text)
        test_split_layout.addWidget(test_split_output_group, stretch=1)

        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        logs_splitter = QSplitter(Qt.Horizontal)

        logs_left_panel = QWidget()
        logs_left_panel.setMinimumWidth(360)
        logs_left_panel.setMaximumWidth(420)
        logs_left_layout = QVBoxLayout(logs_left_panel)

        logs_available_group = QGroupBox("Available Runs")
        logs_available_layout = QVBoxLayout(logs_available_group)
        logs_available_layout.addWidget(self.training_log_available_list)
        logs_available_actions = QGridLayout()
        logs_available_actions.setContentsMargins(0, 0, 0, 0)
        logs_available_actions.setHorizontalSpacing(8)
        logs_available_actions.setVerticalSpacing(8)
        logs_available_actions.addWidget(self.training_log_add_button, 0, 0)
        logs_available_actions.addWidget(self.training_log_refresh_button, 0, 1)
        logs_available_actions.addWidget(self.training_log_delete_button, 0, 2)
        logs_available_actions.addWidget(self.logs_export_button, 0, 3)
        logs_available_actions.addWidget(self.logs_export_include_paths_checkbox, 1, 0, 1, 4)
        logs_available_actions.setColumnStretch(4, 1)
        logs_available_layout.addLayout(logs_available_actions)
        logs_left_layout.addWidget(logs_available_group, stretch=3)

        logs_selected_group = QGroupBox("Selected For Plot")
        logs_selected_layout = QVBoxLayout(logs_selected_group)
        logs_selected_layout.addWidget(self.training_log_selected_list)
        logs_selected_actions = QHBoxLayout()
        logs_selected_actions.addWidget(self.training_log_remove_button)
        logs_selected_actions.addWidget(self.training_log_clear_button)
        logs_selected_actions.addStretch(1)
        logs_selected_layout.addLayout(logs_selected_actions)
        logs_left_layout.addWidget(logs_selected_group, stretch=2)
        logs_left_layout.addWidget(self.training_log_status_label)

        logs_right_splitter = QSplitter(Qt.Vertical)

        logs_top_panel = QWidget()
        logs_top_layout = QVBoxLayout(logs_top_panel)
        logs_plot_group = QGroupBox("Plot")
        logs_plot_form = QFormLayout(logs_plot_group)
        logs_plot_form.addRow(self.training_plot_detail_label, self.training_log_stage_combo)
        logs_plot_form.addRow(self.training_plot_metric_label, self.training_plot_value_combo)
        logs_plot_form.addRow(self.training_plot_stage_label, self.training_plot_stage_combo)
        logs_plot_form.addRow(self.training_plot_timing_label, self.training_plot_timing_combo)
        logs_plot_form.addRow(self.training_plot_efficiency_label, self.training_plot_efficiency_combo)
        logs_plot_form.addRow(self.training_plot_confusion_label, self.training_plot_confusion_spin)
        logs_top_layout.addWidget(logs_plot_group)

        logs_plot_canvas_group = QGroupBox("Run Plot")
        logs_plot_canvas_layout = QVBoxLayout(logs_plot_canvas_group)
        logs_plot_canvas_layout.addWidget(self.training_plot_stack)
        logs_top_layout.addWidget(logs_plot_canvas_group, stretch=1)

        logs_bottom_panel = QWidget()
        logs_bottom_layout = QVBoxLayout(logs_bottom_panel)
        logs_output_group = QGroupBox("Training Run Details")
        logs_output_layout = QVBoxLayout(logs_output_group)
        logs_output_layout.addWidget(self.training_log_text)
        logs_bottom_layout.addWidget(logs_output_group, stretch=1)

        logs_right_splitter.addWidget(logs_top_panel)
        logs_right_splitter.addWidget(logs_bottom_panel)
        logs_right_splitter.setStretchFactor(0, 7)
        logs_right_splitter.setStretchFactor(1, 3)

        logs_splitter.addWidget(logs_left_panel)
        logs_splitter.addWidget(logs_right_splitter)
        logs_splitter.setCollapsible(0, False)
        logs_splitter.setCollapsible(1, False)
        logs_splitter.setStretchFactor(0, 0)
        logs_splitter.setStretchFactor(1, 1)
        logs_splitter.setSizes([320, 980])
        logs_layout.addWidget(logs_splitter, stretch=1)
        custom_models_tab = CustomModelCanvasWidget(on_model_generated=self.on_custom_model_generated, parent=self)

        self.tabs.addTab(training_tab, "Training")
        self.tabs.addTab(predict_tab, "Predicting")
        self.tabs.addTab(test_split_tab, "Test Splits")
        self.tabs.addTab(data_tab, "Data")
        self.tabs.addTab(logs_tab, "Logs")
        self.tabs.addTab(custom_models_tab, "Custom Models")
        self.tabs.setCurrentIndex(0)
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(8)
        corner_layout.addWidget(self.theme_label)
        corner_layout.addWidget(self.theme_combo)
        corner_layout.addWidget(self.global_queue_button)
        self.tabs.setCornerWidget(corner_widget, Qt.TopRightCorner)

        queue_panel = QWidget()
        queue_panel_layout = QVBoxLayout(queue_panel)
        queue_panel_layout.setContentsMargins(10, 10, 10, 10)
        queue_panel_layout.setSpacing(10)
        queue_panel_layout.addWidget(self.global_queue_list)
        queue_buttons = QGridLayout()
        queue_buttons.setContentsMargins(0, 0, 0, 0)
        queue_buttons.setHorizontalSpacing(8)
        queue_buttons.setVerticalSpacing(8)
        queue_buttons.addWidget(self.queue_remove_button, 0, 0)
        queue_buttons.addWidget(self.queue_duplicate_button, 0, 1)
        queue_buttons.addWidget(self.queue_follow_on_test_split_button, 1, 0, 1, 2)
        queue_buttons.addWidget(self.queue_move_up_button, 2, 0)
        queue_buttons.addWidget(self.queue_move_down_button, 2, 1)
        queue_buttons.addWidget(self.queue_run_button, 3, 0)
        queue_buttons.addWidget(self.queue_stop_button, 3, 1)
        queue_buttons.addWidget(self.queue_clear_finished_button, 4, 0, 1, 2)
        queue_panel_layout.addLayout(queue_buttons)
        queue_panel_layout.addWidget(self.global_queue_status_label)

        self.global_queue_dock = QDockWidget("Global Queue", self)
        self.global_queue_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.global_queue_dock.setWidget(queue_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.global_queue_dock)
        self.global_queue_dock.visibilityChanged.connect(self.global_queue_button.setChecked)
        self.global_queue_button.toggled.connect(self.global_queue_dock.setVisible)

        self.model_combo.currentTextChanged.connect(self.on_training_model_changed)
        self.device_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.epochs_spin.valueChanged.connect(self.refresh_command_preview)
        self.batch_size_spin.valueChanged.connect(self.refresh_command_preview)
        self.num_workers_spin.valueChanged.connect(self.refresh_command_preview)
        self.image_size_spin.valueChanged.connect(self.refresh_command_preview)
        self.train_transforms_preset_combo.currentTextChanged.connect(self.on_train_transforms_preset_changed)
        self.lr_spin.valueChanged.connect(self.refresh_command_preview)
        self.optimizer_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.amp_checkbox.toggled.connect(self.refresh_command_preview)
        self.freeze_checkbox.toggled.connect(self.refresh_command_preview)
        self.validation_checkbox.toggled.connect(self.on_validation_toggled)
        self.validation_proportion_spin.valueChanged.connect(self.refresh_command_preview)
        self.resume_checkbox.toggled.connect(self.on_resume_toggled)
        self.resume_path_edit.textChanged.connect(self.refresh_command_preview)
        self.resume_path_edit.editingFinished.connect(self.on_resume_path_edited)
        self.command_preview_toggle.toggled.connect(self.on_command_preview_toggled)
        self.checkpoint_output_combo.currentTextChanged.connect(self.on_checkpoint_output_changed)
        self.checkpoint_output_combo.activated.connect(self.on_checkpoint_output_activated)
        self.on_validation_toggled(self.validation_checkbox.isChecked())
        self.on_resume_toggled(self.resume_checkbox.isChecked())
        self.on_command_preview_toggled(self.command_preview_toggle.isChecked())
        self.refresh_global_queue_view()
        self.on_train_transforms_preset_changed(self.train_transforms_preset_combo.currentText())
        self.on_training_model_changed(self.current_training_model_name())

    def apply_visual_design(self) -> None:
        stylesheet = app_themes.build_stylesheet(self.current_theme_key)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
        else:
            self.setStyleSheet(stylesheet)
        self._set_layout_metrics(self.centralWidget().layout() if self.centralWidget() is not None else None)

    def on_theme_changed(self) -> None:
        theme_key = self.theme_combo.currentData()
        if not isinstance(theme_key, str) or theme_key not in app_themes.THEMES:
            return
        if theme_key == self.current_theme_key:
            return
        self.current_theme_key = theme_key
        self.settings.setValue("ui/theme", theme_key)
        self.apply_visual_design()

    def _set_layout_metrics(self, layout) -> None:
        if layout is None:
            return
        if isinstance(layout, QFormLayout):
            layout.setHorizontalSpacing(14)
            layout.setVerticalSpacing(10)
        else:
            layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child_layout = item.layout()
            if child_layout is not None:
                self._set_layout_metrics(child_layout)
            child_widget = item.widget()
            if child_widget is not None and child_widget.layout() is not None:
                self._set_layout_metrics(child_widget.layout())

    def build_command(self, config: dict[str, object] | None = None) -> list[str]:
        if config is None:
            config = self.collect_training_config_snapshot()
        checkpoint_dir = Path(str(config.get("checkpoint_dir", self.selected_checkpoint_dir()))).expanduser().resolve()
        preset = str(config.get("train_transforms_preset", "baseline"))
        command = [
            "-u",
            str(TRAINING_SCRIPT),
            "--model",
            str(config.get("model", self.current_training_model_name())),
            "--data-root",
            str(config.get("data_root", DEFAULT_DATA_ROOT)),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--epochs",
            str(config.get("epochs", self.epochs_spin.value())),
            "--batch-size",
            str(config.get("batch_size", self.batch_size_spin.value())),
            "--num-workers",
            str(config.get("num_workers", self.num_workers_spin.value())),
            "--image-size",
            str(config.get("image_size", self.image_size_spin.value())),
            "--train-transforms-preset",
            preset,
            "--lr",
            format(float(config.get("lr", self.lr_spin.value())), ".6f"),
            "--optimizer",
            str(config.get("optimizer", self.optimizer_combo.currentText())),
            "--scheduler",
            str(config.get("scheduler", self.scheduler_combo.currentText())),
            "--seed",
            str(config.get("seed", self.seed_spin.value())),
        ]
        if preset == "custom":
            custom = config.get("custom_augmentation")
            if isinstance(custom, dict):
                downsample = custom.get("downsample")
                if isinstance(downsample, dict) and downsample.get("enabled"):
                    command.extend(
                        [
                            "--custom-downsample",
                            "--custom-downsample-prob",
                            format(float(downsample.get("probability", 0.65)), ".2f"),
                            "--custom-downsample-min-scale",
                            format(float(downsample.get("min_scale", 0.18)), ".2f"),
                            "--custom-downsample-max-scale",
                            format(float(downsample.get("max_scale", 0.55)), ".2f"),
                        ]
                    )
                blur = custom.get("mild_blur")
                if isinstance(blur, dict) and blur.get("enabled"):
                    command.extend(
                        [
                            "--custom-mild-blur",
                            "--custom-mild-blur-prob",
                            format(float(blur.get("probability", 0.10)), ".2f"),
                        ]
                    )
                erasing = custom.get("random_erasing")
                if isinstance(erasing, dict) and erasing.get("enabled"):
                    command.extend(
                        [
                            "--custom-random-erasing",
                            "--custom-random-erasing-prob",
                            format(float(erasing.get("probability", 0.08)), ".2f"),
                        ]
                    )
                color_jitter = custom.get("color_jitter")
                if isinstance(color_jitter, dict) and color_jitter.get("enabled"):
                    command.append("--custom-color-jitter")
                horizontal_flip = custom.get("horizontal_flip")
                if isinstance(horizontal_flip, dict) and horizontal_flip.get("enabled"):
                    command.append("--custom-horizontal-flip")
        elif bool(config.get("mild_blur_enabled", self.mild_blur_enabled)):
            command.extend(
                [
                    "--mild-blur",
                    "--mild-blur-prob",
                    format(float(config.get("mild_blur_prob", self.mild_blur_prob)), ".2f"),
                ]
            )

        command.extend(["--progress-format", "gui"])
        command.extend(["--stop-file", str(self.stop_request_path_for(checkpoint_dir))])

        device = str(config.get("device", self.device_combo.currentText()))
        if device != "auto":
            command.extend(["--device", device])
        if bool(config.get("amp", self.amp_checkbox.isChecked())):
            command.append("--amp")

        command.append("--freeze-backbone" if bool(config.get("freeze_backbone", self.freeze_checkbox.isChecked())) else "--no-freeze-backbone")
        if bool(config.get("use_validation_split", self.validation_checkbox.isChecked())):
            command.extend(
                [
                    "--use-validation-split",
                    "--validation-proportion",
                    format(float(config.get("validation_proportion", self.validation_proportion_spin.value())), ".2f"),
                ]
            )

        resume_path = str(config.get("resume_path", self.resume_path_edit.text().strip())).strip()
        if bool(config.get("resume_enabled", self.resume_checkbox.isChecked())) and resume_path:
            command.extend(["--resume", resume_path])
        return command

    def build_training_worker_args(self, config: dict[str, object] | None = None) -> list[str]:
        command = self.build_command(config)
        if len(command) >= 2 and command[0] == "-u":
            return command[2:]
        return command

    def format_command_for_display(self, command: list[str], *, program: str | None = None) -> str:
        parts: list[str] = []
        program_token = program if isinstance(program, str) and program.strip() else sys.executable
        for token in [program_token, *command]:
            text = str(token)
            parts.append(f"\"{text}\"" if " " in text else text)
        return " ".join(parts)

    def _path_expression(self, base_expression: str, path: Path) -> str:
        expression = base_expression
        for part in path.parts:
            if part in {"", "."}:
                continue
            expression = f"{expression} / {part!r}"
        return expression

    def _expression_for_path(self, path: Path, *, notebook_dir: Path | None = None) -> str:
        resolved_path = path.expanduser().resolve()
        try:
            project_relative = resolved_path.relative_to(PROJECT_ROOT)
        except ValueError:
            if notebook_dir is not None:
                try:
                    relative_to_notebook = Path(os.path.relpath(resolved_path, notebook_dir))
                    return f"({self._path_expression('NOTEBOOK_DIR', relative_to_notebook)}).resolve()"
                except ValueError:
                    pass
            return f"Path({str(resolved_path)!r})"
        return f"({self._path_expression('PROJECT_ROOT', project_relative)}).resolve()"

    def _relative_string_for_project_path(self, path: Path) -> str:
        resolved_path = path.expanduser().resolve()
        try:
            relative_path = resolved_path.relative_to(PROJECT_ROOT)
        except ValueError:
            return str(resolved_path)
        return relative_path.as_posix()

    def _project_root_expression_for_path(self, path: Path) -> str:
        resolved_path = path.expanduser().resolve()
        try:
            relative_path = resolved_path.relative_to(PROJECT_ROOT)
        except ValueError:
            return f"Path({str(resolved_path)!r})"
        return f"PROJECT_ROOT / {relative_path.as_posix()!r}"

    def build_notebook_training_code(self, notebook_path: Path, *, include_path_setup: bool) -> str:
        notebook_dir = notebook_path.expanduser().resolve().parent
        notebook_in_project_root = notebook_dir == PROJECT_ROOT
        project_relative = Path(os.path.relpath(PROJECT_ROOT, notebook_dir))
        project_root_expression = f"({self._path_expression('NOTEBOOK_DIR', project_relative)}).resolve()"

        command_lines = [
            "command = [",
            "    sys.executable,",
            "    '-u',",
            "    str(TRAINING_SCRIPT),",
            f"    '--model', {self.current_training_model_name()!r},",
            "    '--data-root', str(DATA_ROOT),",
            "    '--checkpoint-dir', str(CHECKPOINT_DIR),",
            f"    '--epochs', {str(self.epochs_spin.value())!r},",
            f"    '--batch-size', {str(self.batch_size_spin.value())!r},",
            f"    '--num-workers', {str(self.num_workers_spin.value())!r},",
            f"    '--image-size', {str(self.image_size_spin.value())!r},",
            f"    '--train-transforms-preset', {self.train_transforms_preset_combo.currentText()!r},",
            f"    '--lr', {format(self.lr_spin.value(), '.6f')!r},",
            f"    '--optimizer', {self.optimizer_combo.currentText()!r},",
            f"    '--scheduler', {self.scheduler_combo.currentText()!r},",
            f"    '--seed', {str(self.seed_spin.value())!r},",
            "    '--progress-format', 'tqdm',",
        ]
        if self.train_transforms_preset_combo.currentText() != "custom" and self.mild_blur_enabled:
            command_lines.extend(
                [
                    "    '--mild-blur',",
                    f"    '--mild-blur-prob', {format(self.mild_blur_prob, '.2f')!r},",
                ]
            )
        if self.train_transforms_preset_combo.currentText() == "custom":
            custom = self.custom_augmentation_config()
            downsample = custom["downsample"]
            assert isinstance(downsample, dict)
            if downsample.get("enabled"):
                command_lines.extend(
                    [
                        "    '--custom-downsample',",
                        f"    '--custom-downsample-prob', {format(float(downsample.get('probability', 0.65)), '.2f')!r},",
                        f"    '--custom-downsample-min-scale', {format(float(downsample.get('min_scale', 0.18)), '.2f')!r},",
                        f"    '--custom-downsample-max-scale', {format(float(downsample.get('max_scale', 0.55)), '.2f')!r},",
                    ]
                )
            blur = custom["mild_blur"]
            assert isinstance(blur, dict)
            if blur.get("enabled"):
                command_lines.extend(
                    [
                        "    '--custom-mild-blur',",
                        f"    '--custom-mild-blur-prob', {format(float(blur.get('probability', 0.10)), '.2f')!r},",
                    ]
                )
            erasing = custom["random_erasing"]
            assert isinstance(erasing, dict)
            if erasing.get("enabled"):
                command_lines.extend(
                    [
                        "    '--custom-random-erasing',",
                        f"    '--custom-random-erasing-prob', {format(float(erasing.get('probability', 0.08)), '.2f')!r},",
                    ]
                )
            color_jitter = custom.get("color_jitter")
            if isinstance(color_jitter, dict) and color_jitter.get("enabled"):
                command_lines.append("    '--custom-color-jitter',")
            horizontal_flip = custom.get("horizontal_flip")
            if isinstance(horizontal_flip, dict) and horizontal_flip.get("enabled"):
                command_lines.append("    '--custom-horizontal-flip',")

        device = self.device_combo.currentText()
        if device != "auto":
            command_lines.append(f"    '--device', {device!r},")
        if self.amp_checkbox.isChecked():
            command_lines.append("    '--amp',")

        command_lines.append(
            "    '--freeze-backbone'," if self.freeze_checkbox.isChecked() else "    '--no-freeze-backbone',"
        )

        if self.validation_checkbox.isChecked():
            command_lines.extend(
                [
                    "    '--use-validation-split',",
                    f"    '--validation-proportion', {format(self.validation_proportion_spin.value(), '.2f')!r},",
                ]
            )

        resume_path = self.resume_path_edit.text().strip()
        if self.resume_checkbox.isChecked() and resume_path:
            command_lines.extend(
                [
                    "    '--resume',",
                    "    str(RESUME_PATH),",
                ]
            )

        command_lines.append("]")

        code_lines = [f"# Generated for notebook: {notebook_path.name}"]

        if include_path_setup:
            code_lines.extend(["import sys", ""])
            code_lines.insert(1, "from pathlib import Path")
            code_lines.append("")
            if notebook_in_project_root:
                code_lines.extend(
                    [
                        "PROJECT_ROOT = Path.cwd().resolve()",
                        "TRAINING_SCRIPT = (PROJECT_ROOT / 'scripts' / 'entry' / 'training.py').resolve()",
                        f"DATA_ROOT = PROJECT_ROOT / {self._relative_string_for_project_path(DEFAULT_DATA_ROOT)!r}",
                        f"CHECKPOINT_DIR = PROJECT_ROOT / {self._relative_string_for_project_path(self.selected_checkpoint_dir())!r}",
                    ]
                )
                if self.resume_checkbox.isChecked() and resume_path:
                    code_lines.append(
                        f"RESUME_PATH = PROJECT_ROOT / {self._relative_string_for_project_path(Path(resume_path))!r}"
                    )
                else:
                    code_lines.append("RESUME_PATH = None")
            else:
                code_lines.extend(
                    [
                        f"NOTEBOOK_FILE = Path({str(notebook_path.expanduser().resolve())!r})",
                        "NOTEBOOK_DIR = NOTEBOOK_FILE.parent",
                        f"PROJECT_ROOT = {project_root_expression}",
                        "TRAINING_SCRIPT = (PROJECT_ROOT / 'scripts' / 'entry' / 'training.py').resolve()",
                        f"DATA_ROOT = {self._expression_for_path(DEFAULT_DATA_ROOT, notebook_dir=notebook_dir)}",
                        f"CHECKPOINT_DIR = {self._expression_for_path(self.selected_checkpoint_dir(), notebook_dir=notebook_dir)}",
                    ]
                )

                if self.resume_checkbox.isChecked() and resume_path:
                    code_lines.append(
                        f"RESUME_PATH = {self._expression_for_path(Path(resume_path), notebook_dir=notebook_dir)}"
                    )
                else:
                    code_lines.append("RESUME_PATH = None")
        else:
            code_lines.append("")

        code_lines.extend(
            [
                "",
                "from core.notebook_stream import run_and_stream",
                "",
                *command_lines,
                "",
                "print('Project root:', PROJECT_ROOT)",
                "print('Running:', ' '.join(f'\"{part}\"' if ' ' in part else part for part in command))",
                "run_and_stream(command, cwd=PROJECT_ROOT)",
            ]
        )
        return "\n".join(code_lines)

    def export_command_as_python_code(self) -> None:
        checkpoint_name = self.checkpoint_output_name()
        if not checkpoint_name:
            QMessageBox.warning(self, "Checkpoint Name Required", "Choose or enter a checkpoint output folder name before exporting.")
            return

        if self.resume_checkbox.isChecked():
            resume_path = self.resume_path_edit.text().strip()
            if not resume_path:
                QMessageBox.warning(self, "Resume Path Required", "Select a checkpoint file before exporting resume code.")
                return
            if not Path(resume_path).is_file():
                QMessageBox.warning(self, "Invalid Resume Path", f"Checkpoint file does not exist:\n{resume_path}")
                return

        include_path_setup = self.export_include_paths_checkbox.isChecked()
        notebook_path = self._last_export_notebook_path
        if include_path_setup or notebook_path is None:
            selected_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Notebook File",
                str(PROJECT_ROOT),
                "Jupyter Notebook (*.ipynb);;All Files (*.*)",
            )
            if not selected_path:
                return
            notebook_path = Path(selected_path)
            self._last_export_notebook_path = notebook_path

        if notebook_path is None:
            QMessageBox.warning(
                self,
                "Notebook Required",
                "Select a notebook file once with 'Include path setup' enabled before exporting command-only code.",
            )
            return

        code = self.build_notebook_training_code(notebook_path, include_path_setup=include_path_setup)
        QApplication.clipboard().setText(code)
        self.status_label.setText("Python code copied to clipboard.")
        mode_text = "with path setup" if include_path_setup else "command only"
        self.append_output(f"Exported notebook training code ({mode_text}) for {notebook_path}\n")
        if include_path_setup:
            self.export_include_paths_checkbox.setChecked(False)
        QMessageBox.information(
            self,
            "Copied to Clipboard",
            f"Training code has been copied to the clipboard for:\n{notebook_path}\n\nMode: {mode_text}",
        )

    def build_predict_notebook_code(self, notebook_path: Path, *, include_path_setup: bool) -> str:
        notebook_dir = notebook_path.expanduser().resolve().parent
        notebook_in_project_root = notebook_dir == PROJECT_ROOT
        project_relative = Path(os.path.relpath(PROJECT_ROOT, notebook_dir))
        project_root_expression = f"({self._path_expression('NOTEBOOK_DIR', project_relative)}).resolve()"

        image_lines = [
            f"    {self._project_root_expression_for_path(path)},"
            for path in self.predict_image_paths
        ]

        code_lines = [f"# Generated for notebook: {notebook_path.name}"]
        if include_path_setup:
            code_lines.extend(["from pathlib import Path", "import sys", ""])
            if notebook_in_project_root:
                code_lines.extend(
                    [
                        "PROJECT_ROOT = Path.cwd().resolve()",
                        "SCRIPTS_ROOT = (PROJECT_ROOT / 'scripts').resolve()",
                    ]
                )
            else:
                code_lines.extend(
                    [
                        f"NOTEBOOK_FILE = Path({str(notebook_path.expanduser().resolve())!r})",
                        "NOTEBOOK_DIR = NOTEBOOK_FILE.parent",
                        f"PROJECT_ROOT = {project_root_expression}",
                        "SCRIPTS_ROOT = (PROJECT_ROOT / 'scripts').resolve()",
                    ]
                )
            code_lines.extend(
                [
                    "if str(SCRIPTS_ROOT) not in sys.path:",
                    "    sys.path.insert(0, str(SCRIPTS_ROOT))",
                ]
            )
        else:
            code_lines.extend(["from pathlib import Path", ""])

        if self.predict_compare_checkbox.isChecked():
            model_spec_lines = []
            compare_items = self.selected_predict_compare_items(include_main=True, allow_missing_main=True)
            for item in compare_items:
                model_name = str(item.get("detected_model_name", "")).strip()
                checkpoint_path = Path(str(item.get("checkpoint_path", ""))).expanduser()
                if not model_name:
                    continue
                model_spec_lines.append(
                    f"    ({model_name!r}, {self._project_root_expression_for_path(checkpoint_path)}),"
                )
            helper_name = "compare_models_and_display_compact" if self.predict_compact_checkbox.isChecked() else "display_gradcam_comparison"
            code_lines.extend(
                [
                    "",
                    f"from core.notebook_predict import {helper_name}",
                    "",
                    "image_paths = [",
                    *image_lines,
                    "]",
                    "model_specs = [",
                    *model_spec_lines,
                    "]",
                ]
            )
            if self.predict_compact_checkbox.isChecked():
                code_lines.extend(
                    [
                        "",
                        "results = compare_models_and_display_compact(",
                        "    image_paths=image_paths,",
                        "    model_specs=model_specs,",
                        f"    image_size={self.predict_image_size_spin.value()!r},",
                        f"    device={self.predict_device_combo.currentText()!r},",
                        ")",
                    ]
                )
            else:
                current_result = self.predict_results[self.current_predict_index] if self.predict_results and 0 <= self.current_predict_index < len(self.predict_results) else None
                current_image = Path(str(current_result["image_path"])) if isinstance(current_result, dict) and "image_path" in current_result else self.predict_image_paths[0]
                code_lines.extend(
                    [
                        "",
                        f"image_path = {self._project_root_expression_for_path(current_image)}",
                        "display_gradcam_comparison(",
                        "    image_path=image_path,",
                        "    model_specs=model_specs,",
                        f"    image_size={self.predict_image_size_spin.value()!r},",
                        f"    device={self.predict_device_combo.currentText()!r},",
                        ")",
                    ]
                )
        else:
            current_model = self.ensure_predict_model_detected()
            if current_model is None:
                QMessageBox.warning(
                    self,
                    "Model Detection Failed",
                    "Could not auto-detect the checkpoint model type. Choose a valid training checkpoint first.",
                )
                return
            code_lines.extend(
                [
                    "",
                    "from core.notebook_predict import predict_and_display_compact",
                    "",
                    f"checkpoint_path = {self._project_root_expression_for_path(Path(self.predict_checkpoint_edit.text().strip()))}",
                    "image_paths = [",
                    *image_lines,
                    "]",
                    "",
                    "results = predict_and_display_compact(",
                    "    image_paths=image_paths,",
                    "    checkpoint_path=checkpoint_path,",
                    f"    model_name={current_model!r},",
                    f"    image_size={self.predict_image_size_spin.value()!r},",
                    f"    device={self.predict_device_combo.currentText()!r},",
                    ")",
                ]
            )
        return "\n".join(code_lines)

    def export_predicting_as_python_code(self) -> None:
        if not self.predict_image_paths:
            QMessageBox.warning(self, "No Images Selected", "Select one or more images before exporting notebook prediction code.")
            return
        try:
            compare_items = self.selected_predict_compare_items(include_main=True, allow_missing_main=False)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Checkpoint", str(exc))
            return
        if self.predict_compare_checkbox.isChecked():
            model_names = [str(item.get("detected_model_name", "")).strip() for item in compare_items]
            if len([name for name in model_names if name]) != len(set(name for name in model_names if name)):
                QMessageBox.warning(
                    self,
                    "Export Not Supported Yet",
                    "Notebook export for compare mode currently requires unique model architectures.\n"
                    "The desktop Predicting tab can compare duplicate architectures, but the exported notebook helper still expects unique model names.",
                )
                return
        for item in compare_items:
            checkpoint_path = Path(str(item.get("checkpoint_path", ""))).expanduser()
            if not checkpoint_path.is_file():
                item_label = str(item.get("summary_text", checkpoint_path))
                QMessageBox.warning(self, "Invalid Checkpoint", f"Checkpoint file does not exist for {item_label}:\n{checkpoint_path}")
                return

        include_path_setup = self.predict_export_include_paths_checkbox.isChecked()
        notebook_path = self._last_export_notebook_path
        if include_path_setup or notebook_path is None:
            selected_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Notebook File",
                str(PROJECT_ROOT),
                "Jupyter Notebook (*.ipynb);;All Files (*.*)",
            )
            if not selected_path:
                return
            notebook_path = Path(selected_path)
            self._last_export_notebook_path = notebook_path

        if notebook_path is None:
            QMessageBox.warning(
                self,
                "Notebook Required",
                "Select a notebook file once with 'Include path setup' enabled before exporting command-only code.",
            )
            return

        code = self.build_predict_notebook_code(notebook_path, include_path_setup=include_path_setup)
        QApplication.clipboard().setText(code)
        if self.predict_compare_checkbox.isChecked():
            mode_text = "with path setup" if include_path_setup else ("compare compact" if self.predict_compact_checkbox.isChecked() else "Grad-CAM compare")
        else:
            mode_text = "with path setup" if include_path_setup else "compact predict only"
        self.predict_status_label.setText("Predicting code copied to clipboard.")
        if include_path_setup:
            self.predict_export_include_paths_checkbox.setChecked(False)
        QMessageBox.information(
            self,
            "Copied to Clipboard",
            f"Prediction notebook code has been copied to the clipboard for:\n{notebook_path}\n\nMode: {mode_text}",
        )

    def build_logs_notebook_code(self, notebook_path: Path, *, include_path_setup: bool) -> str:
        notebook_dir = notebook_path.expanduser().resolve().parent
        notebook_in_project_root = notebook_dir == PROJECT_ROOT
        project_relative = Path(os.path.relpath(PROJECT_ROOT, notebook_dir))
        project_root_expression = f"({self._path_expression('NOTEBOOK_DIR', project_relative)}).resolve()"

        selected_runs = self.selected_compare_runs()
        if not selected_runs:
            current_run = self.current_available_run()
            selected_runs = [current_run] if current_run is not None else []

        log_path_lines = [
            f"    {self._project_root_expression_for_path(Path(str(run.get('_log_path', ''))))},"
            for run in selected_runs
        ]

        code_lines = [f"# Generated for notebook: {notebook_path.name}"]
        if include_path_setup:
            code_lines.extend(["from pathlib import Path", "import sys", ""])
            if notebook_in_project_root:
                code_lines.extend(
                    [
                        "PROJECT_ROOT = Path.cwd().resolve()",
                        "SCRIPTS_ROOT = (PROJECT_ROOT / 'scripts').resolve()",
                    ]
                )
            else:
                code_lines.extend(
                    [
                        f"NOTEBOOK_FILE = Path({str(notebook_path.expanduser().resolve())!r})",
                        "NOTEBOOK_DIR = NOTEBOOK_FILE.parent",
                        f"PROJECT_ROOT = {project_root_expression}",
                        "SCRIPTS_ROOT = (PROJECT_ROOT / 'scripts').resolve()",
                    ]
                )
            code_lines.extend(
                [
                    "if str(SCRIPTS_ROOT) not in sys.path:",
                    "    sys.path.insert(0, str(SCRIPTS_ROOT))",
                ]
            )
        else:
            code_lines.extend(["from pathlib import Path", ""])

        plot_value = self.training_plot_value_combo.currentText().strip().lower()
        code_lines.extend(["", "log_paths = [", *log_path_lines, "]"])
        if "efficiency" in plot_value:
            code_lines.extend(
                [
                    "from core.notebook_log_analysis import plot_efficiency_tradeoff",
                    "",
                    f"x_metric = {self.training_plot_efficiency_combo.currentText().strip()!r}",
                    "plot_efficiency_tradeoff(log_paths, x_metric=x_metric)",
                ]
            )
        elif "confusion" in plot_value:
            code_lines.extend(
                [
                    "from core.notebook_log_analysis import display_confusion_matrix",
                    "",
                    f"view = {self.training_log_stage_combo.currentText().strip().lower()!r}",
                    f"top_k = {self.training_plot_confusion_spin.value()!r}",
                    "display_confusion_matrix(log_paths, view=view, top_k=top_k)",
                ]
            )
        elif "test splits" in plot_value:
            code_lines.extend(
                [
                    "from core.notebook_log_analysis import plot_test_split_comparison_from_logs",
                    "",
                    "plot_test_split_comparison_from_logs(log_paths)",
                ]
            )
        else:
            code_lines.extend(
                [
                    "from core.notebook_logs import render_log_summary",
                    "",
                    f"view = {self.training_log_stage_combo.currentText().strip().lower()!r}",
                    "print(render_log_summary(log_paths, view=view))",
                ]
            )
        return "\n".join(code_lines)

    def export_logs_as_python_code(self) -> None:
        selected_runs = self.selected_compare_runs()
        current_run = self.current_available_run()
        if not selected_runs and current_run is None:
            QMessageBox.warning(self, "No Logs Selected", "Select or preview at least one log run before exporting notebook code.")
            return

        include_path_setup = self.logs_export_include_paths_checkbox.isChecked()
        notebook_path = self._last_export_notebook_path
        if include_path_setup or notebook_path is None:
            selected_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Notebook File",
                str(PROJECT_ROOT),
                "Jupyter Notebook (*.ipynb);;All Files (*.*)",
            )
            if not selected_path:
                return
            notebook_path = Path(selected_path)
            self._last_export_notebook_path = notebook_path

        if notebook_path is None:
            QMessageBox.warning(
                self,
                "Notebook Required",
                "Select a notebook file once with 'Include path setup' enabled before exporting summary-only code.",
            )
            return

        code = self.build_logs_notebook_code(notebook_path, include_path_setup=include_path_setup)
        QApplication.clipboard().setText(code)
        mode_text = "with path setup" if include_path_setup else "summary only"
        self.training_log_status_label.setText("Log summary code copied to clipboard.")
        if include_path_setup:
            self.logs_export_include_paths_checkbox.setChecked(False)
        QMessageBox.information(
            self,
            "Copied to Clipboard",
            f"Log notebook code has been copied to the clipboard for:\n{notebook_path}\n\nMode: {mode_text}",
        )

    def build_data_command(self, *, check_only: bool = False, force_redownload: bool = False) -> list[str]:
        command = [
            "-u",
            str(DATA_RETRIEVAL_SCRIPT),
            "--data-dir",
            str(DEFAULT_DATA_DIR),
        ]
        if check_only:
            command.append("--check-only")
        if force_redownload:
            command.append("--force-redownload")
        return command

    def build_data_worker_args(self, *, check_only: bool = False, force_redownload: bool = False) -> list[str]:
        command = self.build_data_command(check_only=check_only, force_redownload=force_redownload)
        if len(command) >= 2 and command[0] == "-u":
            return command[2:]
        return command

    def default_predict_checkpoint_path(self) -> Path:
        return DEFAULT_CHECKPOINT_DIR / self.predict_model_combo.currentText() / "best.pth"

    def update_test_split_detected_model(self) -> None:
        checkpoint_text = self.test_split_checkpoint_edit.text().strip()
        checkpoint_path = Path(checkpoint_text).expanduser() if checkpoint_text else None
        detected_model: str | None = None
        if checkpoint_path is not None and checkpoint_path.is_file():
            try:
                from pipeline.predicting import guess_model_name_from_checkpoint_path

                detected_model = guess_model_name_from_checkpoint_path(checkpoint_path.resolve())
            except Exception:
                detected_model = None
        self.test_split_detected_model_name = detected_model
        if detected_model is not None:
            self.test_split_detected_model_label.setText(self.model_source_text(detected_model))
            self.test_split_detected_model_label.setProperty("muted", False)
        elif checkpoint_text:
            self.test_split_detected_model_label.setText("Could not auto-detect model type from this checkpoint.")
            self.test_split_detected_model_label.setProperty("muted", False)
        else:
            self.test_split_detected_model_label.setText("Model will be auto-detected from the checkpoint.")
            self.test_split_detected_model_label.setProperty("muted", True)
        self.test_split_detected_model_label.style().unpolish(self.test_split_detected_model_label)
        self.test_split_detected_model_label.style().polish(self.test_split_detected_model_label)

    def current_predict_model_name(self) -> str | None:
        return self.predict_detected_model_name if isinstance(self.predict_detected_model_name, str) and self.predict_detected_model_name.strip() else None

    def discover_predict_checkpoint_models(self) -> list[str]:
        from pipeline.predicting import discover_checkpoint_model_names

        discovered = discover_checkpoint_model_names(DEFAULT_CHECKPOINT_DIR)
        ordered = sorted(
            discovered,
            key=lambda name: (
                1 if str(model_catalog_entry(name).get("source")) == "legacy" else 0,
                str(name).lower(),
            ),
        )
        return ordered

    def model_source_text(self, model_name: str | None) -> str:
        info = model_catalog_entry(model_name)
        if not bool(info.get("exists")):
            text = str(model_name).strip() if isinstance(model_name, str) else "(unknown)"
            return text
        canonical = str(info.get("model_name", ""))
        provider = str(info.get("provider", "torchvision"))
        family = str(info.get("family", "unknown"))
        variant = str(info.get("variant", "unknown"))
        method = str(info.get("method_type", "unknown"))
        pretrained = info.get("pretrained")
        pre = "pretrained" if pretrained is True else ("scratch" if pretrained is False else "pretrained=?")
        source = str(info.get("source", "handwritten"))
        source_text = "generated" if source == "generated" else ("legacy fallback" if source == "legacy" else "handwritten")
        return f"{canonical} [{provider}/{family}/{variant} | {method} | {pre} | {source_text}]"

    def truncate_checkpoint_selector_model_text(self, text: str, max_chars: int = CHECKPOINT_SELECTOR_MODEL_TEXT_MAX_CHARS) -> str:
        normalized = str(text).strip()
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 1:
            return "…"
        return f"{normalized[: max_chars - 1]}…"

    def training_model_display_name(self, model_name: str) -> str:
        info = model_catalog_entry(model_name)
        source = str(info.get("source", "unknown"))
        variant = str(info.get("variant", "unknown"))
        method = str(info.get("method_type", "unknown"))
        pretrained = info.get("pretrained")
        pre = "pre" if pretrained is True else ("scratch" if pretrained is False else "pre?")
        src = "gen" if source == "generated" else ("legacy" if source == "legacy" else "manual")
        return f"{model_name} [{variant}/{method}/{pre}/{src}]"

    def _set_training_model_combo_items(self, model_names: list[str], *, target_model: str | None = None) -> None:
        self.model_combo.clear()
        for model_name in sort_model_names_for_ui(model_names):
            self.model_combo.addItem(self.training_model_display_name(model_name), model_name)
            row = self.model_combo.count() - 1
            self.model_combo.setItemData(row, model_detailed_tooltip(model_name, include_name=True), Qt.ToolTipRole)
        if target_model:
            index = self.model_combo.findData(target_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

    def current_training_model_name(self) -> str:
        data = self.model_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        text = self.model_combo.currentText().strip()
        if text.endswith(" [legacy]"):
            text = text[:-9]
        return text

    @staticmethod
    def _sanitize_trash_name(value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(value).strip())
        return cleaned.strip("._") or "item"

    def _trash_bundle_dir(self, category: str, item_name: str) -> Path:
        root = LOG_TRASH_DIR if category == "logs" else MODEL_TRASH_DIR
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return root / f"{stamp}_{uuid.uuid4().hex[:8]}_{self._sanitize_trash_name(item_name)}"

    def _prune_trash(self, category: str, limit: int) -> None:
        root = LOG_TRASH_DIR if category == "logs" else MODEL_TRASH_DIR
        if not root.is_dir():
            return
        entries = sorted(root.iterdir(), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
        for stale in entries[limit:]:
            try:
                if stale.is_dir():
                    shutil.rmtree(stale)
                elif stale.exists():
                    stale.unlink()
            except Exception:
                continue

    def _move_paths_to_trash(self, *, category: str, item_name: str, paths: list[Path], limit: int) -> Path | None:
        existing_paths = [path.expanduser().resolve(strict=False) for path in paths if path is not None and path.exists()]
        if not existing_paths:
            return None
        bundle_dir = self._trash_bundle_dir(category, item_name)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest_items: list[dict[str, str]] = []
        for source in existing_paths:
            destination = bundle_dir / source.name
            if destination.exists():
                destination = bundle_dir / f"{uuid.uuid4().hex[:6]}_{source.name}"
            shutil.move(str(source), str(destination))
            manifest_items.append({"original_path": str(source), "trash_path": str(destination)})
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "category": category,
                    "item_name": item_name,
                    "deleted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "items": manifest_items,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._prune_trash(category, limit)
        return bundle_dir

    def _clear_model_registry_caches(self) -> None:
        for attr_name in ("model_metadata", "_load_legacy_migration_pairs"):
            attr = getattr(model_registry_module, attr_name, None)
            if hasattr(attr, "cache_clear"):
                try:
                    attr.cache_clear()
                except Exception:
                    pass

    def _model_paths_for_deletion(self, model_name: str) -> list[Path]:
        metadata = model_registry_module.model_metadata(model_name)
        candidates: list[Path] = [runtime_paths.model_dir() / f"{model_name}.py", DEFAULT_CHECKPOINT_DIR / model_name]
        direct_spec = runtime_paths.model_specs_dir() / f"{model_name}.json"
        candidates.append(direct_spec)
        raw_spec_file = metadata.get("source_spec_file") or metadata.get("spec_file") if isinstance(metadata, dict) else None
        if isinstance(raw_spec_file, str) and raw_spec_file.strip():
            spec_path = Path(raw_spec_file.strip()).expanduser()
            if not spec_path.is_absolute():
                spec_path = PROJECT_ROOT / spec_path
            candidates.append(spec_path.resolve(strict=False))
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            normalized = str(path.resolve(strict=False)).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(path)
        return deduped

    def delete_training_model(self, model_name: str) -> None:
        normalized_name = str(model_name).strip()
        if not normalized_name:
            return
        if len(self.available_models) <= 1:
            QMessageBox.warning(self, "Delete Model", "At least one model must remain available.")
            return
        confirm = QMessageBox.question(
            self,
            "Delete Model",
            f"Move model '{normalized_name}' to the recycle folder?\n\n"
            "This will move the model file, matching spec, and checkpoint directory into .trash/models.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            bundle = self._move_paths_to_trash(
                category="models",
                item_name=normalized_name,
                paths=self._model_paths_for_deletion(normalized_name),
                limit=MODEL_TRASH_LIMIT,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Delete Model Failed", str(exc))
            return
        if bundle is None:
            QMessageBox.information(self, "Delete Model", f"No files were found for model '{normalized_name}'.")
            return
        self._clear_model_registry_caches()
        self.refresh_available_models()
        self.refresh_checkpoint_output_options()
        self.refresh_predict_checkpoint_selector()
        self.refresh_training_log_runs()
        self.training_log_status_label.setText(f"Model '{normalized_name}' moved to {bundle}.")

    def current_log_for_deletion(self) -> dict | None:
        available_item = self.training_log_available_list.currentItem()
        selected_item = self.training_log_selected_list.currentItem()
        available_has_focus = self.training_log_available_list.hasFocus()
        selected_has_focus = self.training_log_selected_list.hasFocus()
        if available_has_focus and available_item is not None:
            return self.current_available_run()
        if selected_has_focus and selected_item is not None:
            return self.current_selected_compare_run()
        if available_item is not None:
            return self.current_available_run()
        if selected_item is not None:
            return self.current_selected_compare_run()
        return None

    def delete_selected_log(self) -> None:
        run = self.current_log_for_deletion()
        if run is None:
            QMessageBox.information(self, "Delete Log", "Select a run log first.")
            return
        log_path_text = str(run.get("_log_path", "")).strip()
        if not log_path_text:
            QMessageBox.warning(self, "Delete Log", "This run does not have a source log file path.")
            return
        log_path = Path(log_path_text).expanduser()
        if not log_path.exists():
            QMessageBox.warning(self, "Delete Log", f"Log file not found:\n{log_path}")
            return
        display_name = self.run_display_name(run)
        confirm = QMessageBox.question(
            self,
            "Delete Log",
            f"Move this log to the recycle folder?\n\n{display_name}\n{log_path}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            bundle = self._move_paths_to_trash(
                category="logs",
                item_name=display_name,
                paths=[log_path],
                limit=LOG_TRASH_LIMIT,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Delete Log Failed", str(exc))
            return
        self.refresh_training_log_runs()
        if bundle is not None:
            self.training_log_status_label.setText(f"Log moved to {bundle}.")

    def update_training_model_source_label(self, model_name: str | None = None) -> None:
        selected = model_name if isinstance(model_name, str) and model_name.strip() else self.current_training_model_name()
        selected_text = self.model_source_text(selected)
        legacy_count = sum(1 for name in self.available_models if str(model_catalog_entry(name).get("source")) == "legacy")
        preferred_count = len(self.available_models) - legacy_count
        self.training_model_variant_label.setText(
            f"{selected_text}. Generated-first ordering active ({preferred_count} preferred, {legacy_count} legacy fallback)."
        )
        self.training_model_variant_label.setToolTip(model_detailed_tooltip(selected, include_name=True))

    def ensure_predict_model_detected(self) -> str | None:
        current_model = self.current_predict_model_name()
        if current_model is not None:
            return current_model
        checkpoint_text = self.predict_checkpoint_edit.text().strip()
        checkpoint_path = Path(checkpoint_text).expanduser() if checkpoint_text else None
        if checkpoint_path is not None and checkpoint_path.is_file():
            try:
                from pipeline.predicting import infer_model_name_from_checkpoint

                detected_model = infer_model_name_from_checkpoint(checkpoint_path.resolve())
            except Exception:
                detected_model = None
            self.predict_detected_model_name = detected_model
            if detected_model is not None:
                if detected_model in self.available_models:
                    self.predict_model_combo.setCurrentText(detected_model)
                self.predict_detected_model_label.setText(self.model_source_text(detected_model))
                self.predict_detected_model_label.setProperty("muted", False)
                self.predict_detected_model_label.style().unpolish(self.predict_detected_model_label)
                self.predict_detected_model_label.style().polish(self.predict_detected_model_label)
                self.refresh_predict_compare_summary()
        return self.current_predict_model_name()

    def update_predict_detected_model(self) -> None:
        checkpoint_text = self.predict_checkpoint_edit.text().strip()
        checkpoint_path = Path(checkpoint_text).expanduser() if checkpoint_text else None
        detected_model: str | None = None
        if checkpoint_path is not None and checkpoint_path.is_file():
            try:
                from pipeline.predicting import infer_model_name_from_checkpoint

                detected_model = infer_model_name_from_checkpoint(checkpoint_path.resolve())
            except Exception:
                detected_model = None

        self.predict_detected_model_name = detected_model
        if detected_model is not None:
            if detected_model in self.available_models:
                self.predict_model_combo.setCurrentText(detected_model)
            self.predict_detected_model_label.setText(self.model_source_text(detected_model))
            self.predict_detected_model_label.setProperty("muted", False)
        elif checkpoint_text:
            self.predict_detected_model_label.setText("Could not auto-detect model type from this checkpoint.")
            self.predict_detected_model_label.setProperty("muted", False)
        else:
            self.predict_detected_model_label.setText("Model will be auto-detected from the checkpoint. Generated variants are preferred when available.")
            self.predict_detected_model_label.setProperty("muted", True)
        self.predict_detected_model_label.style().unpolish(self.predict_detected_model_label)
        self.predict_detected_model_label.style().polish(self.predict_detected_model_label)
        self.refresh_predict_compare_summary()

    def on_predict_model_changed(self) -> None:
        current_path = self.predict_checkpoint_edit.text().strip()
        old_default = DEFAULT_CHECKPOINT_DIR / self._last_predict_model_name / "best.pth"
        old_flat_default = DEFAULT_CHECKPOINT_DIR / f"{self._last_predict_model_name}_best.pth"
        if not current_path or Path(current_path) in {old_default, old_flat_default}:
            self.predict_checkpoint_edit.setText(str(self.default_predict_checkpoint_path()))
        self._last_predict_model_name = self.predict_model_combo.currentText()
        self.refresh_predict_compare_summary()

    def prompt_for_predict_model_confirmation(self, checkpoint_path: Path) -> str | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Checkpoint Model")
        dialog.resize(460, 170)
        layout = QVBoxLayout(dialog)
        prompt = QLabel(
            "The checkpoint model could not be detected automatically.\n"
            "Choose the architecture that matches this checkpoint:",
            dialog,
        )
        prompt.setWordWrap(True)
        layout.addWidget(prompt)
        checkpoint_label = QLabel(str(checkpoint_path), dialog)
        checkpoint_label.setWordWrap(True)
        checkpoint_label.setProperty("muted", True)
        layout.addWidget(checkpoint_label)
        model_combo = QComboBox(dialog)
        model_combo.addItems(self.discover_predict_checkpoint_models())
        layout.addWidget(model_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return model_combo.currentText().strip() or None

    def build_predict_compare_item(
        self,
        checkpoint_path: Path,
        *,
        source: str,
        allow_manual_confirmation: bool,
    ) -> dict[str, object]:
        from pipeline.predicting import validate_prediction_checkpoint

        requested_model_name: str | None = None
        try:
            metadata = validate_prediction_checkpoint(checkpoint_path)
        except ValueError as exc:
            if "Could not determine model type for checkpoint:" not in str(exc) or not allow_manual_confirmation:
                raise
            requested_model_name = self.prompt_for_predict_model_confirmation(checkpoint_path.expanduser().resolve())
            if requested_model_name is None:
                raise ValueError("A model selection is required for checkpoints that cannot be auto-detected.") from exc
            metadata = validate_prediction_checkpoint(checkpoint_path, requested_model_name=requested_model_name)

        resolved_checkpoint = Path(str(metadata["checkpoint_path"])).resolve()
        detected_model_name = str(metadata["resolved_model_name"])
        parent_name = resolved_checkpoint.parent.name
        source_label = self.model_source_text(detected_model_name)
        display_label = f"{source_label} [{parent_name}/{resolved_checkpoint.name}]"
        summary_text = (
            f"{source_label} | {parent_name}/{resolved_checkpoint.name}"
            if parent_name else f"{source_label} | {resolved_checkpoint.name}"
        )
        return {
            "item_id": uuid.uuid4().hex,
            "checkpoint_path": str(resolved_checkpoint),
            "detected_model_name": detected_model_name,
            "display_label": display_label,
            "summary_text": summary_text,
            "source": source,
            "validation_state": "valid",
        }

    def current_predict_main_item(self, *, allow_missing: bool) -> dict[str, object] | None:
        checkpoint_text = self.predict_checkpoint_edit.text().strip()
        if not checkpoint_text:
            if allow_missing:
                return None
            raise ValueError("Choose a prediction checkpoint first.")
        checkpoint_path = Path(checkpoint_text).expanduser()
        if allow_missing and not checkpoint_path.is_file():
            return None
        try:
            return self.build_predict_compare_item(checkpoint_path, source="main", allow_manual_confirmation=False)
        except Exception:
            if allow_missing:
                return None
            raise

    def selected_predict_compare_items(self, *, include_main: bool, allow_missing_main: bool = False) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        if include_main:
            main_item = self.current_predict_main_item(allow_missing=allow_missing_main)
            if main_item is not None:
                items.append(main_item)
        items.extend(self.predict_compare_items)
        return items

    def predict_compare_item_summary(self, item: dict[str, object], *, include_source: bool = True) -> str:
        source = str(item.get("source", "extra"))
        prefix = "Main" if source == "main" else "Add"
        summary = str(item.get("summary_text", "")).strip() or str(item.get("display_label", "")).strip()
        return f"{prefix}: {summary}" if include_source else summary

    def predict_compare_item_compact_label(self, item: dict[str, object]) -> str:
        checkpoint_text = str(item.get("checkpoint_path", "")).strip()
        if checkpoint_text:
            checkpoint_path = Path(checkpoint_text)
            parent_name = checkpoint_path.parent.name
            stem = checkpoint_path.stem
            if parent_name:
                return f"{parent_name}/{stem}"
            if stem:
                return stem
        summary = str(item.get("summary_text", "")).strip()
        if " | " in summary:
            return summary.split(" | ", 1)[1].replace(".pth", "")
        return summary or str(item.get("display_label", "")).strip()

    def iter_predict_checkpoint_selector_items(self) -> list[QTreeWidgetItem]:
        items: list[QTreeWidgetItem] = []
        for index in range(self.predict_checkpoint_tree.topLevelItemCount()):
            model_item = self.predict_checkpoint_tree.topLevelItem(index)
            if model_item is not None:
                items.append(model_item)
        return items

    def refresh_predict_checkpoint_selector(self, *, select_default: bool = False) -> None:
        previous_selected_paths = {
            str(item.get("checkpoint_path", ""))
            for item in self.selected_predict_compare_items(include_main=True, allow_missing_main=True)
        }
        default_checkpoint_path = ""
        if select_default:
            checkpoint_text = self.predict_checkpoint_edit.text().strip()
            if checkpoint_text:
                try:
                    default_checkpoint_path = str(Path(checkpoint_text).expanduser().resolve())
                except Exception:
                    default_checkpoint_path = ""

        self._predict_checkpoint_selector_syncing = True
        self.predict_checkpoint_tree.blockSignals(True)
        self.predict_checkpoint_tree.clear()
        selected_any = False
        fallback_best_item: QTreeWidgetItem | None = None
        for model_name in self.discover_predict_checkpoint_models():
            full_display_model_name = self.model_source_text(model_name)
            display_model_name = self.truncate_checkpoint_selector_model_text(full_display_model_name)
            model_item = QTreeWidgetItem([display_model_name, "", ""])
            model_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.predict_checkpoint_tree.addTopLevelItem(model_item)
            run_tooltip = self.load_latest_run_log_for_checkpoint_dir(DEFAULT_CHECKPOINT_DIR / model_name)
            model_tooltip = full_display_model_name
            if run_tooltip is not None:
                model_tooltip = f"{full_display_model_name}\n\n{self.make_run_tooltip_text(run_tooltip)}"
            model_item.setToolTip(0, model_tooltip)
            for column, checkpoint_kind in ((1, "best"), (2, "last")):
                checkpoint_path = DEFAULT_CHECKPOINT_DIR / model_name / f"{checkpoint_kind}.pth"
                resolved_checkpoint_path = str(checkpoint_path.resolve())
                available = checkpoint_path.is_file()
                model_item.setData(column, Qt.UserRole, {"model_name": model_name, "checkpoint_kind": checkpoint_kind, "checkpoint_path": resolved_checkpoint_path})
                if available:
                    model_item.setFlags(model_item.flags() | Qt.ItemIsUserCheckable)
                    selected = resolved_checkpoint_path in previous_selected_paths or (
                        select_default and default_checkpoint_path and resolved_checkpoint_path == default_checkpoint_path and not previous_selected_paths
                    )
                    model_item.setCheckState(column, Qt.Checked if selected else Qt.Unchecked)
                    model_item.setText(column, "")
                    model_item.setForeground(column, QBrush(QColor("#d7e6f5")))
                    if run_tooltip is not None:
                        model_item.setToolTip(column, self.make_run_tooltip_text(run_tooltip, checkpoint_kind=checkpoint_kind, checkpoint_path=checkpoint_path))
                    if selected:
                        selected_any = True
                    if fallback_best_item is None and checkpoint_kind == "best":
                        fallback_best_item = model_item
                else:
                    model_item.setText(column, "Missing")
                    model_item.setForeground(column, QBrush(QColor("#7f8a98")))
                    model_item.setToolTip(column, f"{checkpoint_kind}: missing\n{checkpoint_path}")
        if select_default and not previous_selected_paths and not selected_any and fallback_best_item is not None:
            fallback_best_item.setCheckState(1, Qt.Checked)
        self.predict_checkpoint_tree.resizeColumnToContents(0)
        current_width = self.predict_checkpoint_tree.columnWidth(0)
        self.predict_checkpoint_tree.setColumnWidth(0, min(current_width, CHECKPOINT_SELECTOR_MODEL_COLUMN_MAX_WIDTH))
        self.predict_checkpoint_tree.resizeColumnToContents(1)
        self.predict_checkpoint_tree.resizeColumnToContents(2)
        self.predict_checkpoint_tree.blockSignals(False)
        self._predict_checkpoint_selector_syncing = False
        self.sync_predict_checkpoint_selection_state(reset_results=False)

    def selected_predict_checkpoint_selector_items(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for tree_item in self.iter_predict_checkpoint_selector_items():
            for column in (1, 2):
                if tree_item.checkState(column) != Qt.Checked:
                    continue
                payload = tree_item.data(column, Qt.UserRole)
                if isinstance(payload, dict):
                    items.append(payload)
        return items

    def sync_predict_checkpoint_selection_state(self, *, reset_results: bool) -> None:
        selected_payloads = self.selected_predict_checkpoint_selector_items()
        ordered_items: list[dict[str, object]] = []
        errors: list[str] = []
        for index, payload in enumerate(selected_payloads):
            checkpoint_path = Path(str(payload.get("checkpoint_path", ""))).expanduser()
            source = "main" if index == 0 else "extra"
            try:
                ordered_items.append(self.build_predict_compare_item(checkpoint_path, source=source, allow_manual_confirmation=False))
            except Exception as exc:
                errors.append(str(exc))

        main_item = ordered_items[0] if ordered_items else None
        self.predict_compare_items = [
            {**item, "source": "extra"}
            for item in ordered_items[1:]
        ]

        self.predict_checkpoint_edit.blockSignals(True)
        self.predict_checkpoint_edit.setText(str(main_item.get("checkpoint_path", "")) if main_item is not None else "")
        self.predict_checkpoint_edit.blockSignals(False)

        compare_enabled = len(ordered_items) >= 2
        self.predict_compare_checkbox.blockSignals(True)
        self.predict_compare_checkbox.setChecked(compare_enabled)
        self.predict_compare_checkbox.blockSignals(False)

        if main_item is not None:
            detected_model = str(main_item.get("detected_model_name", "")).strip() or None
            self.predict_detected_model_name = detected_model
            if detected_model is not None and detected_model in self.available_models:
                self.predict_model_combo.setCurrentText(detected_model)
            if len(ordered_items) == 1 and detected_model is not None:
                self.predict_detected_model_label.setText(self.model_source_text(detected_model))
            elif len(ordered_items) >= 2 and detected_model is not None:
                self.predict_detected_model_label.setText(
                    f"{self.model_source_text(detected_model)} + {len(ordered_items) - 1} more checkpoint(s)"
                )
            else:
                self.predict_detected_model_label.setText(f"{len(ordered_items)} checkpoint(s) selected")
            self.predict_detected_model_label.setProperty("muted", False)
        else:
            self.predict_detected_model_name = None
            if errors:
                self.predict_detected_model_label.setText(errors[0])
                self.predict_detected_model_label.setProperty("muted", False)
            else:
                self.predict_detected_model_label.setText("Select one or more checkpoints from the selector.")
                self.predict_detected_model_label.setProperty("muted", True)
        self.predict_detected_model_label.style().unpolish(self.predict_detected_model_label)
        self.predict_detected_model_label.style().polish(self.predict_detected_model_label)

        if reset_results:
            self.predict_results = []
            self.current_predict_index = -1
            self.predict_compact_built = False
            self.predict_browser_render_key = None

        self.refresh_predict_compare_summary()
        self.refresh_predict_action_states()
        if reset_results:
            self.refresh_predict_page(refresh_compact=True)

    def refresh_predict_action_states(self) -> None:
        selection_count = len(self.selected_predict_checkpoint_selector_items())
        enable_prediction_actions = selection_count > 0 and self.predict_thread is None
        self.predict_run_button.setEnabled(enable_prediction_actions)
        self.predict_queue_button.setEnabled(enable_prediction_actions)
        self.predict_export_button.setEnabled(enable_prediction_actions)
        self.predict_browser_mode_combo.setEnabled(self.predict_thread is None)
        self.predict_select_all_best_button.setEnabled(self.predict_thread is None)
        self.predict_clear_selection_button.setEnabled(selection_count > 0 and self.predict_thread is None)
        self.predict_compare_models_button.setEnabled(self.predict_thread is None)
        self.predict_compare_clear_button.setEnabled(selection_count > 0 and self.predict_thread is None)

    def on_predict_checkpoint_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._predict_checkpoint_selector_syncing or column not in {1, 2}:
            return
        if item.checkState(column) == Qt.Checked:
            other_column = 2 if column == 1 else 1
            self._predict_checkpoint_selector_syncing = True
            self.predict_checkpoint_tree.blockSignals(True)
            item.setCheckState(other_column, Qt.Unchecked)
            self.predict_checkpoint_tree.blockSignals(False)
            self._predict_checkpoint_selector_syncing = False
        self.sync_predict_checkpoint_selection_state(reset_results=True)

    def select_all_predict_best_checkpoints(self) -> None:
        self._predict_checkpoint_selector_syncing = True
        self.predict_checkpoint_tree.blockSignals(True)
        for tree_item in self.iter_predict_checkpoint_selector_items():
            best_payload = tree_item.data(1, Qt.UserRole)
            best_available = isinstance(best_payload, dict) and Path(str(best_payload.get("checkpoint_path", ""))).is_file()
            tree_item.setCheckState(1, Qt.Checked if best_available else Qt.Unchecked)
            tree_item.setCheckState(2, Qt.Unchecked)
        self.predict_checkpoint_tree.blockSignals(False)
        self._predict_checkpoint_selector_syncing = False
        self.sync_predict_checkpoint_selection_state(reset_results=True)

    def clear_predict_checkpoint_selection(self) -> None:
        self._predict_checkpoint_selector_syncing = True
        self.predict_checkpoint_tree.blockSignals(True)
        for tree_item in self.iter_predict_checkpoint_selector_items():
            tree_item.setCheckState(1, Qt.Unchecked)
            tree_item.setCheckState(2, Qt.Unchecked)
        self.predict_checkpoint_tree.blockSignals(False)
        self._predict_checkpoint_selector_syncing = False
        self.sync_predict_checkpoint_selection_state(reset_results=True)

    def refresh_command_preview(self) -> None:
        quoted_parts = []
        for part in [sys.executable, *self.build_command()]:
            text = str(part)
            quoted_parts.append(f'"{text}"' if " " in text else text)
        if quoted_parts:
            preview_text = quoted_parts[0]
            if len(quoted_parts) > 1:
                preview_text += " \\\n    " + " \\\n    ".join(quoted_parts[1:])
        else:
            preview_text = ""
        self.command_preview.setText(preview_text)
        self.refresh_training_settings_summary()

    def refresh_predict_compare_summary(self) -> None:
        main_item = self.current_predict_main_item(allow_missing=True)
        image_count = len(self.predict_image_paths)
        image_text = f"{image_count} image" if image_count == 1 else f"{image_count} images"
        tooltip_lines: list[str] = []
        if main_item is not None:
            tooltip_lines.append(f"Primary: {main_item.get('checkpoint_path', '')}")
        if not self.predict_compare_checkbox.isChecked():
            if main_item is not None:
                summary_label = self.predict_compare_item_compact_label(main_item)
                self.predict_compare_models_label.setText(f"Single | {summary_label} | {image_text}")
                tooltip_lines = [
                    f"Mode: Single",
                    f"Checkpoint: {main_item.get('checkpoint_path', '')}",
                    f"Images: {image_count}",
                ]
            else:
                self.predict_compare_models_label.setText(f"No checkpoint selected | {image_text}")
                tooltip_lines = [f"Images: {image_count}"]
            self.predict_compare_models_label.setToolTip("\n".join(line for line in tooltip_lines if line))
            self.predict_compare_models_button.setEnabled(False)
            self.predict_compare_clear_button.setEnabled(False)
            return
        self.predict_compare_models_button.setEnabled(True)
        self.predict_compare_clear_button.setEnabled(bool(self.predict_compare_items))
        compare_items = self.selected_predict_compare_items(include_main=True, allow_missing_main=True)
        if not compare_items:
            self.predict_compare_models_label.setText(f"Compare | no checkpoints | {image_text}")
            self.predict_compare_models_label.setToolTip(f"Mode: Compare\nImages: {image_count}")
            return
        compact_parts = [self.predict_compare_item_compact_label(item) for item in compare_items]
        self.predict_compare_models_label.setText(
            f"Compare ({len(compare_items)}) | {', '.join(compact_parts)} | {image_text}"
        )
        tooltip_lines = [
            f"Mode: Compare ({len(compare_items)})",
            *(f"{index}. {item.get('checkpoint_path', '')}" for index, item in enumerate(compare_items, start=1)),
            f"Images: {image_count}",
        ]
        self.predict_compare_models_label.setToolTip("\n".join(line for line in tooltip_lines if line))

    def selected_predict_models(self) -> list[str]:
        items = self.selected_predict_compare_items(include_main=self.predict_compare_checkbox.isChecked(), allow_missing_main=True)
        if not self.predict_compare_checkbox.isChecked():
            current_model = self.current_predict_model_name()
            return [current_model] if current_model is not None else []
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            display_label = str(item.get("display_label", "")).strip()
            if display_label and display_label not in seen:
                ordered.append(display_label)
                seen.add(display_label)
        return ordered

    def on_predict_compare_toggled(self, checked: bool) -> None:
        if not checked:
            self.predict_compare_items = []
        self.predict_results = []
        self.current_predict_index = -1
        self.predict_compact_built = False
        self.predict_browser_render_key = None
        self.refresh_predict_compare_summary()
        self.refresh_predict_page(refresh_compact=True)

    def add_predict_compare_model(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Comparison Checkpoint")
        dialog.resize(620, 210)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        checkpoint_edit = QLineEdit(dialog)
        checkpoint_button = QPushButton("Browse...", dialog)
        detected_label = QLabel("Choose a checkpoint. The model will be detected automatically when possible.", dialog)
        detected_label.setWordWrap(True)
        detected_label.setProperty("muted", True)
        selected_item: dict[str, object] | None = None

        def browse_checkpoint() -> None:
            current_path = checkpoint_edit.text().strip() or str(DEFAULT_CHECKPOINT_DIR)
            selected_path, _ = QFileDialog.getOpenFileName(
                dialog,
                "Select Comparison Checkpoint",
                str(self._resolve_dialog_dir(current_path, DEFAULT_CHECKPOINT_DIR)),
                "PyTorch Checkpoints (*.pth *.pt);;All Files (*.*)",
            )
            if selected_path:
                checkpoint_edit.setText(selected_path)
                refresh_detected_item()

        def refresh_detected_item() -> None:
            nonlocal selected_item
            checkpoint_text = checkpoint_edit.text().strip()
            if not checkpoint_text:
                selected_item = None
                detected_label.setText("Choose a checkpoint. The model will be detected automatically when possible.")
                detected_label.setProperty("muted", True)
            else:
                try:
                    selected_item = self.build_predict_compare_item(Path(checkpoint_text), source="extra", allow_manual_confirmation=False)
                except Exception as exc:
                    selected_item = None
                    detected_label.setText(f"Could not validate checkpoint yet:\n{exc}")
                    detected_label.setProperty("muted", False)
                else:
                    detected_label.setText(self.predict_compare_item_summary(selected_item, include_source=False))
                    detected_label.setProperty("muted", False)
            detected_label.style().unpolish(detected_label)
            detected_label.style().polish(detected_label)

        checkpoint_button.clicked.connect(browse_checkpoint)
        checkpoint_edit.editingFinished.connect(refresh_detected_item)
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(checkpoint_edit, stretch=1)
        checkpoint_layout.addWidget(checkpoint_button)
        form.addRow("Checkpoint", checkpoint_layout)
        form.addRow("Detected", detected_label)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return

        checkpoint_text = checkpoint_edit.text().strip()
        if not checkpoint_text:
            return
        try:
            compare_item = self.build_predict_compare_item(Path(checkpoint_text), source="extra", allow_manual_confirmation=True)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Checkpoint", str(exc))
            return
        resolved_checkpoint_text = str(compare_item.get("checkpoint_path", ""))
        existing_paths = {str(item.get("checkpoint_path", "")) for item in self.predict_compare_items}
        main_checkpoint_text = self.predict_checkpoint_edit.text().strip()
        main_checkpoint_path = ""
        if main_checkpoint_text:
            try:
                main_checkpoint_path = str(Path(main_checkpoint_text).expanduser().resolve())
            except Exception:
                main_checkpoint_path = ""
        if resolved_checkpoint_text == main_checkpoint_path or resolved_checkpoint_text in existing_paths:
            QMessageBox.information(
                self,
                "Checkpoint Already Added",
                f"This checkpoint is already part of the comparison set:\n{resolved_checkpoint_text}",
            )
            return
        self.predict_compare_items.append(compare_item)
        self.predict_results = []
        self.current_predict_index = -1
        self.predict_compact_built = False
        self.predict_browser_render_key = None
        self.refresh_predict_compare_summary()
        self.refresh_predict_page(refresh_compact=True)

    def clear_predict_compare_models(self) -> None:
        self.predict_compare_items = []
        self.predict_results = []
        self.current_predict_index = -1
        self.predict_compact_built = False
        self.predict_browser_render_key = None
        self.refresh_predict_compare_summary()
        self.refresh_predict_page(refresh_compact=True)

    def checkpoint_path_for_predict_model(self, model_name: str) -> Path:
        for item in self.selected_predict_compare_items(include_main=True, allow_missing_main=True):
            if str(item.get("display_label", "")) == model_name:
                return Path(str(item.get("checkpoint_path", ""))).expanduser()
        return DEFAULT_CHECKPOINT_DIR / model_name / "best.pth"

    def refresh_available_models(self, *, preferred_model: str | None = None) -> None:
        refreshed = sort_model_names_for_ui(
            discover_model_names_generated_first(include_legacy_fallback=True)
        )
        if not refreshed:
            return
        self.available_models = refreshed
        current_training = self.current_training_model_name()
        current_predict = self.predict_model_combo.currentText().strip()
        preferred_target = resolve_preferred_model_name(preferred_model) if isinstance(preferred_model, str) else None
        current_training_preferred = resolve_preferred_model_name(current_training) if current_training else None
        current_predict_preferred = resolve_preferred_model_name(current_predict) if current_predict else None

        training_target = preferred_target or (
            current_training_preferred if isinstance(current_training_preferred, str) and current_training_preferred in refreshed else refreshed[0]
        )
        predict_target = (
            preferred_target
            if isinstance(preferred_target, str) and preferred_target in refreshed
            else (
                current_predict_preferred
                if isinstance(current_predict_preferred, str) and current_predict_preferred in refreshed
                else refreshed[0]
            )
        )

        self.model_combo.blockSignals(True)
        self._set_training_model_combo_items(refreshed, target_model=training_target)
        self.model_combo.blockSignals(False)

        self.predict_model_combo.blockSignals(True)
        self.predict_model_combo.clear()
        self.predict_model_combo.addItems(refreshed)
        self.predict_model_combo.setCurrentText(predict_target)
        self.predict_model_combo.blockSignals(False)

        self.on_training_model_changed(self.current_training_model_name())
        self._last_predict_model_name = self.predict_model_combo.currentText()
        self.update_training_model_source_label(self.current_training_model_name())
        self.update_predict_detected_model()
        self.update_test_split_detected_model()

    def on_custom_model_generated(self, model_name: str) -> None:
        self.refresh_available_models(preferred_model=model_name)
        self.refresh_checkpoint_output_options(preserve_text=model_name)

    def checkpoint_output_name(self) -> str:
        text = self.checkpoint_output_combo.currentText().strip()
        return "" if text == NEW_CHECKPOINT_NAME_LABEL else text

    def selected_checkpoint_dir(self) -> Path:
        checkpoint_name = self.checkpoint_output_name() or self.current_training_model_name()
        return DEFAULT_CHECKPOINT_DIR / checkpoint_name

    def stop_request_path_for(self, checkpoint_dir: Path | None = None) -> Path:
        target_dir = checkpoint_dir if checkpoint_dir is not None else self.selected_checkpoint_dir()
        return target_dir / ".stop_requested"

    def clear_stop_request_file(self) -> None:
        if self._stop_request_path is not None and self._stop_request_path.exists():
            self._stop_request_path.unlink()

    def refresh_checkpoint_output_options(self, preserve_text: str | None = None) -> None:
        if preserve_text is None:
            preserve_text = self.checkpoint_output_combo.currentText().strip()
        checkpoint_names = sorted(
            path.name for path in DEFAULT_CHECKPOINT_DIR.iterdir()
            if path.is_dir()
        ) if DEFAULT_CHECKPOINT_DIR.is_dir() else []
        items = [*checkpoint_names, NEW_CHECKPOINT_NAME_LABEL]
        self.checkpoint_output_combo.blockSignals(True)
        self.checkpoint_output_combo.clear()
        self.checkpoint_output_combo.addItems(items)
        self.checkpoint_output_combo.blockSignals(False)
        if preserve_text and preserve_text != NEW_CHECKPOINT_NAME_LABEL:
            self.checkpoint_output_combo.setEditText(preserve_text)
        elif self.current_training_model_name():
            self.checkpoint_output_combo.setEditText(self.current_training_model_name())

    def update_checkpoint_dir_label(self) -> None:
        self.checkpoint_dir_label.setText(str(self.selected_checkpoint_dir()))

    def on_training_model_changed(self, model_name: str) -> None:
        resolved_model_name = self.current_training_model_name()
        current_name = self.checkpoint_output_name()
        if self._checkpoint_name_locked_to_model or not current_name or current_name == self._last_training_model_name:
            self.checkpoint_output_combo.setEditText(resolved_model_name)
            self._checkpoint_name_locked_to_model = True
        self._last_training_model_name = resolved_model_name
        self.update_training_model_source_label(resolved_model_name)
        self.update_checkpoint_dir_label()
        self.refresh_command_preview()

    def on_checkpoint_output_changed(self, text: str) -> None:
        checkpoint_name = text.strip()
        self._checkpoint_name_locked_to_model = checkpoint_name in {"", self.current_training_model_name()}
        self.update_checkpoint_dir_label()
        self.refresh_command_preview()

    def on_checkpoint_output_activated(self, index: int) -> None:
        if self.checkpoint_output_combo.itemText(index) == NEW_CHECKPOINT_NAME_LABEL:
            self.checkpoint_output_combo.setEditText("")
            self.checkpoint_output_combo.lineEdit().setFocus()

    def on_resume_toggled(self, checked: bool) -> None:
        self.resume_path_edit.setEnabled(checked)
        self.resume_browse_button.setEnabled(checked)
        self.resume_clear_button.setEnabled(checked)
        if self.training_resume_path_label is not None:
            self.training_resume_path_label.setVisible(checked)
        if self.training_resume_path_widget is not None:
            self.training_resume_path_widget.setVisible(checked)
        self.refresh_command_preview()

    def on_validation_toggled(self, checked: bool) -> None:
        self.validation_proportion_spin.setEnabled(checked)
        if self.training_validation_proportion_label is not None:
            self.training_validation_proportion_label.setVisible(checked)
            self.validation_proportion_spin.setVisible(checked)
        self.refresh_command_preview()

    def on_train_transforms_preset_changed(self, preset: str) -> None:
        self.training_settings_button.setToolTip(
            "Open advanced training settings"
            if preset != "custom"
            else "Open advanced training settings and custom augmentation controls"
        )
        self.refresh_training_settings_summary()
        self.refresh_command_preview()

    def _set_combo_to_value(self, combo: QComboBox, value: object) -> bool:
        text = str(value).strip()
        if not text:
            return False
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
            return True
        if combo.isEditable():
            combo.setEditText(text)
            return True
        return False

    def _apply_custom_augmentation_from_config(self, config: object) -> None:
        if not isinstance(config, dict):
            return
        downsample = config.get("downsample")
        if isinstance(downsample, dict):
            self.custom_downsample_enabled = bool(downsample.get("enabled", self.custom_downsample_enabled))
            if isinstance(downsample.get("probability"), (int, float)):
                self.custom_downsample_prob = float(downsample["probability"])
            if isinstance(downsample.get("min_scale"), (int, float)):
                self.custom_downsample_min_scale = float(downsample["min_scale"])
            if isinstance(downsample.get("max_scale"), (int, float)):
                self.custom_downsample_max_scale = float(downsample["max_scale"])
        blur = config.get("mild_blur")
        if isinstance(blur, dict):
            self.custom_mild_blur_enabled = bool(blur.get("enabled", self.custom_mild_blur_enabled))
            if isinstance(blur.get("probability"), (int, float)):
                self.custom_mild_blur_prob = float(blur["probability"])
        erasing = config.get("random_erasing")
        if isinstance(erasing, dict):
            self.custom_random_erasing_enabled = bool(erasing.get("enabled", self.custom_random_erasing_enabled))
            if isinstance(erasing.get("probability"), (int, float)):
                self.custom_random_erasing_prob = float(erasing["probability"])
        color_jitter = config.get("color_jitter")
        if isinstance(color_jitter, dict):
            self.custom_color_jitter_enabled = bool(color_jitter.get("enabled", self.custom_color_jitter_enabled))
        horizontal_flip = config.get("horizontal_flip")
        if isinstance(horizontal_flip, dict):
            self.custom_horizontal_flip_enabled = bool(horizontal_flip.get("enabled", self.custom_horizontal_flip_enabled))

    def load_resume_checkpoint_training_state(self, checkpoint_path: Path) -> tuple[dict[str, object], list[str]]:
        import torch

        resolved_path = checkpoint_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise ValueError(f"Checkpoint file does not exist:\n{resolved_path}")

        recovered: dict[str, object] = {}
        notes: list[str] = []
        run = self.load_latest_run_log_for_checkpoint_dir(resolved_path.parent)
        if run is not None:
            args = run.get("args") if isinstance(run.get("args"), dict) else {}
            if isinstance(args, dict):
                recovered.update(
                    {
                        "model": args.get("model"),
                        "epochs": args.get("epochs"),
                        "batch_size": args.get("batch_size"),
                        "num_workers": args.get("num_workers"),
                        "image_size": args.get("image_size"),
                        "train_transforms_preset": args.get("train_transforms_preset"),
                        "lr": args.get("lr"),
                        "optimizer": args.get("optimizer"),
                        "scheduler": args.get("scheduler"),
                        "seed": args.get("seed"),
                        "device": args.get("device"),
                        "amp": args.get("amp"),
                        "freeze_backbone": args.get("freeze_backbone"),
                        "use_validation_split": args.get("use_validation_split"),
                        "validation_proportion": args.get("validation_proportion"),
                        "checkpoint_name": Path(str(args.get("checkpoint_dir", resolved_path.parent))).name,
                        "mild_blur_enabled": args.get("mild_blur_enabled"),
                        "mild_blur_prob": args.get("mild_blur_prob"),
                        "custom_augmentation": args.get("augmentation_config"),
                    }
                )
                notes.append("Loaded training parameters from the latest run log in this checkpoint folder.")

        try:
            checkpoint = torch.load(resolved_path, map_location="cpu")
        except Exception as exc:
            if not recovered:
                raise ValueError(f"Could not read checkpoint metadata:\n{resolved_path}\n{exc}") from exc
            notes.append(f"Checkpoint metadata could not be inspected directly: {exc}")
            return recovered, notes

        if isinstance(checkpoint, dict):
            recovered.setdefault("model", checkpoint.get("model_name"))
            recovered.setdefault("optimizer", checkpoint.get("optimizer"))
            recovered.setdefault("scheduler", checkpoint.get("scheduler"))
            recovered.setdefault("amp", checkpoint.get("amp"))
            recovered.setdefault("seed", checkpoint.get("seed"))
            recovered.setdefault("use_validation_split", checkpoint.get("use_validation_split"))
            recovered.setdefault("validation_proportion", checkpoint.get("validation_proportion"))
            recovered.setdefault("checkpoint_name", resolved_path.parent.name)
            if "epoch" in checkpoint and isinstance(checkpoint.get("epoch"), (int, float)):
                notes.append(f"Checkpoint epoch: {int(checkpoint['epoch'])}")
            if "best_acc" in checkpoint and isinstance(checkpoint.get("best_acc"), (int, float)):
                notes.append(f"Checkpoint best acc: {float(checkpoint['best_acc']):.4f}")
        elif not recovered:
            raise ValueError(f"Checkpoint payload is not a dictionary:\n{resolved_path}")

        return recovered, notes

    def apply_resume_checkpoint_to_training_ui(self, checkpoint_path: Path) -> None:
        recovered, notes = self.load_resume_checkpoint_training_state(checkpoint_path)
        resolved_path = checkpoint_path.expanduser().resolve()

        model_name = recovered.get("model")
        if isinstance(model_name, str):
            normalized_model = model_name.strip()
            preferred_model = resolve_preferred_model_name(normalized_model)
            target_model = preferred_model if isinstance(preferred_model, str) else normalized_model
            for available_model in self.available_models:
                if available_model.lower() == target_model.lower():
                    index = self.model_combo.findData(available_model)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                    break

        checkpoint_name = recovered.get("checkpoint_name")
        if isinstance(checkpoint_name, str) and checkpoint_name.strip():
            self.refresh_checkpoint_output_options(preserve_text=checkpoint_name.strip())
            self.checkpoint_output_combo.setEditText(checkpoint_name.strip())

        if isinstance(recovered.get("epochs"), (int, float)):
            self.epochs_spin.setValue(int(recovered["epochs"]))
        if isinstance(recovered.get("batch_size"), (int, float)):
            self.batch_size_spin.setValue(int(recovered["batch_size"]))
        if isinstance(recovered.get("num_workers"), (int, float)):
            self.num_workers_spin.setValue(int(recovered["num_workers"]))
        if isinstance(recovered.get("image_size"), (int, float)):
            self.image_size_spin.setValue(int(recovered["image_size"]))
        if isinstance(recovered.get("lr"), (int, float)):
            self.lr_spin.setValue(float(recovered["lr"]))

        self._set_combo_to_value(self.optimizer_combo, recovered.get("optimizer"))
        self._set_combo_to_value(self.scheduler_combo, recovered.get("scheduler"))
        self._set_combo_to_value(self.train_transforms_preset_combo, recovered.get("train_transforms_preset"))
        self._set_combo_to_value(self.device_combo, recovered.get("device"))

        if isinstance(recovered.get("seed"), (int, float)):
            self.seed_spin.setValue(int(recovered["seed"]))
        if isinstance(recovered.get("amp"), bool):
            self.amp_checkbox.setChecked(bool(recovered["amp"]))
        if isinstance(recovered.get("freeze_backbone"), bool):
            self.freeze_checkbox.setChecked(bool(recovered["freeze_backbone"]))
        if isinstance(recovered.get("use_validation_split"), bool):
            self.validation_checkbox.setChecked(bool(recovered["use_validation_split"]))
        if isinstance(recovered.get("validation_proportion"), (int, float)):
            self.validation_proportion_spin.setValue(float(recovered["validation_proportion"]))

        if isinstance(recovered.get("mild_blur_enabled"), bool):
            self.mild_blur_enabled = bool(recovered["mild_blur_enabled"])
        if isinstance(recovered.get("mild_blur_prob"), (int, float)):
            self.mild_blur_prob = float(recovered["mild_blur_prob"])
        self._apply_custom_augmentation_from_config(recovered.get("custom_augmentation"))

        self.resume_checkbox.setChecked(True)
        self.resume_path_edit.setText(str(resolved_path))
        self.refresh_checkpoint_output_options(preserve_text=self.checkpoint_output_combo.currentText().strip())
        self.update_checkpoint_dir_label()
        self.refresh_training_settings_summary()
        self.refresh_command_preview()
        note_text = " ".join(note for note in notes if note).strip()
        self.status_label.setText("Resume loaded")
        self.progress_label.setText(
            f"Loaded training settings from {resolved_path.name}."
            + (f" {note_text}" if note_text else "")
        )

    def on_resume_path_edited(self) -> None:
        resume_path = self.resume_path_edit.text().strip()
        if not resume_path:
            return
        checkpoint_path = Path(resume_path).expanduser()
        if not checkpoint_path.is_file():
            return
        try:
            self.apply_resume_checkpoint_to_training_ui(checkpoint_path)
        except Exception as exc:
            QMessageBox.warning(self, "Resume Checkpoint Load Failed", str(exc))

    def validate_training_config_snapshot(self, config: dict[str, object]) -> str | None:
        checkpoint_name = str(config.get("checkpoint_name", "")).strip()
        if not checkpoint_name:
            return "Choose or enter a checkpoint output folder name."
        if bool(config.get("resume_enabled")):
            resume_path = str(config.get("resume_path", "")).strip()
            if not resume_path:
                return "Select a checkpoint file before starting resume training."
            if not Path(resume_path).is_file():
                return f"Checkpoint file does not exist:\n{resume_path}"
        preset = str(config.get("train_transforms_preset", "baseline"))
        if preset == "custom":
            custom = config.get("custom_augmentation")
            if isinstance(custom, dict):
                downsample = custom.get("downsample")
                if isinstance(downsample, dict) and downsample.get("enabled"):
                    min_scale = float(downsample.get("min_scale", 0.0))
                    max_scale = float(downsample.get("max_scale", 0.0))
                    if not (0.0 < min_scale <= max_scale <= 1.0):
                        return "Custom downsample min/max scale must satisfy 0 < min <= max <= 1."
        return None

    def add_current_training_config_to_queue(self) -> None:
        config = self.collect_training_config_snapshot()
        error = self.validate_training_config_snapshot(config)
        if error is not None:
            QMessageBox.warning(self, "Invalid Training Config", error)
            return
        title = str(config.get("checkpoint_name", "training run")) or "training run"
        job = global_job_queue.create_queue_job(
            job_type="training",
            title=title,
            source_tab="training",
            config_snapshot=config,
            summary_text=self.training_config_summary(config),
        )
        self.global_queue_jobs.append(job)
        self.refresh_global_queue_view(select_job_id=str(job["job_id"]))

    def collect_predict_config_snapshot(self) -> dict[str, object]:
        if not self.predict_image_paths:
            raise ValueError("Select one or more images before predicting.")
        readable_samples, validation_errors = validate_predict_image_paths(self.predict_image_paths)
        if validation_errors:
            message = "Some selected images are not readable by Python right now.\n\n"
            message += "\n".join(validation_errors[:5])
            if not readable_samples:
                message += "\n\nNo readable sample images were found, so prediction was not started."
            else:
                message += "\n\nPrediction was not started to avoid hanging on unreadable inputs."
            raise ValueError(message)

        compare_items = self.selected_predict_compare_items(include_main=True, allow_missing_main=False)
        compare_enabled = len(compare_items) >= 2
        model_specs: list[dict[str, object]] = []
        if compare_enabled:
            if len(compare_items) < 2:
                raise ValueError("Add at least one extra comparison checkpoint before running compare mode.")
            for item in compare_items:
                checkpoint_path = Path(str(item.get("checkpoint_path", ""))).expanduser().resolve()
                model_name = str(item.get("detected_model_name", "")).strip()
                if not checkpoint_path.is_file():
                    raise ValueError(f"Checkpoint file does not exist:\n{checkpoint_path}")
                if not model_name:
                    raise ValueError(f"Could not determine model type for checkpoint:\n{checkpoint_path}")
                model_specs.append(
                    {
                        "model_name_hint": model_name,
                        "display_label": str(item.get("display_label", model_name)),
                        "checkpoint_path": str(checkpoint_path),
                    }
                )
        else:
            main_item = self.current_predict_main_item(allow_missing=False)
            assert main_item is not None
            checkpoint_path = Path(str(main_item.get("checkpoint_path", ""))).expanduser().resolve()
            model_name = str(main_item.get("detected_model_name", "")).strip()
            if not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint file does not exist:\n{checkpoint_path}")
            if not model_name:
                raise ValueError(f"Could not determine model type for checkpoint:\n{checkpoint_path}")
            model_specs.append(
                {
                    "model_name_hint": model_name,
                    "display_label": str(main_item.get("display_label", model_name)),
                    "checkpoint_path": str(checkpoint_path),
                }
            )

        return {
            "image_paths": [str(path.expanduser().resolve()) for path in self.predict_image_paths],
            "model_specs": model_specs,
            "image_size": int(self.predict_image_size_spin.value()),
            "device": self.predict_device_combo.currentText(),
            "compare_enabled": compare_enabled,
        }

    def predict_config_summary(self, config: dict[str, object]) -> str:
        image_count = len(config.get("image_paths", [])) if isinstance(config.get("image_paths"), list) else 0
        model_specs = config.get("model_specs") if isinstance(config.get("model_specs"), list) else []
        model_count = len(model_specs)
        mode = "compare" if bool(config.get("compare_enabled")) else "single"
        return (
            f"{image_count} image(s)"
            f" | {model_count} model(s)"
            f" | mode={mode}"
            f" | size={config.get('image_size', '-')}"
            f" | device={config.get('device', '-')}"
        )

    def add_current_predict_config_to_queue(self) -> None:
        try:
            config = self.collect_predict_config_snapshot()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Predict Config", str(exc))
            return
        title = f"Predict {len(config.get('image_paths', []))} image(s)"
        job = global_job_queue.create_queue_job(
            job_type="predicting",
            title=title,
            source_tab="predicting",
            config_snapshot=config,
            summary_text=self.predict_config_summary(config),
        )
        self.global_queue_jobs.append(job)
        self.refresh_global_queue_view(select_job_id=str(job["job_id"]))

    def collect_test_split_config_snapshot(self) -> dict[str, object]:
        checkpoint_path = Path(self.test_split_checkpoint_edit.text().strip()).expanduser()
        test_splits_root = Path(self.test_split_root_edit.text().strip()).expanduser()
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file does not exist:\n{checkpoint_path}")
        if not test_splits_root.is_dir():
            raise ValueError(f"Directory does not exist:\n{test_splits_root}")
        return {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "model_name": self.test_split_detected_model_name,
            "test_splits_root": str(test_splits_root.resolve()),
            "image_size": int(self.test_split_image_size_spin.value()),
            "batch_size": int(self.test_split_batch_size_spin.value()),
            "amp_requested": bool(self.test_split_amp_checkbox.isChecked()),
            "device": self.test_split_device_combo.currentText(),
        }

    def collect_test_split_follow_on_snapshot(self) -> dict[str, object]:
        test_splits_root = Path(self.test_split_root_edit.text().strip()).expanduser()
        if not test_splits_root.is_dir():
            raise ValueError(f"Directory does not exist:\n{test_splits_root}")
        return {
            "checkpoint_path": None,
            "model_name": self.test_split_detected_model_name,
            "test_splits_root": str(test_splits_root.resolve()),
            "image_size": int(self.test_split_image_size_spin.value()),
            "batch_size": int(self.test_split_batch_size_spin.value()),
            "amp_requested": bool(self.test_split_amp_checkbox.isChecked()),
            "device": self.test_split_device_combo.currentText(),
        }

    def test_split_config_summary(self, config: dict[str, object]) -> str:
        checkpoint_source = config.get("checkpoint_source") if isinstance(config.get("checkpoint_source"), dict) else None
        if isinstance(checkpoint_source, dict) and checkpoint_source.get("mode") == "best_from_parent_job":
            checkpoint_text = f"best-from-parent:{checkpoint_source.get('parent_job_id', '-')}"
        else:
            checkpoint_text = Path(str(config.get("checkpoint_path", "-"))).name
        return (
            f"{checkpoint_text}"
            f" | size={config.get('image_size', '-')}"
            f" | batch={config.get('batch_size', '-')}"
            f" | amp={config.get('amp_requested', '-')}"
            f" | device={config.get('device', '-')}"
        )

    def add_current_test_split_config_to_queue(self) -> None:
        try:
            config = self.collect_test_split_config_snapshot()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Test Split Config", str(exc))
            return
        title = f"Test Splits: {Path(str(config.get('checkpoint_path', '-'))).name}"
        job = global_job_queue.create_queue_job(
            job_type="test_split_eval",
            title=title,
            source_tab="test_splits",
            config_snapshot=config,
            summary_text=self.test_split_config_summary(config),
        )
        self.global_queue_jobs.append(job)
        self.refresh_global_queue_view(select_job_id=str(job["job_id"]))

    def add_follow_on_test_split_for_selected_job(self) -> None:
        parent_job = self.current_global_queue_job()
        if not isinstance(parent_job, dict) or str(parent_job.get("job_type")) != "training":
            return

        try:
            base_config = self.collect_test_split_follow_on_snapshot()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Test Split Config", str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Follow-on Test Split")
        dialog.resize(420, 220)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        parent_title = QLabel(str(parent_job.get("title", "training job")), dialog)
        parent_title.setWordWrap(True)
        placement_combo = QComboBox(dialog)
        placement_combo.addItems(["Append to queue tail", "Insert directly after parent"])
        placement_combo.setCurrentIndex(0)
        settings_summary = QLabel(self.test_split_config_summary(base_config), dialog)
        settings_summary.setWordWrap(True)
        settings_summary.setProperty("readonlyDisplay", True)

        form.addRow("Parent Training Job", parent_title)
        form.addRow("Placement", placement_combo)
        form.addRow("Test Split Settings", settings_summary)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return

        parent_job_id = str(parent_job.get("job_id", ""))
        parent_config = parent_job.get("config_snapshot") if isinstance(parent_job.get("config_snapshot"), dict) else {}
        follow_on_config = dict(base_config)
        follow_on_config["model_name"] = str(parent_config.get("model", base_config.get("model_name", ""))) or base_config.get("model_name")
        follow_on_config["checkpoint_source"] = {
            "mode": "best_from_parent_job",
            "parent_job_id": parent_job_id,
        }
        follow_on_config["checkpoint_path"] = None

        child_job = global_job_queue.create_queue_job(
            job_type="test_split_eval",
            title=f"Follow-on Test Splits: {parent_job.get('title', 'training job')}",
            source_tab="test_splits",
            config_snapshot=follow_on_config,
            summary_text=self.test_split_config_summary(follow_on_config),
            parent_job_id=parent_job_id,
            status="waiting_on_parent",
        )

        insert_after_parent = placement_combo.currentIndex() == 1
        if insert_after_parent:
            parent_index = self.current_global_queue_index()
            insert_index = parent_index + 1 if parent_index >= 0 else len(self.global_queue_jobs)
            self.global_queue_jobs.insert(insert_index, child_job)
        else:
            self.global_queue_jobs.append(child_job)

        self.resolve_follow_on_children_for_parent(parent_job_id)
        self.refresh_global_queue_view(select_job_id=str(child_job["job_id"]))

    def current_global_queue_index(self) -> int:
        item = self.global_queue_list.currentItem()
        if item is None:
            return -1
        job_id = item.data(Qt.UserRole)
        for index, job in enumerate(self.global_queue_jobs):
            if job.get("job_id") == job_id:
                return index
        return -1

    def current_global_queue_job(self) -> dict[str, object] | None:
        index = self.current_global_queue_index()
        if index < 0 or index >= len(self.global_queue_jobs):
            return None
        return self.global_queue_jobs[index]

    def on_global_queue_selection_changed(self) -> None:
        job = self.current_global_queue_job()
        can_add_follow_on = bool(job and str(job.get("job_type")) == "training")
        self.queue_follow_on_test_split_button.setEnabled(can_add_follow_on)

    def remove_selected_global_queue_job(self) -> None:
        index = self.current_global_queue_index()
        if index < 0:
            return
        job = self.global_queue_jobs[index]
        if job.get("status") == "running":
            return
        del self.global_queue_jobs[index]
        self.refresh_global_queue_view(select_row=min(index, len(self.global_queue_jobs) - 1))

    def duplicate_selected_global_queue_job(self) -> None:
        index = self.current_global_queue_index()
        if index < 0:
            return
        duplicated_job = global_job_queue.clone_queue_job(self.global_queue_jobs[index])
        self.global_queue_jobs.insert(index + 1, duplicated_job)
        self.refresh_global_queue_view(select_job_id=str(duplicated_job["job_id"]))

    def move_selected_global_queue_job_up(self) -> None:
        index = self.current_global_queue_index()
        if index <= 0:
            return
        if self.global_queue_jobs[index].get("status") == "running":
            return
        self.global_queue_jobs[index - 1], self.global_queue_jobs[index] = self.global_queue_jobs[index], self.global_queue_jobs[index - 1]
        self.refresh_global_queue_view(select_row=index - 1)

    def move_selected_global_queue_job_down(self) -> None:
        index = self.current_global_queue_index()
        if index < 0 or index >= len(self.global_queue_jobs) - 1:
            return
        if self.global_queue_jobs[index].get("status") == "running":
            return
        self.global_queue_jobs[index + 1], self.global_queue_jobs[index] = self.global_queue_jobs[index], self.global_queue_jobs[index + 1]
        self.refresh_global_queue_view(select_row=index + 1)

    def clear_finished_global_queue_jobs(self) -> None:
        self.global_queue_jobs = [
            job for job in self.global_queue_jobs
            if str(job.get("status", "queued")) not in {"completed", "failed", "cancelled", "skipped"}
        ]
        self.refresh_global_queue_view()

    def refresh_global_queue_view(self, *, select_job_id: str | None = None, select_row: int | None = None) -> None:
        self.global_queue_list.blockSignals(True)
        self.global_queue_list.clear()
        for order_index, job in enumerate(self.global_queue_jobs, start=1):
            item = QListWidgetItem(f"{order_index}. {global_job_queue.format_queue_job_label(job)}")
            item.setData(Qt.UserRole, job.get("job_id"))
            self.global_queue_list.addItem(item)
        self.global_queue_list.blockSignals(False)
        if select_job_id is not None:
            for row in range(self.global_queue_list.count()):
                item = self.global_queue_list.item(row)
                if item.data(Qt.UserRole) == select_job_id:
                    self.global_queue_list.setCurrentRow(row)
                    break
        elif select_row is not None and self.global_queue_list.count() > 0:
            self.global_queue_list.setCurrentRow(max(0, min(select_row, self.global_queue_list.count() - 1)))
        elif self.global_queue_list.count() > 0 and self.global_queue_list.currentRow() < 0:
            self.global_queue_list.setCurrentRow(0)

        queued_count = sum(1 for job in self.global_queue_jobs if str(job.get("status", "queued")) == "queued")
        waiting_count = sum(1 for job in self.global_queue_jobs if str(job.get("status", "queued")) == "waiting_on_parent")
        running_count = sum(1 for job in self.global_queue_jobs if str(job.get("status", "queued")) == "running")
        if not self.global_queue_jobs:
            self.global_queue_status_label.setText("Queue is empty.")
        elif running_count > 0:
            self.global_queue_status_label.setText(
                f"Queue active. {queued_count} queued, {waiting_count} waiting on parent."
            )
        else:
            self.global_queue_status_label.setText(
                f"{len(self.global_queue_jobs)} job(s). {queued_count} queued, {waiting_count} waiting on parent."
            )
        self.on_global_queue_selection_changed()

    def get_global_queue_job_by_id(self, job_id: str | None) -> dict[str, object] | None:
        if not job_id:
            return None
        for job in self.global_queue_jobs:
            if str(job.get("job_id", "")) == str(job_id):
                return job
        return None

    def resolve_best_checkpoint_from_training_job(self, parent_job: dict[str, object]) -> tuple[str | None, str | None, str | None]:
        artifacts = parent_job.get("artifacts") if isinstance(parent_job.get("artifacts"), dict) else {}
        runtime_best = artifacts.get("best_checkpoint_path")
        if isinstance(runtime_best, str) and runtime_best:
            best_path = Path(runtime_best).expanduser()
            if best_path.is_file():
                run_log_path = artifacts.get("run_log_path")
                return str(best_path.resolve()), (str(run_log_path) if isinstance(run_log_path, str) else None), None

        checkpoint_dir_raw = artifacts.get("checkpoint_dir")
        if not isinstance(checkpoint_dir_raw, str) or not checkpoint_dir_raw:
            config_snapshot = parent_job.get("config_snapshot")
            if isinstance(config_snapshot, dict):
                checkpoint_dir_raw = str(config_snapshot.get("checkpoint_dir", ""))
        checkpoint_dir = Path(str(checkpoint_dir_raw)).expanduser()
        run_logs_dir = checkpoint_dir / RUN_LOG_DIRNAME
        if run_logs_dir.is_dir():
            log_paths = sorted(run_logs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime if path.is_file() else 0.0, reverse=True)
            for log_path in log_paths:
                run_data = run_log_compat.load_run_log(log_path)
                if not isinstance(run_data, dict):
                    continue
                log_artifacts = run_data.get("artifacts") if isinstance(run_data.get("artifacts"), dict) else {}
                best_info = log_artifacts.get("best_checkpoint") if isinstance(log_artifacts.get("best_checkpoint"), dict) else {}
                best_path_raw = best_info.get("path")
                if isinstance(best_path_raw, str) and best_path_raw:
                    best_path = Path(best_path_raw).expanduser()
                    if best_path.is_file():
                        return str(best_path.resolve()), str(log_path.resolve()), None
            return None, None, f"No best checkpoint artifact could be resolved from {run_logs_dir}."
        return None, None, f"No run log directory found for training checkpoint dir: {checkpoint_dir}"

    def resolve_follow_on_children_for_parent(self, parent_job_id: str | None) -> None:
        parent_job = self.get_global_queue_job_by_id(parent_job_id)
        if parent_job is None:
            return
        parent_status = str(parent_job.get("status", "queued"))
        for child_job in self.global_queue_jobs:
            if str(child_job.get("parent_job_id", "")) != str(parent_job_id):
                continue
            if str(child_job.get("status", "")) != "waiting_on_parent":
                continue
            config_snapshot = child_job.get("config_snapshot")
            if not isinstance(config_snapshot, dict):
                child_job["status"] = "skipped"
                child_job["error_message"] = "Child job is missing a valid config snapshot."
                continue
            checkpoint_source = config_snapshot.get("checkpoint_source") if isinstance(config_snapshot.get("checkpoint_source"), dict) else {}
            if not isinstance(checkpoint_source, dict) or checkpoint_source.get("mode") != "best_from_parent_job":
                child_job["status"] = "skipped"
                child_job["error_message"] = "Unsupported follow-on checkpoint source."
                continue
            if parent_status != "completed":
                if global_job_queue.is_terminal_status(parent_status):
                    child_job["status"] = "skipped"
                    child_job["error_message"] = f"Parent training job ended with status={parent_status}; best-checkpoint follow-on was skipped."
                continue
            resolved_best, run_log_path, error_message = self.resolve_best_checkpoint_from_training_job(parent_job)
            if resolved_best is None:
                child_job["status"] = "skipped"
                child_job["error_message"] = error_message or "Best checkpoint could not be resolved from parent training job."
                continue
            config_snapshot["checkpoint_path"] = resolved_best
            child_job["status"] = "queued"
            child_job["error_message"] = None
            child_artifacts = child_job.get("artifacts") if isinstance(child_job.get("artifacts"), dict) else {}
            child_artifacts["resolved_checkpoint_path"] = resolved_best
            if run_log_path is not None:
                child_artifacts["resolved_from_run_log"] = run_log_path
            child_job["artifacts"] = child_artifacts

    def is_global_execution_busy(self) -> bool:
        return (
            self.process.state() != QProcess.NotRunning
            or (self.predict_thread is not None and self.predict_thread.isRunning())
            or (self.test_split_thread is not None and self.test_split_thread.isRunning())
        )

    def start_training_with_config(self, config: dict[str, object], *, origin: str, queue_job_id: str | None = None) -> bool:
        if self.is_global_execution_busy():
            return False
        if runtime_paths.is_frozen_app():
            if not TRAINING_WORKER_EXE.is_file():
                QMessageBox.critical(self, "Missing Worker", f"Could not find packaged training worker:\n{TRAINING_WORKER_EXE}")
                return False
            launch_program = str(TRAINING_WORKER_EXE)
            launch_args = self.build_training_worker_args(config)
        else:
            if not TRAINING_SCRIPT.is_file():
                QMessageBox.critical(self, "Missing Script", f"Could not find training script:\n{TRAINING_SCRIPT}")
                return False
            launch_program = sys.executable
            launch_args = self.build_command(config)
        error = self.validate_training_config_snapshot(config)
        if error is not None:
            QMessageBox.warning(self, "Invalid Training Config", error)
            return False

        checkpoint_dir = Path(str(config.get("checkpoint_dir", ""))).expanduser().resolve()
        self._stop_request_path = self.stop_request_path_for(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.clear_stop_request_file()
        self.training_stop_requested = False
        self.active_job_origin = origin
        self.active_queue_job_type = "training"
        self.active_queue_job_id = queue_job_id
        self.active_job_config_snapshot = json.loads(json.dumps(config))

        self.output_text.clear()
        self._committed_output = ""
        self._stream_buffer = ""
        self.progress_label.setText("Starting training...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.append_output(f"Project root: {PROJECT_ROOT}\n")
        self.append_output(f"Launching: {self.format_command_for_display(launch_args, program=launch_program)}\n\n")
        self.process.start(launch_program, launch_args)
        return True

    def start_predictions_with_config(self, config: dict[str, object], *, origin: str, queue_job_id: str | None = None) -> bool:
        if self.is_global_execution_busy():
            return
        image_paths = [Path(str(path)).expanduser().resolve() for path in config.get("image_paths", [])] if isinstance(config.get("image_paths"), list) else []
        model_specs_raw = config.get("model_specs") if isinstance(config.get("model_specs"), list) else []
        model_specs: list[dict[str, object]] = []
        for item in model_specs_raw:
            if not isinstance(item, dict):
                continue
            checkpoint_path = Path(str(item.get("checkpoint_path", ""))).expanduser().resolve()
            model_name_hint = item.get("model_name_hint") if isinstance(item.get("model_name_hint"), str | type(None)) else None
            display_label = str(item.get("display_label", model_name_hint or checkpoint_path.name))
            model_specs.append(
                {
                    "model_name_hint": model_name_hint,
                    "display_label": display_label,
                    "checkpoint_path": checkpoint_path,
                }
            )
        if not image_paths:
            QMessageBox.warning(self, "Invalid Predict Config", "No images were saved in the queued predict job.")
            return False
        if not model_specs:
            QMessageBox.warning(self, "Invalid Predict Config", "No model checkpoints were saved in the queued predict job.")
            return False

        self.predict_status_label.setText("Loading model and running predictions...")
        self.predict_progress_bar.setRange(0, len(image_paths) * max(len(model_specs), 1))
        self.predict_progress_bar.setValue(0)
        self.set_prediction_running_state(True)
        self.active_job_origin = origin
        self.active_queue_job_type = "predicting"
        self.active_queue_job_id = queue_job_id
        self.active_job_config_snapshot = json.loads(json.dumps(config))

        self.predict_thread = QThread(self)
        self.predict_worker = PredictionWorker(
            image_paths=image_paths,
            model_specs=model_specs,
            image_size=int(config.get("image_size", self.predict_image_size_spin.value())),
            device=str(config.get("device", self.predict_device_combo.currentText())),
        )
        self.predict_worker.moveToThread(self.predict_thread)
        self.predict_thread.started.connect(self.predict_worker.run)
        self.predict_worker.status.connect(self.on_prediction_status)
        self.predict_worker.progress.connect(self.on_prediction_progress)
        self.predict_worker.finished.connect(self.on_prediction_finished)
        self.predict_worker.failed.connect(self.on_prediction_failed)
        self.predict_worker.finished.connect(self.predict_thread.quit)
        self.predict_worker.failed.connect(self.predict_thread.quit)
        self.predict_thread.finished.connect(self.predict_thread.deleteLater)
        self.predict_thread.start()
        return True

    def start_test_split_with_config(self, config: dict[str, object], *, origin: str, queue_job_id: str | None = None) -> bool:
        if self.is_global_execution_busy():
            return False
        checkpoint_path = Path(str(config.get("checkpoint_path", ""))).expanduser().resolve()
        test_splits_root = Path(str(config.get("test_splits_root", ""))).expanduser().resolve()
        if not checkpoint_path.is_file():
            QMessageBox.warning(self, "Invalid Checkpoint", f"Checkpoint file does not exist:\n{checkpoint_path}")
            return False
        if not test_splits_root.is_dir():
            QMessageBox.warning(self, "Invalid Test Splits Root", f"Directory does not exist:\n{test_splits_root}")
            return False

        self.test_split_status_label.setText("Preparing test split evaluation...")
        self.test_split_progress_bar.setRange(0, 0)
        self.test_split_progress_bar.setFormat("Working...")
        self.test_split_output_text.clear()
        self.set_test_split_running_state(True)
        self.active_job_origin = origin
        self.active_queue_job_type = "test_split_eval"
        self.active_queue_job_id = queue_job_id
        self.active_job_config_snapshot = json.loads(json.dumps(config))

        self.test_split_thread = QThread(self)
        self.test_split_worker = TestSplitEvaluationWorker(
            checkpoint_path=checkpoint_path,
            model_name=config.get("model_name") if isinstance(config.get("model_name"), str | type(None)) else None,
            test_splits_root=test_splits_root,
            image_size=int(config.get("image_size", self.test_split_image_size_spin.value())),
            batch_size=int(config.get("batch_size", self.test_split_batch_size_spin.value())),
            amp_requested=bool(config.get("amp_requested", self.test_split_amp_checkbox.isChecked())),
            device=str(config.get("device", self.test_split_device_combo.currentText())),
        )
        self.test_split_worker.moveToThread(self.test_split_thread)
        self.test_split_thread.started.connect(self.test_split_worker.run)
        self.test_split_worker.status.connect(self.on_test_split_status)
        self.test_split_worker.progress.connect(self.on_test_split_progress)
        self.test_split_worker.finished.connect(self.on_test_split_finished)
        self.test_split_worker.failed.connect(self.on_test_split_failed)
        self.test_split_worker.finished.connect(self.test_split_thread.quit)
        self.test_split_worker.failed.connect(self.test_split_thread.quit)
        self.test_split_thread.finished.connect(self.test_split_thread.deleteLater)
        self.test_split_thread.start()
        return True

    def run_global_queue(self) -> None:
        if self.is_global_execution_busy():
            return
        self.global_queue_stop_requested = False
        self.global_queue_running = True
        if not self.start_next_global_queue_job():
            self.global_queue_running = False
            QMessageBox.information(self, "Queue Empty", "There are no queued jobs to run.")

    def start_next_global_queue_job(self) -> bool:
        for job in self.global_queue_jobs:
            if str(job.get("status", "queued")) != "queued":
                continue
            job["status"] = "running"
            job_id = str(job.get("job_id", ""))
            self.refresh_global_queue_view(select_job_id=job_id)
            if self.start_queue_job(job):
                return True
            job["status"] = "failed"
        self.refresh_global_queue_view()
        return False

    def start_queue_job(self, job: dict[str, object]) -> bool:
        config = job.get("config_snapshot")
        if not isinstance(config, dict):
            job["status"] = "skipped"
            return False
        job_type = str(job.get("job_type", ""))
        job_id = str(job.get("job_id", ""))
        if job_type == "training":
            return self.start_training_with_config(config, origin="queue", queue_job_id=job_id)
        if job_type == "predicting":
            return self.start_predictions_with_config(config, origin="queue", queue_job_id=job_id)
        if job_type == "test_split_eval":
            return self.start_test_split_with_config(config, origin="queue", queue_job_id=job_id)
        job["status"] = "skipped"
        job["error_message"] = f"Unsupported queue job type: {job_type}"
        return False

    def complete_global_queue_job(
        self,
        job_id: str | None,
        status: str,
        *,
        artifacts: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> None:
        if not job_id:
            return
        for job in self.global_queue_jobs:
            if job.get("job_id") == job_id:
                job["status"] = status
                job["error_message"] = error_message
                if artifacts:
                    job["artifacts"] = artifacts
                break
        if global_job_queue.is_terminal_status(status):
            self.resolve_follow_on_children_for_parent(job_id)
        self.refresh_global_queue_view(select_job_id=job_id)

    def clear_active_global_job(self) -> None:
        self.active_queue_job_id = None
        self.active_queue_job_type = None
        self.active_job_origin = "manual"
        self.active_job_config_snapshot = None
        self.training_stop_requested = False

    def stop_current_global_job(self) -> None:
        self.global_queue_stop_requested = True
        if self.active_queue_job_type == "training" and self.process.state() != QProcess.NotRunning:
            self.stop_training()
            return
        if self.active_queue_job_type == "predicting":
            self.global_queue_status_label.setText("Queue pause requested. The current prediction job will finish, then the queue will stop.")
            return
        if self.active_queue_job_type == "test_split_eval":
            self.global_queue_status_label.setText("Queue pause requested. The current test split job will finish, then the queue will stop.")

    def training_settings_summary_text(self) -> str:
        if self.train_transforms_preset_combo.currentText() == "custom":
            transform_text = f"Custom={self.custom_augmentation_summary()}"
        else:
            transform_text = (
                f"Blur={self.mild_blur_prob:.2f}"
                if self.mild_blur_enabled
                else "Blur=off"
            )
        return (
            f"Device={self.device_combo.currentText()} | "
            f"Workers={self.num_workers_spin.value()} | "
            f"Scheduler={self.scheduler_combo.currentText()} | "
            f"Seed={self.seed_spin.value()} | "
            f"{transform_text}"
        )

    def refresh_training_settings_summary(self) -> None:
        self.training_settings_summary.setText(self.training_settings_summary_text())

    def open_training_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Training Settings")
        dialog.resize(520, 520)

        layout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget(scroll)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        def create_section(title: str) -> tuple[QGroupBox, QFormLayout]:
            group = QGroupBox(title, scroll_content)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(10, 10, 10, 10)
            group_layout.setSpacing(10)
            form_layout = QFormLayout()
            form_layout.setContentsMargins(0, 0, 0, 0)
            form_layout.setHorizontalSpacing(16)
            form_layout.setVerticalSpacing(10)
            form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
            group_layout.addLayout(form_layout)
            return group, form_layout

        device_combo = QComboBox(dialog)
        device_combo.addItems(["auto", "cpu", "cuda"])
        device_combo.setCurrentText(self.device_combo.currentText())
        device_combo.setToolTip("Select the execution device. Use auto to let the training script choose.")

        num_workers_spin = QSpinBox(dialog)
        num_workers_spin.setRange(0, 64)
        num_workers_spin.setValue(self.num_workers_spin.value())
        num_workers_spin.setToolTip("Number of dataloader worker processes used while training.")

        scheduler_combo = QComboBox(dialog)
        scheduler_combo.addItems(["none", "cosine", "step", "plateau"])
        scheduler_combo.setCurrentText(self.scheduler_combo.currentText())
        scheduler_combo.setToolTip("Learning-rate scheduler applied after each epoch.")

        seed_spin = QSpinBox(dialog)
        seed_spin.setRange(0, 2_147_483_647)
        seed_spin.setValue(self.seed_spin.value())
        seed_spin.setToolTip("Random seed used for training reproducibility.")

        mild_blur_checkbox = QCheckBox("Enable mild blur", dialog)
        mild_blur_checkbox.setChecked(self.mild_blur_enabled)
        mild_blur_checkbox.setToolTip("Add a low-probability mild Gaussian blur on top of the selected training preset.")

        mild_blur_prob_spin = QDoubleSpinBox(dialog)
        mild_blur_prob_spin.setRange(0.01, 0.50)
        mild_blur_prob_spin.setDecimals(2)
        mild_blur_prob_spin.setSingleStep(0.01)
        mild_blur_prob_spin.setValue(self.mild_blur_prob)
        mild_blur_prob_spin.setToolTip("Probability of applying the mild blur augmentation during training.")
        mild_blur_prob_label = QLabel("Blur Probability", dialog)

        def update_blur_controls(checked: bool) -> None:
            preset_mode = self.train_transforms_preset_combo.currentText() != "custom"
            mild_blur_checkbox.setVisible(preset_mode)
            mild_blur_prob_label.setVisible(checked and preset_mode)
            mild_blur_prob_spin.setVisible(checked and preset_mode)
            mild_blur_prob_spin.setEnabled(checked)

        mild_blur_checkbox.toggled.connect(update_blur_controls)
        update_blur_controls(mild_blur_checkbox.isChecked())

        current_preset = self.train_transforms_preset_combo.currentText()
        mode_summary_label = QLabel(
            "Current mode: custom"
            if current_preset == "custom"
            else f"Current mode: preset: {current_preset}",
            dialog,
        )
        mode_summary_label.setWordWrap(True)
        mode_summary_label.setProperty("readonlyDisplay", True)
        mode_summary_hint = QLabel("Validation and test transforms remain deterministic.", dialog)
        mode_summary_hint.setWordWrap(True)
        mode_summary_hint.setProperty("muted", True)

        custom_section_label = QLabel("Custom augmentation options only apply when preset = custom.", dialog)
        custom_section_label.setWordWrap(True)
        custom_section_label.setProperty("muted", True)

        custom_downsample_checkbox = QCheckBox("Enable downsample augmentation", dialog)
        custom_downsample_checkbox.setChecked(self.custom_downsample_enabled)
        custom_downsample_prob_spin = QDoubleSpinBox(dialog)
        custom_downsample_prob_spin.setRange(0.01, 1.00)
        custom_downsample_prob_spin.setDecimals(2)
        custom_downsample_prob_spin.setSingleStep(0.01)
        custom_downsample_prob_spin.setValue(self.custom_downsample_prob)
        custom_downsample_prob_label = QLabel("Downsample Probability", dialog)
        custom_downsample_min_scale_spin = QDoubleSpinBox(dialog)
        custom_downsample_min_scale_spin.setRange(0.05, 1.00)
        custom_downsample_min_scale_spin.setDecimals(2)
        custom_downsample_min_scale_spin.setSingleStep(0.01)
        custom_downsample_min_scale_spin.setValue(self.custom_downsample_min_scale)
        custom_downsample_min_scale_label = QLabel("Downsample Min Scale", dialog)
        custom_downsample_max_scale_spin = QDoubleSpinBox(dialog)
        custom_downsample_max_scale_spin.setRange(0.05, 1.00)
        custom_downsample_max_scale_spin.setDecimals(2)
        custom_downsample_max_scale_spin.setSingleStep(0.01)
        custom_downsample_max_scale_spin.setValue(self.custom_downsample_max_scale)
        custom_downsample_max_scale_label = QLabel("Downsample Max Scale", dialog)

        custom_mild_blur_checkbox = QCheckBox("Enable mild blur", dialog)
        custom_mild_blur_checkbox.setChecked(self.custom_mild_blur_enabled)
        custom_mild_blur_prob_spin = QDoubleSpinBox(dialog)
        custom_mild_blur_prob_spin.setRange(0.01, 0.50)
        custom_mild_blur_prob_spin.setDecimals(2)
        custom_mild_blur_prob_spin.setSingleStep(0.01)
        custom_mild_blur_prob_spin.setValue(self.custom_mild_blur_prob)
        custom_mild_blur_prob_label = QLabel("Custom Blur Probability", dialog)

        custom_random_erasing_checkbox = QCheckBox("Enable random erasing", dialog)
        custom_random_erasing_checkbox.setChecked(self.custom_random_erasing_enabled)
        custom_random_erasing_prob_spin = QDoubleSpinBox(dialog)
        custom_random_erasing_prob_spin.setRange(0.01, 0.50)
        custom_random_erasing_prob_spin.setDecimals(2)
        custom_random_erasing_prob_spin.setSingleStep(0.01)
        custom_random_erasing_prob_spin.setValue(self.custom_random_erasing_prob)
        custom_random_erasing_prob_label = QLabel("Random Erasing Probability", dialog)

        custom_color_jitter_checkbox = QCheckBox("Enable color jitter", dialog)
        custom_color_jitter_checkbox.setChecked(self.custom_color_jitter_enabled)

        custom_horizontal_flip_checkbox = QCheckBox("Enable horizontal flip", dialog)
        custom_horizontal_flip_checkbox.setChecked(self.custom_horizontal_flip_enabled)

        runtime_group, runtime_form = create_section("Runtime & Optimization")
        runtime_form.addRow("Device", device_combo)
        runtime_form.addRow("Num Workers", num_workers_spin)
        runtime_form.addRow("Scheduler", scheduler_combo)
        runtime_form.addRow("Seed", seed_spin)
        runtime_form.addRow("", mild_blur_checkbox)
        runtime_form.addRow(mild_blur_prob_label, mild_blur_prob_spin)

        context_group = QGroupBox("Augmentation Mode / Context", scroll_content)
        context_layout = QVBoxLayout(context_group)
        context_layout.setContentsMargins(10, 10, 10, 10)
        context_layout.setSpacing(8)
        context_layout.addWidget(mode_summary_label)
        context_layout.addWidget(mode_summary_hint)

        custom_group = QGroupBox("Custom Augmentation", scroll_content)
        custom_group_layout = QVBoxLayout(custom_group)
        custom_group_layout.setContentsMargins(10, 10, 10, 10)
        custom_group_layout.setSpacing(12)
        custom_group_layout.addWidget(custom_section_label)

        resolution_group, resolution_form = create_section("Resolution / Degradation")
        resolution_form.addRow("", custom_downsample_checkbox)
        resolution_form.addRow(custom_downsample_prob_label, custom_downsample_prob_spin)
        resolution_form.addRow(custom_downsample_min_scale_label, custom_downsample_min_scale_spin)
        resolution_form.addRow(custom_downsample_max_scale_label, custom_downsample_max_scale_spin)

        blur_group, blur_form = create_section("Blur / Occlusion")
        blur_form.addRow("", custom_mild_blur_checkbox)
        blur_form.addRow(custom_mild_blur_prob_label, custom_mild_blur_prob_spin)
        blur_form.addRow("", custom_random_erasing_checkbox)
        blur_form.addRow(custom_random_erasing_prob_label, custom_random_erasing_prob_spin)

        basic_group, basic_form = create_section("Basic Augmentations")
        basic_form.addRow("", custom_color_jitter_checkbox)
        basic_form.addRow("", custom_horizontal_flip_checkbox)

        custom_group_layout.addWidget(resolution_group)
        custom_group_layout.addWidget(blur_group)
        custom_group_layout.addWidget(basic_group)

        custom_widgets: list[QWidget] = [
            custom_downsample_checkbox,
            custom_downsample_prob_label,
            custom_downsample_prob_spin,
            custom_downsample_min_scale_label,
            custom_downsample_min_scale_spin,
            custom_downsample_max_scale_label,
            custom_downsample_max_scale_spin,
            custom_mild_blur_checkbox,
            custom_mild_blur_prob_label,
            custom_mild_blur_prob_spin,
            custom_random_erasing_checkbox,
            custom_random_erasing_prob_label,
            custom_random_erasing_prob_spin,
            custom_color_jitter_checkbox,
            custom_horizontal_flip_checkbox,
        ]

        def update_custom_downsample_controls(checked: bool) -> None:
            for widget in (
                custom_downsample_prob_label,
                custom_downsample_prob_spin,
                custom_downsample_min_scale_label,
                custom_downsample_min_scale_spin,
                custom_downsample_max_scale_label,
                custom_downsample_max_scale_spin,
            ):
                widget.setVisible(checked and self.train_transforms_preset_combo.currentText() == "custom")
                widget.setEnabled(checked)

        def update_custom_blur_controls(checked: bool) -> None:
            custom_mild_blur_prob_label.setVisible(checked and self.train_transforms_preset_combo.currentText() == "custom")
            custom_mild_blur_prob_spin.setVisible(checked and self.train_transforms_preset_combo.currentText() == "custom")
            custom_mild_blur_prob_spin.setEnabled(checked)

        def update_custom_erasing_controls(checked: bool) -> None:
            custom_random_erasing_prob_label.setVisible(checked and self.train_transforms_preset_combo.currentText() == "custom")
            custom_random_erasing_prob_spin.setVisible(checked and self.train_transforms_preset_combo.currentText() == "custom")
            custom_random_erasing_prob_spin.setEnabled(checked)

        def update_custom_section_visibility() -> None:
            custom_mode = self.train_transforms_preset_combo.currentText() == "custom"
            custom_group.setVisible(custom_mode)
            custom_group.setEnabled(custom_mode)
            for widget in custom_widgets:
                widget.setEnabled(custom_mode)
            update_blur_controls(mild_blur_checkbox.isChecked())
            update_custom_downsample_controls(custom_downsample_checkbox.isChecked())
            update_custom_blur_controls(custom_mild_blur_checkbox.isChecked())
            update_custom_erasing_controls(custom_random_erasing_checkbox.isChecked())

        custom_downsample_checkbox.toggled.connect(update_custom_downsample_controls)
        custom_mild_blur_checkbox.toggled.connect(update_custom_blur_controls)
        custom_random_erasing_checkbox.toggled.connect(update_custom_erasing_controls)
        update_custom_section_visibility()

        scroll_layout.addWidget(runtime_group)
        scroll_layout.addWidget(context_group)
        scroll_layout.addWidget(custom_group)
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.device_combo.setCurrentText(device_combo.currentText())
        self.num_workers_spin.setValue(num_workers_spin.value())
        self.scheduler_combo.setCurrentText(scheduler_combo.currentText())
        self.seed_spin.setValue(seed_spin.value())
        self.mild_blur_enabled = mild_blur_checkbox.isChecked()
        self.mild_blur_prob = mild_blur_prob_spin.value()
        self.custom_downsample_enabled = custom_downsample_checkbox.isChecked()
        self.custom_downsample_prob = custom_downsample_prob_spin.value()
        self.custom_downsample_min_scale = custom_downsample_min_scale_spin.value()
        self.custom_downsample_max_scale = custom_downsample_max_scale_spin.value()
        self.custom_mild_blur_enabled = custom_mild_blur_checkbox.isChecked()
        self.custom_mild_blur_prob = custom_mild_blur_prob_spin.value()
        self.custom_random_erasing_enabled = custom_random_erasing_checkbox.isChecked()
        self.custom_random_erasing_prob = custom_random_erasing_prob_spin.value()
        self.custom_color_jitter_enabled = custom_color_jitter_checkbox.isChecked()
        self.custom_horizontal_flip_enabled = custom_horizontal_flip_checkbox.isChecked()
        self.refresh_training_settings_summary()
        self.refresh_command_preview()

    def on_command_preview_toggled(self, checked: bool) -> None:
        if self.command_preview_body is not None:
            self.command_preview_body.setVisible(checked)

    def choose_resume_path(self) -> None:
        start_dir = self._resolve_dialog_dir(self.resume_path_edit.text().strip(), self.selected_checkpoint_dir())
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Resume Checkpoint",
            str(start_dir),
            "PyTorch Checkpoints (*.pth *.pt);;All Files (*.*)",
        )
        if selected_path:
            try:
                self.apply_resume_checkpoint_to_training_ui(Path(selected_path))
            except Exception as exc:
                QMessageBox.warning(self, "Resume Checkpoint Load Failed", str(exc))

    def clear_resume_path(self) -> None:
        self.resume_path_edit.clear()
        self.refresh_command_preview()

    def set_running_state(self, running: bool) -> None:
        self.train_button.setEnabled(not running)
        self.train_queue_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.model_combo.setEnabled(not running)
        self.epochs_spin.setEnabled(not running)
        self.batch_size_spin.setEnabled(not running)
        self.image_size_spin.setEnabled(not running)
        self.train_transforms_preset_combo.setEnabled(not running)
        self.lr_spin.setEnabled(not running)
        self.optimizer_combo.setEnabled(not running)
        self.amp_checkbox.setEnabled(not running)
        self.training_settings_button.setEnabled(not running)
        self.freeze_checkbox.setEnabled(not running)
        self.validation_checkbox.setEnabled(not running)
        self.validation_proportion_spin.setEnabled(not running and self.validation_checkbox.isChecked())
        self.checkpoint_output_combo.setEnabled(not running)
        self.resume_checkbox.setEnabled(not running)
        self.resume_path_edit.setEnabled(not running and self.resume_checkbox.isChecked())
        self.resume_browse_button.setEnabled(not running and self.resume_checkbox.isChecked())
        self.resume_clear_button.setEnabled(not running and self.resume_checkbox.isChecked())
        self.set_global_queue_running_state(running)

    def append_output(self, text: str) -> None:
        if not text:
            return
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self._committed_output += normalized
        self.output_text.setPlainText(self._committed_output)
        self.output_text.moveCursor(QTextCursor.End)

    def append_data_output(self, text: str) -> None:
        if not text:
            return
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self._data_committed_output += normalized
        self.data_output_text.setPlainText(self._data_committed_output)
        self.data_output_text.moveCursor(QTextCursor.End)

    def append_stream_output(self, text: str) -> None:
        if not text:
            return
        self._stream_buffer += text

        while True:
            cut_positions = [pos for pos in (self._stream_buffer.find("\r"), self._stream_buffer.find("\n")) if pos != -1]
            if not cut_positions:
                break
            cut_index = min(cut_positions)
            delimiter = self._stream_buffer[cut_index]
            line = self._stream_buffer[:cut_index]
            self._stream_buffer = self._stream_buffer[cut_index + 1 :]
            self._handle_stream_line(line, delimiter)

    def append_data_stream_output(self, text: str) -> None:
        if not text:
            return
        self._data_stream_buffer += text

        while True:
            cut_positions = [pos for pos in (self._data_stream_buffer.find("\r"), self._data_stream_buffer.find("\n")) if pos != -1]
            if not cut_positions:
                break
            cut_index = min(cut_positions)
            delimiter = self._data_stream_buffer[cut_index]
            line = self._data_stream_buffer[:cut_index]
            self._data_stream_buffer = self._data_stream_buffer[cut_index + 1 :]
            self._handle_data_stream_line(line, delimiter)

    def _handle_stream_line(self, line: str, delimiter: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        if stripped.startswith("GUI_PROGRESS "):
            self._update_progress_from_payload(stripped.removeprefix("GUI_PROGRESS ").strip())
            return

        self.append_output(stripped + ("\n" if delimiter == "\n" else "\n"))

    def _handle_data_stream_line(self, line: str, delimiter: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        self.data_progress_label.setText(stripped)
        self.append_data_output(stripped + ("\n" if delimiter == "\n" else "\n"))

    def _update_progress_from_payload(self, payload_text: str) -> None:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            self.append_output(payload_text + "\n")
            return

        stage = str(payload.get("stage", "train")).capitalize()
        epoch = payload.get("epoch")
        num_epochs = payload.get("num_epochs")
        step = int(payload.get("step", 0))
        total = int(payload.get("total_steps", 0))
        loss = payload.get("loss")
        acc = payload.get("acc")

        parts = [stage]
        if epoch is not None and num_epochs is not None:
            parts.append(f"Epoch {epoch}/{num_epochs}")
        if total > 0:
            parts.append(f"Step {step}/{total}")
        if isinstance(loss, (int, float)):
            parts.append(f"Loss {loss:.4f}")
        if isinstance(acc, (int, float)):
            parts.append(f"Acc {acc:.4f}")
        self.progress_label.setText(" | ".join(parts))

        if total <= 0:
            self.progress_bar.setRange(0, 0)
            return

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(min(step, total))
        self.progress_bar.setFormat(f"{step}/{total} (%p%)")

    def start_training(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            return
        config = self.collect_training_config_snapshot()
        if not self.start_training_with_config(config, origin="manual"):
            QMessageBox.information(self, "Job Already Running", "Another training, predicting, or test-split job is already running.")

    def set_data_running_state(self, running: bool) -> None:
        self.data_check_button.setEnabled(not running)
        self.data_prepare_button.setEnabled(not running)
        self.data_force_button.setEnabled(not running)

    def start_data_command(self, command: list[str], status_text: str) -> None:
        if self.data_process.state() != QProcess.NotRunning:
            return
        if runtime_paths.is_frozen_app():
            if not DATA_WORKER_EXE.is_file():
                QMessageBox.critical(self, "Missing Worker", f"Could not find packaged data worker:\n{DATA_WORKER_EXE}")
                return
            launch_program = str(DATA_WORKER_EXE)
            launch_args = command[2:] if len(command) >= 2 and command[0] == "-u" else command
        else:
            if not DATA_RETRIEVAL_SCRIPT.is_file():
                QMessageBox.critical(self, "Missing Script", f"Could not find data retrieval script:\n{DATA_RETRIEVAL_SCRIPT}")
                return
            launch_program = sys.executable
            launch_args = command

        self.data_output_text.clear()
        self._data_committed_output = ""
        self._data_stream_buffer = ""
        self.data_status_label.setText(status_text)
        self.data_task_value_label.setText(status_text)
        self.data_state_value_label.setText("Starting")
        self.data_last_result_value_label.setText("Task queued.")
        self.data_progress_label.setText(status_text)
        self.data_progress_bar.setRange(0, 0)
        self.append_data_output(f"Project root: {PROJECT_ROOT}\n")
        self.append_data_output(
            f"Launching: {self.format_command_for_display(launch_args, program=launch_program)}\n\n"
        )
        self.data_process.start(launch_program, launch_args)

    def run_data_check(self) -> None:
        self.start_data_command(self.build_data_command(check_only=True), "Checking dataset integrity...")

    def run_data_prepare(self) -> None:
        self.start_data_command(self.build_data_command(), "Preparing dataset...")

    def run_data_force_redownload(self) -> None:
        self.start_data_command(
            self.build_data_command(force_redownload=True),
            "Force re-downloading and extracting dataset...",
        )

    def load_training_log_files(self) -> list[dict]:
        log_files = sorted(
            DEFAULT_CHECKPOINT_DIR.glob(f"**/{RUN_LOG_DIRNAME}/*.json"),
            key=lambda path: path.stat().st_mtime if path.is_file() else 0.0,
            reverse=True,
        )
        workflow_split_summaries = self.load_workflow_test_split_summaries()
        loaded: list[dict] = []
        for path in log_files:
            data = run_log_compat.load_run_log(path)
            if data is not None:
                log_key = str(path.expanduser().resolve()).lower()
                if not self.test_split_summary_for_run(data):
                    summary = workflow_split_summaries.get(log_key)
                    if summary is not None:
                        data["test_split_summary"] = summary
                loaded.append(data)
        return loaded

    def load_workflow_test_split_summaries(self) -> dict[str, dict]:
        workflow_dir = PROJECT_ROOT / "logs" / "workflow_runs"
        if not workflow_dir.is_dir():
            return {}

        summaries: dict[str, dict] = {}
        for path in sorted(workflow_dir.glob("*.json"), key=lambda item: item.stat().st_mtime if item.is_file() else 0.0, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
            training_log = artifacts.get("training_run_log")
            if not training_log:
                continue
            try:
                training_key = str(Path(str(training_log)).expanduser().resolve()).lower()
            except Exception:
                training_key = str(training_log).lower()
            if training_key in summaries:
                continue
            split_summary = self.test_split_summary_for_run(payload)
            if split_summary is None:
                split_summary = self.load_test_split_summary_from_path(artifacts.get("test_split_json"))
            if split_summary is not None:
                summaries[training_key] = split_summary
        return summaries

    @staticmethod
    def load_test_split_summary_from_path(path_text) -> dict | None:
        if not isinstance(path_text, str) or not path_text.strip():
            return None
        try:
            payload = json.loads(Path(path_text).expanduser().resolve().read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("splits"), list) and payload.get("splits"):
            return payload
        return None

    @staticmethod
    def test_split_summary_for_run(run: dict) -> dict | None:
        summary = run.get("test_split_summary") if isinstance(run, dict) else None
        if isinstance(summary, dict) and isinstance(summary.get("splits"), list) and summary.get("splits"):
            return summary
        return None

    def update_training_plot_value_options(self) -> None:
        has_test_splits = any(self.test_split_summary_for_run(run) is not None for run in self.training_log_runs)
        current_text = self.training_plot_value_combo.currentText()
        options = ["Accuracy", "Loss", "Timing", "Efficiency", "Confusion Matrix"]
        if has_test_splits:
            options.append("Test Splits")
        existing = [self.training_plot_value_combo.itemText(index) for index in range(self.training_plot_value_combo.count())]
        if existing == options:
            return
        self.training_plot_value_combo.blockSignals(True)
        self.training_plot_value_combo.clear()
        self.training_plot_value_combo.addItems(options)
        next_text = current_text if current_text in options else "Accuracy"
        self.training_plot_value_combo.setCurrentText(next_text)
        self.training_plot_value_combo.blockSignals(False)

    def get_run_by_id(self, run_id: str | None) -> dict | None:
        if run_id is None:
            return None
        for run in self.training_log_runs:
            if str(run.get("run_id", "")) == str(run_id):
                return run
        return None

    def selected_compare_run_ids(self) -> list[str]:
        run_ids: list[str] = []
        for index in range(self.training_log_selected_list.count()):
            item = self.training_log_selected_list.item(index)
            run_id = item.data(Qt.UserRole)
            if isinstance(run_id, str):
                run_ids.append(run_id)
        return run_ids

    def selected_compare_runs(self) -> list[dict]:
        runs: list[dict] = []
        for run_id in self.selected_compare_run_ids():
            run = self.get_run_by_id(run_id)
            if run is not None:
                runs.append(run)
        return runs

    def current_available_run(self) -> dict | None:
        item = self.training_log_available_list.currentItem()
        run_id = item.data(Qt.UserRole) if item is not None else None
        if isinstance(run_id, str):
            return self.get_run_by_id(run_id)
        return None

    def current_selected_compare_run(self) -> dict | None:
        item = self.training_log_selected_list.currentItem()
        run_id = item.data(Qt.UserRole) if item is not None else None
        if isinstance(run_id, str):
            return self.get_run_by_id(run_id)
        selected_runs = self.selected_compare_runs()
        return selected_runs[0] if selected_runs else None

    def make_run_list_label(self, run: dict) -> str:
        args = (run.get("args") or {}) if isinstance(run.get("args"), dict) else {}
        model_name = str(args.get("model", "unknown"))
        status = self.normalize_run_status(run)
        started = str(run.get("start_time_utc", "unknown"))[:16].replace("T", " ")
        best_eval = self.format_metric(self.infer_best_eval_acc(run))
        final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else {}
        test_acc = self.format_metric(final_test.get("acc"))
        return f"{started}  {model_name}\n{status}  best={best_eval}  test={test_acc}"

    @staticmethod
    def format_custom_transform_details(args: dict) -> list[str]:
        augmentation_config = args.get("augmentation_config") if isinstance(args.get("augmentation_config"), dict) else {}
        if not isinstance(augmentation_config, dict):
            return []

        lines: list[str] = []
        downsample = augmentation_config.get("downsample")
        if isinstance(downsample, dict) and downsample.get("enabled"):
            lines.append(
                "custom.downsample: "
                f"p={downsample.get('probability', '-')}, "
                f"scale=({downsample.get('min_scale', '-')}, {downsample.get('max_scale', '-')})"
            )
        mild_blur = augmentation_config.get("mild_blur")
        if isinstance(mild_blur, dict) and mild_blur.get("enabled"):
            lines.append(f"custom.mild_blur: p={mild_blur.get('probability', '-')}")
        random_erasing = augmentation_config.get("random_erasing")
        if isinstance(random_erasing, dict) and random_erasing.get("enabled"):
            lines.append(f"custom.random_erasing: p={random_erasing.get('probability', '-')}")
        color_jitter = augmentation_config.get("color_jitter")
        if isinstance(color_jitter, dict):
            lines.append(f"custom.color_jitter: enabled={color_jitter.get('enabled', '-')}")
        horizontal_flip = augmentation_config.get("horizontal_flip")
        if isinstance(horizontal_flip, dict):
            lines.append(f"custom.horizontal_flip: enabled={horizontal_flip.get('enabled', '-')}")
        return lines

    def load_latest_run_log_for_checkpoint_dir(self, checkpoint_dir: Path) -> dict | None:
        run_logs_dir = checkpoint_dir / RUN_LOG_DIRNAME
        if not run_logs_dir.is_dir():
            return None
        log_paths = sorted(
            run_logs_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime if path.is_file() else 0.0,
            reverse=True,
        )
        for path in log_paths:
            data = run_log_compat.load_run_log(path)
            if data is not None:
                return data
        return None

    def make_run_tooltip_text(self, run: dict, *, checkpoint_kind: str | None = None, checkpoint_path: Path | None = None) -> str:
        args = (run.get("args") or {}) if isinstance(run.get("args"), dict) else {}
        summary = (run.get("summary") or {}) if isinstance(run.get("summary"), dict) else {}
        final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else {}
        lines = [self.run_display_name(run)]
        if checkpoint_kind and checkpoint_path is not None:
            lines.append(f"checkpoint: {checkpoint_kind} ({checkpoint_path.name})")
        lines.extend(
            [
                f"status: {self.normalize_run_status(run)}",
                f"best_eval_acc: {self.format_metric(self.infer_best_eval_acc(run))}",
                f"test_acc: {self.format_metric(final_test.get('acc'))}",
                f"epochs: {summary.get('last_completed_epoch', args.get('epochs', '-'))}/{args.get('epochs', '-')}",
                f"image_size: {args.get('image_size', '-')}",
                f"batch_size: {args.get('batch_size', '-')}",
                f"optimizer: {args.get('optimizer', '-')}",
                f"lr: {args.get('lr', '-')}",
                f"scheduler: {args.get('scheduler', '-')}",
                f"freeze_backbone: {args.get('freeze_backbone', '-')}",
                f"amp: {args.get('amp', '-')}",
                f"validation_split: {args.get('use_validation_split', '-')}",
                f"seed: {args.get('seed', '-')}",
                f"transforms: {args.get('train_transforms_preset', '-')}",
            ]
        )
        if str(args.get("train_transforms_preset", "-")) == "custom":
            lines.extend(self.format_custom_transform_details(args))
        return "\n".join(str(line) for line in lines)

    def refresh_training_log_runs(self) -> None:
        self.training_log_runs = self.load_training_log_files()
        self.update_training_plot_value_options()
        previous_available_id = None
        available_item = self.training_log_available_list.currentItem()
        if available_item is not None:
            data = available_item.data(Qt.UserRole)
            if isinstance(data, str):
                previous_available_id = data
        previous_selected_ids = self.selected_compare_run_ids()

        self.training_log_available_list.blockSignals(True)
        self.training_log_selected_list.blockSignals(True)
        self.training_log_available_list.clear()
        self.training_log_selected_list.clear()
        if not self.training_log_runs:
            self.training_log_available_list.blockSignals(False)
            self.training_log_selected_list.blockSignals(False)
            self.training_log_status_label.setText(
                f"No run logs found under {DEFAULT_CHECKPOINT_DIR}. "
                "Start training once to create logs."
            )
            self.training_log_text.setPlainText("")
            self.training_plot_widget.set_plot(
                title="Run Plot",
                x_label="Epoch",
                y_label="Value",
                series=[],
                note="No run logs available yet.",
            )
            return

        selected_row = 0
        for index, run in enumerate(self.training_log_runs):
            run_id = str(run.get("run_id", "unknown"))
            run_tooltip = self.make_run_tooltip_text(run)
            if run_id in previous_selected_ids:
                selected_item = QListWidgetItem(self.make_run_list_label(run))
                selected_item.setData(Qt.UserRole, run_id)
                selected_item.setToolTip(run_tooltip)
                self.training_log_selected_list.addItem(selected_item)
                continue

            available_item = QListWidgetItem(self.make_run_list_label(run))
            available_item.setData(Qt.UserRole, run_id)
            available_item.setToolTip(run_tooltip)
            self.training_log_available_list.addItem(available_item)
            if previous_available_id is not None and run_id == previous_available_id:
                selected_row = self.training_log_available_list.count() - 1

        if self.training_log_available_list.count() > 0:
            self.training_log_available_list.setCurrentRow(selected_row)
        self.training_log_available_list.blockSignals(False)
        self.training_log_selected_list.blockSignals(False)
        self.refresh_training_log_view()

    def on_available_log_selection_changed(self) -> None:
        self.refresh_training_log_view()

    def on_selected_log_selection_changed(self) -> None:
        self.refresh_training_log_view()

    def add_selected_log_to_compare(self) -> None:
        item = self.training_log_available_list.currentItem()
        current_run = self.current_available_run()
        if current_run is None or item is None:
            return
        run_id = str(current_run.get("run_id", "unknown"))
        if run_id not in self.selected_compare_run_ids():
            new_item = QListWidgetItem(self.make_run_list_label(current_run))
            new_item.setData(Qt.UserRole, run_id)
            new_item.setToolTip(self.make_run_tooltip_text(current_run))
            row = self.training_log_available_list.row(item)
            self.training_log_available_list.takeItem(row)
            self.training_log_selected_list.addItem(new_item)
            self.training_log_selected_list.setCurrentItem(new_item)
            if self.training_log_available_list.count() > 0:
                self.training_log_available_list.setCurrentRow(min(row, self.training_log_available_list.count() - 1))
        self.refresh_training_log_view()

    def remove_selected_log_from_compare(self) -> None:
        row = self.training_log_selected_list.currentRow()
        if row < 0:
            return
        item = self.training_log_selected_list.takeItem(row)
        if item is not None:
            self.training_log_available_list.addItem(item)
            self.training_log_available_list.sortItems()
        if self.training_log_selected_list.count() > 0:
            self.training_log_selected_list.setCurrentRow(min(row, self.training_log_selected_list.count() - 1))
        elif self.training_log_available_list.count() > 0:
            self.training_log_available_list.setCurrentRow(0)
        self.refresh_training_log_view()

    def clear_selected_logs(self) -> None:
        while self.training_log_selected_list.count() > 0:
            item = self.training_log_selected_list.takeItem(0)
            if item is not None:
                self.training_log_available_list.addItem(item)
        self.training_log_available_list.sortItems()
        if self.training_log_available_list.count() > 0:
            self.training_log_available_list.setCurrentRow(0)
        self.refresh_training_log_view()

    @staticmethod
    def signature_matches(saved: dict | None, current: dict | None) -> bool:
        if not isinstance(saved, dict) or not isinstance(current, dict):
            return False
        return saved.get("exists") == current.get("exists") and saved.get("size") == current.get("size") and saved.get(
            "mtime_ns"
        ) == current.get("mtime_ns")

    @staticmethod
    def current_file_signature(path: Path) -> dict:
        if not path.is_file():
            return {"exists": False}
        stat = path.stat()
        return {"exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}

    def describe_artifact_state(self, artifact: dict) -> str:
        path_text = str(artifact.get("path", ""))
        if not path_text:
            return "unknown"
        current_sig = self.current_file_signature(Path(path_text))
        final_sig = artifact.get("final_signature")
        initial_sig = artifact.get("initial_signature")

        if not current_sig.get("exists", False):
            return "missing"
        if self.signature_matches(final_sig if isinstance(final_sig, dict) else None, current_sig):
            return "exists (same as saved in this run)"
        if self.signature_matches(initial_sig if isinstance(initial_sig, dict) else None, current_sig):
            return "exists (same as before this run)"
        return "exists (overwritten after this run)"

    def normalize_run_status(self, run: dict) -> str:
        return run_log_compat.normalize_run_status(run)

    @staticmethod
    def safe_float(value) -> float | None:
        return run_log_compat.safe_float(value)

    @staticmethod
    def format_metric(value) -> str:
        return run_log_compat.format_metric(value)

    @staticmethod
    def format_ratio(numerator, denominator) -> str:
        return run_log_compat.format_ratio(numerator, denominator)

    @staticmethod
    def safe_int(value) -> int | None:
        return run_log_compat.safe_int(value)

    @staticmethod
    def infer_last_completed_epoch(run: dict) -> int:
        return run_log_compat.infer_last_completed_epoch(run)

    @staticmethod
    def infer_eval_name(run: dict) -> str:
        return run_log_compat.infer_eval_name(run)

    @staticmethod
    def infer_best_eval_acc(run: dict) -> float | None:
        return run_log_compat.infer_best_eval_acc(run)

    @staticmethod
    def extract_analysis_block(run: dict, stage_name: str | None = None) -> dict | None:
        return run_log_compat.extract_analysis_block(run, stage_name=stage_name)

    @staticmethod
    def summarize_error_block(analysis: dict | None, *, limit: int = 5) -> list[str]:
        return run_log_compat.summarize_error_block(analysis, limit=limit)

    def efficiency_point_for_run(self, run: dict, metric_name: str) -> tuple[float | None, str]:
        timing_summary = run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {}
        stage_totals = timing_summary.get("stage_totals") if isinstance(timing_summary.get("stage_totals"), dict) else {}
        model_info = run.get("model") if isinstance(run.get("model"), dict) else {}
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else {}

        if metric_name == "Train Wall Time":
            train_stage = stage_totals.get("train") if isinstance(stage_totals.get("train"), dict) else {}
            return self.safe_float(train_stage.get("total_seconds")), "Train Wall Time (s)"
        if metric_name == "Train Pure Time":
            train_stage = stage_totals.get("train") if isinstance(stage_totals.get("train"), dict) else {}
            return self.safe_float(train_stage.get("pure_seconds")), "Train Pure Time (s)"
        if metric_name == "Test Avg Pure / Batch":
            test_timing = final_test.get("timing") if isinstance(final_test.get("timing"), dict) else None
            if test_timing is None:
                last_epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
                if last_epochs and isinstance(last_epochs[-1], dict):
                    eval_name = self.infer_eval_name(run)
                    stage = last_epochs[-1].get(eval_name)
                    test_timing = stage.get("timing") if isinstance(stage, dict) and isinstance(stage.get("timing"), dict) else None
            if isinstance(test_timing, dict):
                pure = self.safe_float(test_timing.get("pure_seconds"))
                batches = self.safe_float(test_timing.get("batches"))
                if pure is not None and batches is not None and batches > 0:
                    return pure / batches, "Test Avg Pure / Batch (s)"
            return None, "Test Avg Pure / Batch (s)"
        if metric_name == "Trainable Params":
            return self.safe_float(model_info.get("trainable_params")), "Trainable Params"
        return self.safe_float(summary.get("final_test_acc")), "Accuracy"

    def build_efficiency_plot(self, runs: list[dict]) -> dict:
        x_metric_name = self.training_plot_efficiency_combo.currentText().strip()
        points: list[dict[str, object]] = []
        for index, run in enumerate(runs):
            x_value, x_label = self.efficiency_point_for_run(run, x_metric_name)
            summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
            y_value = self.safe_float(summary.get("final_test_acc"))
            if y_value is None:
                y_value = self.infer_best_eval_acc(run)
            model_info = run.get("model") if isinstance(run.get("model"), dict) else {}
            size = self.safe_float(model_info.get("trainable_params")) or 1.0
            if x_value is None or y_value is None:
                continue
            points.append(
                {
                    "label": self.run_display_name(run),
                    "x": x_value,
                    "y": y_value,
                    "size": size,
                    "color": self.stage_color(f"eff_{index}", index),
                }
            )
        return {
            "title": "Performance vs Efficiency",
            "x_label": x_label if points else x_metric_name,
            "y_label": "Accuracy",
            "points": points,
            "note": "Bubble size represents trainable parameter count.",
        }

    def build_confusion_matrix(self, run: dict) -> dict:
        selected_view = self.training_log_stage_combo.currentText().strip().lower()
        analysis_stage = "final_test" if selected_view == "summary" else selected_view
        analysis = self.extract_analysis_block(run, stage_name=analysis_stage)
        if not isinstance(analysis, dict):
            return {"title": "Confusion Matrix", "labels": [], "matrix": [], "note": "No confusion data recorded for this run."}

        class_names = analysis.get("class_names") if isinstance(analysis.get("class_names"), list) else []
        pair_entries = analysis.get("confusion_pairs") if isinstance(analysis.get("confusion_pairs"), list) else []
        if not pair_entries:
            return {"title": "Confusion Matrix", "labels": [], "matrix": [], "note": "No confusion pairs recorded."}

        involvement: dict[int, int] = {}
        for entry in pair_entries:
            if not isinstance(entry, dict):
                continue
            true_idx = self.safe_int(entry.get("true_idx"))
            pred_idx = self.safe_int(entry.get("pred_idx"))
            count = self.safe_int(entry.get("count")) or 0
            if true_idx is None or pred_idx is None:
                continue
            if true_idx != pred_idx:
                involvement[true_idx] = involvement.get(true_idx, 0) + count
                involvement[pred_idx] = involvement.get(pred_idx, 0) + count

        if not involvement:
            for entry in pair_entries:
                if not isinstance(entry, dict):
                    continue
                true_idx = self.safe_int(entry.get("true_idx"))
                count = self.safe_int(entry.get("count")) or 0
                if true_idx is not None:
                    involvement[true_idx] = involvement.get(true_idx, 0) + count

        top_k = self.training_plot_confusion_spin.value()
        selected_indices = [idx for idx, _ in sorted(involvement.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
        if not selected_indices:
            return {"title": "Confusion Matrix", "labels": [], "matrix": [], "note": "No confusion data recorded for this run."}

        selected_lookup = {idx: position for position, idx in enumerate(selected_indices)}
        matrix = [[0 for _ in selected_indices] for _ in selected_indices]
        for entry in pair_entries:
            if not isinstance(entry, dict):
                continue
            true_idx = self.safe_int(entry.get("true_idx"))
            pred_idx = self.safe_int(entry.get("pred_idx"))
            count = self.safe_int(entry.get("count")) or 0
            if true_idx in selected_lookup and pred_idx in selected_lookup:
                matrix[selected_lookup[true_idx]][selected_lookup[pred_idx]] = count

        labels = [
            str(class_names[idx]) if 0 <= idx < len(class_names) else str(idx)
            for idx in selected_indices
        ]
        return {
            "title": f"Top-{len(selected_indices)} Confusion Matrix",
            "labels": labels,
            "matrix": matrix,
            "note": "Classes are chosen by highest confusion involvement.",
        }

    @staticmethod
    def stage_color(stage_name: str, fallback_index: int = 0) -> str:
        fixed = {
            "train": "#f59e0b",
            "val": "#22c55e",
            "test": "#4e8cff",
        }
        if stage_name in fixed:
            return fixed[stage_name]
        palette = ["#4e8cff", "#f97316", "#14b8a6", "#ef4444", "#a855f7", "#eab308", "#10b981", "#f43f5e"]
        return palette[fallback_index % len(palette)]

    @staticmethod
    def timing_value_from_stage(stage: dict, timing_metric: str) -> float | None:
        return run_log_compat.timing_value_from_stage(stage, timing_metric)

    def extract_stage_points(self, run: dict, stage_name: str, value_kind: str, timing_metric: str | None = None) -> list[tuple[float, float]]:
        return run_log_compat.extract_epoch_metrics(run, stage_name, value_kind, timing_metric)

    def current_selected_run(self) -> dict | None:
        selected_runs = self.selected_compare_runs()
        if len(selected_runs) == 1:
            return selected_runs[0]
        if len(selected_runs) > 1:
            return self.current_selected_compare_run()
        return self.current_available_run()

    def run_display_name(self, run: dict, include_stage: str | None = None) -> str:
        return run_log_compat.run_display_name(run, include_stage=include_stage)

    def build_selected_run_plot(self, run: dict, *, value_kind: str, timing_metric: str) -> dict:
        stage_choice = self.training_plot_stage_combo.currentText().strip().lower()
        stages = ["train", "val", "test"] if stage_choice.startswith("all") else [stage_choice]
        series: list[dict] = []
        for index, stage_name in enumerate(stages):
            points = self.extract_stage_points(run, stage_name, value_kind, timing_metric)
            if not points:
                continue
            series.append({"label": stage_name, "color": self.stage_color(stage_name, index), "points": points})

        timing_label = {"total": "Total Time (s)", "pure": "Pure Time (s)", "avg": "Avg Pure / Batch (s)"}[timing_metric]
        if value_kind == "accuracy":
            title = "Run Accuracy"
            y_label = "Accuracy"
        elif value_kind == "loss":
            title = "Run Loss"
            y_label = "Loss"
        else:
            title = "Run Timing"
            y_label = timing_label

        return {
            "title": title,
            "x_label": "Epoch",
            "y_label": y_label,
            "series": series,
            "note": "All available stage curves are shown together." if stage_choice.startswith("all") else "",
        }

    def build_compare_plot(self, runs: list[dict], *, value_kind: str, timing_metric: str) -> dict:
        stage_choice = self.training_plot_stage_combo.currentText().strip().lower()
        series: list[dict] = []
        for index, run in enumerate(runs):
            stage_name = self.infer_eval_name(run) if stage_choice.startswith("all") else stage_choice
            points = self.extract_stage_points(run, stage_name, value_kind, timing_metric)
            if not points:
                continue
            series.append(
                {
                    "label": self.run_display_name(run, include_stage=stage_name),
                    "color": self.stage_color(f"compare_{index}", index),
                    "points": points,
                }
            )

        timing_label = {"total": "Total Time (s)", "pure": "Pure Time (s)", "avg": "Avg Pure / Batch (s)"}[timing_metric]
        note = (
            "Auto stage uses each run's epoch-wise evaluation stage and leaves missing epochs blank."
            if stage_choice.startswith("all")
            else "Missing epochs are left blank for runs that start later or end earlier."
        )
        if value_kind == "accuracy":
            title = "Compare Accuracy Across Runs"
            y_label = "Accuracy"
        elif value_kind == "loss":
            title = "Compare Loss Across Runs"
            y_label = "Loss"
        else:
            title = "Compare Timing Across Runs"
            y_label = timing_label

        return {
            "title": title,
            "x_label": "Epoch",
            "y_label": y_label,
            "series": series,
            "note": note,
        }

    def build_test_split_plot(self, runs: list[dict]) -> dict:
        split_names: list[str] = []
        split_rows_by_run: list[tuple[dict, list[dict]]] = []
        for run in runs:
            split_summary = self.test_split_summary_for_run(run)
            if split_summary is None:
                continue
            split_rows = [item for item in split_summary.get("splits", []) if isinstance(item, dict)]
            if not split_rows:
                continue
            split_rows_by_run.append((run, split_rows))
            for item in split_rows:
                split_name = str(item.get("split", "")).strip()
                if split_name and split_name not in split_names:
                    split_names.append(split_name)

        split_index = {split_name: index + 1 for index, split_name in enumerate(split_names)}
        series: list[dict] = []
        for index, (run, split_rows) in enumerate(split_rows_by_run):
            points: list[tuple[float, float]] = []
            for item in split_rows:
                split_name = str(item.get("split", "")).strip()
                accuracy = self.safe_float(item.get("accuracy"))
                if split_name in split_index and accuracy is not None:
                    points.append((float(split_index[split_name]), accuracy))
            if points:
                series.append(
                    {
                        "label": self.run_display_name(run),
                        "color": self.stage_color(f"test_split_{index}", index),
                        "points": points,
                    }
                )

        skipped_count = len(runs) - len(split_rows_by_run)
        note = "X axis follows the split order recorded in the logs."
        if skipped_count > 0:
            note += f" {skipped_count} selected run(s) had no test split summary."
        return {
            "title": "Test Split Accuracy",
            "x_label": "Test Split",
            "y_label": "Accuracy",
            "series": series,
            "note": note if series else "No test split summary recorded for this selection.",
            "x_tick_labels": {float(index): name.replace("_", " ") for name, index in split_index.items()},
        }

    def refresh_training_log_plot(self) -> None:
        selected_runs = self.selected_compare_runs()
        if not selected_runs:
            current_run = self.current_available_run()
            selected_runs = [current_run] if current_run is not None else []
        timing_metric_label = self.training_plot_timing_combo.currentText().strip().lower()
        timing_metric = "avg" if "avg" in timing_metric_label else ("pure" if "pure" in timing_metric_label else "total")
        plot_value = self.training_plot_value_combo.currentText().strip().lower()
        is_accuracy = "accuracy" in plot_value
        is_loss = plot_value == "loss"
        is_timing = plot_value == "timing"
        is_efficiency = "efficiency" in plot_value
        is_confusion = "confusion" in plot_value
        is_test_splits = "test splits" in plot_value

        self.training_plot_stage_label.setVisible(not is_efficiency and not is_confusion and not is_test_splits)
        self.training_plot_stage_combo.setVisible(not is_efficiency and not is_confusion and not is_test_splits)
        self.training_plot_timing_label.setVisible(is_timing)
        self.training_plot_timing_combo.setVisible(is_timing)
        self.training_plot_timing_combo.setEnabled(is_timing)
        self.training_plot_efficiency_label.setVisible(is_efficiency)
        self.training_plot_efficiency_combo.setVisible(is_efficiency)
        self.training_plot_confusion_label.setVisible(is_confusion)
        self.training_plot_confusion_spin.setVisible(is_confusion)

        if is_efficiency:
            self.training_plot_stack.setCurrentWidget(self.training_efficiency_plot_widget)
            plot = self.build_efficiency_plot(selected_runs)
            self.training_efficiency_plot_widget.set_plot(**plot)
            return

        if is_confusion:
            self.training_plot_stack.setCurrentWidget(self.training_confusion_widget)
            if len(selected_runs) != 1:
                self.training_confusion_widget.set_matrix(
                    title="Confusion Matrix",
                    labels=[],
                    matrix=[],
                    note="Select exactly one run to view a confusion matrix.",
                )
                return
            matrix_plot = self.build_confusion_matrix(selected_runs[0])
            self.training_confusion_widget.set_matrix(**matrix_plot)
            return

        self.training_plot_stack.setCurrentWidget(self.training_plot_widget)
        if is_test_splits:
            self.training_plot_widget.set_plot(**self.build_test_split_plot(selected_runs))
            return

        value_kind = "accuracy" if is_accuracy else ("loss" if is_loss else "timing")
        if len(selected_runs) >= 2:
            plot = self.build_compare_plot(selected_runs, value_kind=value_kind, timing_metric=timing_metric)
        elif len(selected_runs) == 1:
            plot = self.build_selected_run_plot(selected_runs[0], value_kind=value_kind, timing_metric=timing_metric)
        else:
            plot = {
                "title": "Run Plot",
                "x_label": "Epoch",
                "y_label": "Value",
                "series": [],
                "note": "Add one run from the left to view a plot, or add multiple runs to compare them.",
            }
        self.training_plot_widget.set_plot(**plot)

    def render_compare_runs(self) -> str:
        runs = self.selected_compare_runs()
        if not runs:
            return "No selected runs. Add one or more runs from the left list to compare."

        header = (
            f"{'Started':<22} {'Model':<12} {'Status':<14} {'Progress':<9} "
            f"{'BestEval':<10} {'FinalTest':<10} {'Eval':<6} {'Batch':<6} {'LR':<10} {'Checkpoint'}"
        )
        separator = "-" * len(header)
        lines = [header, separator]
        lines.append("")
        lines.append("Average Timing Compare:")
        for run in runs:
            args = run.get("args") if isinstance(run.get("args"), dict) else {}
            summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
            timing_summary = run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {}
            stage_totals = timing_summary.get("stage_totals") if isinstance(timing_summary.get("stage_totals"), dict) else {}
            started = str(run.get("start_time_utc", "-"))[:19]
            model = str(args.get("model", "-"))[:12]
            status = self.normalize_run_status(run)[:14]
            progress = self.format_ratio(self.infer_last_completed_epoch(run), args.get("planned_epochs_this_run"))
            best_eval = self.format_metric(self.infer_best_eval_acc(run))
            final_test = self.format_metric(summary.get("final_test_acc"))
            eval_name = self.infer_eval_name(run)[:6]
            batch_size = str(args.get("batch_size", "-"))[:6]
            lr = str(args.get("lr", "-"))[:10]
            checkpoint_name = Path(str(args.get("checkpoint_dir", "-"))).name[:20]
            lines.append(
                f"{started:<22} {model:<12} {status:<14} {progress:<9} "
                f"{best_eval:<10} {final_test:<10} {eval_name:<6} {batch_size:<6} {lr:<10} {checkpoint_name}"
            )
            train_stage = stage_totals.get("train") if isinstance(stage_totals.get("train"), dict) else {}
            test_stage = stage_totals.get("test") if isinstance(stage_totals.get("test"), dict) else {}
            train_batches = float(train_stage.get("batches", 0.0)) if isinstance(train_stage.get("batches"), (int, float)) else 0.0
            test_batches = float(test_stage.get("batches", 0.0)) if isinstance(test_stage.get("batches"), (int, float)) else 0.0
            train_avg_epoch = (
                float(train_stage.get("total_seconds", 0.0)) / max(float(self.infer_last_completed_epoch(run)), 1.0)
                if self.infer_last_completed_epoch(run) > 0 and isinstance(train_stage.get("total_seconds"), (int, float))
                else None
            )
            test_avg_epoch = (
                float(test_stage.get("total_seconds", 0.0)) / max(float(self.infer_last_completed_epoch(run)), 1.0)
                if self.infer_last_completed_epoch(run) > 0 and isinstance(test_stage.get("total_seconds"), (int, float))
                else None
            )
            train_avg_batch = (
                float(train_stage.get("pure_seconds", 0.0)) / train_batches
                if train_batches > 0 and isinstance(train_stage.get("pure_seconds"), (int, float))
                else None
            )
            test_avg_batch = (
                float(test_stage.get("pure_seconds", 0.0)) / test_batches
                if test_batches > 0 and isinstance(test_stage.get("pure_seconds"), (int, float))
                else None
            )
            lines.append(
                f"  avg_train_time_per_epoch={self.format_metric(train_avg_epoch)}s, "
                f"avg_test_time_per_epoch={self.format_metric(test_avg_epoch)}s, "
                f"avg_train_pure_per_batch={self.format_metric(train_avg_batch)}s, "
                f"avg_test_pure_per_batch={self.format_metric(test_avg_batch)}s"
            )
        split_compare = self.render_compare_test_splits(runs)
        if split_compare:
            lines.extend(["", split_compare])
        return "\n".join(lines)

    def render_compare_test_splits(self, runs: list[dict]) -> str:
        split_names: list[str] = []
        rows: list[tuple[str, dict[str, float]]] = []
        for run in runs:
            split_summary = self.test_split_summary_for_run(run)
            if split_summary is None:
                continue
            split_values: dict[str, float] = {}
            for item in split_summary.get("splits", []):
                if not isinstance(item, dict):
                    continue
                split_name = str(item.get("split", "")).strip()
                accuracy = self.safe_float(item.get("accuracy"))
                if not split_name or accuracy is None:
                    continue
                split_values[split_name] = accuracy
                if split_name not in split_names:
                    split_names.append(split_name)
            if split_values:
                rows.append((self.run_display_name(run), split_values))
        if not rows:
            return ""

        label_width = min(max(len(label) for label, _ in rows), 42)
        header = f"{'Run':<{label_width}} " + " ".join(f"{name[:12]:>12}" for name in split_names)
        lines = ["Test Split Accuracy Compare:", header, "-" * len(header)]
        for label, split_values in rows:
            cells = [self.format_metric(split_values.get(name)) for name in split_names]
            lines.append(f"{label[:label_width]:<{label_width}} " + " ".join(f"{cell:>12}" for cell in cells))
        return "\n".join(lines)

    def render_test_split_summary(self, run: dict) -> list[str]:
        split_summary = self.test_split_summary_for_run(run)
        if split_summary is None:
            return []

        lines = [
            "",
            "Test Split Summary:",
            f"- model_name: {split_summary.get('model_name', '-')}",
            f"- test_splits_root: {split_summary.get('test_splits_root', '-')}",
            f"- clean_accuracy: {self.format_metric(split_summary.get('clean_accuracy'))}",
            f"- robustness_average: {self.format_metric(split_summary.get('robustness_average'))}",
            f"- total_seconds: {self.format_metric(split_summary.get('total_seconds'))}",
            "- splits:",
        ]
        for item in split_summary.get("splits", []):
            if not isinstance(item, dict):
                continue
            lines.append(
                "  "
                f"{item.get('split', '-')}: "
                f"accuracy={self.format_metric(item.get('accuracy'))}, "
                f"avg_confidence={self.format_metric(item.get('avg_confidence'))}, "
                f"evaluated={item.get('evaluated_images', '-')}/{item.get('total_images', '-')}, "
                f"skipped={item.get('skipped_images', '-')}"
            )
        return lines

    def render_run_summary(self, run: dict) -> str:
        args = run.get("args") if isinstance(run.get("args"), dict) else {}
        dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
        model_info = run.get("model") if isinstance(run.get("model"), dict) else {}
        expected = run.get("expected") if isinstance(run.get("expected"), dict) else {}
        epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        timing_summary = run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {}
        artifacts = run.get("artifacts") if isinstance(run.get("artifacts"), dict) else {}
        best_ckpt = artifacts.get("best_checkpoint") if isinstance(artifacts.get("best_checkpoint"), dict) else {}
        last_ckpt = artifacts.get("last_checkpoint") if isinstance(artifacts.get("last_checkpoint"), dict) else {}

        planned_epochs = int(args.get("planned_epochs_this_run", 0)) if isinstance(args.get("planned_epochs_this_run"), int | float) else 0
        completed_epochs = len(epochs)
        progress_text = f"{completed_epochs}/{planned_epochs}" if planned_epochs > 0 else str(completed_epochs)

        lines = [
            f"Run ID: {run.get('run_id', 'unknown')}",
            f"Status: {self.normalize_run_status(run)}",
            f"Status Reason: {run.get('status_reason', '-')}",
            f"Started (UTC): {run.get('start_time_utc', '-')}",
            f"Ended (UTC): {run.get('end_time_utc', '-')}",
            f"Model: {args.get('model', '-')}",
            f"Device: {args.get('device', '-')}",
            f"Command: {run.get('command', '-')}",
            f"Planned Epochs / Completed Epochs: {progress_text}",
            "",
            "Dataset Summary:",
            f"- data_root: {args.get('data_root', '-')}",
            f"- eval_name: {dataset.get('eval_name', '-')}",
            f"- train_transforms_preset: {args.get('train_transforms_preset', dataset.get('train_transforms_preset', '-'))}",
            f"- mild_blur_enabled: {args.get('mild_blur_enabled', dataset.get('mild_blur_enabled', '-'))}",
            f"- mild_blur_prob: {args.get('mild_blur_prob', dataset.get('mild_blur_prob', '-'))}",
            f"- augmentation_config: {dataset.get('augmentation_config', args.get('augmentation_config', '-'))}",
            f"- num_classes: {dataset.get('num_classes', '-')}",
            f"- train_examples: {dataset.get('train_examples', '-')}",
            f"- eval_examples: {dataset.get('eval_examples', '-')}",
            f"- test_examples: {dataset.get('test_examples', '-')}",
            f"- validation_split: {dataset.get('use_validation_split', '-')}",
            f"- validation_proportion: {dataset.get('validation_proportion', '-')}",
            "",
            "Model Summary:",
            f"- total_params: {model_info.get('total_params', '-')}",
            f"- trainable_params: {model_info.get('trainable_params', '-')}",
            f"- frozen_params: {model_info.get('frozen_params', '-')}",
            f"- batch_size: {args.get('batch_size', '-')}",
            f"- lr: {args.get('lr', '-')}",
            f"- optimizer: {args.get('optimizer', '-')}",
            f"- scheduler: {args.get('scheduler', '-')}",
            f"- amp_requested: {args.get('amp', '-')}",
            f"- amp_enabled: {model_info.get('amp_enabled', args.get('amp', '-'))}",
            f"- seed: {args.get('seed', '-')}",
            f"- checkpoint_dir: {args.get('checkpoint_dir', '-')}",
            "",
            "Run Summary:",
            f"- best_eval_acc: {self.format_metric(summary.get('best_eval_acc'))}",
            f"- best_eval_epoch: {summary.get('best_eval_epoch', '-')}",
            f"- last_completed_epoch: {summary.get('last_completed_epoch', '-')}",
            f"- last_eval_acc: {self.format_metric(summary.get('last_eval_acc'))}",
            f"- last_eval_loss: {self.format_metric(summary.get('last_eval_loss'))}",
            f"- final_test_acc: {self.format_metric(summary.get('final_test_acc'))}",
            f"- final_test_loss: {self.format_metric(summary.get('final_test_loss'))}",
            "",
            f"Expected Train Batches/Epoch: {expected.get('train_batches_per_epoch', '-')}",
            f"Expected Val Batches/Epoch: {expected.get('val_batches_per_epoch', '-')}",
            f"Expected Test Batches/Epoch: {expected.get('test_batches_per_epoch', '-')}",
            f"Expected Final Test Batches: {expected.get('final_test_batches', '-')}",
            f"Error Message: {run.get('error_message', '-')}",
            "",
            "Checkpoint Files:",
            f"- best.pth path: {best_ckpt.get('path', '-')}",
            f"- best.pth state: {self.describe_artifact_state(best_ckpt)}",
            f"- best.pth saved epoch: {best_ckpt.get('saved_epoch', '-')}",
            f"- best.pth best_acc: {best_ckpt.get('saved_best_acc', '-')}",
            f"- last.pth path: {last_ckpt.get('path', '-')}",
            f"- last.pth state: {self.describe_artifact_state(last_ckpt)}",
            "",
            "Timing Summary:",
            f"- total_wall_time_seconds: {timing_summary.get('total_wall_time_seconds', '-')}",
            f"- total_pure_execution_time_seconds: {timing_summary.get('total_pure_execution_time_seconds', '-')}",
            f"- initialization_and_overhead_time_seconds: {timing_summary.get('initialization_and_overhead_time_seconds', '-')}",
        ]

        final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else None
        if final_test:
            lines.extend(
                [
                    "",
                    "Final Test:",
                    f"- loss: {final_test.get('loss', '-')}",
                    f"- acc: {final_test.get('acc', '-')}",
                    f"- timing: {final_test.get('timing', '-')}",
                ]
            )

        lines.extend(self.render_test_split_summary(run))
        lines.extend(["", *self.summarize_error_block(self.extract_analysis_block(run, stage_name="final_test"))])

        return "\n".join(lines)

    def render_stage_epochs(self, run: dict, stage_name: str) -> str:
        epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
        stage_key = stage_name.lower()
        if stage_key == "test":
            final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else None
            if final_test:
                timing = final_test.get("timing", {})
                final_text = (
                    f"Final test: loss={final_test.get('loss', '-')}, acc={final_test.get('acc', '-')}, "
                    f"total_time={timing.get('total_seconds', '-')}, "
                    f"pure_time={timing.get('pure_seconds', '-')}, "
                    f"batches={timing.get('batches', '-')}"
                )
                epoch_test_text = self.render_stage_epochs({**run, "final_test": None}, "test")
                if epoch_test_text != "No test record in this run.":
                    return final_text + "\n\nPer-epoch test:\n" + epoch_test_text
                return final_text

        if not epochs:
            return "No epoch records in this run."

        lines: list[str] = []
        for epoch_record in epochs:
            if not isinstance(epoch_record, dict):
                continue
            epoch_idx = epoch_record.get("epoch", "?")
            stage = epoch_record.get(stage_key)
            if not isinstance(stage, dict):
                continue
            timing = stage.get("timing", {})
            lr_text = self.format_metric(epoch_record.get("lr"))
            best_text = self.format_metric(epoch_record.get("best_eval_acc_after_epoch"))
            best_flag = "yes" if epoch_record.get("is_best_checkpoint") else "no"
            lines.append(
                (
                    f"Epoch {epoch_idx}: "
                    f"loss={stage.get('loss', '-')}, acc={stage.get('acc', '-')}, "
                    f"lr={lr_text}, best_eval_acc={best_text}, saved_best={best_flag}, "
                    f"total_time={timing.get('total_seconds', '-')}, "
                    f"pure_time={timing.get('pure_seconds', '-')}, "
                    f"batches={timing.get('batches', '-')}"
                )
            )

        if not lines:
            return f"No {stage_key} records in this run."
        return "\n".join(lines)

    def refresh_training_log_view(self) -> None:
        if not self.training_log_runs:
            return
        selected_runs = self.selected_compare_runs()
        if selected_runs:
            selected_run = self.current_selected_compare_run() or selected_runs[0]
            status_text = f"Selected for plot: {len(selected_runs)} run(s)"
        else:
            selected_run = self.current_available_run()
            if selected_run is None:
                self.training_log_status_label.setText("Choose a run on the left, then press + Add to plot it.")
                self.training_log_text.setPlainText("No run selected for details.")
                self.refresh_training_log_plot()
                return
            status_text = f"Previewing available run: {selected_run.get('_log_path', '-')}"
        self.training_log_status_label.setText(status_text)

        selected_view = self.training_log_stage_combo.currentText().strip().lower()
        if len(selected_runs) >= 2 and selected_view == "summary":
            self.training_log_text.setPlainText(self.render_compare_runs())
            self.refresh_training_log_plot()
            return
        if len(selected_runs) >= 2 and selected_view in {"train", "val", "test"}:
            blocks: list[str] = []
            for run in selected_runs:
                blocks.append(self.run_display_name(run))
                blocks.append(self.render_stage_epochs(run, selected_view))
                blocks.append("")
            self.training_log_text.setPlainText("\n".join(blocks).strip())
            self.refresh_training_log_plot()
            return
        if selected_view == "summary":
            self.training_log_text.setPlainText(self.render_run_summary(selected_run))
            self.refresh_training_log_plot()
            return
        stage_text = self.render_stage_epochs(selected_run, selected_view)
        analysis = self.extract_analysis_block(selected_run, stage_name=selected_view)
        self.training_log_text.setPlainText(stage_text + "\n\n" + "\n".join(self.summarize_error_block(analysis)))
        self.refresh_training_log_plot()

    def stop_training(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        self.training_stop_requested = True
        if self.active_job_origin == "queue":
            self.global_queue_stop_requested = True
        if self._stop_request_path is None:
            self._stop_request_path = self.stop_request_path_for()
        self._stop_request_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_request_path.write_text("stop requested\n", encoding="utf-8")
        self.append_output("\nGraceful stop requested. Waiting for the current step to finish...\n")
        self.status_label.setText("Stopping")
        self.progress_label.setText("Graceful stop requested. Training will stop after the current batch.")
        self.stop_button.setEnabled(False)
        self.queue_stop_button.setEnabled(False)

    def handle_output(self) -> None:
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.append_stream_output(data)

    def handle_data_output(self) -> None:
        data = self.data_process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.append_data_stream_output(data)

    def on_process_started(self) -> None:
        self.set_running_state(True)
        self.status_label.setText("Running")
        if self.active_job_origin == "queue" and self.active_queue_job_id is not None:
            self.complete_global_queue_job(self.active_queue_job_id, "running")
            self.progress_label.setText("Queued process started. Waiting for training progress...")
        else:
            self.progress_label.setText("Process started. Waiting for training progress...")

    def on_data_process_started(self) -> None:
        self.set_data_running_state(True)
        self.data_status_label.setText("Running")
        self.data_state_value_label.setText("Running")
        self.data_progress_label.setText("Data task started...")

    def on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.set_running_state(False)
        self.clear_stop_request_file()
        self.refresh_checkpoint_output_options(preserve_text=self.checkpoint_output_name())
        self.refresh_training_log_runs()
        status_text = "NormalExit" if exit_status == QProcess.NormalExit else "CrashExit"
        self.status_label.setText(f"Finished ({exit_code})")
        if self._stream_buffer.strip():
            self._handle_stream_line(self._stream_buffer, "\n")
        self._stream_buffer = ""
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.progress_label.setText("Training finished successfully.")
            if self.progress_bar.maximum() > 0:
                self.progress_bar.setValue(self.progress_bar.maximum())
        else:
            self.progress_label.setText(f"Training stopped with exit code {exit_code} ({status_text}).")
        self.append_output(f"\nProcess finished with exit code {exit_code} ({status_text}).\n")

        active_job_id = self.active_queue_job_id
        if self.active_job_origin == "queue" and active_job_id is not None:
            if self.training_stop_requested:
                final_status = "cancelled"
            elif exit_code == 0 and exit_status == QProcess.NormalExit:
                final_status = "completed"
            else:
                final_status = "failed"
            current_config = self.active_job_config_snapshot if isinstance(self.active_job_config_snapshot, dict) else {}
            checkpoint_dir = Path(str(current_config.get("checkpoint_dir", self.selected_checkpoint_dir()))).expanduser().resolve()
            artifacts = {
                "checkpoint_dir": str(checkpoint_dir),
                "last_checkpoint_path": str((checkpoint_dir / "last.pth").resolve()) if (checkpoint_dir / "last.pth").exists() else None,
            }
            resolved_best, run_log_path, _ = self.resolve_best_checkpoint_from_training_job(
                {"artifacts": artifacts, "config_snapshot": current_config}
            )
            artifacts["best_checkpoint_path"] = resolved_best
            if run_log_path is not None:
                artifacts["run_log_path"] = run_log_path
            self.complete_global_queue_job(active_job_id, final_status, artifacts=artifacts)
            should_continue = not self.global_queue_stop_requested
            self.clear_active_global_job()
            if should_continue and self.start_next_global_queue_job():
                return
            self.global_queue_running = False
            self.global_queue_stop_requested = False
            self.refresh_global_queue_view()
        else:
            self.clear_active_global_job()

    def on_data_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.set_data_running_state(False)
        status_text = "NormalExit" if exit_status == QProcess.NormalExit else "CrashExit"
        self.data_progress_bar.setRange(0, 100)
        if self._data_stream_buffer.strip():
            self._handle_data_stream_line(self._data_stream_buffer, "\n")
        self._data_stream_buffer = ""
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            self.data_status_label.setText("Finished")
            self.data_state_value_label.setText("Completed")
            self.data_progress_label.setText("Dataset task finished successfully.")
            self.data_last_result_value_label.setText("Last run completed successfully.")
            self.data_progress_bar.setValue(100)
        else:
            self.data_status_label.setText(f"Finished ({exit_code})")
            self.data_state_value_label.setText("Failed")
            self.data_progress_label.setText(f"Dataset task stopped with exit code {exit_code} ({status_text}).")
            self.data_last_result_value_label.setText(
                f"Last run stopped with exit code {exit_code} ({status_text})."
            )
            self.data_progress_bar.setValue(0)
        self.append_data_output(f"\nProcess finished with exit code {exit_code} ({status_text}).\n")

    def on_process_error(self, error: QProcess.ProcessError) -> None:
        self.set_running_state(False)
        self.clear_stop_request_file()
        self.refresh_training_log_runs()
        self.status_label.setText("Error")
        self.progress_label.setText(f"Process error: {error}")
        self.append_output(f"\nProcess error: {error}\n")
        if self.active_job_origin == "queue" and self.active_queue_job_id is not None:
            active_job_id = self.active_queue_job_id
            self.complete_global_queue_job(active_job_id, "failed", error_message=str(error))
            self.clear_active_global_job()
            if not self.global_queue_stop_requested and self.start_next_global_queue_job():
                return
            self.global_queue_running = False
            self.global_queue_stop_requested = False
            self.refresh_global_queue_view(select_job_id=active_job_id)
        else:
            self.clear_active_global_job()

    def on_data_process_error(self, error: QProcess.ProcessError) -> None:
        self.set_data_running_state(False)
        self.data_status_label.setText("Error")
        self.data_state_value_label.setText("Error")
        self.data_progress_bar.setRange(0, 100)
        self.data_progress_bar.setValue(0)
        self.data_progress_label.setText(f"Process error: {error}")
        self.data_last_result_value_label.setText(f"Process error: {error}")
        self.append_data_output(f"\nProcess error: {error}\n")

    def choose_predict_checkpoint(self) -> None:
        start_dir = self._resolve_dialog_dir(self.predict_checkpoint_edit.text().strip(), DEFAULT_CHECKPOINT_DIR)
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prediction Checkpoint",
            str(start_dir),
            "PyTorch Checkpoints (*.pth *.pt);;All Files (*.*)",
        )
        if selected_path:
            self.predict_checkpoint_edit.setText(selected_path)
            self.update_predict_detected_model()

    def choose_test_split_checkpoint(self) -> None:
        start_dir = self._resolve_dialog_dir(self.test_split_checkpoint_edit.text().strip(), DEFAULT_CHECKPOINT_DIR)
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Evaluation Checkpoint",
            str(start_dir),
            "PyTorch Checkpoints (*.pth *.pt);;All Files (*.*)",
        )
        if selected_path:
            self.test_split_checkpoint_edit.setText(selected_path)
            self.update_test_split_detected_model()

    def choose_test_splits_root(self) -> None:
        start_dir = self._resolve_dialog_dir(self.test_split_root_edit.text().strip(), DEFAULT_TEST_SPLITS_ROOT)
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Test Splits Root", str(start_dir))
        if selected_dir:
            self.test_split_root_edit.setText(selected_dir)

    def choose_predict_images(self) -> None:
        selected_paths = self.select_multiple_files(
            title="Select Images to Predict",
            start_dir=DEFAULT_DATA_ROOT / "images",
            file_filter="Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if selected_paths:
            self.predict_image_paths = [Path(path) for path in selected_paths]
            self.predict_results = []
            self.current_predict_index = 0 if self.predict_image_paths else -1
            self.predict_compact_built = False
            self.predict_compact_loading = False
            self.predict_compact_pending_indices = []
            self.predict_browser_render_key = None
            self.clear_predict_visual_caches()
            self.predict_progress_bar.setValue(0)
            self.refresh_predict_page(refresh_compact=True)

    def choose_predict_folders(self) -> None:
        selected_dirs = self.select_multiple_directories(
            title="Select Folder(s) to Predict",
            start_dir=DEFAULT_DATA_ROOT / "images",
        )
        if not selected_dirs:
            return

        image_paths = []
        for folder_path in selected_dirs:
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                image_paths.extend(folder_path.glob(pattern))

        self.predict_image_paths = sorted(path.resolve() for path in image_paths if path.is_file())
        self.predict_results = []
        self.current_predict_index = 0 if self.predict_image_paths else -1
        self.predict_compact_built = False
        self.predict_compact_loading = False
        self.predict_compact_pending_indices = []
        self.predict_browser_render_key = None
        self.clear_predict_visual_caches()
        self.predict_progress_bar.setValue(0)
        if not self.predict_image_paths:
            self.predict_status_label.setText("No supported images found in the selected folder(s).")
        else:
            self.predict_status_label.setText(
                f"Loaded {len(self.predict_image_paths)} image(s) from {len(selected_dirs)} folder(s)."
            )
        self.refresh_predict_page(refresh_compact=True)

    def select_multiple_files(self, title: str, start_dir: Path, file_filter: str) -> list[str]:
        dialog = QFileDialog(self, title, str(start_dir))
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter(file_filter)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        list_view = dialog.findChild(QListView, "listView")
        if list_view is not None:
            list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        tree_view = dialog.findChild(QTreeView)
        if tree_view is not None:
            tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if not dialog.exec():
            return []

        return dialog.selectedFiles()

    def select_multiple_directories(self, title: str, start_dir: Path) -> list[Path]:
        dialog = QFileDialog(self, title, str(start_dir))
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)

        list_view = dialog.findChild(QListView, "listView")
        if list_view is not None:
            list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        tree_view = dialog.findChild(QTreeView)
        if tree_view is not None:
            tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if not dialog.exec():
            return []

        return [Path(path) for path in dialog.selectedFiles()]

    def run_predictions(self) -> None:
        try:
            config = self.collect_predict_config_snapshot()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Predict Config", str(exc))
            self.predict_status_label.setText("Prediction blocked: invalid configuration.")
            self.predict_progress_bar.setRange(0, 100)
            self.predict_progress_bar.setValue(0)
            return
        if not self.start_predictions_with_config(config, origin="manual"):
            QMessageBox.information(self, "Job Already Running", "Another training, predicting, or test-split job is already running.")

    def refresh_predict_page(self, refresh_compact: bool = False) -> None:
        if not self.predict_results and self.predict_image_paths and not (0 <= self.current_predict_index < len(self.predict_image_paths)):
            self.current_predict_index = 0

        if self.predict_image_paths:
            if len(self.predict_image_paths) == 1:
                self.predict_selected_label.setText(str(self.predict_image_paths[0]))
            else:
                self.predict_selected_label.setText(
                    f"{len(self.predict_image_paths)} images selected.\nFirst: {self.predict_image_paths[0]}"
                )
        else:
            self.predict_selected_label.setText("No images selected.")

        has_results = bool(self.predict_results) and 0 <= self.current_predict_index < len(self.predict_results)
        total_items = len(self.predict_results) if self.predict_results else len(self.predict_image_paths)
        has_selection = total_items > 0 and 0 <= self.current_predict_index < total_items
        self.predict_prev_button.setEnabled(has_selection and self.current_predict_index > 0)
        self.predict_next_button.setEnabled(has_selection and self.current_predict_index < total_items - 1)
        self.predict_gradcam_button.setEnabled(has_results and isinstance(self.predict_results[self.current_predict_index], dict) and self.is_predict_compare_result(self.predict_results[self.current_predict_index]) if has_results else False)
        self.predict_page_label.setText(
            f"{self.current_predict_index + 1 if has_selection else 0} / {total_items}"
        )
        self.refresh_predict_compact_view()
        self.predict_display_stack.setCurrentIndex(0)
        self.update_predict_gradcam_ui_state()

        if not has_results:
            if has_selection and self.predict_image_paths:
                image_path = self.predict_image_paths[self.current_predict_index]
                self.set_predict_preview_pixmap(self.predict_image_label, image_path)
                self.predict_result_label.setText(
                    f"Image: {image_path}\n"
                    f"Ready to predict this image.\n"
                    f"Selected {self.current_predict_index + 1} of {len(self.predict_image_paths)}."
                )
            else:
                self.predict_image_label.setPixmap(QPixmap())
                self.predict_image_label.setText("Select images and click Predict.")
                self.predict_result_label.setText("Prediction result will appear here.")
            return

        result = self.predict_results[self.current_predict_index]
        compare_active = (
            self.predict_compare_checkbox.isChecked()
            and isinstance(result, dict)
            and isinstance(result.get("comparisons"), dict)
            and len(result.get("comparisons", {})) >= 2
        )
        if compare_active:
            self.refresh_predict_compare_page(result)
            return

        image_path = Path(str(result["image_path"]))
        self.set_predict_preview_pixmap(self.predict_image_label, image_path)

        actual_label = result.get("actual_label")
        is_correct = result.get("is_correct")
        if is_correct is True:
            correctness_text = "Yes"
        elif is_correct is False:
            correctness_text = "No"
        else:
            correctness_text = "Unknown"

        actual_text = actual_label if actual_label is not None else "Unknown (folder name not recognized as a class)"
        self.predict_result_label.setText(
            f"Image: {image_path}\n"
            f"Predicted: {result['predicted_class']}\n"
            f"Confidence: {float(result['confidence']):.4f}\n"
            f"Ground Truth: {actual_text}\n"
            f"Predict Correct: {correctness_text}"
        )
        self.update_predict_gradcam_ui_state()

    def is_predict_compare_result(self, result: dict) -> bool:
        comparisons = result.get("comparisons")
        return isinstance(comparisons, dict) and len(comparisons) > 1

    def refresh_predict_compare_page(self, result: dict) -> None:
        image_path = Path(str(result["image_path"]))
        comparisons = result.get("comparisons") if isinstance(result.get("comparisons"), dict) else {}
        self.predict_display_stack.setCurrentIndex(1)
        self.set_predict_preview_pixmap(self.predict_compare_shared_image_label, image_path)
        actual_label = result.get("actual_label")
        actual_text = actual_label if actual_label is not None else "Unknown"
        self.predict_compare_context_label.setText(
            f"Image: {image_path}\n"
            f"Ground Truth: {actual_text}\n"
            f"Compared Items: {len(comparisons)}"
        )
        self.predict_compare_context_summary_label.setText(
            f"{image_path.name} | GT: {actual_text} | Compared: {len(comparisons)}"
        )
        self.predict_compare_context_summary_label.setToolTip(str(image_path))
        self.populate_predict_compare_cards(image_path, comparisons)
        self.update_predict_gradcam_ui_state()

    def current_predict_gradcam_request(self) -> dict[str, object] | None:
        if not self.predict_results or not (0 <= self.current_predict_index < len(self.predict_results)):
            return None
        current_result = self.predict_results[self.current_predict_index]
        if not isinstance(current_result, dict) or not self.is_predict_compare_result(current_result):
            return None
        comparisons = current_result.get("comparisons") if isinstance(current_result.get("comparisons"), dict) else {}
        return self.build_predict_gradcam_request(Path(str(current_result["image_path"])), comparisons)

    def update_predict_gradcam_ui_state(self) -> None:
        request = self.current_predict_gradcam_request()
        if request is None:
            self.predict_gradcam_button.setText("Generate / Show Grad-CAM")
            self.predict_gradcam_button.setToolTip("Grad-CAM is available for compare results after prediction.")
            return
        missing_specs = request.get("missing_specs")
        request_key = request.get("request_key")
        if isinstance(request_key, tuple) and self.predict_gradcam_request_key == request_key:
            self.predict_gradcam_button.setText("Generating Grad-CAM...")
            self.predict_gradcam_button.setToolTip("Grad-CAM overlays are currently being generated for this compare image.")
        elif isinstance(missing_specs, list) and missing_specs:
            self.predict_gradcam_button.setText("Generate Grad-CAM")
            self.predict_gradcam_button.setToolTip("Generate missing Grad-CAM overlays for the current compare image.")
        else:
            self.predict_gradcam_button.setText("Show Grad-CAM")
            self.predict_gradcam_button.setToolTip("All Grad-CAM overlays are cached for the current compare image.")

    def on_predict_compare_context_toggled(self, checked: bool) -> None:
        self.predict_compare_context_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.predict_compare_context_content.setVisible(checked)

    def set_predict_preview_pixmap(self, target_label: QLabel, image_path: Path, *, source_pixmap: QPixmap | None = None) -> None:
        if source_pixmap is not None and not source_pixmap.isNull():
            pixmap = source_pixmap.scaled(
                target_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            cache_key = (str(image_path), max(target_label.width(), 1), max(target_label.height(), 1))
            pixmap = self.predict_cache_get(self.predict_display_cache, cache_key)
            if pixmap is None:
                loaded = QPixmap(str(image_path))
                if not loaded.isNull():
                    pixmap = loaded.scaled(
                        target_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self.predict_cache_put(self.predict_display_cache, cache_key, pixmap, PREDICT_DISPLAY_CACHE_LIMIT)
                else:
                    pixmap = QPixmap()
        if pixmap.isNull():
            target_label.setPixmap(QPixmap())
            target_label.setText(f"Could not load image:\n{image_path}")
        else:
            target_label.setText("")
            target_label.setPixmap(pixmap)

    def predict_cache_get(self, cache: OrderedDict, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def predict_cache_put(self, cache: OrderedDict, key, value, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            oldest_key, _ = cache.popitem(last=False)
            if cache is self.predict_gradcam_cache:
                self.predict_gradcam_diagnostics.pop(oldest_key, None)

    def clear_predict_visual_caches(self, *, keep_gradcam: bool = False) -> None:
        self.predict_thumbnail_cache.clear()
        self.predict_display_cache.clear()
        self.predict_compare_display_cache.clear()
        if not keep_gradcam:
            self.predict_gradcam_cache.clear()
            self.predict_gradcam_diagnostics.clear()

    def is_predict_overlay_meaningful(self, image_path: Path, overlay_pixmap: QPixmap | None) -> bool:
        if overlay_pixmap is None or overlay_pixmap.isNull():
            return False
        original_pixmap = QPixmap(str(image_path))
        if original_pixmap.isNull():
            return True
        sample_size = QSize(48, 48)
        overlay_image = overlay_pixmap.scaled(sample_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation).toImage()
        original_image = original_pixmap.scaled(sample_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation).toImage()
        if overlay_image.size() != original_image.size():
            return True
        total_delta = 0
        pixel_count = max(overlay_image.width() * overlay_image.height(), 1)
        for y in range(overlay_image.height()):
            for x in range(overlay_image.width()):
                overlay_color = overlay_image.pixelColor(x, y)
                original_color = original_image.pixelColor(x, y)
                total_delta += abs(overlay_color.red() - original_color.red())
                total_delta += abs(overlay_color.green() - original_color.green())
                total_delta += abs(overlay_color.blue() - original_color.blue())
        average_delta = total_delta / float(pixel_count * 3)
        return average_delta >= 6.0

    def clear_predict_compare_cards(self) -> None:
        while self.predict_compare_cards_layout.count():
            item = self.predict_compare_cards_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    child_widget = child_item.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    def populate_predict_compare_cards(self, image_path: Path, comparisons: dict) -> None:
        self.clear_predict_compare_cards()
        request = self.build_predict_gradcam_request(image_path, comparisons)
        overlay_lookup: dict[str, QPixmap] = {}
        diagnostic_lookup: dict[str, str] = {}
        loading_labels: set[str] = set()
        if request is not None:
            model_specs = request.get("model_specs")
            if isinstance(model_specs, list):
                for display_label, _model_name, checkpoint_path in model_specs:
                    cache_key = (
                        str(image_path.resolve()),
                        str(display_label),
                        str(Path(checkpoint_path)),
                        self.predict_image_size_spin.value(),
                        self.predict_device_combo.currentText(),
                    )
                    overlay = self.predict_cache_get(self.predict_gradcam_cache, cache_key)
                    if overlay is not None:
                        overlay_lookup[str(display_label)] = overlay
                    diagnostic_reason = self.predict_cache_get(self.predict_gradcam_diagnostics, cache_key)
                    if isinstance(diagnostic_reason, str) and diagnostic_reason.strip():
                        diagnostic_lookup[str(display_label)] = diagnostic_reason.strip()
            request_key = request.get("request_key")
            missing_specs = request.get("missing_specs")
            if (
                isinstance(request_key, tuple)
                and request_key == self.predict_gradcam_request_key
                and isinstance(missing_specs, list)
            ):
                for display_label, _model_name, _checkpoint_path in missing_specs:
                    loading_labels.add(str(display_label))

        ordered_labels = self.selected_predict_models()
        for index, display_label in enumerate(ordered_labels):
            model_result = comparisons.get(display_label)
            if not isinstance(model_result, dict):
                continue
            card = QFrame()
            card.setObjectName("PredictCompareCard")
            card.setToolTip(str(model_result.get("checkpoint_path", "")))
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(8)

            title_label = QLabel(display_label)
            title_label.setProperty("sectionTitle", True)
            title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            card_layout.addWidget(title_label)

            model_name = str(model_result.get("model_name", display_label)).strip() or display_label
            checkpoint_hint = Path(str(model_result.get("checkpoint_path", ""))).name
            header_label = QLabel(f"Model: {model_name}\nCheckpoint: {checkpoint_hint}")
            header_label.setWordWrap(True)
            header_label.setProperty("muted", True)
            card_layout.addWidget(header_label)

            divider = QFrame()
            divider.setFrameShape(QFrame.HLine)
            divider.setFrameShadow(QFrame.Plain)
            divider.setProperty("divider", True)
            divider.setMaximumHeight(1)
            card_layout.addWidget(divider)

            preview_label = QLabel("Preview unavailable.")
            preview_label.setObjectName("PredictPreviewCard")
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumHeight(170)
            preview_label.setMaximumHeight(210)
            preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            overlay_pixmap = overlay_lookup.get(display_label)
            overlay_is_meaningful = self.is_predict_overlay_meaningful(image_path, overlay_pixmap)
            self.set_predict_preview_pixmap(preview_label, image_path, source_pixmap=overlay_pixmap)
            card_layout.addWidget(preview_label, stretch=1)

            predicted = str(model_result.get("predicted_class", "-"))
            confidence = float(model_result.get("confidence", 0.0))
            is_correct = model_result.get("is_correct")
            status = "Correct" if is_correct is True else ("Wrong" if is_correct is False else "Unknown")
            if display_label in loading_labels:
                preview_mode = "Grad-CAM loading..."
                preview_note = "Generating overlay for this model"
            elif overlay_pixmap is None:
                diagnostic_reason = diagnostic_lookup.get(display_label)
                if diagnostic_reason:
                    preview_mode = "Grad-CAM unavailable"
                    preview_note = diagnostic_reason
                else:
                    preview_mode = "Original preview"
                    preview_note = "Grad-CAM not generated yet"
            elif overlay_is_meaningful:
                preview_mode = "Grad-CAM overlay"
                preview_note = None
            else:
                preview_mode = "Fallback to original preview"
                preview_note = diagnostic_lookup.get(display_label, "Overlay too similar to original")
            details_text = (
                f"Predicted: {predicted}\n"
                f"Confidence: {confidence:.4f}\n"
                f"Correctness: {status}\n"
                f"Display: {preview_mode}"
            )
            if preview_note:
                details_text += f"\nReason: {preview_note}"
            details_label = QLabel(
                details_text
            )
            details_label.setWordWrap(True)
            details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            details_label.setProperty("detailText", True)
            card_layout.addWidget(details_label)

            row = index // 2
            column = index % 2
            self.predict_compare_cards_layout.addWidget(card, row, column)

        if self.predict_compare_cards_layout.count() == 0:
            empty_label = QLabel("No compare details are available for this result.")
            empty_label.setWordWrap(True)
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setProperty("muted", True)
            self.predict_compare_cards_layout.addWidget(empty_label, 0, 0, 1, 2)

        self.predict_compare_cards_layout.setRowStretch((len(ordered_labels) + 1) // 2, 1)

    def build_predict_compare_pixmap(self, image_path: Path, comparisons: dict) -> QPixmap:
        request = self.build_predict_gradcam_request(image_path, comparisons)
        if request is None:
            return QPixmap()
        request_key = request["request_key"]
        assert isinstance(request_key, tuple)
        cached = self.predict_cache_get(self.predict_compare_display_cache, request_key)
        if cached is not None:
            return cached

        original = QPixmap(str(image_path))
        if original.isNull():
            return QPixmap()
        columns: list[tuple[str, QPixmap]] = [("Original", original)]
        missing_specs = request["missing_specs"]
        assert isinstance(missing_specs, list)
        model_specs = request["model_specs"]
        assert isinstance(model_specs, list)

        for display_label, model_name, checkpoint_path in model_specs:
            cache_key = (
                str(image_path.resolve()),
                display_label,
                str(checkpoint_path),
                self.predict_image_size_spin.value(),
                self.predict_device_combo.currentText(),
            )
            overlay = self.predict_cache_get(self.predict_gradcam_cache, cache_key)
            if overlay is None:
                columns.append((f"{display_label} (Preview)", original))
            else:
                columns.append((display_label, overlay))

        compare_pixmap = self.compose_labeled_pixmap(columns)
        if not missing_specs:
            self.predict_cache_put(self.predict_compare_display_cache, request_key, compare_pixmap, PREDICT_COMPARE_DISPLAY_CACHE_LIMIT)
        return compare_pixmap

    def build_predict_gradcam_request(self, image_path: Path, comparisons: dict) -> dict[str, object] | None:
        if not isinstance(comparisons, dict):
            return None
        image_path = image_path.resolve()
        image_size = self.predict_image_size_spin.value()
        device = self.predict_device_combo.currentText()
        model_specs: list[tuple[str, str, Path]] = []
        missing_specs: list[tuple[str, str, Path]] = []
        cache_keys: list[tuple[str, str, str, int, str]] = []

        for display_label in self.selected_predict_models():
            model_result = comparisons.get(display_label)
            if not isinstance(model_result, dict):
                continue
            model_name = str(model_result.get("model_name", display_label)).strip()
            if not model_name:
                continue
            checkpoint_raw = model_result.get("checkpoint_path", "")
            checkpoint_path = Path(str(checkpoint_raw)).expanduser().resolve()
            model_specs.append((display_label, model_name, checkpoint_path))
            cache_key = (str(image_path), display_label, str(checkpoint_path), image_size, device)
            cache_keys.append(cache_key)
            if self.predict_cache_get(self.predict_gradcam_cache, cache_key) is None:
                missing_specs.append((display_label, model_name, checkpoint_path))

        request_key: tuple[object, ...] = (
            str(image_path),
            tuple((display_label, model_name, str(checkpoint_path)) for display_label, model_name, checkpoint_path in model_specs),
            image_size,
            device,
        )
        return {
            "image_path": image_path,
            "image_size": image_size,
            "device": device,
            "model_specs": model_specs,
            "missing_specs": missing_specs,
            "cache_keys": cache_keys,
            "request_key": request_key,
        }

    def start_predict_gradcam_generation(self, request: dict[str, object]) -> None:
        request_key = request.get("request_key")
        if not isinstance(request_key, tuple):
            return
        missing_specs = request.get("missing_specs")
        if not isinstance(missing_specs, list) or not missing_specs:
            return
        if self.predict_gradcam_request_key == request_key:
            return
        if self.predict_gradcam_thread is not None:
            self.predict_gradcam_pending_request = request
            return

        image_path = request.get("image_path")
        image_size = request.get("image_size")
        device = request.get("device")
        if not isinstance(image_path, Path) or not isinstance(image_size, int) or not isinstance(device, str):
            return

        model_specs: list[tuple[str, str, Path]] = []
        for display_label, model_name, checkpoint_path in missing_specs:
            model_specs.append((str(display_label), str(model_name), Path(checkpoint_path)))

        self.predict_gradcam_request_key = request_key
        self.predict_gradcam_thread = QThread(self)
        self.predict_gradcam_worker = GradCamComparisonWorker(
            image_path=image_path,
            model_specs=model_specs,
            image_size=image_size,
            device=device,
            request_key=request_key,
        )
        self.predict_gradcam_worker.moveToThread(self.predict_gradcam_thread)
        self.predict_gradcam_thread.started.connect(self.predict_gradcam_worker.run)
        self.predict_gradcam_worker.finished.connect(self.on_predict_gradcam_finished)
        self.predict_gradcam_worker.failed.connect(self.on_predict_gradcam_failed)
        self.predict_gradcam_worker.finished.connect(self.predict_gradcam_thread.quit)
        self.predict_gradcam_worker.failed.connect(self.predict_gradcam_thread.quit)
        self.predict_gradcam_worker.finished.connect(self.predict_gradcam_worker.deleteLater)
        self.predict_gradcam_worker.failed.connect(self.predict_gradcam_worker.deleteLater)
        self.predict_gradcam_thread.finished.connect(self.predict_gradcam_thread.deleteLater)
        self.predict_gradcam_thread.start()

    def show_predict_gradcam_for_current_page(self) -> None:
        if not self.predict_results or not (0 <= self.current_predict_index < len(self.predict_results)):
            return
        current_result = self.predict_results[self.current_predict_index]
        if not isinstance(current_result, dict) or not self.is_predict_compare_result(current_result):
            return
        request = self.build_predict_gradcam_request(
            Path(str(current_result["image_path"])),
            current_result.get("comparisons") if isinstance(current_result.get("comparisons"), dict) else {},
        )
        if request is None:
            return
        missing_specs = request.get("missing_specs")
        if isinstance(missing_specs, list) and missing_specs:
            self.predict_status_label.setText("Generating Grad-CAM for current page...")
            self.start_predict_gradcam_generation(request)
            self.refresh_predict_compare_page(current_result)
        else:
            self.refresh_predict_compare_page(current_result)

    def on_predict_gradcam_finished(self, request_key: object, overlays: object) -> None:
        if isinstance(overlays, list):
            for item in overlays:
                if not isinstance(item, tuple) or len(item) not in {2, 3}:
                    continue
                cache_key, image_data = item[0], item[1]
                diagnostic_reason = item[2] if len(item) == 3 else None
                if not isinstance(cache_key, tuple):
                    continue
                if isinstance(image_data, bytes):
                    pixmap = QPixmap()
                    pixmap.loadFromData(image_data, "PNG")
                    if not pixmap.isNull():
                        self.predict_cache_put(self.predict_gradcam_cache, cache_key, pixmap, PREDICT_GRADCAM_CACHE_LIMIT)
                if isinstance(diagnostic_reason, str) and diagnostic_reason.strip():
                    self.predict_cache_put(self.predict_gradcam_diagnostics, cache_key, diagnostic_reason.strip(), PREDICT_GRADCAM_CACHE_LIMIT)
                else:
                    self.predict_gradcam_diagnostics.pop(cache_key, None)
        if isinstance(request_key, tuple):
            self.predict_compare_display_cache.pop(request_key, None)
        self.finish_predict_gradcam_request(request_key)

        if self.predict_results and 0 <= self.current_predict_index < len(self.predict_results):
            current_result = self.predict_results[self.current_predict_index]
            if self.is_predict_compare_result(current_result):
                current_request = self.build_predict_gradcam_request(
                    Path(str(current_result["image_path"])),
                    current_result.get("comparisons") if isinstance(current_result.get("comparisons"), dict) else {},
                )
                if current_request is not None and current_request.get("request_key") == request_key:
                    self.refresh_predict_compare_page(current_result)

    def on_predict_gradcam_failed(self, request_key: object, error_message: str) -> None:
        self.finish_predict_gradcam_request(request_key)
        if self.predict_results and 0 <= self.current_predict_index < len(self.predict_results):
            self.predict_status_label.setText(f"Grad-CAM preview fallback: {error_message}")

    def finish_predict_gradcam_request(self, request_key: object) -> None:
        self.predict_gradcam_worker = None
        self.predict_gradcam_thread = None
        if self.predict_gradcam_request_key == request_key:
            self.predict_gradcam_request_key = None
        if self.predict_gradcam_pending_request is not None:
            pending_request = self.predict_gradcam_pending_request
            self.predict_gradcam_pending_request = None
            self.start_predict_gradcam_generation(pending_request)
        else:
            self.update_predict_gradcam_ui_state()

    def compose_labeled_pixmap(self, columns: list[tuple[str, QPixmap]]) -> QPixmap:
        valid_columns = [(label, pixmap) for label, pixmap in columns if not pixmap.isNull()]
        if not valid_columns:
            return QPixmap()
        thumb_width = 220
        thumb_height = 220
        header_height = 28
        spacing = 16
        total_width = len(valid_columns) * thumb_width + max(len(valid_columns) - 1, 0) * spacing
        total_height = header_height + thumb_height
        canvas = QPixmap(total_width, total_height)
        canvas.fill(QColor("#11151a"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("#eef4fb"))
        for index, (label, pixmap) in enumerate(valid_columns):
            x = index * (thumb_width + spacing)
            painter.drawText(QRectF(x, 0, thumb_width, header_height), Qt.AlignCenter, label)
            target = pixmap.scaled(QSize(thumb_width, thumb_height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            draw_x = x + (thumb_width - target.width()) / 2
            draw_y = header_height + (thumb_height - target.height()) / 2
            painter.drawPixmap(int(draw_x), int(draw_y), target)
        painter.end()
        return canvas

    def refresh_predict_compact_view(self) -> None:
        browser_mode = str(self.predict_browser_mode_combo.currentData() or "thumbnails")
        render_key = (len(self.predict_image_paths), len(self.predict_results), browser_mode)
        if self.predict_compact_built and self.predict_browser_render_key == render_key and self.predict_compact_list.count() == len(self.predict_image_paths):
            if 0 <= self.current_predict_index < self.predict_compact_list.count():
                self.predict_compact_list.setCurrentRow(self.current_predict_index)
            self.schedule_predict_visible_thumbnail_update()
            return

        self.predict_compact_list.clear()
        if not self.predict_image_paths:
            self.predict_compact_built = False
            self.predict_compact_loading = False
            self.predict_compact_pending_indices = []
            self.predict_browser_render_key = None
            return

        for index, image_path in enumerate(self.predict_image_paths):
            item = QListWidgetItem()
            icon = self.predict_thumbnail_cache.get(str(image_path))
            if icon is not None:
                item.setIcon(icon)

            result = self.predict_results[index] if index < len(self.predict_results) else None
            if isinstance(result, dict) and self.is_predict_compare_result(result):
                actual_label = result.get("actual_label")
                actual_text = actual_label if actual_label is not None else "Unknown"
                comparisons = result.get("comparisons") if isinstance(result.get("comparisons"), dict) else {}
                if browser_mode == "thumbnails":
                    item.setText(f"{image_path.name}\nGT: {actual_text}\nCompared {len(comparisons)}")
                else:
                    item.setText(f"{image_path.name} | GT: {actual_text} | Compared {len(comparisons)}")
            elif isinstance(result, dict):
                is_correct = result.get("is_correct")
                if is_correct is True:
                    correctness_text = "Yes"
                elif is_correct is False:
                    correctness_text = "No"
                else:
                    correctness_text = "Unknown"
                actual_label = result.get("actual_label")
                actual_text = actual_label if actual_label is not None else "Unknown"
                if browser_mode == "thumbnails":
                    item.setText(
                        f"{image_path.name}\n"
                        f"{result['predicted_class']} | {float(result['confidence']):.2%}\n"
                        f"GT: {actual_text}"
                    )
                else:
                    item.setText(
                        f"{image_path.name} | Pred: {result['predicted_class']} | "
                        f"True: {actual_text} | {float(result['confidence']):.2%} | Correct: {correctness_text}"
                    )
            else:
                item.setText(
                    f"{image_path.name}\n{image_path.parent.name}" if browser_mode == "thumbnails"
                    else str(image_path)
                )
            item.setTextAlignment(Qt.AlignHCenter if browser_mode == "thumbnails" else Qt.AlignLeft | Qt.AlignVCenter)
            item.setSizeHint(QSize(190, 220) if browser_mode == "thumbnails" else QSize(260, 56))
            item.setData(Qt.UserRole, index)
            self.predict_compact_list.addItem(item)

        if 0 <= self.current_predict_index < self.predict_compact_list.count():
            self.predict_compact_list.setCurrentRow(self.current_predict_index)
        self.predict_compact_built = True
        self.predict_browser_render_key = render_key
        self.predict_compact_pending_indices = []
        self.predict_compact_loading = False
        self.schedule_predict_visible_thumbnail_update()

    def predict_browser_priority_indices(self) -> list[int]:
        count = self.predict_compact_list.count()
        if count <= 0:
            return []
        viewport_rect = self.predict_compact_list.viewport().rect()
        visible_indices: list[int] = []
        for index in range(count):
            item = self.predict_compact_list.item(index)
            if item is None:
                continue
            item_rect = self.predict_compact_list.visualItemRect(item)
            if item_rect.isValid() and item_rect.intersects(viewport_rect):
                visible_indices.append(index)
        if not visible_indices:
            visible_indices = [index for index in range(min(count, 20))]
        window_start = max(min(visible_indices) - 8, 0)
        window_end = min(max(visible_indices) + 8, count - 1)
        ordered: list[int] = []
        seen: set[int] = set()
        if 0 <= self.current_predict_index < count:
            for index in range(max(self.current_predict_index - 4, 0), min(self.current_predict_index + 5, count)):
                if index not in seen:
                    ordered.append(index)
                    seen.add(index)
        for index in visible_indices:
            if index not in seen:
                ordered.append(index)
                seen.add(index)
        for index in range(window_start, window_end + 1):
            if index not in seen:
                ordered.append(index)
                seen.add(index)
        return ordered

    def schedule_predict_visible_thumbnail_update(self, *_args) -> None:
        if not self.predict_image_paths or self.predict_compact_list.count() == 0:
            return
        priority_indices = [
            index for index in self.predict_browser_priority_indices()
            if 0 <= index < len(self.predict_image_paths)
            and self.predict_cache_get(self.predict_thumbnail_cache, str(self.predict_image_paths[index])) is None
        ]
        if not priority_indices:
            self.predict_compact_pending_indices = []
            self.predict_compact_loading = False
            return
        existing = list(self.predict_compact_pending_indices)
        merged: list[int] = []
        seen: set[int] = set()
        for index in [*priority_indices, *existing]:
            if index not in seen:
                merged.append(index)
                seen.add(index)
        self.predict_compact_pending_indices = merged
        if not self.predict_compact_loading:
            self.predict_compact_loading = True
            self.predict_browser_thumbnail_timer.start(0)

    def process_predict_compact_thumbnail_batch(self) -> None:
        if not self.predict_compact_pending_indices:
            self.predict_compact_loading = False
            return

        batch_size = 12
        batch = self.predict_compact_pending_indices[:batch_size]
        self.predict_compact_pending_indices = self.predict_compact_pending_indices[batch_size:]

        for index in batch:
            if index >= len(self.predict_image_paths) or index >= self.predict_compact_list.count():
                continue
            image_path = self.predict_image_paths[index]
            icon = self.predict_cache_get(self.predict_thumbnail_cache, str(image_path))
            if icon is None:
                pixmap = QPixmap(str(image_path))
                if not pixmap.isNull():
                    target_size = self.predict_compact_list.iconSize()
                    pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    icon = QIcon(pixmap)
                    self.predict_cache_put(self.predict_thumbnail_cache, str(image_path), icon, PREDICT_THUMBNAIL_CACHE_LIMIT)
            if icon is not None:
                self.predict_compact_list.item(index).setIcon(icon)

        if self.predict_compact_pending_indices:
            self.predict_browser_thumbnail_timer.start(0)
        else:
            self.predict_compact_loading = False

    def on_predict_compact_toggled(self, checked: bool) -> None:
        self.predict_display_stack.setCurrentIndex(0)
        self.refresh_predict_compact_view()

    def on_predict_compact_item_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.UserRole)
        if isinstance(index, int):
            self.current_predict_index = index
            self.refresh_predict_page()

    def on_predict_browser_mode_changed(self, *_args) -> None:
        browser_mode = str(self.predict_browser_mode_combo.currentData() or "thumbnails")
        if browser_mode == "list":
            self.predict_compact_list.setViewMode(QListView.ListMode)
            self.predict_compact_list.setResizeMode(QListView.Adjust)
            self.predict_compact_list.setMovement(QListView.Static)
            self.predict_compact_list.setSpacing(4)
            self.predict_compact_list.setIconSize(QSize(56, 56))
            self.predict_compact_list.setGridSize(QSize())
        else:
            self.predict_compact_list.setViewMode(QListView.IconMode)
            self.predict_compact_list.setResizeMode(QListView.Adjust)
            self.predict_compact_list.setMovement(QListView.Static)
            self.predict_compact_list.setSpacing(10)
            self.predict_compact_list.setIconSize(QSize(120, 120))
            self.predict_compact_list.setGridSize(QSize(148, 196))
        self.predict_compact_built = False
        self.predict_browser_render_key = None
        self.predict_compact_pending_indices = []
        self.refresh_predict_compact_view()

    def show_previous_prediction(self) -> None:
        if self.current_predict_index > 0:
            self.current_predict_index -= 1
            if 0 <= self.current_predict_index < self.predict_compact_list.count():
                self.predict_compact_list.setCurrentRow(self.current_predict_index)
            self.refresh_predict_page()

    def show_next_prediction(self) -> None:
        total_items = len(self.predict_results) if self.predict_results else len(self.predict_image_paths)
        if self.current_predict_index < total_items - 1:
            self.current_predict_index += 1
            if 0 <= self.current_predict_index < self.predict_compact_list.count():
                self.predict_compact_list.setCurrentRow(self.current_predict_index)
            self.refresh_predict_page()

    def _resolve_dialog_dir(self, current_text: str, fallback: Path) -> Path:
        if current_text:
            current_path = Path(current_text)
            if current_path.exists():
                return current_path.parent if current_path.is_file() else current_path
        return fallback

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.predict_results and 0 <= self.current_predict_index < len(self.predict_results):
            self.predict_resize_timer.start(90)
        self.schedule_predict_visible_thumbnail_update()

    def _refresh_predict_after_resize(self) -> None:
        if self.predict_results and 0 <= self.current_predict_index < len(self.predict_results):
            self.refresh_predict_page()

    def set_prediction_running_state(self, running: bool) -> None:
        selection_count = len(self.selected_predict_checkpoint_selector_items())
        self.predict_run_button.setEnabled(not running and selection_count > 0)
        self.predict_queue_button.setEnabled(not running and selection_count > 0)
        self.predict_select_images_button.setEnabled(not running)
        self.predict_select_folder_button.setEnabled(not running)
        self.predict_browser_mode_combo.setEnabled(not running)
        self.predict_checkpoint_tree.setEnabled(not running)
        self.predict_select_all_best_button.setEnabled(not running)
        self.predict_clear_selection_button.setEnabled(not running and selection_count > 0)
        self.predict_checkpoint_browse_button.setEnabled(False)
        self.predict_model_combo.setEnabled(not running)
        self.predict_device_combo.setEnabled(not running)
        self.predict_image_size_spin.setEnabled(not running)
        self.predict_compare_checkbox.setEnabled(False)
        self.predict_compare_models_button.setEnabled(not running)
        self.predict_compare_clear_button.setEnabled(not running and selection_count > 0)
        self.predict_export_button.setEnabled(not running and selection_count > 0)
        self.predict_gradcam_button.setEnabled(not running and bool(self.predict_results) and 0 <= self.current_predict_index < len(self.predict_results) and isinstance(self.predict_results[self.current_predict_index], dict) and self.is_predict_compare_result(self.predict_results[self.current_predict_index]) if self.predict_results else False)
        self.set_global_queue_running_state(running)

    def set_test_split_running_state(self, running: bool) -> None:
        self.test_split_run_button.setEnabled(not running)
        self.test_split_queue_button.setEnabled(not running)
        self.test_split_checkpoint_browse_button.setEnabled(not running)
        self.test_split_root_browse_button.setEnabled(not running)
        self.test_split_device_combo.setEnabled(not running)
        self.test_split_image_size_spin.setEnabled(not running)
        self.test_split_batch_size_spin.setEnabled(not running)
        self.test_split_amp_checkbox.setEnabled(not running)
        self.test_split_checkpoint_edit.setEnabled(not running)
        self.test_split_root_edit.setEnabled(not running)
        self.set_global_queue_running_state(running)

    def set_global_queue_running_state(self, running: bool) -> None:
        self.global_queue_list.setEnabled(not running)
        self.queue_remove_button.setEnabled(not running)
        self.queue_duplicate_button.setEnabled(not running)
        self.queue_move_up_button.setEnabled(not running)
        self.queue_move_down_button.setEnabled(not running)
        self.queue_run_button.setEnabled(not running)
        self.queue_clear_finished_button.setEnabled(not running)
        self.queue_stop_button.setEnabled(running)

    def on_prediction_progress(self, processed: int, total: int) -> None:
        self.predict_progress_bar.setRange(0, max(total, 1))
        self.predict_progress_bar.setValue(processed)
        self.predict_progress_bar.setFormat(f"{processed}/{total} (%p%)")
        self.predict_status_label.setText(f"Predicting images... {processed}/{total}")

    def on_prediction_status(self, message: str, indeterminate: bool) -> None:
        self.predict_status_label.setText(message)
        if indeterminate:
            self.predict_progress_bar.setRange(0, 0)
            self.predict_progress_bar.setFormat("Working...")

    def on_prediction_finished(self, results: list, timing: dict) -> None:
        self.predict_results = results
        self.current_predict_index = 0 if results else -1
        self.predict_compact_built = False
        self.predict_compact_loading = False
        self.predict_compact_pending_indices = []
        self.predict_browser_render_key = None
        self.predict_compare_display_cache.clear()
        self.predict_resize_timer.stop()
        total_seconds = float(timing.get("total_seconds", 0.0))
        model_count = int(timing.get("model_count", 1))
        if model_count > 1:
            self.predict_status_label.setText(
                f"Compared {model_count} model(s) across {len(results)} image(s). Total={total_seconds:.2f}s"
            )
        else:
            per_model = timing.get("per_model") if isinstance(timing.get("per_model"), dict) else {}
            first_timing = next(iter(per_model.values()), {})
            pure_seconds = float(first_timing.get("pure_seconds", 0.0)) if isinstance(first_timing, dict) else 0.0
            avg_pure_per_image = float(first_timing.get("avg_pure_per_image_seconds", 0.0)) if isinstance(first_timing, dict) else 0.0
            avg_pure_per_batch = float(first_timing.get("avg_pure_per_batch_seconds", 0.0)) if isinstance(first_timing, dict) else 0.0
            self.predict_status_label.setText(
                f"Predicted {len(results)} image(s). "
                f"Total={total_seconds:.2f}s, Pure={pure_seconds:.2f}s, "
                f"AvgPure/Image={avg_pure_per_image:.4f}s, AvgPure/Batch={avg_pure_per_batch:.4f}s"
            )
        if self.predict_progress_bar.maximum() > 0:
            self.predict_progress_bar.setValue(self.predict_progress_bar.maximum())
        self.set_prediction_running_state(False)
        self.predict_worker = None
        self.predict_thread = None
        self.refresh_predict_page(refresh_compact=self.predict_compact_checkbox.isChecked())
        if self.active_queue_job_type == "predicting" and self.active_job_origin == "queue" and self.active_queue_job_id is not None:
            artifacts = {
                "result_count": len(results),
                "timing": timing,
            }
            active_job_id = self.active_queue_job_id
            self.complete_global_queue_job(active_job_id, "completed", artifacts=artifacts)
            self.clear_active_global_job()
            if not self.global_queue_stop_requested and self.start_next_global_queue_job():
                return
            self.global_queue_running = False
            self.global_queue_stop_requested = False
            self.refresh_global_queue_view(select_job_id=active_job_id)
        else:
            self.clear_active_global_job()

    def on_prediction_failed(self, error_message: str) -> None:
        self.predict_status_label.setText("Prediction failed.")
        self.predict_progress_bar.setValue(0)
        self.set_prediction_running_state(False)
        self.predict_worker = None
        self.predict_thread = None
        if self.active_queue_job_type == "predicting" and self.active_job_origin == "queue" and self.active_queue_job_id is not None:
            active_job_id = self.active_queue_job_id
            final_status = "cancelled" if self.global_queue_stop_requested else "failed"
            self.complete_global_queue_job(active_job_id, final_status, error_message=error_message)
            self.clear_active_global_job()
            if not self.global_queue_stop_requested and self.start_next_global_queue_job():
                return
            self.global_queue_running = False
            self.global_queue_stop_requested = False
            self.refresh_global_queue_view(select_job_id=active_job_id)
            return
        self.clear_active_global_job()
        QMessageBox.critical(self, "Prediction Failed", error_message)

    def handle_predict_process_output(self) -> None:
        try:
            data = bytes(self.predict_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            data = ""
        if data:
            self._predict_process_output += data

    def on_predict_process_started(self) -> None:
        self.predict_status_label.setText("Single-model prediction started...")
        self.predict_progress_bar.setRange(0, 0)
        self.predict_progress_bar.setFormat("Working...")

    def on_predict_process_finished(self, exit_code: int, exit_status) -> None:
        output_path = self._predict_process_json_path
        input_list_path = self._predict_process_input_list_path
        elapsed = 0.0 if self._predict_process_started_at is None else max(time.perf_counter() - self._predict_process_started_at, 0.0)
        self._predict_process_started_at = None
        self._predict_process_json_path = None
        self._predict_process_input_list_path = None

        try:
            if exit_code != 0 or output_path is None or not output_path.is_file():
                detail = self._predict_process_output.strip() or f"Predicting subprocess exited with code {exit_code}."
                raise RuntimeError(detail)

            raw_results = json.loads(output_path.read_text(encoding="utf-8"))
            results: list[dict[str, object]] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                image_path = Path(str(item.get("image_path", ""))).resolve()
                actual_label = image_path.parent.name if image_path.parent.name else None
                predicted_class = str(item.get("predicted_class", ""))
                results.append(
                    {
                        "image_path": image_path,
                        "predicted_class": predicted_class,
                        "confidence": float(item.get("confidence", 0.0)),
                        "actual_label": actual_label,
                        "is_correct": None if actual_label is None else predicted_class == actual_label,
                    }
                )

            timing = {
                "total_seconds": elapsed,
                "num_images": len(results),
                "model_count": 1,
                "per_model": {},
            }
            self.on_prediction_finished(results, timing)
        except Exception as exc:
            self.predict_status_label.setText("Prediction failed.")
            self.predict_progress_bar.setRange(0, 100)
            self.predict_progress_bar.setValue(0)
            self.set_prediction_running_state(False)
            message = f"{exc}"
            if self._predict_process_output.strip():
                message = f"{message}\n\n{self._predict_process_output.strip()}"
            QMessageBox.critical(self, "Prediction Failed", message)
        finally:
            if output_path is not None and output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            if input_list_path is not None and input_list_path.exists():
                try:
                    input_list_path.unlink()
                except Exception:
                    pass
            self._predict_process_output = ""

    def on_predict_process_error(self, error: QProcess.ProcessError) -> None:
        input_list_path = self._predict_process_input_list_path
        self._predict_process_input_list_path = None
        self._predict_process_started_at = None
        self.predict_status_label.setText("Prediction failed.")
        self.predict_progress_bar.setRange(0, 100)
        self.predict_progress_bar.setValue(0)
        self.set_prediction_running_state(False)
        detail = self._predict_process_output.strip()
        if detail:
            detail = f"{error}\n\n{detail}"
        else:
            detail = str(error)
        if input_list_path is not None and input_list_path.exists():
            try:
                input_list_path.unlink()
            except Exception:
                pass
        QMessageBox.critical(self, "Prediction Failed", detail)

    def run_test_split_evaluation(self) -> None:
        try:
            config = self.collect_test_split_config_snapshot()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Test Split Config", str(exc))
            return
        if not self.start_test_split_with_config(config, origin="manual"):
            QMessageBox.information(self, "Job Already Running", "Another training, predicting, or test-split job is already running.")

    def on_test_split_status(self, message: str, indeterminate: bool) -> None:
        self.test_split_status_label.setText(message)
        if indeterminate:
            self.test_split_progress_bar.setRange(0, 0)
            self.test_split_progress_bar.setFormat("Working...")

    def on_test_split_progress(self, processed: int, total: int) -> None:
        self.test_split_progress_bar.setRange(0, max(total, 1))
        self.test_split_progress_bar.setValue(processed)
        self.test_split_progress_bar.setFormat(f"{processed}/{total} (%p%)")

    def on_test_split_finished(self, payload: dict, json_path: str, csv_path: str) -> None:
        splits = payload.get("splits", []) if isinstance(payload, dict) else []
        lines = []
        for item in splits:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"{item.get('split', '-')}: "
                f"acc={float(item.get('accuracy', 0.0)):.4f}, "
                f"correct={int(item.get('correct_images', 0))}/{int(item.get('evaluated_images', 0))}, "
                f"avg_conf={float(item.get('avg_confidence', 0.0)):.4f}"
            )
        self.test_split_output_text.setPlainText(
            "\n".join(lines + ["", f"JSON: {json_path}", f"CSV: {csv_path}"])
        )
        self.test_split_result_label.setText(
            f"Model: {payload.get('model_name', '-')}\n"
            f"Device: {payload.get('device', '-')}\n"
            f"AMP Requested: {payload.get('amp_requested', '-')}\n"
            f"AMP Enabled: {payload.get('amp_enabled', '-')}\n"
            f"Batch Size: {payload.get('batch_size', '-')}\n"
            f"Clean Accuracy: {float(payload.get('clean_accuracy', 0.0)):.4f}\n"
            f"Robustness Average: {float(payload.get('robustness_average', 0.0)):.4f}\n"
            f"Total Time: {float(payload.get('total_seconds', 0.0)):.2f}s"
        )
        self.test_split_status_label.setText("Test split evaluation finished.")
        self.test_split_progress_bar.setRange(0, 100)
        self.test_split_progress_bar.setValue(100)
        self.set_test_split_running_state(False)
        self.test_split_worker = None
        self.test_split_thread = None
        if self.active_queue_job_type == "test_split_eval" and self.active_job_origin == "queue" and self.active_queue_job_id is not None:
            active_job_id = self.active_queue_job_id
            artifacts = {"json_path": json_path, "csv_path": csv_path, "payload": payload}
            self.complete_global_queue_job(active_job_id, "completed", artifacts=artifacts)
            self.clear_active_global_job()
            if not self.global_queue_stop_requested and self.start_next_global_queue_job():
                return
            self.global_queue_running = False
            self.global_queue_stop_requested = False
            self.refresh_global_queue_view(select_job_id=active_job_id)
        else:
            self.clear_active_global_job()

    def on_test_split_failed(self, error_message: str) -> None:
        self.test_split_status_label.setText("Test split evaluation failed.")
        self.test_split_progress_bar.setRange(0, 100)
        self.test_split_progress_bar.setValue(0)
        self.set_test_split_running_state(False)
        self.test_split_worker = None
        self.test_split_thread = None
        if self.active_queue_job_type == "test_split_eval" and self.active_job_origin == "queue" and self.active_queue_job_id is not None:
            active_job_id = self.active_queue_job_id
            final_status = "cancelled" if self.global_queue_stop_requested else "failed"
            self.complete_global_queue_job(active_job_id, final_status, error_message=error_message)
            self.clear_active_global_job()
            if not self.global_queue_stop_requested and self.start_next_global_queue_job():
                return
            self.global_queue_running = False
            self.global_queue_stop_requested = False
            self.refresh_global_queue_view(select_job_id=active_job_id)
            return
        self.clear_active_global_job()
        QMessageBox.critical(self, "Test Split Evaluation Failed", error_message)


class PredictionWorker(QObject):
    progress = Signal(int, int)
    status = Signal(str, bool)
    finished = Signal(list, dict)
    failed = Signal(str)

    def __init__(
        self,
        *,
        image_paths: list[Path],
        model_specs: list[dict[str, object]],
        image_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.model_specs = model_specs
        self.image_size = image_size
        self.device = device

    def run(self) -> None:
        try:
            total_start = time.perf_counter()
            import torch
            from pipeline.predicting import build_transform, infer_model_name_from_checkpoint, load_model, predict_images_batch

            resolved_device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            transform = build_transform(self.image_size)
            predict_batch_size = 16
            aggregate_progress_total = len(self.image_paths) * max(len(self.model_specs), 1)
            aggregate_processed = 0
            results_by_path: dict[str, dict[str, object]] = {
                str(path.resolve()): {
                    "image_path": path.resolve(),
                    "comparisons": {},
                }
                for path in self.image_paths
            }
            timing_by_model: dict[str, dict[str, float | str]] = {}

            for model_index, spec in enumerate(self.model_specs, start=1):
                checkpoint_path = spec.get("checkpoint_path") if isinstance(spec, dict) else None
                model_name_hint = spec.get("model_name_hint") if isinstance(spec, dict) else None
                display_label = spec.get("display_label") if isinstance(spec, dict) else None
                if not isinstance(checkpoint_path, Path):
                    raise ValueError("Prediction worker received an invalid checkpoint path.")
                resolved_checkpoint = checkpoint_path.expanduser().resolve()
                resolved_model_name = model_name_hint if isinstance(model_name_hint, str) else None
                if resolved_model_name is None:
                    self.status.emit(f"Detecting model {model_index}/{len(self.model_specs)} from checkpoint...", True)
                    resolved_model_name = infer_model_name_from_checkpoint(resolved_checkpoint)
                if not resolved_model_name:
                    raise ValueError(f"Could not determine model type for checkpoint: {resolved_checkpoint}")
                model_name = str(resolved_model_name)
                result_label = str(display_label).strip() if isinstance(display_label, str) and str(display_label).strip() else model_name
                self.status.emit(f"Loading model {model_index}/{len(self.model_specs)}: {result_label}", True)
                model, class_to_idx = load_model(resolved_checkpoint, model_name, resolved_device)
                idx_to_class = {idx: name for name, idx in class_to_idx.items()}
                self.status.emit(
                    f"Running {result_label} on {len(self.image_paths)} image(s) ({model_index}/{len(self.model_specs)})",
                    False,
                )

                pure_start = time.perf_counter()
                batch_results = predict_images_batch(
                    model,
                    self.image_paths,
                    transform,
                    idx_to_class,
                    resolved_device,
                    batch_size=predict_batch_size,
                    progress_callback=lambda processed, total, base=aggregate_processed: self.progress.emit(base + processed, aggregate_progress_total),
                )
                pure_seconds = time.perf_counter() - pure_start
                aggregate_processed += len(self.image_paths)
                self.progress.emit(aggregate_processed, aggregate_progress_total)

                for result in batch_results:
                    resolved_image = Path(str(result["image_path"])).resolve()
                    actual_label = resolved_image.parent.name if resolved_image.parent.name in class_to_idx else None
                    result_entry = results_by_path[str(resolved_image)]
                    comparisons = result_entry["comparisons"]
                    assert isinstance(comparisons, dict)
                    comparisons[result_label] = {
                        **result,
                        "model_name": model_name,
                        "display_label": result_label,
                        "checkpoint_path": str(resolved_checkpoint),
                        "actual_label": actual_label,
                        "is_correct": None if actual_label is None else result["predicted_class"] == actual_label,
                    }

                num_images = len(self.image_paths)
                num_batches = (num_images + predict_batch_size - 1) // predict_batch_size if num_images > 0 else 0
                timing_by_model[result_label] = {
                    "model_name": model_name,
                    "checkpoint_path": str(resolved_checkpoint),
                    "pure_seconds": pure_seconds,
                    "avg_pure_per_image_seconds": (pure_seconds / num_images) if num_images > 0 else 0.0,
                    "avg_pure_per_batch_seconds": (pure_seconds / num_batches) if num_batches > 0 else 0.0,
                    "num_images": num_images,
                    "num_batches": num_batches,
                }

            total_seconds = time.perf_counter() - total_start
            results: list[dict[str, object]] = []
            for image_path in self.image_paths:
                resolved_image = image_path.resolve()
                result_entry = results_by_path[str(resolved_image)]
                comparisons = result_entry.get("comparisons")
                actual_label = None
                if isinstance(comparisons, dict) and comparisons:
                    first_item = next(iter(comparisons.values()))
                    if isinstance(first_item, dict):
                        actual_label = first_item.get("actual_label")
                flattened: dict[str, object] = {
                    "image_path": resolved_image,
                    "actual_label": actual_label,
                    "comparisons": comparisons if isinstance(comparisons, dict) else {},
                }
                if isinstance(comparisons, dict) and len(comparisons) == 1:
                    single_result = next(iter(comparisons.values()))
                    if isinstance(single_result, dict):
                        flattened.update(single_result)
                results.append(flattened)

            num_images = len(self.image_paths)
            timing = {
                "total_seconds": total_seconds,
                "num_images": num_images,
                "model_count": len(self.model_specs),
                "per_model": timing_by_model,
            }
            self.finished.emit(results, timing)
        except Exception as exc:
            self.failed.emit(str(exc))


class GradCamComparisonWorker(QObject):
    finished = Signal(object, object)
    failed = Signal(object, str)

    def __init__(
        self,
        *,
        image_path: Path,
        model_specs: list[tuple[str, str, Path]],
        image_size: int,
        device: str,
        request_key: tuple[object, ...],
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.model_specs = model_specs
        self.image_size = image_size
        self.device = device
        self.request_key = request_key

    def run(self) -> None:
        try:
            from core.gradcam import render_gradcam_overlay_bytes_with_diagnostics

            overlays: list[tuple[tuple[str, str, str, int, str], bytes, str | None]] = []
            resolved_image_path = self.image_path.resolve()
            for display_label, model_name, checkpoint_path in self.model_specs:
                resolved_checkpoint = checkpoint_path.expanduser().resolve()
                cache_key = (
                    str(resolved_image_path),
                    display_label,
                    str(resolved_checkpoint),
                    self.image_size,
                    self.device,
                )
                try:
                    image_data, diagnostic_reason = render_gradcam_overlay_bytes_with_diagnostics(
                        image_path=resolved_image_path,
                        checkpoint_path=resolved_checkpoint,
                        model_name=model_name,
                        image_size=self.image_size,
                        device=self.device,
                    )
                except Exception as exc:
                    overlays.append((cache_key, b"", f"Grad-CAM unavailable: {exc}"))
                    continue
                overlays.append((cache_key, image_data, diagnostic_reason))
            self.finished.emit(self.request_key, overlays)
        except Exception as exc:
            self.failed.emit(self.request_key, str(exc))


class TestSplitEvaluationWorker(QObject):
    status = Signal(str, bool)
    progress = Signal(int, int)
    finished = Signal(dict, str, str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        model_name: str | None,
        test_splits_root: Path,
        image_size: int,
        batch_size: int,
        amp_requested: bool,
        device: str,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.test_splits_root = test_splits_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.amp_requested = amp_requested
        self.device = device

    def run(self) -> None:
        try:
            from pipeline.evaluate_test_splits import evaluate_test_splits

            payload, json_path, csv_path = evaluate_test_splits(
                checkpoint_path=self.checkpoint_path,
                model_name=self.model_name,
                test_splits_root=self.test_splits_root,
                image_size=self.image_size,
                batch_size=self.batch_size,
                amp_requested=self.amp_requested,
                device=self.device,
                output_dir=PROJECT_ROOT / "logs" / "test_split_evaluations",
                status_callback=lambda message, indeterminate: self.status.emit(message, indeterminate),
                progress_callback=lambda processed, total: self.progress.emit(processed, total),
            )
            self.finished.emit(payload, str(json_path), str(csv_path))
        except Exception as exc:
            self.failed.emit(str(exc))


def main() -> None:
    runtime_paths.ensure_working_folders()
    set_windows_app_id()
    app = QApplication(sys.argv)
    if APP_ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    saved_theme = str(settings.value("ui/theme", app_themes.DEFAULT_THEME_KEY))
    current_theme = saved_theme if saved_theme in app_themes.THEMES else app_themes.DEFAULT_THEME_KEY
    app.setStyleSheet(app_themes.build_stylesheet(current_theme))
    splash = build_startup_splash(current_theme)
    splash.show()
    app.processEvents()
    splash_color = QColor(app_themes.get_theme(current_theme)["text_muted"])

    def _on_startup_progress(message: str) -> None:
        splash.showMessage(f"  {message}", Qt.AlignLeft | Qt.AlignBottom, splash_color)
        app.processEvents()

    window = TrainingLauncher(startup_progress_callback=_on_startup_progress)
    window.showMaximized()
    splash.finish(window)
    QTimer.singleShot(0, lambda: apply_windows_taskbar_icon(window))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
