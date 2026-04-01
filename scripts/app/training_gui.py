from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QSpinBox,
    QStackedWidget,
    QTreeView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.model_registry import discover_model_names


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RETRIEVAL_SCRIPT = PROJECT_ROOT / "scripts" / "entry" / "data_retrieval.py"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "entry" / "training.py"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "food-101"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
APP_ICON_PATH = PROJECT_ROOT / "scripts" / "assets" / "training_launcher_icon.ico"
APP_ID = "DATA586Project.TrainingLauncher"
WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_DEFAULTSIZE = 0x00000040
NEW_CHECKPOINT_NAME_LABEL = "New checkpoint name..."


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


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


class TrainingLauncher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DATA586 Training Launcher")
        self.resize(1080, 820)
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
        self.predict_thumbnail_cache: dict[str, QIcon] = {}
        self.predict_display_cache: dict[tuple[str, int, int], QPixmap] = {}
        self.available_models = discover_model_names()
        self._checkpoint_name_locked_to_model = True
        self._last_training_model_name = self.available_models[0] if self.available_models else ""
        self._last_predict_model_name = self.available_models[0] if self.available_models else ""

        self._init_data_controls()
        self._init_training_controls()
        self._init_prediction_controls()
        self._build_ui()
        self.refresh_command_preview()
        self.refresh_predict_page()
        self.on_predict_compact_toggled(self.predict_compact_checkbox.isChecked())

    def _init_training_controls(self) -> None:
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.available_models)

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

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0, 10.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)

        self.freeze_checkbox = QCheckBox("Freeze backbone")
        self.freeze_checkbox.setChecked(True)

        self.validation_checkbox = QCheckBox("Use validation split")
        self.validation_checkbox.setChecked(False)

        self.validation_proportion_spin = QDoubleSpinBox()
        self.validation_proportion_spin.setRange(0.01, 0.99)
        self.validation_proportion_spin.setDecimals(2)
        self.validation_proportion_spin.setSingleStep(0.01)
        self.validation_proportion_spin.setValue(0.10)

        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        self.resume_checkbox.setChecked(False)

        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText(str(DEFAULT_CHECKPOINT_DIR))

        self.resume_browse_button = QPushButton("Browse...")
        self.resume_browse_button.clicked.connect(self.choose_resume_path)

        self.resume_clear_button = QPushButton("Clear")
        self.resume_clear_button.clicked.connect(self.clear_resume_path)

        self.checkpoint_output_combo = QComboBox()
        self.checkpoint_output_combo.setEditable(True)
        self.refresh_checkpoint_output_options()

        self.data_root_label = QLabel(str(DEFAULT_DATA_ROOT))
        self.data_root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_root_label.setWordWrap(True)

        self.checkpoint_dir_label = QLabel(str(self.selected_checkpoint_dir()))
        self.checkpoint_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.checkpoint_dir_label.setWordWrap(True)

        self.command_preview = QLabel()
        self.command_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command_preview.setWordWrap(True)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(170)

        self.progress_label = QLabel("Progress will appear here after training starts.")
        self.progress_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.start_training)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)

        self.status_label = QLabel("Idle")

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

        self.data_progress_bar = QProgressBar()
        self.data_progress_bar.setRange(0, 100)
        self.data_progress_bar.setValue(0)
        self.data_progress_bar.setFormat("%p%")

        self.data_output_text = QPlainTextEdit()
        self.data_output_text.setReadOnly(True)
        self.data_output_text.setMaximumHeight(220)

    def _init_prediction_controls(self) -> None:
        self.predict_model_combo = QComboBox()
        self.predict_model_combo.addItems(self.available_models)

        self.predict_device_combo = QComboBox()
        self.predict_device_combo.addItems(["auto", "cpu", "cuda"])

        self.predict_image_size_spin = QSpinBox()
        self.predict_image_size_spin.setRange(32, 2048)
        self.predict_image_size_spin.setValue(224)

        self.predict_checkpoint_edit = QLineEdit(str(self.default_predict_checkpoint_path()))
        self.predict_checkpoint_browse_button = QPushButton("Browse...")
        self.predict_checkpoint_browse_button.clicked.connect(self.choose_predict_checkpoint)
        self.predict_model_combo.currentTextChanged.connect(self.on_predict_model_changed)

        self.predict_select_images_button = QPushButton("Select Images")
        self.predict_select_images_button.clicked.connect(self.choose_predict_images)

        self.predict_select_folder_button = QPushButton("Select Folders")
        self.predict_select_folder_button.clicked.connect(self.choose_predict_folders)

        self.predict_run_button = QPushButton("Predict")
        self.predict_run_button.clicked.connect(self.run_predictions)

        self.predict_compact_checkbox = QCheckBox("Compact Mode")
        self.predict_compact_checkbox.toggled.connect(self.on_predict_compact_toggled)

        self.predict_prev_button = QPushButton("Previous")
        self.predict_prev_button.clicked.connect(self.show_previous_prediction)

        self.predict_next_button = QPushButton("Next")
        self.predict_next_button.clicked.connect(self.show_next_prediction)

        self.predict_selected_label = QLabel("No images selected.")
        self.predict_selected_label.setWordWrap(True)

        self.predict_status_label = QLabel("Ready.")
        self.predict_status_label.setWordWrap(True)

        self.predict_progress_bar = QProgressBar()
        self.predict_progress_bar.setRange(0, 100)
        self.predict_progress_bar.setValue(0)
        self.predict_progress_bar.setFormat("%p%")

        self.predict_page_label = QLabel("0 / 0")

        self.predict_image_label = QLabel("Select images and click Predict.")
        self.predict_image_label.setAlignment(Qt.AlignCenter)
        self.predict_image_label.setMinimumHeight(420)
        self.predict_image_label.setStyleSheet(
            "QLabel { border: 1px solid #4a4a4a; background: #1f1f1f; color: #d0d0d0; }"
        )

        self.predict_result_label = QLabel("Prediction result will appear here.")
        self.predict_result_label.setWordWrap(True)
        self.predict_result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.predict_compact_list = QListWidget()
        self.predict_compact_list.setViewMode(QListView.IconMode)
        self.predict_compact_list.setResizeMode(QListView.Adjust)
        self.predict_compact_list.setMovement(QListView.Static)
        self.predict_compact_list.setSpacing(12)
        self.predict_compact_list.setIconSize(QSize(160, 160))
        self.predict_compact_list.setGridSize(QSize(190, 250))
        self.predict_compact_list.setWordWrap(True)
        self.predict_compact_list.setUniformItemSizes(True)
        self.predict_compact_list.itemClicked.connect(self.on_predict_compact_item_clicked)

        self.predict_display_stack = QStackedWidget()

        single_predict_page = QWidget()
        single_predict_layout = QVBoxLayout(single_predict_page)
        single_predict_layout.addWidget(self.predict_image_label, stretch=1)

        predict_result_group = QGroupBox("Prediction Result")
        predict_result_layout = QVBoxLayout(predict_result_group)
        predict_result_layout.addWidget(self.predict_result_label)
        single_predict_layout.addWidget(predict_result_group)

        compact_predict_page = QWidget()
        compact_predict_layout = QVBoxLayout(compact_predict_page)
        compact_predict_layout.addWidget(self.predict_compact_list)

        self.predict_display_stack.addWidget(single_predict_page)
        self.predict_display_stack.addWidget(compact_predict_page)

    def _build_ui(self) -> None:
        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)

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
        training_layout = QVBoxLayout(training_tab)

        config_group = QGroupBox("Training Config")
        form = QFormLayout(config_group)
        form.addRow("Model", self.model_combo)
        form.addRow("Device", self.device_combo)
        form.addRow("Epochs", self.epochs_spin)
        form.addRow("Batch Size", self.batch_size_spin)
        form.addRow("Num Workers", self.num_workers_spin)
        form.addRow("Image Size", self.image_size_spin)
        form.addRow("Learning Rate", self.lr_spin)
        form.addRow("", self.freeze_checkbox)
        form.addRow("", self.validation_checkbox)
        form.addRow("Validation Proportion", self.validation_proportion_spin)
        form.addRow("", self.resume_checkbox)
        form.addRow("Checkpoint Output", self.checkpoint_output_combo)
        resume_layout = QHBoxLayout()
        resume_layout.addWidget(self.resume_path_edit, stretch=1)
        resume_layout.addWidget(self.resume_browse_button)
        resume_layout.addWidget(self.resume_clear_button)
        form.addRow("Resume Checkpoint", resume_layout)
        form.addRow("Data Root", self.data_root_label)
        form.addRow("Checkpoint Dir", self.checkpoint_dir_label)
        form.addRow("Command", self.command_preview)
        training_layout.addWidget(config_group)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)
        training_layout.addLayout(controls_layout)

        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        training_layout.addWidget(progress_group)

        log_group = QGroupBox("Training Output")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.output_text)
        training_layout.addWidget(log_group)

        predict_tab = QWidget()
        predict_layout = QVBoxLayout(predict_tab)

        predict_config_group = QGroupBox("Predict Config")
        predict_form = QFormLayout(predict_config_group)
        predict_form.addRow("Model", self.predict_model_combo)
        predict_form.addRow("Device", self.predict_device_combo)
        predict_form.addRow("Image Size", self.predict_image_size_spin)
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.predict_checkpoint_edit, stretch=1)
        checkpoint_layout.addWidget(self.predict_checkpoint_browse_button)
        predict_form.addRow("Checkpoint", checkpoint_layout)
        predict_form.addRow("Images", self.predict_selected_label)
        predict_layout.addWidget(predict_config_group)

        predict_controls = QHBoxLayout()
        predict_controls.addWidget(self.predict_select_images_button)
        predict_controls.addWidget(self.predict_select_folder_button)
        predict_controls.addWidget(self.predict_run_button)
        predict_controls.addWidget(self.predict_compact_checkbox)
        predict_controls.addStretch(1)
        predict_controls.addWidget(self.predict_prev_button)
        predict_controls.addWidget(self.predict_page_label)
        predict_controls.addWidget(self.predict_next_button)
        predict_layout.addLayout(predict_controls)

        predict_layout.addWidget(self.predict_status_label)
        predict_layout.addWidget(self.predict_progress_bar)
        predict_layout.addWidget(self.predict_display_stack, stretch=1)

        tabs.addTab(training_tab, "Training")
        tabs.addTab(predict_tab, "Predicting")
        tabs.addTab(data_tab, "Data")
        tabs.setCurrentIndex(0)

        self.model_combo.currentTextChanged.connect(self.on_training_model_changed)
        self.device_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.epochs_spin.valueChanged.connect(self.refresh_command_preview)
        self.batch_size_spin.valueChanged.connect(self.refresh_command_preview)
        self.num_workers_spin.valueChanged.connect(self.refresh_command_preview)
        self.image_size_spin.valueChanged.connect(self.refresh_command_preview)
        self.lr_spin.valueChanged.connect(self.refresh_command_preview)
        self.freeze_checkbox.toggled.connect(self.refresh_command_preview)
        self.validation_checkbox.toggled.connect(self.on_validation_toggled)
        self.validation_proportion_spin.valueChanged.connect(self.refresh_command_preview)
        self.resume_checkbox.toggled.connect(self.on_resume_toggled)
        self.resume_path_edit.textChanged.connect(self.refresh_command_preview)
        self.checkpoint_output_combo.currentTextChanged.connect(self.on_checkpoint_output_changed)
        self.checkpoint_output_combo.activated.connect(self.on_checkpoint_output_activated)
        self.on_validation_toggled(self.validation_checkbox.isChecked())
        self.on_resume_toggled(self.resume_checkbox.isChecked())
        self.on_training_model_changed(self.model_combo.currentText())

    def build_command(self) -> list[str]:
        checkpoint_dir = self.selected_checkpoint_dir()
        command = [
            "-u",
            str(TRAINING_SCRIPT),
            "--model",
            self.model_combo.currentText(),
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch-size",
            str(self.batch_size_spin.value()),
            "--num-workers",
            str(self.num_workers_spin.value()),
            "--image-size",
            str(self.image_size_spin.value()),
            "--lr",
            format(self.lr_spin.value(), ".6f"),
            "--progress-format",
            "gui",
        ]

        device = self.device_combo.currentText()
        if device != "auto":
            command.extend(["--device", device])

        command.append("--freeze-backbone" if self.freeze_checkbox.isChecked() else "--no-freeze-backbone")
        if self.validation_checkbox.isChecked():
            command.extend(
                [
                    "--use-validation-split",
                    "--validation-proportion",
                    format(self.validation_proportion_spin.value(), ".2f"),
                ]
            )

        resume_path = self.resume_path_edit.text().strip()
        if self.resume_checkbox.isChecked() and resume_path:
            command.extend(["--resume", resume_path])
        return command

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

    def default_predict_checkpoint_path(self) -> Path:
        return DEFAULT_CHECKPOINT_DIR / self.predict_model_combo.currentText() / "best.pth"

    def on_predict_model_changed(self) -> None:
        current_path = self.predict_checkpoint_edit.text().strip()
        old_default = DEFAULT_CHECKPOINT_DIR / self._last_predict_model_name / "best.pth"
        old_flat_default = DEFAULT_CHECKPOINT_DIR / f"{self._last_predict_model_name}_best.pth"
        if not current_path or Path(current_path) in {old_default, old_flat_default}:
            self.predict_checkpoint_edit.setText(str(self.default_predict_checkpoint_path()))
        self._last_predict_model_name = self.predict_model_combo.currentText()

    def refresh_command_preview(self) -> None:
        parts = [sys.executable, *self.build_command()]
        self.command_preview.setText(" ".join(f'"{part}"' if " " in part else part for part in parts))

    def checkpoint_output_name(self) -> str:
        text = self.checkpoint_output_combo.currentText().strip()
        return "" if text == NEW_CHECKPOINT_NAME_LABEL else text

    def selected_checkpoint_dir(self) -> Path:
        checkpoint_name = self.checkpoint_output_name() or self.model_combo.currentText()
        return DEFAULT_CHECKPOINT_DIR / checkpoint_name

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
        elif self.model_combo.currentText():
            self.checkpoint_output_combo.setEditText(self.model_combo.currentText())

    def update_checkpoint_dir_label(self) -> None:
        self.checkpoint_dir_label.setText(str(self.selected_checkpoint_dir()))

    def on_training_model_changed(self, model_name: str) -> None:
        current_name = self.checkpoint_output_name()
        if self._checkpoint_name_locked_to_model or not current_name or current_name == self._last_training_model_name:
            self.checkpoint_output_combo.setEditText(model_name)
            self._checkpoint_name_locked_to_model = True
        self._last_training_model_name = model_name
        self.update_checkpoint_dir_label()
        self.refresh_command_preview()

    def on_checkpoint_output_changed(self, text: str) -> None:
        checkpoint_name = text.strip()
        self._checkpoint_name_locked_to_model = checkpoint_name in {"", self.model_combo.currentText()}
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
        self.refresh_command_preview()

    def on_validation_toggled(self, checked: bool) -> None:
        self.validation_proportion_spin.setEnabled(checked)
        self.refresh_command_preview()

    def choose_resume_path(self) -> None:
        start_dir = self._resolve_dialog_dir(self.resume_path_edit.text().strip(), self.selected_checkpoint_dir())
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Resume Checkpoint",
            str(start_dir),
            "PyTorch Checkpoints (*.pth *.pt);;All Files (*.*)",
        )
        if selected_path:
            self.resume_checkbox.setChecked(True)
            self.resume_path_edit.setText(selected_path)

    def clear_resume_path(self) -> None:
        self.resume_path_edit.clear()
        self.refresh_command_preview()

    def set_running_state(self, running: bool) -> None:
        self.train_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.model_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.epochs_spin.setEnabled(not running)
        self.batch_size_spin.setEnabled(not running)
        self.num_workers_spin.setEnabled(not running)
        self.image_size_spin.setEnabled(not running)
        self.lr_spin.setEnabled(not running)
        self.freeze_checkbox.setEnabled(not running)
        self.validation_checkbox.setEnabled(not running)
        self.validation_proportion_spin.setEnabled(not running and self.validation_checkbox.isChecked())
        self.checkpoint_output_combo.setEnabled(not running)
        self.resume_checkbox.setEnabled(not running)
        self.resume_path_edit.setEnabled(not running and self.resume_checkbox.isChecked())
        self.resume_browse_button.setEnabled(not running and self.resume_checkbox.isChecked())
        self.resume_clear_button.setEnabled(not running and self.resume_checkbox.isChecked())

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

        if not TRAINING_SCRIPT.is_file():
            QMessageBox.critical(self, "Missing Script", f"Could not find training script:\n{TRAINING_SCRIPT}")
            return

        if self.resume_checkbox.isChecked():
            resume_path = self.resume_path_edit.text().strip()
            if not resume_path:
                QMessageBox.warning(self, "Resume Path Required", "Select a checkpoint file before starting resume training.")
                return
            if not Path(resume_path).is_file():
                QMessageBox.warning(self, "Invalid Resume Path", f"Checkpoint file does not exist:\n{resume_path}")
                return
        checkpoint_name = self.checkpoint_output_name()
        if not checkpoint_name:
            QMessageBox.warning(self, "Checkpoint Name Required", "Choose or enter a checkpoint output folder name.")
            return

        self.output_text.clear()
        self._committed_output = ""
        self._stream_buffer = ""
        self.progress_label.setText("Starting training...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.append_output(f"Project root: {PROJECT_ROOT}\n")
        self.append_output(f"Launching: {self.command_preview.text()}\n\n")

        self.process.start(sys.executable, self.build_command())

    def set_data_running_state(self, running: bool) -> None:
        self.data_check_button.setEnabled(not running)
        self.data_prepare_button.setEnabled(not running)
        self.data_force_button.setEnabled(not running)

    def start_data_command(self, command: list[str], status_text: str) -> None:
        if self.data_process.state() != QProcess.NotRunning:
            return
        if not DATA_RETRIEVAL_SCRIPT.is_file():
            QMessageBox.critical(self, "Missing Script", f"Could not find data retrieval script:\n{DATA_RETRIEVAL_SCRIPT}")
            return

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
            f"Launching: {' '.join(f'\"{part}\"' if ' ' in part else part for part in [sys.executable, *command])}\n\n"
        )
        self.data_process.start(sys.executable, command)

    def run_data_check(self) -> None:
        self.start_data_command(self.build_data_command(check_only=True), "Checking dataset integrity...")

    def run_data_prepare(self) -> None:
        self.start_data_command(self.build_data_command(), "Preparing dataset...")

    def run_data_force_redownload(self) -> None:
        self.start_data_command(
            self.build_data_command(force_redownload=True),
            "Force re-downloading and extracting dataset...",
        )

    def stop_training(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        self.append_output("\nStopping training process...\n")
        self.process.kill()

    def handle_output(self) -> None:
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.append_stream_output(data)

    def handle_data_output(self) -> None:
        data = self.data_process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.append_data_stream_output(data)

    def on_process_started(self) -> None:
        self.set_running_state(True)
        self.status_label.setText("Running")
        self.progress_label.setText("Process started. Waiting for training progress...")

    def on_data_process_started(self) -> None:
        self.set_data_running_state(True)
        self.data_status_label.setText("Running")
        self.data_state_value_label.setText("Running")
        self.data_progress_label.setText("Data task started...")

    def on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.set_running_state(False)
        self.refresh_checkpoint_output_options(preserve_text=self.checkpoint_output_name())
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
        self.status_label.setText("Error")
        self.progress_label.setText(f"Process error: {error}")
        self.append_output(f"\nProcess error: {error}\n")

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

    def choose_predict_images(self) -> None:
        selected_paths = self.select_multiple_files(
            title="Select Images to Predict",
            start_dir=DEFAULT_DATA_ROOT / "images",
            file_filter="Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if selected_paths:
            self.predict_image_paths = [Path(path) for path in selected_paths]
            self.predict_results = []
            self.current_predict_index = -1
            self.predict_compact_built = False
            self.predict_compact_loading = False
            self.predict_compact_pending_indices = []
            self.predict_progress_bar.setValue(0)
            self.refresh_predict_page()

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
        self.current_predict_index = -1
        self.predict_compact_built = False
        self.predict_compact_loading = False
        self.predict_compact_pending_indices = []
        self.predict_progress_bar.setValue(0)
        if not self.predict_image_paths:
            self.predict_status_label.setText("No supported images found in the selected folder(s).")
        else:
            self.predict_status_label.setText(
                f"Loaded {len(self.predict_image_paths)} image(s) from {len(selected_dirs)} folder(s)."
            )
        self.refresh_predict_page()

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
        if self.predict_thread is not None and self.predict_thread.isRunning():
            return
        checkpoint_path = Path(self.predict_checkpoint_edit.text().strip()).expanduser()
        if not checkpoint_path.is_file():
            QMessageBox.warning(self, "Invalid Checkpoint", f"Checkpoint file does not exist:\n{checkpoint_path}")
            return
        if not self.predict_image_paths:
            QMessageBox.warning(self, "No Images Selected", "Select one or more images before predicting.")
            return

        device = self.predict_device_combo.currentText()
        self.predict_status_label.setText("Loading model and running predictions...")
        self.predict_progress_bar.setRange(0, len(self.predict_image_paths))
        self.predict_progress_bar.setValue(0)
        self.set_prediction_running_state(True)

        self.predict_thread = QThread(self)
        self.predict_worker = PredictionWorker(
            image_paths=[path.expanduser().resolve() for path in self.predict_image_paths],
            checkpoint_path=checkpoint_path.resolve(),
            model_name=self.predict_model_combo.currentText(),
            image_size=self.predict_image_size_spin.value(),
            device=device,
        )
        self.predict_worker.moveToThread(self.predict_thread)
        self.predict_thread.started.connect(self.predict_worker.run)
        self.predict_worker.progress.connect(self.on_prediction_progress)
        self.predict_worker.finished.connect(self.on_prediction_finished)
        self.predict_worker.failed.connect(self.on_prediction_failed)
        self.predict_worker.finished.connect(self.predict_thread.quit)
        self.predict_worker.failed.connect(self.predict_thread.quit)
        self.predict_thread.finished.connect(self.predict_thread.deleteLater)
        self.predict_thread.start()

    def refresh_predict_page(self, refresh_compact: bool = False) -> None:
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
        self.predict_prev_button.setEnabled(has_results and self.current_predict_index > 0)
        self.predict_next_button.setEnabled(has_results and self.current_predict_index < len(self.predict_results) - 1)
        self.predict_page_label.setText(
            f"{self.current_predict_index + 1 if has_results else 0} / {len(self.predict_results)}"
        )
        if refresh_compact:
            self.refresh_predict_compact_view()

        if not has_results:
            self.predict_image_label.setPixmap(QPixmap())
            self.predict_image_label.setText("Select images and click Predict.")
            self.predict_result_label.setText("Prediction result will appear here.")
            return

        result = self.predict_results[self.current_predict_index]
        image_path = Path(str(result["image_path"]))
        cache_key = (str(image_path), max(self.predict_image_label.width(), 1), max(self.predict_image_label.height(), 1))
        pixmap = self.predict_display_cache.get(cache_key)
        if pixmap is None:
            loaded = QPixmap(str(image_path))
            if not loaded.isNull():
                pixmap = loaded.scaled(
                    self.predict_image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.predict_display_cache[cache_key] = pixmap
            else:
                pixmap = QPixmap()
        if pixmap.isNull():
            self.predict_image_label.setPixmap(QPixmap())
            self.predict_image_label.setText(f"Could not load image:\n{image_path}")
        else:
            self.predict_image_label.setText("")
            self.predict_image_label.setPixmap(pixmap)

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

    def refresh_predict_compact_view(self) -> None:
        if self.predict_compact_built and self.predict_compact_list.count() == len(self.predict_results):
            if 0 <= self.current_predict_index < self.predict_compact_list.count():
                self.predict_compact_list.setCurrentRow(self.current_predict_index)
            return

        self.predict_compact_list.clear()
        if not self.predict_results:
            self.predict_compact_built = False
            self.predict_compact_loading = False
            self.predict_compact_pending_indices = []
            return

        for index, result in enumerate(self.predict_results):
            item = QListWidgetItem()
            image_path = Path(str(result["image_path"]))
            icon = self.predict_thumbnail_cache.get(str(image_path))
            if icon is not None:
                item.setIcon(icon)

            is_correct = result.get("is_correct")
            if is_correct is True:
                correctness_text = "Yes"
            elif is_correct is False:
                correctness_text = "No"
            else:
                correctness_text = "Unknown"
            actual_label = result.get("actual_label")
            actual_text = actual_label if actual_label is not None else "Unknown"

            item.setText(
                f"{result['predicted_class']}\n"
                f"True: {actual_text}\n"
                f"{float(result['confidence']):.2%}\n"
                f"Correct: {correctness_text}"
            )
            item.setTextAlignment(Qt.AlignHCenter)
            item.setSizeHint(QSize(190, 250))
            item.setData(Qt.UserRole, index)
            self.predict_compact_list.addItem(item)

        if 0 <= self.current_predict_index < self.predict_compact_list.count():
            self.predict_compact_list.setCurrentRow(self.current_predict_index)
        self.predict_compact_built = True
        self.predict_compact_pending_indices = [
            index for index, result in enumerate(self.predict_results)
            if str(result["image_path"]) not in self.predict_thumbnail_cache
        ]
        if self.predict_compact_pending_indices:
            self.predict_compact_loading = True
            QTimer.singleShot(0, self.process_predict_compact_thumbnail_batch)
        else:
            self.predict_compact_loading = False

    def process_predict_compact_thumbnail_batch(self) -> None:
        if not self.predict_compact_pending_indices:
            self.predict_compact_loading = False
            return

        batch_size = 12
        batch = self.predict_compact_pending_indices[:batch_size]
        self.predict_compact_pending_indices = self.predict_compact_pending_indices[batch_size:]

        for index in batch:
            if index >= len(self.predict_results) or index >= self.predict_compact_list.count():
                continue
            result = self.predict_results[index]
            image_path = Path(str(result["image_path"]))
            icon = self.predict_thumbnail_cache.get(str(image_path))
            if icon is None:
                pixmap = QPixmap(str(image_path))
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    icon = QIcon(pixmap)
                    self.predict_thumbnail_cache[str(image_path)] = icon
            if icon is not None:
                self.predict_compact_list.item(index).setIcon(icon)

        if self.predict_compact_pending_indices:
            QTimer.singleShot(0, self.process_predict_compact_thumbnail_batch)
        else:
            self.predict_compact_loading = False

    def on_predict_compact_toggled(self, checked: bool) -> None:
        self.predict_display_stack.setCurrentIndex(1 if checked else 0)
        self.predict_prev_button.setVisible(not checked)
        self.predict_next_button.setVisible(not checked)
        self.predict_page_label.setVisible(not checked)
        if checked:
            self.refresh_predict_compact_view()

    def on_predict_compact_item_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.UserRole)
        if isinstance(index, int):
            self.current_predict_index = index
            if not self.predict_compact_checkbox.isChecked():
                self.refresh_predict_page()

    def show_previous_prediction(self) -> None:
        if self.current_predict_index > 0:
            self.current_predict_index -= 1
            self.refresh_predict_page()

    def show_next_prediction(self) -> None:
        if self.current_predict_index < len(self.predict_results) - 1:
            self.current_predict_index += 1
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
            self.refresh_predict_page()

    def set_prediction_running_state(self, running: bool) -> None:
        self.predict_run_button.setEnabled(not running)
        self.predict_select_images_button.setEnabled(not running)
        self.predict_select_folder_button.setEnabled(not running)
        self.predict_checkpoint_browse_button.setEnabled(not running)
        self.predict_model_combo.setEnabled(not running)
        self.predict_device_combo.setEnabled(not running)
        self.predict_image_size_spin.setEnabled(not running)

    def on_prediction_progress(self, processed: int, total: int) -> None:
        self.predict_progress_bar.setRange(0, max(total, 1))
        self.predict_progress_bar.setValue(processed)
        self.predict_progress_bar.setFormat(f"{processed}/{total} (%p%)")
        self.predict_status_label.setText(f"Predicting images... {processed}/{total}")

    def on_prediction_finished(self, results: list, timing: dict) -> None:
        self.predict_results = results
        self.current_predict_index = 0 if results else -1
        self.predict_compact_built = False
        self.predict_compact_loading = False
        self.predict_compact_pending_indices = []
        total_seconds = float(timing.get("total_seconds", 0.0))
        pure_seconds = float(timing.get("pure_seconds", 0.0))
        avg_pure_per_image = float(timing.get("avg_pure_per_image_seconds", 0.0))
        avg_pure_per_batch = float(timing.get("avg_pure_per_batch_seconds", 0.0))
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
        self.refresh_predict_page()

    def on_prediction_failed(self, error_message: str) -> None:
        self.predict_status_label.setText("Prediction failed.")
        self.predict_progress_bar.setValue(0)
        self.set_prediction_running_state(False)
        self.predict_worker = None
        self.predict_thread = None
        QMessageBox.critical(self, "Prediction Failed", error_message)


class PredictionWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(list, dict)
    failed = Signal(str)

    def __init__(
        self,
        *,
        image_paths: list[Path],
        checkpoint_path: Path,
        model_name: str,
        image_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.image_size = image_size
        self.device = device

    def run(self) -> None:
        try:
            total_start = time.perf_counter()
            import torch
            from pipeline.predicting import build_transform, load_model, predict_images_batch

            resolved_device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            model, class_to_idx = load_model(self.checkpoint_path, self.model_name, resolved_device)
            transform = build_transform(self.image_size)
            idx_to_class = {idx: name for name, idx in class_to_idx.items()}
            predict_batch_size = 16

            pure_start = time.perf_counter()
            batch_results = predict_images_batch(
                model,
                self.image_paths,
                transform,
                idx_to_class,
                resolved_device,
                batch_size=predict_batch_size,
                progress_callback=lambda processed, total: self.progress.emit(processed, total),
            )
            pure_seconds = time.perf_counter() - pure_start
            total_seconds = time.perf_counter() - total_start

            results: list[dict[str, str | float | bool | None]] = []
            for result in batch_results:
                resolved_image = Path(str(result["image_path"])).resolve()
                actual_label = resolved_image.parent.name if resolved_image.parent.name in class_to_idx else None
                results.append(
                    {
                        **result,
                        "actual_label": actual_label,
                        "is_correct": None if actual_label is None else result["predicted_class"] == actual_label,
                    }
                )
            num_images = len(self.image_paths)
            num_batches = (num_images + predict_batch_size - 1) // predict_batch_size if num_images > 0 else 0
            timing = {
                "total_seconds": total_seconds,
                "pure_seconds": pure_seconds,
                "avg_pure_per_image_seconds": (pure_seconds / num_images) if num_images > 0 else 0.0,
                "avg_pure_per_batch_seconds": (pure_seconds / num_batches) if num_batches > 0 else 0.0,
                "num_images": num_images,
                "num_batches": num_batches,
            }
            self.finished.emit(results, timing)
        except Exception as exc:
            self.failed.emit(str(exc))


def main() -> None:
    set_windows_app_id()
    app = QApplication(sys.argv)
    if APP_ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = TrainingLauncher()
    QTimer.singleShot(0, lambda: apply_windows_taskbar_icon(window))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
