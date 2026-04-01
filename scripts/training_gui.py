from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (
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
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from model_registry import discover_model_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RETRIEVAL_SCRIPT = PROJECT_ROOT / "scripts" / "data_retrieval.py"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training.py"
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
        self.available_models = discover_model_names()

        self._init_data_controls()
        self._init_training_controls()
        self._init_prediction_controls()
        self._build_ui()
        self.refresh_command_preview()
        self.refresh_predict_page()

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

        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        self.resume_checkbox.setChecked(False)

        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText(str(DEFAULT_CHECKPOINT_DIR))

        self.resume_browse_button = QPushButton("Browse...")
        self.resume_browse_button.clicked.connect(self.choose_resume_path)

        self.resume_clear_button = QPushButton("Clear")
        self.resume_clear_button.clicked.connect(self.clear_resume_path)

        self.data_root_label = QLabel(str(DEFAULT_DATA_ROOT))
        self.data_root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.data_root_label.setWordWrap(True)

        self.checkpoint_dir_label = QLabel(str(DEFAULT_CHECKPOINT_DIR))
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

        self.predict_run_button = QPushButton("Predict")
        self.predict_run_button.clicked.connect(self.run_predictions)

        self.predict_prev_button = QPushButton("Previous")
        self.predict_prev_button.clicked.connect(self.show_previous_prediction)

        self.predict_next_button = QPushButton("Next")
        self.predict_next_button.clicked.connect(self.show_next_prediction)

        self.predict_selected_label = QLabel("No images selected.")
        self.predict_selected_label.setWordWrap(True)

        self.predict_status_label = QLabel("Ready.")
        self.predict_status_label.setWordWrap(True)

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
        form.addRow("", self.resume_checkbox)
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
        predict_controls.addWidget(self.predict_run_button)
        predict_controls.addStretch(1)
        predict_controls.addWidget(self.predict_prev_button)
        predict_controls.addWidget(self.predict_page_label)
        predict_controls.addWidget(self.predict_next_button)
        predict_layout.addLayout(predict_controls)

        predict_layout.addWidget(self.predict_status_label)
        predict_layout.addWidget(self.predict_image_label, stretch=1)

        predict_result_group = QGroupBox("Prediction Result")
        predict_result_layout = QVBoxLayout(predict_result_group)
        predict_result_layout.addWidget(self.predict_result_label)
        predict_layout.addWidget(predict_result_group)

        tabs.addTab(training_tab, "Training")
        tabs.addTab(predict_tab, "Predicting")
        tabs.addTab(data_tab, "Data")
        tabs.setCurrentIndex(0)

        self.model_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.device_combo.currentTextChanged.connect(self.refresh_command_preview)
        self.epochs_spin.valueChanged.connect(self.refresh_command_preview)
        self.batch_size_spin.valueChanged.connect(self.refresh_command_preview)
        self.num_workers_spin.valueChanged.connect(self.refresh_command_preview)
        self.image_size_spin.valueChanged.connect(self.refresh_command_preview)
        self.lr_spin.valueChanged.connect(self.refresh_command_preview)
        self.freeze_checkbox.toggled.connect(self.refresh_command_preview)
        self.resume_checkbox.toggled.connect(self.on_resume_toggled)
        self.resume_path_edit.textChanged.connect(self.refresh_command_preview)
        self.on_resume_toggled(self.resume_checkbox.isChecked())

    def build_command(self) -> list[str]:
        command = [
            "-u",
            str(TRAINING_SCRIPT),
            "--model",
            self.model_combo.currentText(),
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--checkpoint-dir",
            str(DEFAULT_CHECKPOINT_DIR),
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
        return DEFAULT_CHECKPOINT_DIR / f"{self.predict_model_combo.currentText()}_best.pth"

    def on_predict_model_changed(self) -> None:
        current_path = self.predict_checkpoint_edit.text().strip()
        if not current_path or Path(current_path).parent == DEFAULT_CHECKPOINT_DIR:
            self.predict_checkpoint_edit.setText(str(self.default_predict_checkpoint_path()))

    def refresh_command_preview(self) -> None:
        parts = [sys.executable, *self.build_command()]
        self.command_preview.setText(" ".join(f'"{part}"' if " " in part else part for part in parts))

    def on_resume_toggled(self, checked: bool) -> None:
        self.resume_path_edit.setEnabled(checked)
        self.resume_browse_button.setEnabled(checked)
        self.resume_clear_button.setEnabled(checked)
        self.refresh_command_preview()

    def choose_resume_path(self) -> None:
        start_dir = self._resolve_dialog_dir(self.resume_path_edit.text().strip(), DEFAULT_CHECKPOINT_DIR)
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
        self.data_progress_label.setText("Data task started...")

    def on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.set_running_state(False)
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
            self.data_progress_label.setText("Dataset task finished successfully.")
            self.data_progress_bar.setValue(100)
        else:
            self.data_status_label.setText(f"Finished ({exit_code})")
            self.data_progress_label.setText(f"Dataset task stopped with exit code {exit_code} ({status_text}).")
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
        self.data_progress_bar.setRange(0, 100)
        self.data_progress_bar.setValue(0)
        self.data_progress_label.setText(f"Process error: {error}")
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
        start_dir = str(DEFAULT_DATA_ROOT / "images")
        selected_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images to Predict",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)",
        )
        if selected_paths:
            self.predict_image_paths = [Path(path) for path in selected_paths]
            self.predict_results = []
            self.current_predict_index = -1
            self.refresh_predict_page()

    def run_predictions(self) -> None:
        checkpoint_path = Path(self.predict_checkpoint_edit.text().strip()).expanduser()
        if not checkpoint_path.is_file():
            QMessageBox.warning(self, "Invalid Checkpoint", f"Checkpoint file does not exist:\n{checkpoint_path}")
            return
        if not self.predict_image_paths:
            QMessageBox.warning(self, "No Images Selected", "Select one or more images before predicting.")
            return

        import torch
        from predicting import build_transform, load_model, predict_image

        device = self.predict_device_combo.currentText()
        resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.predict_status_label.setText("Loading model and running predictions...")
        QApplication.processEvents()

        try:
            model_name = self.predict_model_combo.currentText()
            model, class_to_idx = load_model(checkpoint_path.resolve(), model_name, resolved_device)
            transform = build_transform(self.predict_image_size_spin.value())
            idx_to_class = {idx: name for name, idx in class_to_idx.items()}
            results: list[dict[str, str | float | bool | None]] = []
            for image_path in self.predict_image_paths:
                resolved_image = image_path.expanduser().resolve()
                result = predict_image(model, resolved_image, transform, idx_to_class, resolved_device)
                actual_label = resolved_image.parent.name if resolved_image.parent.name in class_to_idx else None
                results.append(
                    {
                        **result,
                        "actual_label": actual_label,
                        "is_correct": None if actual_label is None else result["predicted_class"] == actual_label,
                    }
                )
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Failed", str(exc))
            self.predict_status_label.setText("Prediction failed.")
            return

        self.predict_results = results
        self.current_predict_index = 0 if results else -1
        self.predict_status_label.setText(f"Predicted {len(results)} image(s).")
        self.refresh_predict_page()

    def refresh_predict_page(self) -> None:
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

        if not has_results:
            self.predict_image_label.setPixmap(QPixmap())
            self.predict_image_label.setText("Select images and click Predict.")
            self.predict_result_label.setText("Prediction result will appear here.")
            return

        result = self.predict_results[self.current_predict_index]
        image_path = Path(str(result["image_path"]))
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.predict_image_label.setPixmap(QPixmap())
            self.predict_image_label.setText(f"Could not load image:\n{image_path}")
        else:
            scaled = pixmap.scaled(
                self.predict_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.predict_image_label.setText("")
            self.predict_image_label.setPixmap(scaled)

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
