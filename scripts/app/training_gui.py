from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QProcess, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen, QPixmap, QTextCursor
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
APP_ID = "MLWorkbench.TrainingLauncher"
WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_DEFAULTSIZE = 0x00000040
NEW_CHECKPOINT_NAME_LABEL = "New checkpoint name..."
RUN_LOG_DIRNAME = "_run_logs"
APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #17191d;
    color: #eef2f7;
    font-family: "Segoe UI";
    font-size: 10.5pt;
}
QTabWidget::pane {
    border: 1px solid #2e3642;
    border-radius: 14px;
    background: #1d2128;
    top: -1px;
}
QTabBar::tab {
    background: #20252d;
    color: #aeb8c6;
    border: 1px solid #303846;
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-width: 90px;
}
QTabBar::tab:selected {
    background: #2c6df2;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background: #27303b;
    color: #edf3ff;
}
QGroupBox {
    background: #1f242c;
    border: 1px solid #313a47;
    border-radius: 14px;
    margin-top: 14px;
    padding: 12px 14px 14px 14px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #f8fbff;
}
QLabel {
    color: #e8edf5;
}
QLabel[muted="true"] {
    color: #9ca8b8;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {
    background: #14181e;
    color: #eff4fb;
    border: 1px solid #364152;
    border-radius: 10px;
    padding: 7px 10px;
    selection-background-color: #2c6df2;
    selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QListWidget:focus {
    border: 1px solid #4e8cff;
}
QComboBox::drop-down, QSpinBox::down-button, QSpinBox::up-button, QDoubleSpinBox::down-button, QDoubleSpinBox::up-button {
    border: none;
    width: 22px;
}
QPushButton {
    background: #2c6df2;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #3b7bfd;
}
QPushButton:pressed {
    background: #2258c5;
}
QPushButton:disabled {
    background: #2a3039;
    color: #748092;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #485364;
    background: #14181e;
}
QCheckBox::indicator:checked {
    background: #2c6df2;
    border: 1px solid #2c6df2;
}
QProgressBar {
    border: 1px solid #364152;
    border-radius: 9px;
    background: #12161b;
    text-align: center;
    min-height: 18px;
    color: #f5f8ff;
}
QProgressBar::chunk {
    background: #2c6df2;
    border-radius: 8px;
}
QScrollBar:vertical {
    background: #171b21;
    width: 12px;
    margin: 8px 0 8px 0;
}
QScrollBar::handle:vertical {
    background: #3a4454;
    min-height: 28px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QListWidget::item {
    border-radius: 10px;
    padding: 6px;
}
QListWidget::item:selected {
    background: #243a64;
    border: 1px solid #4e8cff;
}
QPlainTextEdit {
    background: #11151a;
    font-family: "Cascadia Code";
    font-size: 10pt;
}
QLabel#ImagePreview {
    border: 1px solid #364152;
    border-radius: 16px;
    background: #11151a;
    color: #93a0b2;
}
QLabel#SectionStatus {
    background: #202832;
    border: 1px solid #354050;
    border-radius: 10px;
    padding: 6px 10px;
    color: #f0f4fa;
    font-weight: 600;
}
"""


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


class LogPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.plot_title = "Run Plot"
        self.x_label = "Epoch"
        self.y_label = "Value"
        self.note = ""
        self.series: list[dict] = []

    def set_plot(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        series: list[dict],
        note: str = "",
    ) -> None:
        self.plot_title = title
        self.x_label = x_label
        self.y_label = y_label
        self.series = series
        self.note = note
        self.update()

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

        for tick in range(5):
            fraction = tick / 4 if 4 > 0 else 0
            y = plot_rect.bottom() - fraction * plot_rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
            tick_value = y_min + fraction * (y_max - y_min)
            painter.setPen(label_pen)
            painter.drawText(QRectF(plot_rect.left() - 52, y - 10, 46, 20), Qt.AlignRight | Qt.AlignVCenter, f"{tick_value:.3g}")

        x_tick_count = min(6, max(2, int(x_max - x_min) + 1))
        for tick in range(x_tick_count):
            fraction = tick / (x_tick_count - 1) if x_tick_count > 1 else 0
            x = plot_rect.left() + fraction * plot_rect.width()
            painter.setPen(grid_pen)
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
            tick_value = x_min + fraction * (x_max - x_min)
            painter.setPen(label_pen)
            painter.drawText(QRectF(x - 20, plot_rect.bottom() + 6, 40, 18), Qt.AlignHCenter | Qt.AlignTop, f"{tick_value:.0f}")

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

        for series in self.series:
            points = [(float(x), float(y)) for x, y in series.get("points", [])]
            if not points:
                continue
            color = QColor(series.get("color", "#4e8cff"))
            pen = QPen(color, 2.2)
            painter.setPen(pen)
            mapped_points = [map_point(x_value, y_value) for x_value, y_value in points]
            for point_index in range(len(mapped_points) - 1):
                painter.drawLine(mapped_points[point_index], mapped_points[point_index + 1])
            painter.setBrush(color)
            for point in mapped_points:
                painter.drawEllipse(point, 3.2, 3.2)

        painter.setPen(QColor("#aeb8c6"))
        painter.drawText(QRectF(plot_rect.left(), plot_rect.bottom() + 24, plot_rect.width(), 20), Qt.AlignCenter, self.x_label)

        painter.save()
        painter.translate(plot_rect.left() - 48, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_rect.height() / 2, -16, plot_rect.height(), 20), Qt.AlignCenter, self.y_label)
        painter.restore()

        metrics = QFontMetrics(painter.font())
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


class TrainingLauncher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
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
        self._stop_request_path: Path | None = None

        self._init_data_controls()
        self._init_training_controls()
        self._init_prediction_controls()
        self._init_log_controls()
        self._build_ui()
        self.apply_visual_design()
        self.refresh_command_preview()
        self.refresh_predict_page()
        self.on_predict_compact_toggled(self.predict_compact_checkbox.isChecked())
        self.refresh_training_log_runs()

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

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("SectionStatus")

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
        self.predict_selected_label.setProperty("muted", True)

        self.predict_status_label = QLabel("Ready.")
        self.predict_status_label.setWordWrap(True)
        self.predict_status_label.setObjectName("SectionStatus")

        self.predict_progress_bar = QProgressBar()
        self.predict_progress_bar.setRange(0, 100)
        self.predict_progress_bar.setValue(0)
        self.predict_progress_bar.setFormat("%p%")

        self.predict_page_label = QLabel("0 / 0")
        self.predict_page_label.setProperty("muted", True)

        self.predict_image_label = QLabel("Select images and click Predict.")
        self.predict_image_label.setObjectName("ImagePreview")
        self.predict_image_label.setAlignment(Qt.AlignCenter)
        self.predict_image_label.setMinimumHeight(420)

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

    def _init_log_controls(self) -> None:
        self.training_log_runs: list[dict] = []
        self.training_log_run_combo = QComboBox()
        self.training_log_run_combo.currentIndexChanged.connect(self.on_training_log_run_changed)

        self.training_log_stage_combo = QComboBox()
        self.training_log_stage_combo.addItems(["Summary", "Compare", "Train", "Val", "Test"])
        self.training_log_stage_combo.currentIndexChanged.connect(self.refresh_training_log_view)

        self.training_log_refresh_button = QPushButton("Refresh Logs")
        self.training_log_refresh_button.clicked.connect(self.refresh_training_log_runs)

        self.training_log_status_label = QLabel("No training logs loaded.")
        self.training_log_status_label.setWordWrap(True)
        self.training_log_status_label.setObjectName("SectionStatus")

        self.training_plot_mode_combo = QComboBox()
        self.training_plot_mode_combo.addItems(
            ["Selected Run Accuracy", "Selected Run Timing", "Compare Accuracy", "Compare Timing"]
        )
        self.training_plot_mode_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_stage_combo = QComboBox()
        self.training_plot_stage_combo.addItems(["All / Auto", "Train", "Val", "Test"])
        self.training_plot_stage_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_timing_combo = QComboBox()
        self.training_plot_timing_combo.addItems(["Total Time", "Pure Time", "Avg Pure / Batch"])
        self.training_plot_timing_combo.currentIndexChanged.connect(self.refresh_training_log_plot)

        self.training_plot_widget = LogPlotWidget()

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

        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)

        logs_controls_group = QGroupBox("Run Selection")
        logs_controls_form = QFormLayout(logs_controls_group)
        logs_controls_form.addRow("Run", self.training_log_run_combo)
        logs_controls_form.addRow("View", self.training_log_stage_combo)
        logs_layout.addWidget(logs_controls_group)

        logs_plot_group = QGroupBox("Plot")
        logs_plot_form = QFormLayout(logs_plot_group)
        logs_plot_form.addRow("Mode", self.training_plot_mode_combo)
        logs_plot_form.addRow("Stage", self.training_plot_stage_combo)
        logs_plot_form.addRow("Timing Metric", self.training_plot_timing_combo)
        logs_layout.addWidget(logs_plot_group)

        logs_actions = QHBoxLayout()
        logs_actions.addWidget(self.training_log_refresh_button)
        logs_actions.addWidget(self.training_log_status_label, stretch=1)
        logs_layout.addLayout(logs_actions)

        logs_plot_canvas_group = QGroupBox("Run Plot")
        logs_plot_canvas_layout = QVBoxLayout(logs_plot_canvas_group)
        logs_plot_canvas_layout.addWidget(self.training_plot_widget)
        logs_layout.addWidget(logs_plot_canvas_group)

        logs_output_group = QGroupBox("Training Run Details")
        logs_output_layout = QVBoxLayout(logs_output_group)
        logs_output_layout.addWidget(self.training_log_text)
        logs_layout.addWidget(logs_output_group, stretch=1)

        self.tabs.addTab(training_tab, "Training")
        self.tabs.addTab(predict_tab, "Predicting")
        self.tabs.addTab(data_tab, "Data")
        self.tabs.addTab(logs_tab, "Logs")
        self.tabs.setCurrentIndex(0)

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

    def apply_visual_design(self) -> None:
        self.setStyleSheet(APP_STYLESHEET)
        self._set_layout_metrics(self.centralWidget().layout() if self.centralWidget() is not None else None)

    def _set_layout_metrics(self, layout) -> None:
        if layout is None:
            return
        if isinstance(layout, QFormLayout):
            layout.setHorizontalSpacing(16)
            layout.setVerticalSpacing(12)
        else:
            layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child_layout = item.layout()
            if child_layout is not None:
                self._set_layout_metrics(child_layout)
            child_widget = item.widget()
            if child_widget is not None and child_widget.layout() is not None:
                self._set_layout_metrics(child_widget.layout())

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
            "--stop-file",
            str(self.stop_request_path_for(checkpoint_dir)),
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
        checkpoint_dir = self.selected_checkpoint_dir()
        self._stop_request_path = self.stop_request_path_for(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.clear_stop_request_file()

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

    def load_training_log_files(self) -> list[dict]:
        log_files = sorted(
            DEFAULT_CHECKPOINT_DIR.glob(f"**/{RUN_LOG_DIRNAME}/*.json"),
            key=lambda path: path.stat().st_mtime if path.is_file() else 0.0,
            reverse=True,
        )
        loaded: list[dict] = []
        for path in log_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                data["_log_path"] = str(path)
                loaded.append(data)
            except Exception:
                continue
        return loaded

    def refresh_training_log_runs(self) -> None:
        self.training_log_runs = self.load_training_log_files()
        previous_run_id = self.training_log_run_combo.currentData()

        self.training_log_run_combo.blockSignals(True)
        self.training_log_run_combo.clear()
        if not self.training_log_runs:
            self.training_log_run_combo.addItem("No logs found", None)
            self.training_log_run_combo.blockSignals(False)
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

        selected_index = 0
        for index, run in enumerate(self.training_log_runs):
            run_id = str(run.get("run_id", "unknown"))
            args = (run.get("args") or {}) if isinstance(run.get("args"), dict) else {}
            model_name = str(args.get("model", "unknown"))
            status = str(run.get("status", "unknown"))
            started = str(run.get("start_time_utc", "unknown"))
            best_eval = self.format_metric(self.infer_best_eval_acc(run))
            final_test = self.format_metric(((run.get("summary") or {}) if isinstance(run.get("summary"), dict) else {}).get("final_test_acc"))
            label = f"{started} | {model_name} | {status} | best={best_eval} | final_test={final_test} | {run_id}"
            self.training_log_run_combo.addItem(label, run_id)
            if previous_run_id is not None and run_id == previous_run_id:
                selected_index = index
        self.training_log_run_combo.setCurrentIndex(selected_index)
        self.training_log_run_combo.blockSignals(False)
        self.on_training_log_run_changed()

    def on_training_log_run_changed(self) -> None:
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
        status = str(run.get("status", "unknown"))
        if status == "running":
            return "incomplete_or_interrupted"
        return status

    @staticmethod
    def safe_float(value) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def format_metric(value) -> str:
        numeric = TrainingLauncher.safe_float(value)
        return f"{numeric:.4f}" if numeric is not None else "-"

    @staticmethod
    def format_ratio(numerator, denominator) -> str:
        left = int(numerator) if isinstance(numerator, (int, float)) else 0
        right = int(denominator) if isinstance(denominator, (int, float)) else 0
        return f"{left}/{right}" if right > 0 else str(left)

    @staticmethod
    def infer_last_completed_epoch(run: dict) -> int:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        if isinstance(summary.get("last_completed_epoch"), (int, float)):
            return int(summary["last_completed_epoch"])
        epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
        return len(epochs)

    @staticmethod
    def infer_eval_name(run: dict) -> str:
        dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
        if isinstance(dataset.get("eval_name"), str):
            return str(dataset["eval_name"])
        expected = run.get("expected") if isinstance(run.get("expected"), dict) else {}
        if "val_batches_per_epoch" in expected:
            return "val"
        if "test_batches_per_epoch" in expected:
            return "test"
        return "-"

    @staticmethod
    def infer_best_eval_acc(run: dict) -> float | None:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        best = summary.get("best_eval_acc") if isinstance(summary, dict) else None
        if isinstance(best, (int, float)):
            return float(best)
        epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
        best_value: float | None = None
        for epoch_record in epochs:
            if not isinstance(epoch_record, dict):
                continue
            for key, stage in epoch_record.items():
                if key in {"epoch", "lr", "best_eval_acc_after_epoch", "is_best_checkpoint"}:
                    continue
                if isinstance(stage, dict) and isinstance(stage.get("acc"), (int, float)):
                    value = float(stage["acc"])
                    if best_value is None or value > best_value:
                        best_value = value
        return best_value

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
        timing = stage.get("timing", {}) if isinstance(stage, dict) else {}
        if not isinstance(timing, dict):
            return None
        if timing_metric == "total":
            return float(timing["total_seconds"]) if isinstance(timing.get("total_seconds"), (int, float)) else None
        if timing_metric == "pure":
            return float(timing["pure_seconds"]) if isinstance(timing.get("pure_seconds"), (int, float)) else None
        pure_seconds = timing.get("pure_seconds")
        batches = timing.get("batches")
        if isinstance(pure_seconds, (int, float)) and isinstance(batches, (int, float)) and float(batches) > 0:
            return float(pure_seconds) / float(batches)
        return None

    def extract_stage_points(self, run: dict, stage_name: str, value_kind: str, timing_metric: str | None = None) -> list[tuple[float, float]]:
        epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
        stage_key = stage_name.lower()
        points: list[tuple[float, float]] = []
        for epoch_record in epochs:
            if not isinstance(epoch_record, dict):
                continue
            epoch_index = epoch_record.get("epoch")
            stage = epoch_record.get(stage_key)
            if not isinstance(epoch_index, (int, float)) or not isinstance(stage, dict):
                continue
            if value_kind == "accuracy":
                value = float(stage["acc"]) if isinstance(stage.get("acc"), (int, float)) else None
            else:
                value = self.timing_value_from_stage(stage, timing_metric or "total")
            if value is not None:
                points.append((float(epoch_index), float(value)))

        if stage_key == "test" and not points:
            final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else None
            if isinstance(final_test, dict):
                epoch_index = float(self.infer_last_completed_epoch(run))
                if value_kind == "accuracy":
                    value = final_test.get("acc")
                    if isinstance(value, (int, float)):
                        points.append((epoch_index, float(value)))
                else:
                    value = self.timing_value_from_stage(final_test, timing_metric or "total")
                    if value is not None:
                        points.append((epoch_index, value))
        return points

    def current_selected_run(self) -> dict | None:
        if not self.training_log_runs:
            return None
        run_id = self.training_log_run_combo.currentData()
        for run in self.training_log_runs:
            if str(run.get("run_id", "")) == str(run_id):
                return run
        return self.training_log_runs[0] if self.training_log_runs else None

    def run_display_name(self, run: dict, include_stage: str | None = None) -> str:
        args = run.get("args") if isinstance(run.get("args"), dict) else {}
        started = str(run.get("start_time_utc", "-"))[:10]
        model = str(args.get("model", "run"))
        checkpoint_name = Path(str(args.get("checkpoint_dir", "-"))).name
        base = f"{started} {model} ({checkpoint_name})"
        return f"{base} [{include_stage}]" if include_stage else base

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
        return {
            "title": "Selected Run Accuracy" if value_kind == "accuracy" else "Selected Run Timing",
            "x_label": "Epoch",
            "y_label": "Accuracy" if value_kind == "accuracy" else timing_label,
            "series": series,
            "note": "All available stage curves are shown together." if stage_choice.startswith("all") else "",
        }

    def build_compare_plot(self, *, value_kind: str, timing_metric: str) -> dict:
        stage_choice = self.training_plot_stage_combo.currentText().strip().lower()
        series: list[dict] = []
        for index, run in enumerate(self.training_log_runs):
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
        return {
            "title": "Compare Accuracy Across Runs" if value_kind == "accuracy" else "Compare Timing Across Runs",
            "x_label": "Epoch",
            "y_label": "Accuracy" if value_kind == "accuracy" else timing_label,
            "series": series,
            "note": note,
        }

    def refresh_training_log_plot(self) -> None:
        mode = self.training_plot_mode_combo.currentText().strip().lower()
        timing_metric_label = self.training_plot_timing_combo.currentText().strip().lower()
        timing_metric = "avg" if "avg" in timing_metric_label else ("pure" if "pure" in timing_metric_label else "total")

        self.training_plot_timing_combo.setEnabled("timing" in mode)
        if "compare" in mode:
            plot = self.build_compare_plot(
                value_kind="accuracy" if "accuracy" in mode else "timing",
                timing_metric=timing_metric,
            )
        else:
            selected_run = self.current_selected_run()
            if selected_run is None:
                plot = {
                    "title": "Run Plot",
                    "x_label": "Epoch",
                    "y_label": "Value",
                    "series": [],
                    "note": "No run selected.",
                }
            else:
                plot = self.build_selected_run_plot(
                    selected_run,
                    value_kind="accuracy" if "accuracy" in mode else "timing",
                    timing_metric=timing_metric,
                )
        self.training_plot_widget.set_plot(**plot)

    def render_compare_runs(self) -> str:
        if not self.training_log_runs:
            return "No run logs available."

        header = (
            f"{'Started':<22} {'Model':<12} {'Status':<14} {'Progress':<9} "
            f"{'BestEval':<10} {'FinalTest':<10} {'Eval':<6} {'Batch':<6} {'LR':<10} {'Checkpoint'}"
        )
        separator = "-" * len(header)
        lines = [header, separator]
        for run in self.training_log_runs:
            args = run.get("args") if isinstance(run.get("args"), dict) else {}
            summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
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
        return "\n".join(lines)

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
        run_id = self.training_log_run_combo.currentData()
        selected_run = None
        for run in self.training_log_runs:
            if str(run.get("run_id", "")) == str(run_id):
                selected_run = run
                break
        if selected_run is None:
            selected_run = self.training_log_runs[0]

        status_text = f"Loaded log: {selected_run.get('_log_path', '-')}"
        self.training_log_status_label.setText(status_text)

        selected_view = self.training_log_stage_combo.currentText().strip().lower()
        if selected_view == "compare":
            self.training_log_text.setPlainText(self.render_compare_runs())
            self.refresh_training_log_plot()
            return
        if selected_view == "summary":
            self.training_log_text.setPlainText(self.render_run_summary(selected_run))
            self.refresh_training_log_plot()
            return
        self.training_log_text.setPlainText(self.render_stage_epochs(selected_run, selected_view))
        self.refresh_training_log_plot()

    def stop_training(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        if self._stop_request_path is None:
            self._stop_request_path = self.stop_request_path_for()
        self._stop_request_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_request_path.write_text("stop requested\n", encoding="utf-8")
        self.append_output("\nGraceful stop requested. Waiting for the current step to finish...\n")
        self.status_label.setText("Stopping")
        self.progress_label.setText("Graceful stop requested. Training will stop after the current batch.")
        self.stop_button.setEnabled(False)

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
    window.showMaximized()
    QTimer.singleShot(0, lambda: apply_windows_taskbar_icon(window))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
