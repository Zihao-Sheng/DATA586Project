from __future__ import annotations


THEMES: dict[str, dict[str, str]] = {
    "graphite_metal": {
        "display_name": "Graphite Metal",
        "font_family": "Segoe UI",
        "mono_font_family": "Cascadia Code",
        "window_bg": "#171c23",
        "base_bg": "#1c222b",
        "panel_bg": "#252d38",
        "panel_alt_bg": "#222a34",
        "input_bg": "#2a323d",
        "input_alt_bg": "#262e39",
        "code_bg": "#161c24",
        "scroll_track": "#202732",
        "scroll_handle": "#505d6f",
        "border": "#3a4552",
        "border_strong": "#556376",
        "text": "#e2e8ef",
        "text_muted": "#a5b0bf",
        "text_inverse": "#f6f9fc",
        "text_dark": "#1f2b39",
        "accent": "#5f84b4",
        "accent_hover": "#6f93c1",
        "accent_pressed": "#4f739f",
        "accent_soft": "#2a3a4f",
        "accent_soft_border": "#5879a4",
        "selection_bg": "#37506f",
        "selection_text": "#f4f8fd",
        "disabled_bg": "#2a3038",
        "disabled_text": "#7d8795",
        "image_preview_bg": "#171d26",
        "status_bg": "#24303d",
        "status_border": "#46566a",
        "warning": "#d0ab73",
        "error": "#d2838b",
        "checkbox_bg": "#252d37",
    },
    "soft_dark": {
        "display_name": "Soft Dark",
        "font_family": "Segoe UI",
        "mono_font_family": "Cascadia Code",
        "window_bg": "#20252d",
        "base_bg": "#232831",
        "panel_bg": "#2a3038",
        "panel_alt_bg": "#262c34",
        "input_bg": "#313844",
        "input_alt_bg": "#2b323d",
        "code_bg": "#1b2027",
        "scroll_track": "#242a33",
        "scroll_handle": "#485362",
        "border": "#3a4350",
        "border_strong": "#4f5a68",
        "text": "#e6eaf0",
        "text_muted": "#aeb7c2",
        "text_inverse": "#ffffff",
        "text_dark": "#233040",
        "accent": "#5b8def",
        "accent_hover": "#6d9bf4",
        "accent_pressed": "#4978d2",
        "accent_soft": "#2a3f60",
        "accent_soft_border": "#4c74b5",
        "selection_bg": "#355180",
        "selection_text": "#f7faff",
        "disabled_bg": "#2a2f37",
        "disabled_text": "#7e8896",
        "image_preview_bg": "#1e242c",
        "status_bg": "#27313d",
        "status_border": "#445161",
        "warning": "#d7a66a",
        "error": "#d98181",
        "checkbox_bg": "#242a32",
    },
    "slate_dark": {
        "display_name": "Slate Dark",
        "font_family": "Segoe UI",
        "mono_font_family": "Cascadia Code",
        "window_bg": "#1b222b",
        "base_bg": "#1f2832",
        "panel_bg": "#273440",
        "panel_alt_bg": "#24303b",
        "input_bg": "#2f3d4b",
        "input_alt_bg": "#2b3947",
        "code_bg": "#182028",
        "scroll_track": "#202933",
        "scroll_handle": "#496173",
        "border": "#3a4b5a",
        "border_strong": "#536779",
        "text": "#e3eaf2",
        "text_muted": "#a6b3c2",
        "text_inverse": "#fafdff",
        "text_dark": "#203244",
        "accent": "#4f9bcf",
        "accent_hover": "#61a9da",
        "accent_pressed": "#3f86b8",
        "accent_soft": "#234253",
        "accent_soft_border": "#4d82a3",
        "selection_bg": "#2f5370",
        "selection_text": "#f7fbff",
        "disabled_bg": "#28323c",
        "disabled_text": "#7d8c9c",
        "image_preview_bg": "#192029",
        "status_bg": "#22303c",
        "status_border": "#476174",
        "warning": "#d2ac72",
        "error": "#d5828a",
        "checkbox_bg": "#24303a",
    },
    "warm_gray_dark": {
        "display_name": "Warm Gray Dark",
        "font_family": "Segoe UI",
        "mono_font_family": "Cascadia Code",
        "window_bg": "#23211f",
        "base_bg": "#282522",
        "panel_bg": "#312d2a",
        "panel_alt_bg": "#2d2a27",
        "input_bg": "#3a3531",
        "input_alt_bg": "#36322f",
        "code_bg": "#211e1b",
        "scroll_track": "#2a2623",
        "scroll_handle": "#5a524b",
        "border": "#4a433d",
        "border_strong": "#625851",
        "text": "#ece7e1",
        "text_muted": "#b9b1a8",
        "text_inverse": "#fffdfa",
        "text_dark": "#352d27",
        "accent": "#c28b5c",
        "accent_hover": "#cf9a6d",
        "accent_pressed": "#ad7649",
        "accent_soft": "#4f3c2c",
        "accent_soft_border": "#8f6b4d",
        "selection_bg": "#5a4532",
        "selection_text": "#fff9f2",
        "disabled_bg": "#34302c",
        "disabled_text": "#8c837b",
        "image_preview_bg": "#221f1c",
        "status_bg": "#36312d",
        "status_border": "#5d554d",
        "warning": "#d7ae74",
        "error": "#d38b85",
        "checkbox_bg": "#2f2b28",
    },
    "neutral_light": {
        "display_name": "Neutral Light",
        "font_family": "Segoe UI",
        "mono_font_family": "Cascadia Code",
        "window_bg": "#f2f4f7",
        "base_bg": "#eef1f5",
        "panel_bg": "#fafbfc",
        "panel_alt_bg": "#ffffff",
        "input_bg": "#ffffff",
        "input_alt_bg": "#f8fafc",
        "code_bg": "#f5f7fa",
        "scroll_track": "#e1e6ee",
        "scroll_handle": "#bac4d1",
        "border": "#ced6e0",
        "border_strong": "#b7c1cd",
        "text": "#1f2933",
        "text_muted": "#5b6673",
        "text_inverse": "#ffffff",
        "text_dark": "#1f2933",
        "accent": "#4f7fd8",
        "accent_hover": "#6090e5",
        "accent_pressed": "#3f6ec7",
        "accent_soft": "#dce7fb",
        "accent_soft_border": "#8cabde",
        "selection_bg": "#cfe0ff",
        "selection_text": "#17222d",
        "disabled_bg": "#edf1f5",
        "disabled_text": "#8b95a1",
        "image_preview_bg": "#edf2f7",
        "status_bg": "#e9eff7",
        "status_border": "#c6d2df",
        "warning": "#9a6b32",
        "error": "#b24f56",
        "checkbox_bg": "#ffffff",
    },
    "legacy_dark": {
        "display_name": "Legacy Dark",
        "font_family": "Segoe UI",
        "mono_font_family": "Cascadia Code",
        "window_bg": "#17191d",
        "base_bg": "#1d2128",
        "panel_bg": "#1f242c",
        "panel_alt_bg": "#20252d",
        "input_bg": "#14181e",
        "input_alt_bg": "#161b22",
        "code_bg": "#11151a",
        "scroll_track": "#171b21",
        "scroll_handle": "#3a4454",
        "border": "#364152",
        "border_strong": "#485364",
        "text": "#eef2f7",
        "text_muted": "#9ca8b8",
        "text_inverse": "#ffffff",
        "text_dark": "#1f2e42",
        "accent": "#2c6df2",
        "accent_hover": "#3b7bfd",
        "accent_pressed": "#2258c5",
        "accent_soft": "#22314a",
        "accent_soft_border": "#365c9a",
        "selection_bg": "#2c6df2",
        "selection_text": "#ffffff",
        "disabled_bg": "#2a3039",
        "disabled_text": "#748092",
        "image_preview_bg": "#11151a",
        "status_bg": "#202832",
        "status_border": "#354050",
        "warning": "#d8aa67",
        "error": "#d57d84",
        "checkbox_bg": "#14181e",
    },
}


DEFAULT_THEME_KEY = "graphite_metal"


def theme_display_names() -> list[tuple[str, str]]:
    return [(key, values["display_name"]) for key, values in THEMES.items()]


def get_theme(theme_key: str | None) -> dict[str, str]:
    key = theme_key if theme_key in THEMES else DEFAULT_THEME_KEY
    return THEMES[key]


def build_stylesheet(theme_key: str | None) -> str:
    theme = get_theme(theme_key)
    return """
QMainWindow {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_alt_bg}, stop:0.2 {window_bg}, stop:1 {window_bg});
    color: {text};
    font-family: "{font_family}";
    font-size: 10.25pt;
}}
QWidget {{
    background: {base_bg};
    color: {text};
    font-family: "{font_family}";
    font-size: 10.25pt;
}}
QToolTip {{
    background: {panel_bg};
    color: {text};
    border: 1px solid {border_strong};
    padding: 8px 10px;
    border-radius: 8px;
}}
QMenu {{
    background: {panel_bg};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{
    padding: 7px 12px;
    border-radius: 6px;
}}
QMenu::item:selected {{
    background: {accent_soft};
    color: {selection_text};
}}
QDockWidget {{
    background: {base_bg};
    color: {text};
}}
QDockWidget::title {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    color: {text};
    border-bottom: 1px solid {border_strong};
    padding: 8px 10px;
    text-align: left;
}}
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 12px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {base_bg}, stop:1 {panel_alt_bg});
    top: -1px;
}}
QTabBar::tab {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    color: {text_muted};
    border: 1px solid {border};
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    min-width: 90px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent_hover}, stop:1 {accent});
    color: {text_inverse};
    border-color: {accent};
}}
QTabBar::tab:hover:!selected {{
    background: {panel_alt_bg};
    color: {text};
}}
QGroupBox {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    border: 1px solid {border};
    border-radius: 12px;
    margin-top: 10px;
    padding: 11px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {text_muted};
    background: transparent;
}}
QLabel {{
    background: transparent;
    color: {text};
}}
QLabel[sectionTitle="true"] {{
    font-weight: 700;
}}
QLabel[muted="true"] {{
    color: {text_muted};
}}
QLabel[readonlyDisplay="true"] {{
    background: {input_alt_bg};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 8px 10px;
    color: {text};
}}
QLabel[sectionHint="true"] {{
    color: {text_muted};
    font-size: 9.6pt;
    letter-spacing: 0.02em;
}}
QLabel[detailText="true"] {{
    padding-top: 2px;
}}
QLabel[codeblock="true"] {{
    background: {code_bg};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 12px 14px;
    color: {text};
    font-family: "{mono_font_family}";
    font-size: 10pt;
}}
QLabel[statusType="warning"] {{
    color: {warning};
}}
QLabel[statusType="error"] {{
    color: {error};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QListWidget, QTreeWidget, QTableWidget, QTreeView, QListView {{
    background: {input_bg};
    color: {text};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 5px 10px;
    selection-background-color: {selection_bg};
    selection-color: {selection_text};
}}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 30px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus, QListWidget:focus, QTreeWidget:focus, QTableWidget:focus, QTreeView:focus, QListView:focus {{
    border: 1px solid {accent};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QListWidget:disabled, QTreeWidget:disabled, QTableWidget:disabled {{
    background: {disabled_bg};
    color: {disabled_text};
    border-color: {border};
}}
QComboBox::drop-down, QSpinBox::down-button, QSpinBox::up-button, QDoubleSpinBox::down-button, QDoubleSpinBox::up-button {{
    border: none;
    width: 22px;
}}
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent_hover}, stop:1 {accent});
    color: {text_inverse};
    border: 1px solid {accent};
    border-radius: 10px;
    padding: 7px 13px;
    min-height: 18px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent}, stop:1 {accent_hover});
    border-color: {accent_hover};
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent_pressed}, stop:1 {accent});
    border-color: {accent_pressed};
}}
QPushButton:checked {{
    background: {accent_pressed};
    border-color: {accent_pressed};
}}
QPushButton:disabled {{
    background: {disabled_bg};
    color: {disabled_text};
    border-color: {border};
}}
QToolButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    color: {text};
    border: 1px solid {border};
    border-radius: 9px;
    padding: 5px 10px;
}}
QToolButton:hover {{
    background: {panel_bg};
    border-color: {border_strong};
}}
QToolButton:checked {{
    background: {accent_soft};
    border-color: {accent_soft_border};
}}
QCheckBox, QRadioButton {{
    background: transparent;
    spacing: 8px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid {border_strong};
    background: {checkbox_bg};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {accent};
    border: 1px solid {accent};
}}
QProgressBar {{
    border: 1px solid {border};
    border-radius: 8px;
    background: {input_alt_bg};
    text-align: center;
    min-height: 18px;
    color: {text};
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 7px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {scroll_track};
    width: 12px;
    margin: 8px 0 8px 0;
    border-radius: 6px;
}}
QScrollBar::handle:vertical {{
    background: {scroll_handle};
    min-height: 28px;
    border-radius: 6px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {scroll_track};
    height: 12px;
    margin: 0 8px 0 8px;
    border-radius: 6px;
}}
QScrollBar::handle:horizontal {{
    background: {scroll_handle};
    min-width: 28px;
    border-radius: 6px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QListWidget::item, QTreeWidget::item, QTableWidget::item {{
    border-radius: 8px;
    padding: 5px 8px;
    margin: 2px 4px;
}}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background: {accent_soft};
    border: 1px solid {accent_soft_border};
    color: {selection_text};
}}
QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected, QTableWidget::item:hover:!selected {{
    background: {panel_alt_bg};
}}
QPlainTextEdit, QTextEdit {{
    background: {code_bg};
    font-family: "{mono_font_family}";
    font-size: 10pt;
}}
QTableCornerButton::section, QHeaderView::section {{
    background: {panel_alt_bg};
    color: {text_muted};
    border: 1px solid {border};
    border-left: none;
    border-top: none;
    padding: 6px 8px;
    font-weight: 600;
}}
QHeaderView::section:first {{
    border-left: 1px solid {border};
}}
QLabel#ImagePreview {{
    border: 1px solid {border};
    border-radius: 14px;
    background: {image_preview_bg};
    color: {text_muted};
}}
QLabel#SectionStatus {{
    background: {status_bg};
    border: 1px solid {status_border};
    border-radius: 10px;
    padding: 6px 10px;
    color: {text};
    font-weight: 600;
}}
QDialogButtonBox QPushButton {{
    min-width: 92px;
}}
QSplitter::handle {{
    background: {border};
    margin: 0 1px 0 1px;
}}
QFrame#PredictCompareCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    border: 1px solid {border_strong};
    border-radius: 12px;
}}
QLabel#PredictPreviewCard {{
    background: {image_preview_bg};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 4px;
}}
QFrame[divider="true"] {{
    background: {border};
    color: {border};
}}
QFrame#CanvasStageNode {{
    border: 1px solid {border_strong};
    border-radius: 10px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    padding: 2px;
}}
QFrame#CanvasStageNode[editable="false"] {{
    background: {panel_alt_bg};
    border-color: {border};
}}
QFrame#CanvasStageNode[selected="true"] {{
    border: 1px solid {accent};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {accent_soft}, stop:1 {panel_alt_bg});
}}
QListWidget#StrategyPaletteList {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    border: 1px solid {border_strong};
    border-radius: 10px;
    padding: 6px;
}}
QListWidget#StrategyPaletteList::item {{
    margin: 3px 2px;
    padding: 8px 10px;
    border-radius: 8px;
}}
QStatusBar {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {panel_bg}, stop:1 {panel_alt_bg});
    color: {text_muted};
    border-top: 1px solid {border_strong};
}}
    """.format(**theme)
