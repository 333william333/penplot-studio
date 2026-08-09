"""Cura-inspired theme, in light and dark, following the system setting.

The paper stays white in both.  What the machine will do is black ink on a white
sheet, and a preview that showed light strokes on a dark bed would be a picture
of something that is never going to happen.  The room around the paper is what
gets dark.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    name: str
    background: str
    panel: str
    panel_alt: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str
    success: str
    warning: str
    danger: str
    header_bg: str
    canvas: str        # the surround the sheet of paper sits on
    bed: str
    bed_edge: str
    grid_minor: str
    grid_major: str
    travel: str
    ghost: str


#: Adobe Spectrum, dark theme, by the actual ramp values.
#:
#:   gray-50 #1D1D1D · gray-75 #262626 · gray-100 #323232 · gray-200 #3F3F3F
#:   gray-300 #545454 · gray-500 #909090 · gray-600 #B2B2B2 · gray-700 #D0D0D0
#:   blue-400 #2680EB · blue-500 #378EF0 · red-600 #F76D74
#:   orange-600 #F9A43F · green-600 #39B98D
#:
#: Two places deviate on purpose, and only by moving along the same ramp:
#: filled buttons use #1473E6 because white on blue-400 measures 3.91:1, below
#: AA for 12 px text, while #1473E6 gives 4.54:1; and the semantic colours are
#: taken one step up the ramp (red-600 rather than red-400) for the same reason
#: - 3.24:1 is not a colour you can set small text in.
DARK = Palette(
    name="dark",
    background="#1D1D1D",      # gray-50
    panel="#323232",           # gray-100
    panel_alt="#262626",       # gray-75
    border="#3F3F3F",          # gray-200
    border_strong="#545454",   # gray-300
    text="#D0D0D0",            # gray-700   8.31:1 on the panel
    text_muted="#B2B2B2",      # gray-600   6.05:1
    text_faint="#909090",      # gray-500   4.02:1 - labels and disabled only
    accent="#1473E6",          # filled buttons, white text at 4.54:1
    accent_hover="#2680EB",    # blue-400
    accent_pressed="#0D66D0",
    accent_soft="#1D3A5C",
    success="#39B98D",         # green-600
    warning="#F9A43F",         # orange-600
    danger="#F76D74",          # red-600
    header_bg="#262626",       # gray-75
    canvas="#1D1D1D",
    # the sheet of paper is white, because that is what it is
    bed="#FFFFFF", bed_edge="#545454",
    grid_minor="#EDF1F4", grid_major="#DCE3EA",
    travel="#C3CDD6", ghost="#E2E7EC",
)

#: The app ships dark only.  A pen plotter is looked at next to a bright sheet
#: of paper, and one theme that is right beats two that are nearly right.
LIGHT = DARK


class _Signals(QObject):
    #: emitted after the palette changes, for the few widgets that build their
    #: own stylesheet instead of inheriting the application one
    changed = Signal()


signals = _Signals()

ACTIVE = LIGHT

#: How big the interface is drawn, 1.0 = as designed.  Every px in the
#: stylesheet and every fixed size in the window goes through this, so the whole
#: thing grows and shrinks together instead of the text outgrowing its boxes.
SCALE = 1.0
SCALE_MIN = 0.7
SCALE_MAX = 1.6
BASE_POINT_SIZE = 10.5


def px(value: float) -> int:
    """A designed size in the size the user has actually asked for."""
    return max(int(round(value * SCALE)), 1)

# --------------------------------------------------------------------------
# Module-level names, kept because panels read `theme.PANEL` at paint time.
BACKGROUND = LIGHT.background
PANEL = LIGHT.panel
PANEL_ALT = LIGHT.panel_alt
BORDER = LIGHT.border
BORDER_STRONG = LIGHT.border_strong
TEXT = LIGHT.text
TEXT_MUTED = LIGHT.text_muted
TEXT_FAINT = LIGHT.text_faint

ACCENT = LIGHT.accent
ACCENT_HOVER = LIGHT.accent_hover
ACCENT_PRESSED = LIGHT.accent_pressed
ACCENT_SOFT = LIGHT.accent_soft

SUCCESS = LIGHT.success
WARNING = LIGHT.warning
DANGER = LIGHT.danger

HEADER_BG = LIGHT.header_bg
CANVAS = LIGHT.canvas
BED = LIGHT.bed
BED_EDGE = LIGHT.bed_edge
GRID_MINOR = LIGHT.grid_minor
GRID_MAJOR = LIGHT.grid_major
TRAVEL = LIGHT.travel
GHOST = LIGHT.ghost
PANEL_LIFT = "#404040"
PANEL_LIFT_HOVER = "#4C4C4C"
PANEL_ALT_HOVER = "#3A3A3A"

RADIUS = "6px"        # controls
RADIUS_LARGE = "9px"  # panels and cards
RADIUS_SMALL = "4px"   # chips and small toggles


def _lift(hex_colour: str, amount: int) -> str:
    """A hair lighter, for the top of a button - Apple's one-pixel highlight."""
    value = hex_colour.lstrip("#")
    channels = [min(int(value[i : i + 2], 16) + amount, 255) for i in (0, 2, 4)]
    return "#" + "".join(f"{c:02X}" for c in channels)


def _activate(palette: Palette) -> None:
    """Rebind the module-level names so `theme.PANEL` follows the theme."""
    global ACTIVE, BACKGROUND, PANEL, PANEL_ALT, BORDER, BORDER_STRONG
    global TEXT, TEXT_MUTED, TEXT_FAINT, ACCENT, ACCENT_HOVER, ACCENT_PRESSED
    global ACCENT_SOFT, SUCCESS, WARNING, DANGER, HEADER_BG, CANVAS, BED
    global BED_EDGE, GRID_MINOR, GRID_MAJOR, TRAVEL, GHOST
    global PANEL_LIFT, PANEL_LIFT_HOVER, PANEL_ALT_HOVER
    ACTIVE = palette
    BACKGROUND = palette.background
    PANEL = palette.panel
    PANEL_ALT = palette.panel_alt
    BORDER = palette.border
    BORDER_STRONG = palette.border_strong
    TEXT = palette.text
    TEXT_MUTED = palette.text_muted
    TEXT_FAINT = palette.text_faint
    ACCENT = palette.accent
    ACCENT_HOVER = palette.accent_hover
    ACCENT_PRESSED = palette.accent_pressed
    ACCENT_SOFT = palette.accent_soft
    SUCCESS = palette.success
    WARNING = palette.warning
    DANGER = palette.danger
    HEADER_BG = palette.header_bg
    CANVAS = palette.canvas
    BED = palette.bed
    BED_EDGE = palette.bed_edge
    GRID_MINOR = palette.grid_minor
    GRID_MAJOR = palette.grid_major
    TRAVEL = palette.travel
    GHOST = palette.ghost
    PANEL_LIFT = _lift(palette.panel, 14)
    PANEL_LIFT_HOVER = _lift(palette.panel, 26)
    PANEL_ALT_HOVER = _lift(palette.panel, 8)


def qcolor(value: str, alpha: int = 255) -> QColor:
    color = QColor(value)
    color.setAlpha(alpha)
    return color


def _stylesheet() -> str:
    return f"""
QWidget {{
    color: {TEXT};
    font-size: 11px;
}}

QLabel, QCheckBox, QRadioButton, QSlider, QSplitter, QStackedWidget, QTabWidget {{
    background: transparent;
}}

QToolTip {{
    background: {TEXT};
    color: {PANEL};
    border: none;
    padding: 4px 6px;
    border-radius: {RADIUS_SMALL};
    font-size: 10px;
}}

/* ---------- structural ---------- */
QFrame#Card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LARGE};
}}

QFrame#CardHeader {{
    background: transparent;
    border: none;
}}

QLabel#CardTitle {{
    font-size: 11px;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: 0.3px;
}}

QLabel#SectionLabel {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

QLabel#Hint {{
    color: {TEXT_FAINT};
    font-size: 10px;
}}

QLabel#Muted {{ color: {TEXT_MUTED}; }}
/* the maker's mark: legible if you look at it, silent while you work */
QLabel#Credit {{
    color: {BORDER_STRONG};
    font-size: 9px;
    letter-spacing: 0.6px;
    padding-left: 8px;
    padding-right: 2px;
}}
QLabel#Warning {{ color: {WARNING}; font-weight: 600; }}
QLabel#Danger {{ color: {DANGER}; font-weight: 600; }}
QLabel#Success {{ color: {SUCCESS}; font-weight: 600; }}

QFrame#SidePanel {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}

QFrame#Filmstrip {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
}}

QFrame#OptionsBar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
}}

QFrame#StatusStrip {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
}}

QLabel#ToolGlyph {{ color: {TEXT}; font-size: 12px; }}

QFrame#VLine {{ color: {BORDER}; background: {BORDER}; max-width: 1px; }}

QSplitter#Dock {{ background: {BACKGROUND}; }}
QSplitter#Dock::handle {{ background: {BORDER}; }}

QTabWidget::pane {{
    border: none;
    border-left: 1px solid {BORDER};
    background: {PANEL};
}}
QTabBar {{ background: {HEADER_BG}; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    border: 1px solid transparent;
    border-radius: {RADIUS};
    margin: 4px 2px;
    padding: 4px 9px;
    font-size: 10px;
}}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; border-color: {BORDER}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QFrame#ToolRail {{
    background: {HEADER_BG};
    border-right: 1px solid {BORDER};
}}

QFrame#ToolRail QToolButton {{
    border: 1px solid transparent;
    border-radius: {RADIUS};
    background: transparent;
    font-size: 13px;
}}
QFrame#ToolRail QToolButton:hover {{ background: {PANEL_ALT}; border-color: {BORDER}; }}
QFrame#ToolRail QToolButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: #FFFFFF; }}

QFrame#Header {{
    background: {HEADER_BG};
    border-bottom: 1px solid {BORDER};
}}

QFrame#ActionBar {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LARGE};
}}

QFrame#Separator {{
    background: {BORDER};
    max-height: 1px;
    border: none;
}}

QLabel#Brand {{
    font-size: 13px;
    font-weight: 700;
    color: {TEXT};
}}

QLabel#BrandMark {{
    font-size: 13px;
    font-weight: 700;
    color: {ACCENT};
}}

/* ---------- buttons ---------- */
/* Apple's button geometry on Adobe's colour: a soft rectangle with a hair of
   vertical shading, a 1 px top highlight to catch the light, and a real focus
   ring rather than a dotted outline. */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 {PANEL_LIFT}, stop:1 {PANEL});
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS};
    padding: 5px 10px;
    color: {TEXT};
    min-height: 14px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 {PANEL_LIFT_HOVER}, stop:1 {PANEL_ALT_HOVER});
    border-color: {BORDER_STRONG};
}}
QPushButton:pressed {{
    background: {PANEL_ALT};
    border-color: {BORDER_STRONG};
}}
QPushButton:focus {{ border-color: {ACCENT_HOVER}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; background: {PANEL_ALT}; }}

/* Narrow buttons - the arrows in the layer list - have no room for the
   comfortable padding, and Qt clips the glyph rather than the padding. */
QPushButton#Compact {{ padding: 4px 2px; }}

QPushButton#Primary {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 {ACCENT_HOVER}, stop:1 {ACCENT});
    border: 1px solid {ACCENT_PRESSED};
    color: #FFFFFF;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: {RADIUS};
}}
QPushButton#Primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3D8DEF, stop:1 {ACCENT_HOVER});
}}
QPushButton#Primary:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#Primary:disabled {{ background: {PANEL_ALT}; border-color: {BORDER}; color: {TEXT_FAINT}; }}

QPushButton#Danger {{
    background: {PANEL};
    border: 1px solid {DANGER};
    color: {DANGER};
    font-weight: 600;
}}
QPushButton#Danger:hover {{ background: #FCEDED; }}

QPushButton#Ghost {{
    background: transparent;
    border: 1px solid transparent;
    color: {ACCENT};
    padding: 3px 6px;
}}
QPushButton#Ghost:hover {{ background: {ACCENT_SOFT}; }}

QPushButton#Stage {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS};
    padding: 6px 13px;
    color: {TEXT_MUTED};
    font-weight: 600;
}}
QPushButton#Stage:hover {{ color: {TEXT}; background: {PANEL_ALT}; }}
QPushButton#Stage:checked {{ color: #FFFFFF; background: {ACCENT}; border-color: {ACCENT}; }}

QPushButton#Segment, QPushButton#SegmentFirst, QPushButton#SegmentLast {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    margin: 1px;
    padding: 4px 7px;
    color: {TEXT_MUTED};
}}
QPushButton#Segment:hover, QPushButton#SegmentFirst:hover, QPushButton#SegmentLast:hover {{
    color: {TEXT};
}}
QPushButton#Segment:checked, QPushButton#SegmentFirst:checked, QPushButton#SegmentLast:checked {{
    background: {ACCENT};
    color: #FFFFFF;
    border-color: {ACCENT};
    font-weight: 600;
}}

QToolButton {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    padding: 3px 5px;
    color: {TEXT};
}}
QToolButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QToolButton:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT}; }}

/* ---------- inputs ---------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS};
    padding: 3px 5px;
    selection-background-color: {ACCENT};
    selection-color: #FFFFFF;
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {PANEL_ALT}; color: {TEXT_FAINT};
}}

QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
    padding: 2px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 15px; border: none; background: transparent; }}

QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_SMALL};
    background: {PANEL};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{{CHECK_ICON}}");
}}
QRadioButton::indicator:checked {{ background: {ACCENT}; border: 4px solid {PANEL}; outline: 1px solid {ACCENT}; }}

QSlider::groove:horizontal {{
    height: 3px; background: {BORDER}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {PANEL}; border: 2px solid {ACCENT};
    width: 11px; height: 11px; margin: -5px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_SOFT}; }}
QSlider::handle:horizontal:disabled {{ border-color: {BORDER_STRONG}; }}

QProgressBar {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: {TEXT};
    font-size: 9px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}

/* ---------- containers ---------- */
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 5px; min-width: 24px; }}

QTabWidget::pane {{ border: none; background: transparent; }}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 5px 10px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QListWidget {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    outline: none;
}}
QListWidget::item {{ padding: 4px; border-radius: {RADIUS_SMALL}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}

QPlainTextEdit#Console {{
    background: #1F2933;
    color: #D7E0E8;
    border: 1px solid #16202A;
    font-family: Menlo, Monaco, monospace;
    font-size: 10px;
}}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {BORDER}; }}

QMenuBar {{ background: {HEADER_BG}; }}
QMenuBar::item:selected {{ background: {ACCENT_SOFT}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER_STRONG}; padding: 3px; }}
QMenu::item {{ padding: 4px 16px 4px 9px; border-radius: {RADIUS_SMALL}; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
"""



def _checkmark_icon() -> str:
    """Draw the tick used inside checked checkboxes and cache it on disk."""
    from pathlib import Path

    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QPainter, QPen, QPixmap

    from ..core.settings import config_dir

    path = Path(config_dir()) / "checkmark.png"
    if not path.exists():
        size = 28
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#FFFFFF"))
        pen.setWidthF(size * 0.16)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(
            [
                QPointF(size * 0.24, size * 0.52),
                QPointF(size * 0.43, size * 0.71),
                QPointF(size * 0.77, size * 0.30),
            ]
        )
        painter.end()
        pixmap.save(str(path))
    return str(path).replace("\\", "/")


def system_is_dark(app: QApplication | None = None) -> bool:
    """What the operating system is asking for, right now."""
    app = app or QApplication.instance()
    if app is not None:
        try:
            scheme = app.styleHints().colorScheme()
            if scheme == Qt.ColorScheme.Dark:
                return True
            if scheme == Qt.ColorScheme.Light:
                return False
        except (AttributeError, TypeError):  # pragma: no cover - older Qt
            pass
    # Qt reports Unknown on some platforms and offscreen; ask macOS directly
    import subprocess

    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True, text=True, timeout=1.5,
        )
        return "dark" in result.stdout.strip().lower()
    except Exception:  # pragma: no cover - not macOS, or defaults missing
        return False


def _scaled_stylesheet() -> str:
    """The stylesheet with every px measurement taken through the scale."""
    import re

    sheet = _stylesheet()
    if abs(SCALE - 1.0) < 1e-6:
        return sheet

    def scale_px(match) -> str:
        value = int(match.group(1))
        # hairlines stay hairlines: a 1 px border scaled to 2 px reads as a
        # heavier design rather than a bigger one
        return match.group(0) if value <= 1 else f"{max(int(round(value * SCALE)), 1)}px"

    return re.sub(r"(\d+)px", scale_px, sheet)


def set_scale(app: QApplication, factor: float) -> bool:
    """Resize the whole interface.  Returns False when already at the end."""
    global SCALE

    wanted = round(min(max(factor, SCALE_MIN), SCALE_MAX), 2)
    if abs(wanted - SCALE) < 1e-6:
        return False
    SCALE = wanted
    font = QFont()
    font.setPointSizeF(BASE_POINT_SIZE * SCALE)
    app.setFont(font)
    _paint(app)
    signals.changed.emit()
    return True


def _paint(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(PANEL_ALT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(PANEL))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipBase, QColor(TEXT))
    palette.setColor(QPalette.ToolTipText, QColor(PANEL))
    app.setPalette(palette)
    app.setStyleSheet(_scaled_stylesheet().replace("{CHECK_ICON}", _checkmark_icon()))


def set_palette(app: QApplication, palette: Palette) -> None:
    """Switch theme and tell the few widgets that style themselves."""
    if palette is ACTIVE:
        return
    _activate(palette)
    _paint(app)
    signals.changed.emit()


def apply_theme(app: QApplication, follow_system: bool = False) -> None:
    """The app is dark, always.

    `follow_system` is kept so existing calls still work, and ignored: there is
    one theme now, and it is the one the interface was actually designed and
    measured in.
    """
    app.setStyle("Fusion")
    _activate(DARK)
    font = QFont()
    font.setPointSizeF(BASE_POINT_SIZE * SCALE)
    app.setFont(font)
    _paint(app)


