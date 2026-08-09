"""The application window: header, stages, preview and the action bar."""

from __future__ import annotations

import bisect
import copy
import dataclasses
import os

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QTabBar,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QDockWidget,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import gcode, svgexport, techniques, testpattern
from ..core.pipeline import PlotJob
from ..core.printer import PrinterLink
from ..core.settings import AppSettings
from . import theme
from .panels.layers_panel import LayersPanel
from .panels.layout_panel import LayoutPanel
from .panels.machine_panel import MachinePanel
from .panels.monitor_panel import ConnectionPanel, JobPanel
from .panels.pens_panel import PensPanel
from .panels.source_panel import SourcePanel
from .preview import PreviewCanvas
from .widgets import Card, Separator
from .worker import AutoTuneRunner, PipelineRunner

REBUILD_DELAY_MS = 140


#: (scroll area, designed width) for everything the scale has to resize
_SCROLL_AREAS: list = []


def _scroll(widget: QWidget, width: int = 330) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    # never silently crop a control: show the bar if the content really is wider
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(9, 9, 9, 9)
    layout.setSpacing(7)
    layout.addWidget(widget)
    layout.addStretch(1)
    area.setWidget(holder)
    # Pinned width: letting the side columns grow squeezed the preview toolbar
    # until its buttons were cut in half on a smaller screen.
    area.setFixedWidth(theme.px(width))
    _SCROLL_AREAS.append((area, width))
    return area


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.job: PlotJob | None = None
        self.program: gcode.GCodeProgram | None = None
        self._path_lines: list[int] = []
        self._suppress_rebuild = False
        self._pending_action: str | None = None
        #: how long the last build took, so the debounce can follow the cost of
        #: the technique instead of a fixed guess
        self._last_build_ms = 0.0
        self._submitted_draft = False
        self._full_scheduled = False
        #: a freshly loaded file should be brought into view, wherever the
        #: camera happened to be left
        self._fit_after_load = False
        self._printing = False
        # bumped on every settings change; a render remembers the value it was
        # submitted with, so a job that lands late cannot clear a newer change
        self._settings_serial = 0
        self._submitted_serial = -1

        self.setWindowTitle("PenPlot Studio")
        self.resize(1620, 1000)
        self.setMinimumSize(1280, 680)

        self.printer = PrinterLink(self)
        self.runner = PipelineRunner(self)
        self.tuner = AutoTuneRunner(self)

        self.preview = PreviewCanvas()
        self.preview_holder = QWidget()
        self._preview_layout = QVBoxLayout(self.preview_holder)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_layout.setSpacing(0)

        self.layers_panel = LayersPanel(settings)
        self.source_panel = SourcePanel(settings)
        self.layout_panel = LayoutPanel(settings)
        self.pens_panel = PensPanel(settings)
        self.machine_panel = MachinePanel(settings)
        self.connection_panel = ConnectionPanel(settings, self.printer)
        self.job_panel = JobPanel(settings, self.printer)

        _SCROLL_AREAS.clear()
        self._build_ui()
        self._scroll_areas = list(_SCROLL_AREAS)
        self._connect_signals()
        theme.signals.changed.connect(self._apply_scale)

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(REBUILD_DELAY_MS)
        self._rebuild_timer.timeout.connect(self._rebuild_now)

        self.source_panel.set_pen_count(len(settings.library))
        self.source_panel.rebuild_source()
        QTimer.singleShot(60, self.preview.fit_view)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """The standard Qt application shell.

        Hand-rolled columns with pinned widths were the reason none of this
        resized: a QHBoxLayout holding three fixed-width children has nothing to
        give when the window changes.  QMainWindow already solves this properly
        and every desktop application uses it - dock widgets the user can drag,
        resize, tab together, tear off or hide, a central widget that takes
        whatever is left, toolbars, and a status bar.  The arrangement is saved
        between sessions, so a layout you set up once stays set up.
        """
        self.setDockOptions(
            QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks
        )
        self.setDockNestingEnabled(True)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)

        self.stages = QStackedWidget()
        self.stages.addWidget(self.preview_holder)
        self.stages.addWidget(self._build_monitor_stage())
        self.setCentralWidget(self.stages)

        self.addToolBar(Qt.TopToolBarArea, self._build_header_bar())
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, self._build_options_bar())
        self.addToolBar(Qt.LeftToolBarArea, self._build_tool_rail())

        self._docks: dict[str, QDockWidget] = {}
        self._dock_visible: dict[str, bool] = {}
        image = self._dock_for("Image", self.source_panel, Qt.RightDockWidgetArea, 300)
        technique, self._technique_settings = self._build_technique_docks()
        layout_dock = self._dock_for("Layout", self.layout_panel, Qt.RightDockWidgetArea, 300)
        pens = self._dock_for("Pens", self.pens_panel, Qt.RightDockWidgetArea, 300)
        printer = self._dock_for("Printer", self.machine_panel, Qt.RightDockWidgetArea, 300)
        layers = self._dock_for("Layers", self.layers_panel, Qt.RightDockWidgetArea, 300)

        # Image on the left where the picture is; the rest tabbed on the right
        self.removeDockWidget(image)
        self.addDockWidget(Qt.LeftDockWidgetArea, image)
        image.show()
        # Split before tabifying: splitting a dock that is already part of a tab
        # group moves the whole group, which is how Layers ended up as a fifth
        # tab instead of a panel of its own underneath.
        self.splitDockWidget(technique, layers, Qt.Vertical)
        for dock in (layout_dock, pens, printer):
            self.tabifyDockWidget(technique, dock)
        technique.raise_()
        self._tidy_dock_tabs()
        self.resizeDocks([technique, layers], [theme.px(520), theme.px(300)], Qt.Vertical)
        self.resizeDocks([image, technique], [theme.px(320), theme.px(346)], Qt.Horizontal)

        self._build_status_bar()
        self._build_menu()
        self._set_stage(0)
        self._default_state = self.saveState()
        self._restore_layout()

    def _tidy_dock_tabs(self) -> None:
        """Let the dock tabs read.

        Qt elides tabified dock titles by default, so "Technique" became
        "Techn…" in a bar with room to spare.  Scroll buttons are the honest
        alternative: full labels when they fit, arrows when the dock has
        genuinely been dragged too narrow.  Expanding spreads them over the
        whole bar instead of leaving a quarter of it empty.
        """
        for bar in self.findChildren(QTabBar):
            if bar.parent() is not self:
                continue        # a QTabWidget's own bar, not the dock group
            bar.setElideMode(Qt.ElideNone)
            bar.setUsesScrollButtons(True)
            bar.setExpanding(True)
            bar.setDrawBase(False)

    def showEvent(self, event) -> None:   # noqa: N802 - Qt
        super().showEvent(event)
        # the dock tab bar is created lazily, so once more when it exists
        self._tidy_dock_tabs()

    def _dock_for(self, title: str, widget: QWidget, area, minimum: int) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock:{title}")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        holder = QScrollArea()
        holder.setWidgetResizable(True)
        holder.setFrameShape(QFrame.NoFrame)
        holder.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(7)
        box.addWidget(widget)
        box.addStretch(1)
        holder.setWidget(inner)
        # a minimum, not a fixed width: the user drags the edge, the canvas keeps
        # whatever is left over
        holder.setMinimumWidth(theme.px(minimum))
        dock.setWidget(holder)
        self.addDockWidget(area, dock)
        self._docks[title] = dock
        return dock

    def _build_technique_docks(self):
        """The picker along the bottom, its settings on the right."""
        gallery_card, settings_card = self.source_panel.detach_wide_cards()
        gallery_card.set_title("")
        gallery_card.hide_header()
        self._relayout_gallery(columns=len(self.source_panel.gallery.tiles))

        strip = QDockWidget("Techniques", self)
        strip.setObjectName("dock:Techniques")
        strip.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 8, 2)
        filter_control = self.source_panel.take_group_filter()
        filter_control.setMinimumWidth(filter_control.sizeHint().width())
        header.addWidget(filter_control)
        header.addStretch(1)
        box.addLayout(header)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setWidget(gallery_card)
        self.technique_strip = area
        box.addWidget(area)
        strip.setWidget(holder)
        strip.setMinimumHeight(theme.px(150))
        self.addDockWidget(Qt.BottomDockWidgetArea, strip)
        self._docks["Techniques"] = strip
        self._strip_host = strip

        settings_dock = self._dock_for("Technique", settings_card, Qt.RightDockWidgetArea, 300)
        return settings_dock, settings_card

    def _build_header_bar(self) -> QToolBar:
        bar = QToolBar("Workspace", self)
        bar.setObjectName("toolbar:workspace")
        bar.setMovable(False)
        mark = QLabel("  ✏  ")
        mark.setObjectName("BrandMark")
        brand = QLabel("PenPlot Studio   ")
        brand.setObjectName("Brand")
        bar.addWidget(mark)
        bar.addWidget(brand)

        self.stage_group = QButtonGroup(self)
        self.stage_group.setExclusive(True)
        for index, name in enumerate(("Prepare", "Monitor")):
            button = QPushButton(name)
            button.setObjectName("Stage")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, i=index: self._set_stage(i))
            self.stage_group.addButton(button, index)
            bar.addWidget(button)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.machine_label = QLabel(self.settings.machine.name + "  ")
        self.machine_label.setObjectName("Muted")
        bar.addWidget(self.machine_label)
        self.busy_label = QLabel("")
        self.busy_label.setObjectName("Hint")
        bar.addWidget(self.busy_label)
        self._header_frame = bar
        return bar

    def _build_tool_rail(self) -> QToolBar:
        """The tools, in a vertical toolbar the user can move or hide."""
        rail = QToolBar("Tools", self)
        rail.setObjectName("toolbar:tools")
        rail.setOrientation(Qt.Vertical)
        rail.setIconSize(QSize(theme.px(18), theme.px(18)))
        self._rail = rail

        self.tool_buttons: dict[str, QToolButton] = {}
        for tool, label, tip in (
            ("", "⟐", "Move and select  (V)"),
            ("free", "✎", "Draw freehand on the bed  (B)"),
            ("dot", "•", "Place dots - the pen taps and lifts  (D)"),
            ("line", "╱", "Straight line  (L)"),
            ("rect", "▭", "Rectangle  (M)"),
            ("ellipse", "◯", "Ellipse  (E)"),
        ):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setChecked(tool == "")
            button.setFixedSize(theme.px(26), theme.px(26))
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, t=tool: self._set_draw_tool(t))
            rail.addWidget(button)
            self.tool_buttons[tool] = button

        rail.addSeparator()
        self.undo_stroke_button = QToolButton()
        self.undo_stroke_button.setText("⌫")
        self.undo_stroke_button.setFixedSize(theme.px(26), theme.px(26))
        self.undo_stroke_button.setToolTip("Undo the last stroke  (⌘Z)")
        self.undo_stroke_button.clicked.connect(self._undo_stroke)
        rail.addWidget(self.undo_stroke_button)
        return rail

    def _vline(self) -> QFrame:
        line = QFrame()
        line.setObjectName("VLine")
        line.setFrameShape(QFrame.VLine)
        line.setFixedWidth(1)
        return line

    def _build_options_bar(self) -> QToolBar:
        """Settings for the tool in your hand, and nothing else."""
        bar = QToolBar("Tool options", self)
        bar.setObjectName("toolbar:options")
        bar.setMovable(False)
        self._options_bar = bar

        self.tool_glyph = QLabel("  ⟐ ")
        self.tool_glyph.setObjectName("ToolGlyph")
        self.tool_name = QLabel("Move")
        self.tool_name.setObjectName("CardTitle")
        self.tool_name.setMinimumWidth(theme.px(62))
        bar.addWidget(self.tool_glyph)
        bar.addWidget(self.tool_name)
        bar.addWidget(self._vline())

        self.move_options = QLabel(
            "  Drag the artwork on the bed · arrow keys nudge 1 mm · ⇧ for 10 mm"
        )
        self.move_options.setObjectName("Hint")
        bar.addWidget(self.move_options)

        self.draw_options = QWidget()
        draw = QHBoxLayout(self.draw_options)
        draw.setContentsMargins(8, 0, 0, 0)
        draw.setSpacing(6)
        pen_label = QLabel("Draws with")
        pen_label.setObjectName("Hint")
        draw.addWidget(pen_label)
        self.draw_pen_combo = QComboBox()
        self.draw_pen_combo.setMinimumWidth(theme.px(150))
        self.draw_pen_combo.currentIndexChanged.connect(self._on_draw_pen)
        draw.addWidget(self.draw_pen_combo)
        undo = QPushButton("Undo stroke")
        undo.clicked.connect(self._undo_stroke)
        draw.addWidget(undo)
        bar.addWidget(self.draw_options)
        self.draw_options.setVisible(False)
        return bar

    def _build_status_bar(self) -> None:
        """Qt's own status bar, with the view controls and the committing action."""
        bar = QStatusBar()
        self.setStatusBar(bar)
        self._status_bar = bar

        left = QWidget()
        row = QHBoxLayout(left)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        zoom_out = QToolButton(); zoom_out.setText("−")
        zoom_out.setToolTip("Zoom out  (−)")
        zoom_out.clicked.connect(self.preview.zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(theme.px(40))
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_in = QToolButton(); zoom_in.setText("+")
        zoom_in.setToolTip("Zoom in  (+)")
        zoom_in.clicked.connect(self.preview.zoom_in)
        fit = QToolButton(); fit.setText("Fit")
        fit.setToolTip("Show the whole bed  (F)")
        fit.clicked.connect(self.preview.fit_view)
        art = QToolButton(); art.setText("Art")
        art.setToolTip("Fill the view with the drawing  (Z)")
        art.clicked.connect(self.preview.zoom_to_artwork)
        self.play_button = QToolButton()
        self.play_button.setText("▶")
        self.play_button.setCheckable(True)
        self.play_button.setToolTip("Play the drawing back")
        self.play_button.toggled.connect(self._toggle_playback)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(1000)
        self.progress_slider.setFixedWidth(theme.px(88))
        self.progress_slider.valueChanged.connect(lambda v: self.preview.set_progress(v / 1000.0))
        for widget in (zoom_out, self.zoom_label, zoom_in, fit, art,
                       self._vline(), self.play_button, self.progress_slider):
            row.addWidget(widget)
        bar.addWidget(left)

        self.summary_label = QLabel("Load a picture, type some text, or drop a PDF")
        self.summary_label.setObjectName("Hint")
        self.summary_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.summary_label, 1)

        self.estimate_label = QLabel("—")
        self.estimate_label.setObjectName("CardTitle")
        self.estimate_label.setMinimumWidth(theme.px(80))
        self.estimate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.save_button = QPushButton("Save G-code…")
        self.save_button.clicked.connect(self.save_gcode)
        self.send_button = QPushButton("Send to printer")
        self.send_button.setObjectName("Primary")
        self.send_button.clicked.connect(self.send_to_printer)
        for widget in (self.estimate_label, self.save_button, self.send_button):
            bar.addPermanentWidget(widget)

        self.coord_label = QLabel("")
        self.coord_label.setObjectName("Hint")
        bar.addPermanentWidget(self.coord_label)

        # A maker's mark, not a banner: present if you look for it, invisible
        # while you work.
        credit = QLabel("WilliamLabs")
        credit.setObjectName("Credit")
        credit.setToolTip("PenPlot Studio - made by WilliamLabs")
        bar.addPermanentWidget(credit)

        self._playback = QTimer(self)
        self._playback.setInterval(33)
        self._playback.timeout.connect(self._advance_playback)

        self.setAcceptDrops(True)

    def _restore_layout(self) -> None:
        saved = self.settings.window_state
        if not saved:
            return
        try:
            from PySide6.QtCore import QByteArray

            self.restoreState(QByteArray.fromBase64(saved.encode("ascii")))
        except Exception:  # pragma: no cover - a corrupt layout must not stop the app
            pass

    def show_panel(self, name: str) -> None:
        """Bring a panel to the front by name - Image, Technique, Layers..."""
        dock = self._docks.get(name)
        if dock is not None:
            dock.show()
            dock.raise_()

    def reset_layout(self) -> None:
        """Put the panels back where they started."""
        self.restoreState(self._default_state)
        self.settings.window_state = ""

    def _relayout_gallery(self, columns: int) -> None:
        gallery = self.source_panel.gallery
        grid = gallery.layout()
        tiles = list(gallery.tiles.values())
        for tile in tiles:
            grid.removeWidget(tile)
        for index, tile in enumerate(tiles):
            grid.addWidget(tile, index // columns, index % columns)
        gallery.columns = columns

    def _on_draw_pen(self, index: int) -> None:
        value = self.draw_pen_combo.itemData(index)
        if value is None or self.settings.item.kind != "shapes":
            return
        self.settings.item.pen = int(value)
        self.source_panel.refresh_shapes_pens()
        self.source_panel.rebuild_source()

    def _sync_options_bar(self, tool: str) -> None:
        names = {
            "": ("⟐", "Move"), "free": ("✎", "Pencil"), "dot": ("•", "Dot"),
            "line": ("╱", "Line"), "rect": ("▭", "Rectangle"), "ellipse": ("◯", "Ellipse"),
        }
        glyph, name = names.get(tool, ("⟐", "Move"))
        self.tool_glyph.setText(f"  {glyph} ")
        self.tool_name.setText(name)
        drawing = bool(tool)
        self.move_options.setVisible(not drawing)
        self.draw_options.setVisible(drawing)
        if drawing:
            self.draw_pen_combo.blockSignals(True)
            self.draw_pen_combo.clear()
            for index, pen in enumerate(self.settings.library):
                mark = " ✂" if pen.cuts else ""
                self.draw_pen_combo.addItem(f"{pen.name} · {pen.width:.2f} mm{mark}", index)
            current = int(getattr(self.settings.item, "pen", 0))
            self.draw_pen_combo.setCurrentIndex(max(0, min(current, len(self.settings.library) - 1)))
            self.draw_pen_combo.blockSignals(False)

    def _build_monitor_stage(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_scroll(self.connection_panel, 386))

        centre = QWidget()
        self.monitor_centre = QVBoxLayout(centre)
        self.monitor_centre.setContentsMargins(12, 12, 12, 12)
        self.monitor_centre.setSpacing(10)
        layout.addWidget(centre, 1)

        layout.addWidget(_scroll(self.job_panel, 398))
        return page

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        for text, slot, shortcut in (
            ("Open image…", self.source_panel.open_image_dialog, QKeySequence.Open),
            ("Open PDF…", self.source_panel.open_pdf_dialog, None),
            ("Save G-code…", self.save_gcode, QKeySequence.Save),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(shortcut)
            file_menu.addAction(action)
        export = QAction("Export SVG…", self)
        export.triggered.connect(self.save_svg)
        file_menu.addAction(export)
        file_menu.addSeparator()
        reset = QAction("Reset all settings", self)
        reset.triggered.connect(self.reset_settings)
        file_menu.addAction(reset)

        view_menu = menu.addMenu("&View")
        for text, slot, shortcut in (
            ("Bigger interface", lambda: self._zoom_interface(0.1), "Ctrl++"),
            ("Smaller interface", lambda: self._zoom_interface(-0.1), "Ctrl+-"),
            ("Interface at 100%", lambda: self._zoom_interface(0.0), "Ctrl+0"),
            ("Refresh preview", self.render_now, "Ctrl+R"),
            ("Fit the bed on screen", self.preview.fit_view, "F"),
            ("Zoom to artwork", self.preview.zoom_to_artwork, "Z"),
            ("Prepare stage", lambda: self._set_stage(0), "1"),
            ("Monitor stage", lambda: self._set_stage(1), "2"),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            action.setShortcut(shortcut)
            view_menu.addAction(action)

        # Photoshop's single-key tool shortcuts, so the hand never leaves the
        # canvas to change tool.
        for key, tool in (("V", ""), ("B", "free"), ("D", "dot"), ("L", "line"),
                          ("M", "rect"), ("E", "ellipse")):
            action = QAction(f"Tool {tool or 'move'}", self)
            action.setShortcut(key)
            action.triggered.connect(lambda _checked=False, t=tool: self._set_draw_tool(t))
            self.addAction(action)
        undo = QAction("Undo stroke", self)
        undo.setShortcut(QKeySequence.Undo)
        undo.triggered.connect(self._undo_stroke)
        self.addAction(undo)

        view_menu.addSeparator()
        panels = view_menu.addMenu("Panels")
        for name, dock in self._docks.items():
            panels.addAction(dock.toggleViewAction())
        reset_layout = QAction("Reset panel layout", self)
        reset_layout.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout)

        tools_menu = menu.addMenu("&Tools")
        fit_action = QAction("Fit the settings to this picture…", self)
        fit_action.setShortcut("Ctrl+Shift+F")
        fit_action.triggered.connect(self.fit_settings)
        tools_menu.addAction(fit_action)
        tools_menu.addSeparator()

        stencils = QAction("Make stencils from this picture…", self)
        stencils.triggered.connect(self.make_stencils)
        tools_menu.addAction(stencils)
        tools_menu.addSeparator()
        tools_menu.addAction(self._pattern_action("Pen height ladder", "z"))
        tools_menu.addAction(self._pattern_action("Speed ladder", "speed"))
        tools_menu.addAction(self._pattern_action("Pen test sheet", "pen"))
        tools_menu.addAction(self._pattern_action("Registration sheet", "registration"))
        tools_menu.addSeparator()
        hint = QAction("What are these?", self)
        hint.triggered.connect(self._explain_patterns)
        tools_menu.addAction(hint)

        help_menu = menu.addMenu("&Help")
        about = QAction("About PenPlot Studio", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.layers_panel.selection_changed.connect(self._on_layer_selected)
        self.layers_panel.structure_changed.connect(self._on_layers_changed)
        self.layers_panel.redraw_needed.connect(self.render_now)
        self.layers_panel.add_requested.connect(self._add_layer)
        self.preview.item_selected.connect(self._on_layer_selected)
        self.preview.stroke_drawn.connect(self._on_stroke_drawn)

        self.source_panel.source_changed.connect(self._on_source_changed)
        self.source_panel.file_loaded.connect(self._on_file_loaded)
        self.source_panel.file_loaded.connect(lambda: self.source_panel.refresh_gallery())
        self.source_panel.settings_changed.connect(self.schedule_rebuild)
        self.source_panel.palette_suggested.connect(self.pens_panel.apply_colours)
        self.source_panel.status.connect(self.statusBar().showMessage)

        self.layout_panel.changed.connect(self.schedule_rebuild)
        self.pens_panel.changed.connect(self._on_pens_changed)
        self.machine_panel.changed.connect(self._on_machine_changed)
        self.connection_panel.status.connect(self.statusBar().showMessage)
        # setting the pen height or keeping a live tweak changes the G-code, so
        # the cached program and the Printer tab both have to follow
        self.connection_panel.settings_changed.connect(self._on_machine_settings_changed)

        self.preview.move_committed.connect(self._on_preview_move)
        self.preview.scale_committed.connect(self._on_preview_scale)
        self.preview.files_dropped.connect(self._on_files_dropped)
        self.preview.hover_position.connect(
            lambda x, y: self.coord_label.setText(f"X {x:7.1f}   Y {y:7.1f} mm")
        )
        self.preview.zoom_changed.connect(self._on_zoom_changed)

        self.source_panel.auto_tune_requested.connect(self._on_auto_tune)
        self.tuner.finished.connect(self._on_tune_ready)
        self.tuner.failed.connect(lambda err: self.source_panel.show_tune_result("Auto-tune failed."))
        self.tuner.busy_changed.connect(self.source_panel.set_tuning)

        self.runner.finished.connect(self._on_job_ready)
        self.runner.failed.connect(self._on_job_failed)
        self.runner.cancelled.connect(self._on_job_cancelled)
        self.runner.busy_changed.connect(
            lambda busy: self.busy_label.setText("preparing…" if busy else "")
        )

        self.printer.paused.connect(self._on_printer_paused)
        self.printer.progress.connect(self._on_printer_progress)
        self.printer.job_finished.connect(self._on_printer_finished)
        self.printer.position.connect(lambda x, y, z: self.preview.set_live_position(x, y))
        self.printer.connected.connect(lambda port: self.statusBar().showMessage(f"Connected to {port}"))

    # ------------------------------------------------------------------
    # stages & preview toggles
    # ------------------------------------------------------------------
    def _set_stage(self, index: int) -> None:
        """Prepare and Monitor are different jobs, so they get different panels.

        Leaving the preparation docks open while watching a print left half the
        window showing settings that do nothing to a drawing already on its way
        to the machine.
        """
        self.stages.setCurrentIndex(index)
        for button in self.stage_group.buttons():
            button.setChecked(self.stage_group.id(button) == index)
        preparing = index == 0
        for name, dock in self._docks.items():
            if preparing:
                dock.setVisible(self._dock_visible.get(name, True))
            else:
                self._dock_visible[name] = dock.isVisible()
                dock.setVisible(False)
        self._rail.setVisible(preparing)
        self._options_bar.setVisible(preparing)
        if preparing:
            self._preview_layout.addWidget(self.preview)
        else:
            self.monitor_centre.addWidget(self.preview)
        self.preview.show()

    def _on_zoom_changed(self, zoom: float) -> None:
        # 100 % means one screen pixel per bed millimetre at the reference DPI
        self.zoom_label.setText(f"{zoom * 100 / 3.78:.0f}%")

    def _set_grid(self, value: bool) -> None:
        self.preview.show_grid = value
        self.preview.update()

    def _set_travels(self, value: bool) -> None:
        self.preview.show_travels = value
        self.preview.update()

    def _set_handles(self, value: bool) -> None:
        self.preview.show_handles = value
        self.preview.update()

    def _set_stats(self, value: bool) -> None:
        self.preview.show_stats = value
        self.preview.update()

    def _toggle_playback(self, playing: bool) -> None:
        self.play_button.setText("❚❚" if playing else "▶")
        if playing:
            if self.progress_slider.value() >= 1000:
                self.progress_slider.setValue(0)
            self._playback.start()
        else:
            self._playback.stop()

    def _advance_playback(self) -> None:
        value = self.progress_slider.value() + 6
        if value >= 1000:
            self.progress_slider.setValue(1000)
            self.play_button.setChecked(False)
            return
        self.progress_slider.setValue(value)

    # ------------------------------------------------------------------
    # rebuild plumbing
    # ------------------------------------------------------------------
    def schedule_rebuild(self) -> None:
        """Something changed: re-draw, on a debounce that follows the cost.

        There is no Render button.  A slider drag fires this on every tick, so
        the interval is taken from how long the last build actually took - a
        cheap technique feels instant, an expensive one waits until the user
        stops moving instead of queueing a build per pixel.
        """
        if self._suppress_rebuild:
            return
        self._settings_serial += 1
        self._rebuild_timer.setInterval(self._debounce_ms())
        self._rebuild_timer.start()
        self.preview.set_stale(True)

    def _debounce_ms(self) -> int:
        return int(max(REBUILD_DELAY_MS, min(self._last_build_ms * 1.3, 480.0)))

    def render_now(self) -> None:
        """Explicit refresh - always full quality, never a draft."""
        self._rebuild_timer.stop()
        self._rebuild_now(draft=False)

    def _rebuild_now(self, draft: bool | None = None) -> None:
        # An expensive technique gets a fast, coarse pass first so the canvas
        # keeps up with the control being dragged; the full-resolution pass
        # follows once nothing has changed for a moment.
        if draft is None:
            draft = self._last_build_ms > 220.0
        self._submitted_serial = self._settings_serial
        self._submitted_draft = bool(draft)
        self.preview.set_stale(True)
        self.runner.submit(self.source_panel.sources, self.settings, draft=bool(draft))

    @property
    def _render_in_flight(self) -> bool:
        return self.runner.pending > 0 or self._rebuild_timer.isActive()

    @property
    def _up_to_date(self) -> bool:
        return not self._render_in_flight and self._submitted_serial == self._settings_serial

    def _on_file_loaded(self) -> None:
        """A file arrived on the selected layer: show it, and bring it into view."""
        self._fit_after_load = True
        self.layers_panel.rebuild()
        self.layout_panel.refresh()
        self.render_now()

    def _on_source_changed(self) -> None:
        source = self.source_panel.source
        if source is None:
            # nothing loaded: do not keep showing the previous drawing with
            # Save and Send still enabled
            self._settings_serial += 1
            self.render_now()
            return
        if source is not None:
            # text and vector PDFs have a real physical size, so keep it
            if source.kind == "text" or (source.kind == "pdf" and source.vector is not None):
                if self.settings.layout.mode == "fit":
                    self.settings.layout.mode = "scale"
                    self.settings.layout.scale_percent = 100.0
                    self.layout_panel.refresh()
            elif self.settings.layout.mode == "scale" and source.native_size_mm() is None:
                self.settings.layout.mode = "fit"
                self.layout_panel.refresh()
        self.schedule_rebuild()

    # ------------------------------------------------------------------
    # layers
    # ------------------------------------------------------------------
    def _on_layer_selected(self, index: int) -> None:
        if index == self.settings.active and self.layers_panel.rows_layout.count():
            return
        self._suppress_rebuild = True
        self.source_panel.select_item(index)
        self.layout_panel.refresh()
        self._suppress_rebuild = False
        self.layers_panel.rebuild()
        self.preview.set_selected_item(index)

    def _on_layers_changed(self) -> None:
        self.layers_panel.rebuild()
        self.render_now()

    def _add_layer(self, kind: str) -> None:
        self.settings.add_item(kind)
        self._suppress_rebuild = True
        self.source_panel.select_item(self.settings.active)
        self.layout_panel.refresh()
        self._suppress_rebuild = False
        self.layers_panel.rebuild()
        self.preview.set_selected_item(self.settings.active)
        if kind == "image":
            self.source_panel.open_image_dialog()
        elif kind == "pdf":
            self.source_panel.open_pdf_dialog()
        else:
            self.render_now()

    # ------------------------------------------------------------------
    # drawing by hand
    # ------------------------------------------------------------------
    def _set_draw_tool(self, tool: str) -> None:
        for key, button in self.tool_buttons.items():
            button.setChecked(key == tool)
        self.preview.set_draw_tool(tool)
        self._sync_options_bar(tool)
        if tool and self.settings.item.kind != "shapes":
            # drawing always goes onto a drawing layer, so make one
            self._add_layer("shapes")
            self.settings.item.name = "Drawing"
            self.layers_panel.rebuild()
        if tool:
            self.statusBar().showMessage(
                "Drag on the bed to draw. The strokes go onto the selected drawing layer.", 6000
            )

    def _on_stroke_drawn(self, points) -> None:
        item = self.settings.item
        if item.kind != "shapes":
            return
        item.strokes.append([[float(x), float(y)] for x, y in points])
        self.source_panel.rebuild_source()
        self.render_now()

    def _undo_stroke(self) -> None:
        item = self.settings.item
        if item.kind == "shapes" and item.strokes:
            item.strokes.pop()
            self.source_panel.rebuild_source()
            self.render_now()

    def _on_theme_changed(self) -> None:
        """The system switched between light and dark while we were running."""
        for separator in self.findChildren(QFrame):
            if separator.frameShape() == QFrame.VLine:
                separator.setStyleSheet(f"color: {theme.BORDER};")
        self.zoom_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self.layers_panel.rebuild()
        self.pens_panel.rebuild()
        self.source_panel.restyle()
        self.preview.update()

    def _on_pens_changed(self) -> None:
        self.source_panel.set_pen_count(len(self.settings.library))
        self.source_panel.refresh_shapes_pens()
        self.source_panel.refresh_pen_scaled_hints()
        # every pen-relative setting - hatch spacing, dot pitch, contour count -
        # moves with the width, so the tiles have to be redrawn too
        self.source_panel.refresh_gallery(
            max(self.job.target_size[0], 20.0) if self.job else 120.0
        )
        self.preview.update()
        kind = self.settings.item.kind
        if kind == "shapes" or (kind == "text" and self.settings.text.pen_mode != "single"):
            # these sources bake the pen into their geometry, so they have to be
            # rebuilt rather than just re-optimised
            self.source_panel.rebuild_source()
        else:
            self.schedule_rebuild()

    def _on_machine_settings_changed(self) -> None:
        """Something outside the Prepare stage changed a machine setting."""
        self.program = None
        self._path_lines = []
        self.machine_panel.refresh()
        self.schedule_rebuild()

    def _on_machine_changed(self) -> None:
        self.machine_label.setText(self.settings.machine.name)
        self.schedule_rebuild()

    # ------------------------------------------------------------------
    # automatic setup
    # ------------------------------------------------------------------
    def _on_auto_tune(self, minutes: float, choose_technique: bool) -> None:
        source = self.source_panel.source
        if source is None or not source.is_raster:
            self.source_panel.show_tune_result("Load a picture first.")
            return
        if not self.tuner.submit(source, self.settings, minutes, choose_technique):
            self.source_panel.show_tune_result("Already working…")

    def _on_tune_ready(self, result, tuned) -> None:
        """Copy what the worker decided back onto the live settings."""
        if result is None or tuned is None:
            return
        style = self.settings.style
        for name in (
            "technique", "auto_levels", "brightness", "contrast", "gamma",
            "black_point", "white_point", "saturation",
        ):
            setattr(style, name, getattr(tuned.style, name))
        style.params.update(copy.deepcopy(tuned.style.params))

        self._suppress_rebuild = True
        self.source_panel.refresh()
        self._suppress_rebuild = False

        technique = techniques.REGISTRY.get(result.technique)
        label = technique.label if technique else result.technique
        notes = (" " + " ".join(result.notes)) if result.notes else ""
        self.source_panel.show_tune_result(
            f"{label}: about {result.minutes:.0f} min after {result.iterations} passes.{notes}"
        )
        self.render_now()

    def _zoom_interface(self, delta: float) -> None:
        """Resize the interface itself.  0 puts it back to as designed.

        This is what most people mean by cmd-plus in an application: the panels
        and the type, not the picture.  The canvas keeps plain + and - and F,
        so both zooms are available and neither steals the other's keys.
        """
        wanted = 1.0 if delta == 0.0 else theme.SCALE + delta
        if not theme.set_scale(QApplication.instance(), wanted):
            self.statusBar().showMessage(
                f"Interface already at its {'largest' if delta > 0 else 'smallest'}"
                f" ({theme.SCALE * 100:.0f}%)", 3000
            )
            return
        self.settings.ui_scale = theme.SCALE

    def _apply_scale(self) -> None:
        """Re-apply the pinned sizes after the interface has been resized.

        The stylesheet carries the type and the padding, but a button pinned at
        26 px stays 26 px however big its glyph gets, so the pinned numbers have
        to go back through the scale as well.
        """
        self._options_bar.setIconSize(QSize(theme.px(18), theme.px(18)))
        self._rail.setIconSize(QSize(theme.px(18), theme.px(18)))
        for button in list(self.tool_buttons.values()) + [self.undo_stroke_button]:
            button.setFixedSize(theme.px(26), theme.px(26))
        self.tool_name.setMinimumWidth(theme.px(62))
        self.draw_pen_combo.setMinimumWidth(theme.px(150))
        self.zoom_label.setMinimumWidth(theme.px(40))
        self.progress_slider.setFixedWidth(theme.px(88))
        self.estimate_label.setMinimumWidth(theme.px(80))
        filter_control = self.source_panel.group_filter
        filter_control.setMinimumWidth(0)
        filter_control.setMinimumWidth(filter_control.sizeHint().width())
        for dock in self._docks.values():
            holder = dock.widget()
            if isinstance(holder, QScrollArea):
                # a minimum, never a fixed width - the dock stays draggable
                holder.setMinimumWidth(theme.px(300))
        self._strip_host.setMinimumHeight(theme.px(150))
        self.source_panel.gallery.set_tile_size(theme.px(96))
        self.setMinimumSize(theme.px(1100), theme.px(640))
        self.statusBar().showMessage(f"Interface at {theme.SCALE * 100:.0f}%", 2500)

    def _sync_strip(self) -> None:
        """A vector PDF has no technique to choose, so do not park an empty
        two-hundred-pixel drawer under the canvas for it."""
        kind = self.settings.item.kind
        raster_source = kind == "image" or (kind == "pdf" and self.settings.pdf.mode == "render")
        if hasattr(self, "_strip_host"):
            self._strip_host.setVisible(raster_source)

    def _on_job_ready(self, job: PlotJob) -> None:
        self.job = job
        self.program = None
        self._path_lines = []
        self._last_build_ms = job.stats.build_seconds * 1000.0
        current = self._submitted_serial == self._settings_serial
        if current:
            self.preview.set_stale(False)
        if not self._printing:
            self.progress_slider.setValue(1000)
            self.preview.set_progress(1.0)
        self.preview.set_job(
            job,
            self.settings.library,
            (self.settings.machine.bed_x, self.settings.machine.bed_y),
            self.settings.layout.margin,
            (self.settings.machine.park_x, self.settings.machine.park_y),
        )
        self.preview.set_selected_item(self.settings.active)
        self.layout_panel.update_info(job)
        self._sync_strip()
        if self._fit_after_load and not job.is_empty:
            self._fit_after_load = False
            self.preview.fit_view()

        stats = job.stats
        # A draft is a coarse pass so the canvas can keep up with a dragged
        # control.  Its stroke count and time are not the real ones - a lower
        # working resolution breaks strokes into more, shorter pieces - so the
        # numbers wait for the full pass rather than telling the user that a
        # 37-minute drawing takes 231.
        if self._submitted_draft and not job.is_empty:
            self.estimate_label.setText("…")
            self.summary_label.setText("measuring…")
        elif job.is_empty:
            self.estimate_label.setText("—")
            self.summary_label.setText("Nothing to draw yet")
        else:
            self.estimate_label.setText(gcode.format_duration(stats.estimated_seconds))
            pens = len(job.drawing.used_pens())
            # "818 strokes ... 818 pen lifts" said the same number twice: every
            # stroke is one pen lift by definition.
            parts = [
                f"{stats.path_count:,} strokes",
                f"{stats.draw_length/1000:.1f} m of line",
                f"{stats.travel_length/1000:.1f} m of travel",
                f"{pens} pen{'s' if pens != 1 else ''}",
            ]
            if stats.dot_count:
                parts.insert(1, f"{stats.dot_count:,} dots")
            self.summary_label.setText(" · ".join(parts))
        self.save_button.setEnabled(not job.is_empty)
        self.send_button.setEnabled(not job.is_empty)

        if self.settings.source_kind == "image" or (
            self.settings.source_kind == "pdf" and self.settings.pdf.mode == "render"
        ):
            self.source_panel.refresh_gallery(max(job.target_size[0], 20.0))

        # A draft pass keeps the canvas moving while a control is being dragged;
        # once nothing has changed for a moment, quietly redo it properly.
        if self._submitted_draft and current and not self._full_scheduled:
            # `_submitted_draft` stays true until the full pass actually lands,
            # so nothing - not the action bar, not Save, not a test - can mistake
            # the coarse pass for the finished job during the gap.
            self._full_scheduled = True
            QTimer.singleShot(200, self._full_pass)

        if self._pending_action and self._up_to_date:
            action, self._pending_action = self._pending_action, None
            if action == "save":
                self.save_gcode()
            elif action == "send":
                self.send_to_printer()

        for warning in job.warnings:
            self.statusBar().showMessage(warning, 6000)

    def _full_pass(self) -> None:
        """Re-render the draft at full quality, unless something moved on."""
        self._full_scheduled = False
        if self._render_in_flight or self._submitted_serial != self._settings_serial:
            return
        self._rebuild_now(draft=False)

    def _on_job_failed(self, error: str) -> None:
        self._pending_action = None
        self.preview.clear_temporary_transform()
        self.statusBar().showMessage("Something went wrong while preparing the drawing", 8000)
        print(error)

    def _on_job_cancelled(self) -> None:
        self._pending_action = None
        self.preview.clear_temporary_transform()

    # ------------------------------------------------------------------
    # preview interaction
    # ------------------------------------------------------------------
    def _on_preview_move(self, dx: float, dy: float) -> None:
        layout = self.settings.layout
        if layout.mode == "asis":
            # An as-drawn offset is already absolute millimetres on the bed;
            # seeding it from the bounding box like the centred branch does
            # would make a sideways drag jump vertically as well.
            layout.offset_x += dx
            layout.offset_y += dy
            self.layout_panel.refresh()
            self._settings_serial += 1
            self._rebuild_timer.start()
            return
        if not layout.center:
            layout.center = True
            bounds = self.job.stats.bounds if self.job else None
            if bounds:
                layout.offset_x = (bounds[0] + bounds[2]) / 2.0 - self.settings.machine.bed_x / 2.0
                layout.offset_y = (bounds[1] + bounds[3]) / 2.0 - self.settings.machine.bed_y / 2.0
        layout.offset_x += dx
        layout.offset_y += dy
        self.layout_panel.refresh()
        # Dragging on the canvas always re-renders - waiting for the Render
        # button would look like the artwork refused to move.  The debounce
        # keeps a held-down arrow key from queueing one build per repeat.
        self._settings_serial += 1
        self._rebuild_timer.start()

    def _on_preview_scale(self, factor: float) -> None:
        layout = self.settings.layout
        if layout.mode == "scale":
            layout.scale_percent = max(1.0, layout.scale_percent * factor)
        else:
            # scale the *canvas*, not the ink bounding box, or blank margins
            # would be squeezed out and the artwork would jump
            target = self.job.target_size if self.job else (0.0, 0.0)
            if target[0] > 0:
                layout.width = max(target[0] * factor, 2.0)
                layout.height = max(target[1] * factor, 2.0)
            layout.mode = "size"
        self.layout_panel.refresh()
        self.render_now()

    def _on_files_dropped(self, paths: list) -> None:
        """Every dropped file becomes its own layer.

        Loading into the selected layer would throw away whatever picture was
        already there - with no undo - and drop the rest of a multiple-file
        drop on the floor.
        """
        item = self.settings.item
        # a blank picture layer is fair game; a drawing layer was made on
        # purpose and is not a parking space for someone else's file
        reuse = not item.source_path and not item.strokes and item.kind != "shapes"
        loaded: list[str] = []
        rejected: list[str] = []
        self._suppress_rebuild = True
        try:
            for path in paths:
                kind = "pdf" if path.lower().endswith(".pdf") else "image"
                if not reuse:
                    self.settings.add_item(kind)
                    self.source_panel.select_item(self.settings.active)
                reuse = False
                if self.source_panel.load_any(path):
                    loaded.append(os.path.basename(path))
                else:
                    rejected.append(os.path.basename(path))
                    if len(self.settings.items) > 1:
                        # an empty layer left behind by a file that would not open
                        index = self.settings.active
                        self.settings.remove_item(index)
                        self.source_panel.forget_item(index)
                        self.source_panel.select_item(self.settings.active)
        finally:
            self._suppress_rebuild = False

        self.layers_panel.rebuild()
        self.layout_panel.refresh()
        self.preview.set_selected_item(self.settings.active)
        if loaded:
            self._fit_after_load = True
            self.render_now()
        message = ""
        if loaded:
            message = f"Loaded {', '.join(loaded)}" if len(loaded) < 4 else f"Loaded {len(loaded)} files"
        if rejected:
            message += ("  ·  " if message else "") + f"could not open {', '.join(rejected)}"
        self.statusBar().showMessage(message or "That file type is not supported", 6000)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _render_first(self, action: str) -> bool:
        """Re-render before writing G-code; the action resumes when it lands.

        This has to cover a render that is *already running* too - otherwise the
        G-code is built from the previous job while a newer one is in flight,
        and a draft pass must never become a file.
        """
        if self._up_to_date and not self._submitted_draft:
            return False
        self._pending_action = action
        self.statusBar().showMessage("Finishing the drawing first…", 3000)
        if not self._render_in_flight:
            self.render_now()
        return True

    def _confirm_out_of_bounds(self, action: str) -> bool:
        verb = "written to a file" if action == "save" else "sent to the printer"
        answer = QMessageBox.warning(
            self,
            "The drawing does not fit",
            f"Part of this drawing lies outside the bed, so what is {verb} will be "
            "cropped and the head will hit its limits.\n\nMove or scale it first.",
            QMessageBox.Cancel | QMessageBox.Ignore,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Ignore

    def _ensure_program(self) -> gcode.GCodeProgram | None:
        if self.job is None or self.job.is_empty:
            self.statusBar().showMessage("There is nothing to draw yet", 4000)
            return None
        if self.program is None:
            self.program = gcode.generate(self.job, self.settings, self.settings.library)
            self._path_lines = sorted(self.program.path_at.keys())
        return self.program

    def save_gcode(self) -> None:
        if self._render_first("save"):
            return
        program = self._ensure_program()
        if program is None:
            return
        if self.job.stats.out_of_bounds and not self._confirm_out_of_bounds("save"):
            return
        label = (self.job.drawing.source_label or "drawing").rsplit(".", 1)[0]
        start = os.path.join(self.settings.last_export_dir or os.path.expanduser("~"), f"{label}.gcode")
        path, _ = QFileDialog.getSaveFileName(self, "Save G-code", start, "G-code (*.gcode *.gco *.nc)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(program.text())
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.settings.last_export_dir = os.path.dirname(path)
        self.statusBar().showMessage(f"Saved {len(program.lines):,} lines to {os.path.basename(path)}", 6000)

    def send_to_printer(self) -> None:
        if self._render_first("send"):
            return
        program = self._ensure_program()
        if program is None:
            return
        if not self.printer.is_connected:
            self._set_stage(1)
            QMessageBox.information(
                self,
                "Not connected",
                "Connect to the printer first: pick the USB port on the left and press Connect.",
            )
            return

        if self.job.stats.out_of_bounds and not self._confirm_out_of_bounds("send"):
            return

        pens = [self.settings.library[i].name for i in self.job.drawing.used_pens()]
        cutting = [self.settings.library[i] for i in self.job.drawing.used_pens()
                   if self.settings.library[i].cuts]

        message = (
            f"Ready to draw {self.job.stats.draw_length/1000:.1f} m of line in about "
            f"{gcode.format_duration(self.job.stats.estimated_seconds)}.\n\n"
            f"Pens, in order: {', '.join(pens)}\n"
            f"Pen changes: {program.pen_changes}   ·   sharpening stops: {program.sharpen_stops}\n\n"
        )
        if cutting:
            tools = ", ".join(f"{p.name} ({p.repeats}×)" for p in cutting)
            message += f"This job cuts, it does not draw: {tools}\n\n"
        # warnings belong here, not in a status bar the dialog is about to cover
        for warning in getattr(program, "warnings", []):
            message += f"⚠  {warning}\n\n"
        if self.settings.pen.zero_z_at_start and not getattr(self.printer, "pen_zeroed", False):
            # Without a reference the file homes, lifts and then draws the whole
            # picture in the air - which reads as "it only goes up in Z".
            from .pen_height_dialog import ask_to_calibrate

            box = QMessageBox(self)
            box.setWindowTitle("The pen height is not set")
            box.setIcon(QMessageBox.Warning)
            box.setText("The pen height has not been set since you connected.")
            box.setInformativeText(
                "This file takes the height the pen is at right now as Z0. If the pen is "
                "not touching the paper, the printer will lift and draw the whole picture "
                "in mid-air without ever marking anything."
            )
            setup = box.addButton("Set the pen height…", QMessageBox.AcceptRole)
            anyway = box.addButton("It is already touching", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(setup)
            box.exec()
            if box.clickedButton() is setup:
                if not ask_to_calibrate(self, self.settings, self.printer):
                    return
            elif box.clickedButton() is anyway:
                self.printer.pen_zeroed = True
            else:
                return

        if self.settings.pen.zero_z_at_start:
            message += "The pen height is set. Z0 is where the tip is now."
        else:
            message += "Z is used as the printer currently has it. Make sure the pen is safely above the bed."

        answer = QMessageBox.question(
            self, "Start drawing?", message, QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok
        )
        if answer != QMessageBox.Ok:
            return

        self._set_stage(1)
        self._printing = True
        self.job_panel.job_started(len(program.lines), pens, program.estimated_seconds)
        self.printer.start(program, skip_m0=self.settings.pauses.host_pause)
        self.preview.set_progress(0.0)
        self.progress_slider.setValue(0)

    # ------------------------------------------------------------------
    # printer feedback
    # ------------------------------------------------------------------
    def _on_printer_progress(self, line: int, total: int, drawn: float) -> None:
        if self._path_lines:
            done = bisect.bisect_right(self._path_lines, line)
            self.preview.set_progress(done / len(self._path_lines))
        elif total:
            self.preview.set_progress(line / total)

    def _on_printer_paused(self, message: str) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Paused")
        box.setIcon(QMessageBox.Information)
        box.setText(message)
        box.setInformativeText(
            "The head is parked and waiting. Swap or sharpen the pen, then press Continue.\n"
            "Do not switch the printer off - it would lose the Z reference."
        )
        continue_button = box.addButton("Continue", QMessageBox.AcceptRole)
        box.addButton("Stop the job", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is continue_button:
            self.printer.resume()
        else:
            self.printer.cancel(True)

    def _on_printer_finished(self, ok: bool, message: str) -> None:
        self._printing = False
        self.statusBar().showMessage(message, 8000)
        if ok:
            self.preview.set_progress(1.0)
            self.preview.set_live_position(None)

    # ------------------------------------------------------------------
    def reset_settings(self) -> None:
        answer = QMessageBox.question(
            self, "Reset settings", "Put every setting back to its default value?"
        )
        if answer != QMessageBox.Yes:
            return
        # copy field by field: the panels hold references to these very objects,
        # so replacing them would leave every control writing to a dead copy
        fresh = AppSettings()
        for name in ("machine", "pen", "style", "layout", "optimize", "pauses", "text", "pdf"):
            target = getattr(self.settings, name)
            source = getattr(fresh, name)
            for field in dataclasses.fields(source):
                setattr(target, field.name, getattr(source, field.name))
        self.settings.library.pens = fresh.library.pens
        self._suppress_rebuild = True
        for panel in (self.source_panel, self.layout_panel, self.pens_panel, self.machine_panel):
            panel.refresh()
        self._suppress_rebuild = False
        self.source_panel.rebuild_source()

    # ------------------------------------------------------------------
    # test patterns and SVG
    # ------------------------------------------------------------------
    def fit_settings(self) -> None:
        """Search this technique's own settings for the closest match to the picture.

        Deliberately a thing you ask for rather than something that happens to
        you: it takes a few seconds, it changes settings you may have chosen on
        purpose, and on some techniques it finds nothing worth changing - all of
        which it says afterwards instead of pretending.
        """
        from ..core import ai, fit, geometry as geo, raster, techniques

        rgb = self.source_panel.active_image()
        if rgb is None:
            QMessageBox.information(
                self, "Fit the settings",
                "Select a picture layer first - there is nothing to match against.",
            )
            return
        style = self.settings.style
        technique = techniques.REGISTRY.get(style.technique)
        if technique is None or not fit.TUNABLE.get(style.technique):
            self.statusBar().showMessage(
                f"{technique.label if technique else style.technique} has nothing worth searching", 6000
            )
            return

        target = self.job.target_size[0] if self.job and self.job.target_size[0] > 0 else 150.0
        prepared = raster.prepare(
            rgb, detail=style.detail, brightness=style.brightness, contrast=style.contrast,
            gamma=style.gamma, blur=style.blur, invert=style.invert,
            auto_levels=style.auto_levels, black_point=style.black_point,
            white_point=style.white_point, saturation=style.saturation,
        )
        gray = raster.to_gray(prepared)
        pen = min((p.width for p in self.settings.library if p.enabled), default=0.5)
        context = techniques.Context(
            px_per_mm=prepared.shape[1] / max(target, 1.0),
            pen_width=pen,
            scale_with_pen=self.settings.pen.scale_with_pen_width,
        )
        machine = self.settings.machine

        def minutes_of(paths) -> float:
            metres = sum(geo.path_length(p) for p in paths) / context.px_per_mm / 1000.0
            lifts = len(paths) * (self.settings.pen.lift * 2.0) / max(machine.z_feed, 1.0)
            return metres * 1000.0 / max(machine.draw_feed, 1.0) + lifts

        faces = ()
        try:
            faces = ai.read(rgb).faces
        except Exception:  # pragma: no cover - defensive
            pass

        self.statusBar().showMessage("Fitting the settings to the picture…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = fit.fit_technique(
                gray, style.technique, dict(style.technique_params()), context,
                target_minutes=float(self.source_panel.target_minutes.value()),
                minutes_of=minutes_of, faces=faces, budget_seconds=12.0,
            )
        finally:
            QApplication.restoreOverrideCursor()

        if not result.params:
            self.statusBar().showMessage("Nothing worth changing for this technique", 6000)
            return
        style.technique_params().update(result.params)
        self.source_panel._rebuild_params()
        self.source_panel.binder.refresh()
        self.schedule_rebuild()
        changed = ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                            for k, v in result.params.items())
        notes = ("  ·  " + "; ".join(result.notes)) if result.notes else ""
        self.statusBar().showMessage(f"{result.summary()}  ·  {changed}{notes}", 15000)

    def make_stencils(self) -> None:
        """Split the selected picture into spray stencils, one cut layer each."""
        rgb = self.source_panel.active_image()
        if rgb is None:
            QMessageBox.information(
                self,
                "Make stencils",
                "Select a picture layer first - stencils are cut from an image.\n\n"
                "Open an image, then Tools ▸ Make stencils.",
            )
            return

        from .stencil_dialog import StencilDialog

        dialog = StencilDialog(rgb, self.settings, self)
        if dialog.exec() != QDialog.Accepted or not dialog.sheets:
            return

        pen = dialog.result_pen()
        sheets = dialog.sheet_strokes()
        self._suppress_rebuild = True
        for index, (name, strokes) in enumerate(sheets):
            self.settings.add_item("shapes")
            item = self.settings.item
            item.name = name
            item.strokes = strokes
            item.pen = pen
            # you cut one sheet of card at a time, so only the first is on
            item.visible = index == 0
            self.source_panel.select_item(self.settings.active)
        self.settings.active = len(self.settings.items) - len(sheets)
        self.source_panel.select_item(self.settings.active)
        self._suppress_rebuild = False

        self.pens_panel.refresh()
        self.layers_panel.rebuild()
        self.preview.set_selected_item(self.settings.active)
        self.render_now()
        tool = self.settings.library[pen]
        self.statusBar().showMessage(
            f"{len(sheets)} stencil layers added, cut with {tool.name} "
            f"({tool.repeats} pass{'es' if tool.repeats > 1 else ''}). "
            "Only the first sheet is switched on - cut it, then show the next one.",
            12000,
        )

    def _pattern_action(self, label: str, kind: str) -> QAction:
        action = QAction(label + "…", self)
        action.triggered.connect(lambda _checked=False, k=kind: self._draw_pattern(k))
        return action

    def _draw_pattern(self, kind: str) -> None:
        """Put a calibration pattern on the bed instead of the artwork."""
        machine, pen, library = self.settings.machine, self.settings.pen, self.settings.library
        try:
            if kind == "z":
                job = testpattern.z_ladder(machine, pen, library)
                name = "pen height ladder"
            elif kind == "speed":
                job = testpattern.speed_ladder(machine, pen, library)
                name = "speed ladder"
            elif kind == "pen":
                job = testpattern.pen_test(machine, pen, library)
                name = "pen test sheet"
            else:
                job = testpattern.registration(machine, pen, library)
                name = "registration sheet"
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Could not build the pattern", str(exc))
            return

        self._on_job_ready(job)
        self.preview.fit_view()
        self.statusBar().showMessage(
            f"Showing the {name}. Save or send it like any drawing; "
            "changing a setting and pressing Render brings your artwork back.",
            10000,
        )

    def _explain_patterns(self) -> None:
        QMessageBox.information(
            self,
            "Test patterns",
            "<b>Pen height ladder</b> - the same strokes at thirteen pen heights, labelled. "
            "Draw it once, look at which row is crisp without digging in, and set that as "
            "your drawing Z.<br><br>"
            "<b>Speed ladder</b> - the same strokes at eight feed rates. A pen that skips at "
            "3000 mm/min is often perfect at 1200.<br><br>"
            "<b>Pen test sheet</b> - fills, crosshatch, dots and a spiral at several line "
            "spacings, so you can see how your pen behaves before committing to an hour of "
            "plotting.<br><br>"
            "<b>Registration sheet</b> - corner crosses and a millimetre ruler for checking "
            "that the machine's scale and squareness are right.",
        )

    def save_svg(self) -> None:
        if self.job is None or self.job.is_empty:
            self.statusBar().showMessage("There is nothing to export yet", 4000)
            return
        label = (self.job.drawing.source_label or "drawing").rsplit(".", 1)[0]
        start = os.path.join(self.settings.last_export_dir or os.path.expanduser("~"), f"{label}.svg")
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", start, "SVG (*.svg)")
        if not path:
            return
        try:
            svgexport.save_svg(path, self.job, self.settings.library, self.settings.machine)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.settings.last_export_dir = os.path.dirname(path)
        self.statusBar().showMessage(f"Exported {os.path.basename(path)}", 6000)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "PenPlot Studio",
            "<b>PenPlot Studio</b><br><br>"
            "Turns pictures, text and PDFs into pen drawings on a 3D printer.<br>"
            "Made for an Ender 3 with a pen clamped to the hot end.<br><br>"
            "Nothing in the generated G-code extrudes filament or heats anything up.",
        )

    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._on_files_dropped(paths)
            event.acceptProposedAction()

    def closeEvent(self, event):  # noqa: N802
        if self._printing and self.printer.is_connected:
            answer = QMessageBox.question(
                self,
                "The printer is still drawing",
                "Closing now stops the drawing where it is, and the pen height "
                "reference is lost.\n\nStop and close anyway?",
                QMessageBox.Cancel | QMessageBox.Close,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Close:
                event.ignore()
                return
            self.printer.cancel(True)

        try:
            self.settings.window_state = bytes(self.saveState().toBase64()).decode("ascii")
            self.settings.save()
        except Exception:
            pass
        self.printer.shutdown()
        self.runner.shutdown()
        self.tuner.shutdown()
        self.source_panel.gallery.shutdown()
        self.source_panel.close_documents()
        super().closeEvent(event)
