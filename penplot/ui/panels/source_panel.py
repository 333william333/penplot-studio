"""Left column: what to draw and how it should be turned into strokes."""

from __future__ import annotations

import dataclasses

import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core import ai, autotune, looks, raster, separation, textsource
from ...core.drawing import SourceResult
from ...core.pdfsource import TEXT_MODES, PdfDocument
from ...core import techniques
from ...core.settings import AppSettings, StyleSettings
from .. import theme
from ..gallery import TechniqueGallery
from ..widgets import Binder, Card, FieldRow, Segmented, SliderSpin, hint_label

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp);;All files (*)"
PDF_FILTER = "PDF documents (*.pdf);;All files (*)"


class SourcePanel(QWidget):
    """Image / Text / PDF input plus the drawing-style settings."""

    file_loaded = Signal()         # a new file was opened - always worth rendering
    auto_tune_requested = Signal(float, bool)   # target minutes, also pick technique
    source_changed = Signal()      # the artwork itself changed - reload needed
    settings_changed = Signal()    # only style settings changed - re-run pipeline
    palette_suggested = Signal(list)
    status = Signal(str)

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        # one source per item, keyed by its index in the project
        self.sources: dict[int, SourceResult] = {}
        self._images: dict[int, np.ndarray] = {}
        self._pdfs: dict[int, PdfDocument] = {}
        self._pen_count = 1

        self.binder = Binder(self._on_settings_changed, self, context=settings)
        self.source_binder = Binder(self._on_source_settings_changed, self, context=settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_source_card())
        layout.addWidget(self._build_shapes_card())
        layout.addWidget(self._build_auto_card())
        layout.addWidget(self._build_style_card())
        layout.addWidget(self._build_params_card())
        layout.addWidget(self._build_adjust_card())
        layout.addWidget(self._build_colour_card())
        layout.addStretch(1)

        self._sync_visibility()

    # ------------------------------------------------------------------
    # source card
    # ------------------------------------------------------------------
    def _build_source_card(self) -> Card:
        card = Card("SOURCE")
        self.source_card = card
        self.kind_switch = Segmented([("image", "Image"), ("text", "Text"), ("pdf", "PDF")])
        self.kind_switch.setValue(self.settings.source_kind)
        self.kind_switch.changed.connect(self._on_kind_changed)
        card.add(self.kind_switch)

        # plain show/hide instead of a QStackedWidget: hidden pages then take no
        # vertical space at all, so the card is always exactly as tall as it needs
        self.pages = [self._build_image_page(), self._build_text_page(), self._build_pdf_page()]
        for page in self.pages:
            card.add(page)
        self._show_page(self.page_index())
        return card

    def _build_shapes_card(self) -> Card:
        """Hand-drawn and cut layers: which tool draws them, and what is on them."""
        card = Card("DRAWING LAYER")
        self.shapes_card = card
        self.shapes_pen = QComboBox()
        self.shapes_pen.currentIndexChanged.connect(self._on_shapes_pen)
        card.add(FieldRow("Tool", self.shapes_pen, "Which pen or blade draws this layer"))
        self.shapes_info = hint_label("")
        card.add(self.shapes_info)
        return card

    def refresh_shapes_pens(self) -> None:
        if not hasattr(self, "shapes_pen"):
            return
        self.shapes_pen.blockSignals(True)
        self.shapes_pen.clear()
        for index, pen in enumerate(self.settings.library):
            mark = " ✂" if pen.cuts else ""
            self.shapes_pen.addItem(f"{index + 1}. {pen.name}{mark}", index)
        current = max(0, min(int(getattr(self.settings.item, "pen", 0)), len(self.settings.library) - 1))
        self.shapes_pen.setCurrentIndex(current)
        self.shapes_pen.blockSignals(False)

        item = self.settings.item
        pen = self.settings.library[current]
        if pen.cuts:
            detail = (
                f"{pen.kind_label}. Every stroke is cut {pen.repeats} time"
                f"{'s' if pen.repeats > 1 else ''}"
            )
            if pen.pass_depth > 0:
                detail += f", {pen.pass_depth:.2f} mm deeper each time"
            detail += "."
        else:
            detail = f"{pen.kind_label}, {pen.width:.2f} mm."
        self.shapes_info.setText(f"{len(item.strokes)} strokes on this layer. {detail}")

    def _on_shapes_pen(self, index: int) -> None:
        value = self.shapes_pen.itemData(index)
        if value is None:
            return
        self.settings.item.pen = int(value)
        self.refresh_shapes_pens()
        self.rebuild_source()

    def page_index(self) -> int:
        return {"image": 0, "text": 1, "pdf": 2}.get(self.settings.source_kind, 0)

    def _show_page(self, index: int) -> None:
        for i, page in enumerate(self.pages):
            page.setVisible(i == index)

    def _build_image_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        row = QHBoxLayout()
        open_button = QPushButton("Open image…")
        open_button.clicked.connect(self.open_image_dialog)
        row.addWidget(open_button, 1)
        layout.addLayout(row)

        self.image_name = QLabel("No image loaded")
        self.image_name.setObjectName("Hint")
        self.image_name.setWordWrap(True)
        layout.addWidget(self.image_name)

        self.thumbnail = QLabel()
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setMinimumHeight(96)
        self.thumbnail.setMaximumHeight(210)
        self.thumbnail.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.thumbnail.setStyleSheet(
            f"background: {theme.PANEL_ALT}; border: 1px dashed {theme.BORDER_STRONG}; border-radius: 6px;"
        )
        self.thumbnail.setText("drop an image here")
        layout.addWidget(self.thumbnail)
        return page

    def _build_text_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(7)
        text = self.settings.text

        layout.addWidget(self.source_binder.multiline("text", "text", height=84, placeholder="Type the text to draw…"))

        # The font and its weight have to be applied together: updating the
        # family first and the weight afterwards rendered one build with the
        # previous font's weight, and then left the two out of step.
        families = textsource.available_families()
        self.font_combo = QComboBox()
        for family in families:
            self.font_combo.addItem(family, family)
        index = self.font_combo.findData(text.font)
        self.font_combo.setCurrentIndex(max(index, 0))
        self.font_combo.setMaxVisibleItems(24)
        self.font_combo.currentIndexChanged.connect(self._on_font_changed)
        self.font_row = FieldRow("Font", self.font_combo)
        layout.addWidget(self.font_row)

        self.style_combo = QComboBox()
        self.style_combo.currentIndexChanged.connect(self._on_weight_changed)
        self.style_row = FieldRow("Weight", self.style_combo)
        layout.addWidget(self.style_row)
        self._reload_weights(text.style_name)

        layout.addWidget(self.source_binder.slider("text", "size_mm", 2.0, 200.0, label="Cap height", decimals=1, step=1.0, suffix="mm"))
        layout.addWidget(self.source_binder.slider("text", "line_spacing", 0.8, 4.0, label="Line spacing", decimals=2, step=0.05, suffix="×"))
        layout.addWidget(self.source_binder.slider("text", "letter_spacing", -2.0, 10.0, label="Letter spacing", decimals=2, step=0.1, suffix="mm"))
        layout.addWidget(
            self.source_binder.segmented(
                "text", "align", [("left", "Left"), ("center", "Centre"), ("right", "Right")], label="Align"
            )
        )

        self.text_fill_check = self.source_binder.check("text", "fill", "Fill outline fonts")
        layout.addWidget(self.text_fill_check)
        self.text_fill_spacing = self.source_binder.slider(
            text, "fill_spacing", 0.1, 3.0, label="Fill spacing", decimals=2, step=0.05, suffix="mm"
        )
        layout.addWidget(self.text_fill_spacing)
        self.text_fill_angle = self.source_binder.slider(
            text, "fill_angle", 0.0, 180.0, label="Fill angle", decimals=0, step=5, suffix="°"
        )
        layout.addWidget(self.text_fill_angle)
        self.text_keep_outline = self.source_binder.check("text", "keep_outline", "Draw the outline as well")
        layout.addWidget(self.text_keep_outline)

        self.text_double = self.source_binder.check("text", "double_stroke", "Double stroke (bolder)")
        layout.addWidget(self.text_double)

        layout.addWidget(
            self.source_binder.combo(
                text,
                "pen_mode",
                {
                    "single": "One pen for everything",
                    "line": "New pen per line",
                    "word": "New pen per word",
                    "char": "New pen per character",
                },
                label="Pens",
            )
        )
        self.text_fill_check.toggled.connect(lambda _: self._sync_visibility())
        return page

    def _build_pdf_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(7)
        pdf = self.settings.pdf

        open_button = QPushButton("Open PDF…")
        open_button.clicked.connect(self.open_pdf_dialog)
        layout.addWidget(open_button)

        self.pdf_name = QLabel("No PDF loaded")
        self.pdf_name.setObjectName("Hint")
        self.pdf_name.setWordWrap(True)
        layout.addWidget(self.pdf_name)

        self.pdf_page_row = self.source_binder.slider("pdf", "page", 0, 0, label="Page", decimals=0, step=1)
        layout.addWidget(self.pdf_page_row)

        layout.addWidget(
            self.source_binder.segmented(
                "pdf", "mode", [("vector", "Vector"), ("render", "Render page")], label="Mode",
                hint="Vector follows the real line work in the file. Render turns the page into an image first.",
            )
        )
        self.pdf_text_row = self.source_binder.combo("pdf", "text_mode", TEXT_MODES, label="Text")
        layout.addWidget(self.pdf_text_row)
        self.pdf_fills_row = self.source_binder.check("pdf", "keep_fills", "Include filled shapes")
        layout.addWidget(self.pdf_fills_row)
        self.pdf_dpi_row = self.source_binder.slider("pdf", "dpi", 72, 1200, label="Render DPI", decimals=0, step=25)
        layout.addWidget(self.pdf_dpi_row)
        return page

    def _build_auto_card(self) -> Card:
        """The Look: one decision that carries the technique and its settings.

        Twenty-one techniques is a menu nobody can taste their way through, and
        Hick's law says the choice itself is the cost.  A Look answers the
        question the user actually has - "what kind of picture is this" - and
        the technique falls out of that.  The whole gallery is still there for
        anyone who wants to go picking.
        """
        card = Card("LOOK")
        self.auto_card = card

        self.look_combo = QComboBox()
        for key, label in looks.look_labels().items():
            self.look_combo.addItem(label, key)
        self.look_combo.currentIndexChanged.connect(self._on_look_changed)
        card.add(FieldRow("Look", self.look_combo, label_width=76))

        self.look_note = hint_label("")
        card.add(self.look_note)

        self.look_suggestion = QPushButton("")
        self.look_suggestion.setObjectName("Ghost")
        self.look_suggestion.setVisible(False)
        self.look_suggestion.clicked.connect(self._accept_suggestion)
        card.add(self.look_suggestion)

        # what the engine knows, and how to give it more to work with
        self.engine_note = hint_label("")
        card.add(self.engine_note)
        self.engine_button = QPushButton("Enable face detection…")
        self.engine_button.setObjectName("Ghost")
        self.engine_button.setToolTip(
            "Downloads a 232 KB face detector (YuNet, Apache-2.0, from the OpenCV Zoo) "
            "so the app can tell a person from a bowl of fruit"
        )
        self.engine_button.clicked.connect(self._enable_engine)
        card.add(self.engine_button)

        timing = Card("DRAWING TIME", collapsible=True, expanded=False)
        self.timing_card = timing
        timing.add(hint_label(
            "Tunes the density until the drawing lands near the time you ask for."
        ))
        self.target_minutes = SliderSpin(2, 300, decimals=0, step=5, suffix="min")
        self.target_minutes.setValue(25)
        timing.add(FieldRow("Target", self.target_minutes))
        self.tune_button = QPushButton("Fit to this time")
        self.tune_button.setToolTip("Adjust the density so the drawing takes about that long")
        self.tune_button.clicked.connect(lambda: self._request_tune(False))
        timing.add(self.tune_button)
        self.auto_result = QLabel("")
        self.auto_result.setObjectName("Hint")
        self.auto_result.setWordWrap(True)
        timing.add(self.auto_result)
        card.add(timing)
        return card

    def _on_look_changed(self, index: int) -> None:
        key = self.look_combo.itemData(index)
        if not key:
            return
        stats = self._stats_for_current()
        rgb = self._image_rgb
        reading = self._read_with_engine(rgb) if rgb is not None else None
        looks.apply_look(key, self.settings.style, stats)
        self.settings.style.look = key
        self._rebuild_params()
        self.binder.refresh()
        self._sync_visibility()
        self._update_look_note(stats, reading)
        self.settings_changed.emit()

    def _accept_suggestion(self) -> None:
        key = self.look_suggestion.property("look") or ""
        if key:
            self.look_combo.setCurrentIndex(max(self.look_combo.findData(key), 0))

    def _stats_for_current(self):
        rgb = self._image_rgb
        if rgb is None:
            return None
        try:
            return autotune.analyse(rgb)
        except Exception:  # pragma: no cover - defensive
            return None

    def _update_engine_row(self, reading=None) -> None:
        """Say which engine answered, and never imply more than it knows."""
        installed = ai.have_model()
        self.engine_button.setVisible(not installed)
        if reading is None:
            self.engine_note.setText("")
            return
        if reading.backend == "neural":
            self.engine_note.setText(f"Face detection: {reading.summary()}")
        else:
            self.engine_note.setText(
                f"Reading the picture without a face detector. {reading.summary()}."
            )

    def _enable_engine(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self,
            "Download the face detector?",
            f"This fetches {ai.MODEL['bytes'] / 1024:.0f} KB from the OpenCV Zoo "
            f"({ai.MODEL['licence']}).\n\n"
            "With it the app can tell that a picture is of a person, instead of "
            "guessing from where the sharp part of the frame is.\n\n"
            "Everything works without it.",
            QMessageBox.Cancel | QMessageBox.Ok,
            QMessageBox.Ok,
        )
        if answer != QMessageBox.Ok:
            return
        self.engine_button.setEnabled(False)
        self.engine_button.setText("Downloading…")
        ok, message = ai.download_model()
        self.engine_button.setEnabled(True)
        self.engine_button.setText("Enable face detection…")
        self.status.emit(message)
        if ok:
            self.engine_button.setVisible(False)
            rgb = self._image_rgb
            if rgb is not None:
                stats = self._stats_for_current()
                reading = self._read_with_engine(rgb)
                chosen = looks.choose(stats, reading)
                looks.apply_look(chosen, self.settings.style, stats)
                self.settings.style.look = chosen
                self.look_combo.blockSignals(True)
                self.look_combo.setCurrentIndex(max(self.look_combo.findData(chosen), 0))
                self.look_combo.blockSignals(False)
                self._update_look_note(stats, reading)
                self._rebuild_params()
                self.binder.refresh()
                self.settings_changed.emit()

    def _read_with_engine(self, rgb):
        try:
            return ai.read(rgb)
        except Exception:  # pragma: no cover - the engine must never break loading
            return None

    def _update_look_note(self, stats=None, reading=None) -> None:
        key = self.settings.style.look or ""
        self.look_note.setText(looks.describe(key, stats, reading) if key else "")
        self._update_engine_row(reading)
        nudge = looks.suggestion(stats, reading) if stats is not None else ""
        show = bool(nudge) and nudge != key
        self.look_suggestion.setVisible(show)
        if show:
            self.look_suggestion.setProperty("look", nudge)
            self.look_suggestion.setText(
                f"This looks like it might be a person - try {looks.LOOKS[nudge].label}"
            )

    def _request_tune(self, choose: bool = False) -> None:
        self.auto_result.setText("Working…")
        self.auto_tune_requested.emit(float(self.target_minutes.value()), choose)

    def set_tuning(self, busy: bool) -> None:
        self.tune_button.setEnabled(not busy)
        self.tune_button.setText("Tuning…" if busy else "Auto-tune")

    def show_tune_result(self, text: str) -> None:
        self.auto_result.setText(text)

    # ------------------------------------------------------------------
    # technique gallery + generated settings
    # ------------------------------------------------------------------
    def _build_style_card(self) -> Card:
        card = Card("DRAWING TECHNIQUE")
        self.style_card = card

        self.group_filter = Segmented(
            [("all", "All")] + [(key, label.split()[0]) for key, label in techniques.GROUPS.items()]
        )
        self.group_filter.setValue("all")
        self.group_filter.changed.connect(self._filter_gallery)

        self.gallery = TechniqueGallery(columns=2)
        self.gallery.set_current(self.settings.style.technique)
        self.gallery.technique_chosen.connect(self._on_technique_chosen)
        card.add(self.gallery)

        self.style_hint = hint_label("")
        card.add(self.style_hint)

        self.requirement_note = QLabel("")
        self.requirement_note.setWordWrap(True)
        self.requirement_note.setVisible(False)
        self.requirement_note.setStyleSheet(
            f"color: {theme.WARNING}; background: #FFF7EB; border: 1px solid {theme.WARNING};"
            f" border-radius: 6px; padding: 6px 8px; font-size: 11px;"
        )
        card.add(self.requirement_note)
        return card

    def _build_params_card(self) -> Card:
        card = Card("TECHNIQUE SETTINGS")
        self.params_card = card
        self.params_host = QWidget()
        self.params_layout = QVBoxLayout(self.params_host)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(7)
        card.add(self.params_host)

        card.add_heading("Tone from the machine")
        card.add(
            self.binder.combo(
                "style", "modulation",
                {
                    "none": "Off - every line the same",
                    "pressure": "Pen pressure (Z)",
                    "speed": "Drawing speed",
                },
                label="Modulate",
                hint="On top of whatever the technique draws, the machine can press the pen "
                     "deeper or slow down where the picture is dark, so one pen gives a whole "
                     "range of line weights.",
            )
        )
        self.modulation_amount = self.binder.slider(
            "style", "modulation_amount", 0.0, 1.0,
            label="Amount", decimals=2, step=0.05,
        )
        card.add(self.modulation_amount)
        self.modulation_note = QLabel("")
        self.modulation_note.setWordWrap(True)
        self.modulation_note.setVisible(False)
        self.modulation_note.setStyleSheet(
            f"color: {theme.WARNING}; background: #FFF7EB; border: 1px solid {theme.WARNING};"
            f" border-radius: 6px; padding: 6px 8px; font-size: 11px;"
        )
        card.add(self.modulation_note)

        reset = QPushButton("Reset this technique")
        reset.clicked.connect(self._reset_params)
        card.add(reset)
        card.add(
            self.binder.slider(
                self.settings.style, "detail", 200, 2400, label="Detail", decimals=0, step=50,
                hint="Working resolution. Higher catches finer detail but takes longer to prepare.",
            )
        )
        self._rebuild_params()
        return card

    # ---- parameter controls, generated from the technique registry --------
    def _rebuild_params(self) -> None:
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                # Unparent now, not at the next event loop turn.  deleteLater
                # alone left the old rows occupying space while the new ones
                # were laid out on top of them, and the card kept the height it
                # had before - which is how labels ended up drawn over sliders.
                widget.setParent(None)
                widget.deleteLater()

        technique = techniques.REGISTRY.get(self.settings.style.technique)
        if technique is None:
            return
        values = self.settings.style.technique_params()
        self._param_widgets = {}
        self._pen_scaled_rows = []
        for param in technique.params:
            row = self._param_row(param, values)
            if row is not None:
                self.params_layout.addWidget(row)
        self.params_card.set_title(f"{technique.label.upper()} SETTINGS")
        self.refresh_pen_scaled_hints()
        # the card is inside a scroll area that will not re-measure on its own
        self.params_host.adjustSize()
        self.params_card.updateGeometry()
        self.params_card.adjustSize()

    def _param_row(self, param, values: dict):
        current = values.get(param.key, param.default)

        def store(value) -> None:
            values[param.key] = value
            self._sync_param_visibility()
            self.settings_changed.emit()

        if param.kind == "bool":
            control = QCheckBox(param.label)
            control.setChecked(bool(current))
            control.toggled.connect(lambda state: store(bool(state)))
            if param.hint:
                row = FieldRow("", control, param.hint)
                row.control = control
            else:
                row = control
        elif param.kind == "text":
            control = QLineEdit(str(current))
            control.textChanged.connect(lambda text: store(text))
            row = FieldRow(param.label, control, param.hint)
        elif param.kind == "choice":
            control = QComboBox()
            for key, label in (param.choices or {}).items():
                control.addItem(label, key)
            index = control.findData(current)
            control.setCurrentIndex(max(index, 0))
            control.currentIndexChanged.connect(lambda i: store(control.itemData(i)))
            row = FieldRow(param.label, control, param.hint)
        else:
            control = SliderSpin(
                param.minimum, param.maximum,
                decimals=param.decimals, step=param.step, suffix=param.suffix,
            )
            control.setValue(float(current))
            control.valueChanged.connect(
                lambda value: store(int(round(value)) if param.decimals == 0 else float(value))
            )
            # A paragraph of help under every slider is what turned this card
            # into 860 px of scrolling.  The explanation moves to the tooltip,
            # where Adobe keeps it; the only line that stays visible is the one
            # a number cannot say for itself - what a pen-relative value really
            # measures on the paper.
            control.setToolTip(param.hint)
            hint = ""
            if param.pen_scaled:
                hint = "Scales with the pen."
            row = FieldRow(param.label, control, hint)
            row.setToolTip(param.hint)
            if param.pen_scaled:
                self._pen_scaled_rows.append((param, row, control))

        self._param_widgets[param.key] = row
        return row

    def reset_adjustments(self) -> None:
        """Undo the tonal fiddling.

        Not back to zero: back to what the app read off this picture when it was
        loaded.  Zero is not a neutral state for a photograph - it is the state
        that made every tone land in the middle and hatch the whole sheet.
        """
        style = self.settings.style
        # Every field that shapes the picture, taken from the dataclass rather
        # than a hand-written list - the list had drifted and left enhance,
        # detail and the modulation behind, so a "reset" quietly kept them.
        fresh = StyleSettings()
        for field in dataclasses.fields(fresh):
            if field.name in ("technique", "params", "separation"):
                continue        # what to draw and with how many pens is not a filter
            setattr(style, field.name, getattr(fresh, field.name))
        rgb = self._image_rgb
        if rgb is not None:
            try:
                for key, value in autotune.tune_levels(autotune.analyse(rgb)).items():
                    setattr(style, key, value)
            except Exception:  # pragma: no cover - defensive
                pass
        self.binder.refresh()
        self.settings_changed.emit()
        self.status.emit("Image adjustments reset")

    def take_group_filter(self):
        """The filter is a header control, not part of the scrolling grid.

        Left inside the gallery card it stretched to the card's full width -
        which, once the card held twenty-one tiles in a row, was a blue bar
        thousands of pixels long.
        """
        return self.group_filter

    def detach_wide_cards(self) -> tuple:
        """Hand out the two cards that do not belong in a narrow column.

        The technique gallery wants 1 940 px of height in a 537 px viewport -
        five screens of scrolling for one control.  It is a grid of pictures,
        so it belongs on the wide axis, and its settings belong next to it.
        Re-parenting is enough: the panel keeps every signal it already owns.
        """
        return self.style_card, self.params_card

    def restyle(self) -> None:
        """Re-apply the styles that were baked from theme colours."""
        self.thumbnail.setStyleSheet(
            f"background: {theme.PANEL_ALT}; border: 1px dashed {theme.BORDER_STRONG}; border-radius: 6px;"
        )
        self.gallery.restyle()
        self._sync_visibility()

    def refresh_pen_scaled_hints(self) -> None:
        """Say what a pen-relative slider really means with the pen in use."""
        widths = [pen.width for pen in self.settings.library if pen.enabled]
        width = min(widths) if widths else 0.5
        factor = max(width, 0.05) / 0.5
        for param, row, control in getattr(self, "_pen_scaled_rows", []):
            labels = [w for w in row.findChildren(QLabel) if w.objectName() == "Hint"]
            if not labels:
                continue
            value = control.value()
            if abs(factor - 1.0) < 0.02:
                # with the reference pen the two numbers are the same, and
                # saying so twice is the sort of filler that makes an interface
                # feel like it is padding itself out
                labels[-1].setText("Scales with the pen.")
            else:
                labels[-1].setText(
                    f"{value:.2f} mm becomes {value * factor:.2f} mm with this {width:.2f} mm pen."
                )

    def _sync_param_visibility(self) -> None:
        """A couple of techniques have settings that only matter in some modes."""
        technique = self.settings.style.technique
        values = self.settings.style.technique_params()
        widgets = getattr(self, "_param_widgets", {})

        def show(key: str, visible: bool) -> None:
            widget = widgets.get(key)
            if widget is not None:
                widget.setVisible(visible)

        if technique == "sketch":
            auto = bool(values.get("auto_threshold", True))
            show("sensitivity", auto)
            show("low", not auto)
            show("high", not auto)
        elif technique == "silhouette":
            mode = values.get("threshold_mode", "otsu")
            show("level", mode == "manual")
            show("local_area", mode == "adaptive")
            show("local_offset", mode == "adaptive")
            show("fill_angle", float(values.get("fill", 0.0)) > 0)
        elif technique == "stipple":
            show("vary_size", float(values.get("dot_size", 0.0)) > 0.01)

    def _reset_params(self) -> None:
        self.settings.style.params[self.settings.style.technique] = {}
        self._rebuild_params()
        self._sync_param_visibility()
        self.settings_changed.emit()

    def _on_technique_chosen(self, key: str) -> None:
        self.settings.style.technique = key
        self.gallery.set_current(key)
        self._rebuild_params()
        self._sync_param_visibility()
        self._show_page(self.page_index())
        self._sync_visibility()
        self.settings_changed.emit()

    def _filter_gallery(self, group: str) -> None:
        for key, tile in self.gallery.tiles.items():
            tile.setVisible(group == "all" or techniques.REGISTRY[key].group == group)

    def refresh_gallery(self, width_mm: float = 120.0) -> None:
        """Re-render the preview tiles from the picture that is loaded now."""
        image = self._gallery_image()
        # the finest enabled pen sets how much detail a tile can honestly show;
        # pen 0 might not even be one of the pens the drawing uses
        widths = [p.width for p in self.settings.library if p.enabled]
        self.gallery.refresh(image, width_mm, min(widths) if widths else 0.5)

    def _gallery_image(self):
        source = self.source
        rgb = self._image_rgb
        if rgb is None and source is not None and source.is_raster:
            rgb = source.rgb
        if rgb is None:
            return None
        style = self.settings.style
        prepared = raster.prepare(
            rgb, detail=260,
            brightness=style.brightness, contrast=style.contrast, gamma=style.gamma,
            blur=style.blur, invert=style.invert, auto_levels=style.auto_levels,
            black_point=style.black_point, white_point=style.white_point,
            saturation=style.saturation,
        )
        return raster.to_gray(prepared)

    def _build_adjust_card(self) -> Card:
        card = Card("IMAGE ADJUSTMENTS", collapsible=True, expanded=False)
        self.adjust_card = card
        card.add_reset(
            self.reset_adjustments,
            "Back to what the app worked out from this picture",
        )
        style = self.settings.style
        card.add(self.binder.check("style", "auto_levels", "Auto levels"))
        card.add(self.binder.slider("style", "brightness", -100, 100, label="Brightness", decimals=0, step=1))
        card.add(self.binder.slider("style", "contrast", -100, 100, label="Contrast", decimals=0, step=1))
        card.add(self.binder.slider("style", "gamma", 0.2, 3.0, label="Gamma", decimals=2, step=0.05))
        card.add(self.binder.slider("style", "black_point", 0.0, 1.0, label="Black point", decimals=2, step=0.01))
        card.add(self.binder.slider("style", "white_point", 0.0, 1.0, label="White point", decimals=2, step=0.01))
        card.add(self.binder.slider("style", "saturation", 0.0, 2.0, label="Saturation", decimals=2, step=0.05))
        card.add(self.binder.slider("style", "blur", 0.0, 8.0, label="Blur", decimals=1, step=0.2, suffix="px"))
        card.add(self.binder.check("style", "invert", "Invert (dark paper)"))
        return card

    def _build_colour_card(self) -> Card:
        card = Card("COLOURS", collapsible=True, expanded=True)
        self.colour_card = card
        style = self.settings.style
        card.add(
            self.binder.combo(
                style, "separation", separation.SEPARATION_MODES, label="Separation",
                hint="Match my pen colours splits the picture between the pens you have loaded.",
            )
        )
        self.paper_row = self.binder.slider("style", "paper_lightness", 60.0, 100.0, label="Paper white", decimals=0, step=1)
        card.add(self.paper_row)
        self.ink_row = self.binder.slider("style", "ink_gamma", 0.3, 3.0, label="Ink curve", decimals=2, step=0.05)
        card.add(self.ink_row)

        self.suggest_button = QPushButton("Suggest pen colours")
        self.suggest_button.clicked.connect(self._suggest_palette)
        card.add(self.suggest_button)
        return card

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def active(self) -> int:
        return self.settings.active

    @property
    def source(self) -> SourceResult | None:
        """The source of the layer that is selected right now."""
        return self.sources.get(self.active)

    @property
    def _image_rgb(self):
        return self._images.get(self.active)

    def active_image(self):
        """The picture on the selected layer, or None if it is not a picture."""
        return self._images.get(self.active) if self.settings.item.kind == "image" else None

    @property
    def _pdf(self):
        return self._pdfs.get(self.active)

    @property
    def _image_path(self) -> str:
        return self.settings.item.source_path

    @property
    def _pdf_path(self) -> str:
        return self.settings.item.source_path

    def forget_item(self, index: int) -> None:
        """Drop the cached source of a layer that has been removed."""
        document = self._pdfs.pop(index, None)
        if document is not None:
            document.close()
        self._images.pop(index, None)
        self.sources.pop(index, None)
        # everything above the removed index shifts down one
        for store in (self._images, self._pdfs, self.sources):
            for key in sorted(k for k in store if k > index):
                store[key - 1] = store.pop(key)

    def swap_items(self, a: int, b: int) -> None:
        for store in (self._images, self._pdfs, self.sources):
            first, second = store.pop(a, None), store.pop(b, None)
            if first is not None:
                store[b] = first
            if second is not None:
                store[a] = second

    def duplicate_item_source(self, source_index: int, new_index: int) -> None:
        for store in (self._images, self._pdfs, self.sources):
            for key in sorted((k for k in store if k >= new_index), reverse=True):
                store[key + 1] = store.pop(key)
        if source_index in self._images:
            self._images[new_index] = self._images[source_index]
        if source_index in self.sources:
            self.sources[new_index] = self.sources[source_index]

    def open_image_dialog(self) -> None:
        start = os.path.dirname(self.settings.last_image) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, IMAGE_FILTER)
        if path:
            self.load_image(path)

    def open_pdf_dialog(self) -> None:
        start = os.path.dirname(self.settings.last_pdf) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", start, PDF_FILTER)
        if path:
            self.load_pdf(path)

    def load_any(self, path: str) -> bool:
        lowered = path.lower()
        if lowered.endswith(".pdf"):
            return self.load_pdf(path)
        return self.load_image(path)

    def _adopt_layer(self, kind: str) -> None:
        """Make the selected layer hold a file, and put it back on the bed.

        A layer that used to be a drawing (or a stencil sheet) carries the
        "as drawn" placement, which for a picture means no scaling, no centring
        and a corner pinned to the far edge of the bed - the picture lands
        partly off the sheet.  A layer that was dragged or scaled keeps those
        offsets too.  A newly loaded file has no relationship to any of that, so
        the placement starts again.
        """
        item = self.settings.item
        previous = item.kind
        item.kind = kind
        item.visible = True
        if previous != kind or item.layout.mode not in ("fit", "size", "scale"):
            item.strokes = []
            item.layout.mode = "fit"
            item.layout.center = True
            item.layout.offset_x = 0.0
            item.layout.offset_y = 0.0
            item.layout.rotation = 0.0
            item.layout.mirror_x = False
            item.layout.mirror_y = False
            item.layout.scale_percent = 100.0

    def _setup_for(self, rgb) -> None:
        """Read the picture and set the tone up so it draws well straight away.

        Without this a photograph arrives with black point 0 and white point 1,
        and almost no photograph uses that whole range - so every tone lands in
        the middle and the whole sheet is hatched evenly, which is the "just
        scratches" result.  Only untouched settings are written, so a picture
        dropped onto a layer the user has already adjusted keeps their work.
        """
        style = self.settings.style
        untouched = (
            abs(style.black_point) < 1e-6
            and abs(style.white_point - 1.0) < 1e-6
            and abs(style.gamma - 1.0) < 1e-6
            and abs(style.brightness) < 1e-6
            and abs(style.contrast) < 1e-6
        )
        if not untouched:
            return
        try:
            stats = autotune.analyse(rgb)
        except Exception:  # pragma: no cover - defensive
            return
        for key, value in autotune.tune_levels(stats).items():
            setattr(style, key, value)
        # Pick the look from what the picture is.  Only two answers are decided
        # automatically because only two can be decided honestly; a portrait is
        # offered as a nudge rather than assumed.
        reading = self._read_with_engine(rgb)
        chosen = looks.choose(stats, reading)
        looks.apply_look(chosen, style, stats)
        style.look = chosen
        if hasattr(self, "look_combo"):
            self.look_combo.blockSignals(True)
            self.look_combo.setCurrentIndex(max(self.look_combo.findData(chosen), 0))
            self.look_combo.blockSignals(False)
        self._update_look_note(stats, reading)
        self._rebuild_params()
        self.binder.refresh()

    def load_image(self, path: str) -> bool:
        try:
            self._images[self.active] = raster.load_rgb(path)
        except Exception as exc:
            self.status.emit(f"Could not open the image: {exc}")
            return False
        self._adopt_layer("image")
        self._setup_for(self._images[self.active])
        self.settings.item.source_path = path
        if not self.settings.item.name or self.settings.item.name.startswith("Layer "):
            self.settings.item.name = os.path.splitext(os.path.basename(path))[0][:24]
        self.settings.last_image = path
        self.kind_switch.setValue("image")
        self._show_page(0)
        height, width = self._image_rgb.shape[:2]
        self.image_name.setText(f"{os.path.basename(path)}  ·  {width} × {height} px")
        self._update_thumbnail()
        self._sync_visibility()
        self.rebuild_source()
        self.file_loaded.emit()
        return True

    def load_pdf(self, path: str) -> bool:
        try:
            document = PdfDocument(path)
        except Exception as exc:
            self.status.emit(f"Could not open the PDF: {exc}")
            return False
        old = self._pdfs.get(self.active)
        if old is not None:
            old.close()
        self._pdfs[self.active] = document
        self._adopt_layer("pdf")
        self.settings.item.source_path = path
        if not self.settings.item.name or self.settings.item.name.startswith("Layer "):
            self.settings.item.name = os.path.splitext(os.path.basename(path))[0][:24]
        self.settings.last_pdf = path
        self.settings.pdf.page = 0
        self.kind_switch.setValue("pdf")
        self._show_page(2)

        pages = document.page_count
        slider = self.pdf_page_row.control
        slider.slider.setRange(0, max(pages - 1, 0))
        slider.spin.setRange(0, max(pages - 1, 0))
        slider.setValue(0)
        size = document.page_size_mm(0)
        self.pdf_name.setText(
            f"{os.path.basename(path)}  ·  {pages} page{'s' if pages != 1 else ''}  ·  {size[0]:.0f} × {size[1]:.0f} mm"
        )
        self._sync_visibility()
        self.rebuild_source()
        self.file_loaded.emit()
        return True

    def close_documents(self) -> None:
        """Release every open PDF so no file is held after the window closes."""
        for document in self._pdfs.values():
            try:
                document.close()
            except Exception:
                pass
        self._pdfs.clear()

    def _refresh_source_labels(self) -> None:
        """Filename, thumbnail and page count belong to the selected layer."""
        rgb = self._image_rgb
        if rgb is None:
            self.image_name.setText("No image loaded")
            self.thumbnail.setPixmap(QPixmap())
            self.thumbnail.setText("drop an image here")
        else:
            height, width = rgb.shape[:2]
            self.image_name.setText(f"{os.path.basename(self._image_path)}  ·  {width} × {height} px")
            self._update_thumbnail()

        document = self._pdf
        if document is None:
            self.pdf_name.setText("No PDF loaded")
        else:
            size = document.page_size_mm(self.settings.pdf.page)
            self.pdf_name.setText(
                f"{os.path.basename(self._pdf_path)}  ·  {document.page_count} page"
                f"{'s' if document.page_count != 1 else ''}  ·  {size[0]:.0f} × {size[1]:.0f} mm"
            )
            slider = self.pdf_page_row.control
            slider.slider.setRange(0, max(document.page_count - 1, 0))
            slider.spin.setRange(0, max(document.page_count - 1, 0))

    def _update_thumbnail(self) -> None:
        if self._image_rgb is None:
            return
        preview = raster.resize_long_edge(self._image_rgb, 240)
        buffer, width, height = raster.to_qimage_bytes(preview)
        image = QImage(buffer, width, height, width * 3, QImage.Format_RGB888).copy()
        self.thumbnail.setPixmap(
            QPixmap.fromImage(image).scaled(230, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.thumbnail.setText("")

    # ------------------------------------------------------------------
    def set_pen_count(self, count: int) -> None:
        self._pen_count = max(1, count)

    def rebuild_source(self) -> None:
        """Build the SourceResult for the selected layer.

        Fonts and PDF rendering both want the GUI thread, so the source is
        prepared here and the worker only runs the technique on it.
        """
        index = self.active
        kind = self.settings.source_kind
        try:
            if kind == "text":
                layers = textsource.build_text(self.settings.text, self._pen_count)
                self.sources[index] = SourceResult(
                    kind="text", layers=layers, mm_per_unit=1.0, label=self.settings.item.label()
                )
            elif kind == "pdf" and self._pdf is not None:
                pdf = self.settings.pdf
                if pdf.mode == "render":
                    rgb = self._pdf.render(pdf.page, pdf.dpi)
                    self.sources[index] = SourceResult(
                        kind="pdf",
                        rgb=rgb,
                        mm_per_unit=25.4 / max(pdf.dpi, 1),
                        label=os.path.basename(self._pdf_path),
                        page_count=self._pdf.page_count,
                    )
                else:
                    art, warnings = self._pdf.vectors(pdf)
                    self.sources[index] = SourceResult(
                        kind="pdf",
                        vector=art,
                        mm_per_unit=1.0,
                        label=os.path.basename(self._pdf_path),
                        page_count=self._pdf.page_count,
                        warnings=warnings,
                    )
            elif kind == "image" and self._image_rgb is not None:
                self.sources[index] = SourceResult(
                    kind="image",
                    rgb=self._image_rgb,
                    label=os.path.basename(self._image_path),
                )
            elif kind == "shapes":
                self.sources[index] = self._shapes_source()
            else:
                self.sources.pop(index, None)
        except Exception as exc:  # pragma: no cover - defensive
            self.status.emit(f"Could not prepare the source: {exc}")
            self.sources.pop(index, None)
        self.source_changed.emit()

    def _shapes_source(self) -> SourceResult | None:
        """Freehand and shape geometry is already millimetres on the bed."""
        from ...core.drawing import Layer

        bed_y = self.settings.machine.bed_y
        strokes = []
        for raw in self.settings.item.strokes:
            points = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
            if len(points) < 1:
                continue
            # stored as drawn (Y up on the bed); every source speaks Y down, and
            # the "as drawn" placement mode flips it straight back
            flipped = points.copy()
            flipped[:, 1] = bed_y - flipped[:, 1]
            strokes.append(flipped)
        if not strokes:
            return None
        pen = max(0, min(int(getattr(self.settings.item, "pen", 0)), len(self.settings.library) - 1))
        return SourceResult(
            kind="shapes",
            layers=[Layer(pen=pen, paths=strokes, name=self.settings.library[pen].name)],
            mm_per_unit=1.0,
            label=self.settings.item.label(),
        )

    # ------------------------------------------------------------------
    def _on_kind_changed(self, key: str) -> None:
        self.settings.source_kind = key
        self._show_page(self.page_index())
        self._sync_visibility()
        self.rebuild_source()

    def _on_settings_changed(self) -> None:
        self._sync_visibility()
        self.settings_changed.emit()

    def _on_source_settings_changed(self) -> None:
        self._sync_visibility()
        self.rebuild_source()

    def _reload_weights(self, preferred: str = "") -> None:
        """Refill the weight list for the current family, keeping it if we can."""
        available = textsource.styles_for(self.settings.text.font)
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        for name in available:
            self.style_combo.addItem(name, name)
        keep = preferred if preferred in available else (
            "Regular" if "Regular" in available else (available[0] if available else "Regular")
        )
        self.style_combo.setCurrentIndex(max(self.style_combo.findData(keep), 0))
        self.style_combo.blockSignals(False)
        self.settings.text.style_name = keep

    def _on_font_changed(self, index: int) -> None:
        self.settings.text.font = self.font_combo.itemData(index) or self.settings.text.font
        self._reload_weights(self.settings.text.style_name)
        self._sync_visibility()
        self.rebuild_source()

    def _on_weight_changed(self, index: int) -> None:
        chosen = self.style_combo.itemData(index)
        if not chosen or chosen == self.settings.text.style_name:
            return
        self.settings.text.style_name = chosen
        self.rebuild_source()

    def _suggest_palette(self) -> None:
        image = self._image_rgb
        if image is None and self.source is not None and self.source.is_raster:
            image = self.source.rgb
        if image is None:
            self.status.emit("Load a picture first, then I can pick pen colours from it.")
            return
        count = max(len(self.settings.library), 2)
        colours = separation.suggest_palette(image, count)
        if colours:
            self.palette_suggested.emit(colours)

    # ------------------------------------------------------------------
    def _sync_visibility(self) -> None:
        kind = self.settings.item.kind
        raster_source = kind == "image" or (kind == "pdf" and self.settings.pdf.mode == "render")

        shapes = kind == "shapes"
        self.shapes_card.setVisible(shapes)
        if shapes:
            self.refresh_shapes_pens()
        # a drawing layer has no source to pick, so the whole picker goes away
        self.source_card.setVisible(not shapes)
        self.style_card.setVisible(raster_source)
        self.auto_card.setVisible(raster_source)
        self.adjust_card.setVisible(raster_source)
        self.colour_card.setVisible(True)
        self.suggest_button.setVisible(raster_source)
        self.paper_row.setVisible(self.settings.style.separation == "palette")
        self.ink_row.setVisible(self.settings.style.separation != "mono")

        technique = techniques.REGISTRY.get(self.settings.style.technique)
        self.style_hint.setText(technique.description if technique else "")
        self.style_hint.setVisible(raster_source)
        self.params_card.setVisible(raster_source)
        self._sync_requirement(technique, raster_source)

        mode = self.settings.style.modulation
        self.modulation_amount.setVisible(mode != "none")
        if mode == "pressure":
            self.modulation_note.setText(
                "⚠  Pen pressure pushes the pen up to 0.6 mm below the drawing height. Use a "
                "spring-loaded or otherwise compliant pen holder - a rigid mount will dig into "
                "the paper and can stall the Z axis."
            )
            self.modulation_note.setVisible(True)
        elif mode == "speed":
            self.modulation_note.setText(
                "Works with pens that bleed more when they move slowly: fibre tips, fountain "
                "pens and markers. A ballpoint will look almost the same throughout."
            )
            self.modulation_note.setStyleSheet(
                f"color: {theme.TEXT_MUTED}; background: {theme.PANEL_ALT}; border: 1px solid "
                f"{theme.BORDER}; border-radius: 6px; padding: 6px 8px; font-size: 11px;"
            )
            self.modulation_note.setVisible(True)
        else:
            self.modulation_note.setVisible(False)

        if kind == "text":
            outline_font = not self.settings.text.is_stroke_font
            self.text_fill_check.setVisible(outline_font)
            filling = outline_font and self.settings.text.fill
            self.text_fill_spacing.setVisible(filling)
            self.text_fill_angle.setVisible(filling)
            self.text_keep_outline.setVisible(filling)
            self.text_double.setVisible(not outline_font)
            self.style_row.setVisible(outline_font)

        if kind == "pdf":
            vector = self.settings.pdf.mode == "vector"
            self.pdf_text_row.setVisible(vector)
            self.pdf_fills_row.setVisible(vector)
            self.pdf_dpi_row.setVisible(not vector)

    def _sync_requirement(self, technique, raster_source: bool) -> None:
        """Tell the user when the chosen technique needs a particular kind of pen."""
        if technique is None or not raster_source or technique.requires != "bleeding_pen":
            self.requirement_note.setVisible(False)
            return
        pens = [p for p in self.settings.library if p.enabled]
        wrong = [p for p in pens if not p.bleeds]
        if not wrong:
            self.requirement_note.setVisible(False)
            return
        names = ", ".join(f"{p.name} ({p.kind_label.lower()})" for p in wrong[:3])
        self.requirement_note.setText(
            "⚠  This technique needs a pen that keeps bleeding while it rests - a fibre tip, "
            "fountain pen, gel pen or marker. These will just make identical dots: "
            f"{names}. Set the tip type under Pens."
        )
        self.requirement_note.setVisible(True)

    def select_item(self, index: int) -> None:
        """Point the whole panel at another layer."""
        self.settings.active = max(0, min(index, len(self.settings.items) - 1))
        self.refresh()
        if self.active not in self.sources:
            self.rebuild_source()
        else:
            self.source_changed.emit()

    def refresh(self) -> None:
        self.binder.refresh()
        self.source_binder.refresh()
        self._refresh_source_labels()
        self.gallery.set_current(self.settings.style.technique)
        index = self.font_combo.findData(self.settings.text.font)
        if index >= 0 and index != self.font_combo.currentIndex():
            self.font_combo.blockSignals(True)
            self.font_combo.setCurrentIndex(index)
            self.font_combo.blockSignals(False)
            self._reload_weights(self.settings.text.style_name)
        self._rebuild_params()
        self._sync_param_visibility()
        self.kind_switch.setValue(self.settings.source_kind)
        self._show_page(self.page_index())
        self._sync_visibility()
