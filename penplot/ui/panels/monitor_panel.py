"""Monitor stage: connect over USB, jog the head, set the pen height, watch the job."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.gcode import format_duration
from ...core.printer import PrinterLink, available_ports
from ...core.settings import AppSettings
from .. import theme
from ..widgets import Card, FieldRow, SliderSpin, hint_label

STEPS = [("0.05", 0.05), ("0.1", 0.1), ("1", 1.0), ("10", 10.0), ("50", 50.0)]


class ConnectionPanel(QWidget):
    """Port selection, jogging and the pen-height calibration helper."""

    status = Signal(str)
    settings_changed = Signal()

    def __init__(self, settings: AppSettings, printer: PrinterLink, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.printer = printer
        self._step = 1.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # ---- connection ----
        card = Card("CONNECTION")
        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(32)
        refresh.setToolTip("Rescan ports")
        refresh.clicked.connect(self.refresh_ports)
        row.addWidget(self.port_combo, 1)
        row.addWidget(refresh)
        card.add_layout(row)

        self.baud_combo = QComboBox()
        for baud in (115200, 250000, 57600, 250000, 500000, 1000000):
            if self.baud_combo.findData(baud) < 0:
                self.baud_combo.addItem(str(baud), baud)
        index = self.baud_combo.findData(settings.machine.baud)
        self.baud_combo.setCurrentIndex(max(index, 0))
        self.baud_combo.currentIndexChanged.connect(self._on_baud)
        card.add(FieldRow("Baud rate", self.baud_combo))

        self.simple_mode = QCheckBox("Simple mode (no line numbers)")
        self.simple_mode.setChecked(not settings.machine.use_checksums)
        self.simple_mode.setToolTip(
            "Sends plain commands instead of numbered, checksummed ones.\n"
            "Try this if the console reports line-number errors."
        )
        self.simple_mode.toggled.connect(self._on_simple_mode)
        card.add(self.simple_mode)

        self.connect_button = QPushButton("Connect")
        self.connect_button.setObjectName("Primary")
        self.connect_button.clicked.connect(self._toggle_connection)
        card.add(self.connect_button)

        self.state_label = QLabel("Not connected")
        self.state_label.setObjectName("Hint")
        card.add(self.state_label)
        outer.addWidget(card)

        # ---- manual control ----
        control = Card("MANUAL CONTROL")
        self.step_buttons: list[QPushButton] = []
        steps = QHBoxLayout()
        steps.setSpacing(4)
        for label, value in STEPS:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(abs(value - 1.0) < 1e-6)
            button.clicked.connect(lambda _checked=False, v=value: self._set_step(v))
            steps.addWidget(button)
            self.step_buttons.append(button)
        control.add(FieldRow("Step (mm)", self._wrap(steps), label_width=70))

        grid = QGridLayout()
        grid.setSpacing(4)
        #: every control that would move the head behind the job's back
        self._manual_widgets: list = []
        self._jog_button(grid, "Y+", 0, 1, lambda: self._jog(0, self._step, 0))
        self._jog_button(grid, "X−", 1, 0, lambda: self._jog(-self._step, 0, 0))
        home = QPushButton("⌂ XY")
        home.setToolTip("Home X and Y (the pen lifts first)")
        home.clicked.connect(self._home_xy)
        grid.addWidget(home, 1, 1)
        self._manual_widgets.append(home)
        self._jog_button(grid, "X+", 1, 2, lambda: self._jog(self._step, 0, 0))
        self._jog_button(grid, "Y−", 2, 1, lambda: self._jog(0, -self._step, 0))
        self._jog_button(grid, "Z↑", 0, 3, lambda: self._jog(0, 0, self._step))
        self._jog_button(grid, "Z↓", 2, 3, lambda: self._jog(0, 0, -self._step))
        control.add_layout(grid)

        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        centre = QPushButton("Bed centre")
        centre.clicked.connect(self._go_centre)
        park = QPushButton("Park")
        park.clicked.connect(self._go_park)
        buttons.addWidget(centre)
        buttons.addWidget(park)
        self._manual_widgets += [centre, park]
        control.add_layout(buttons)

        pen_buttons = QHBoxLayout()
        pen_buttons.setSpacing(5)
        down = QPushButton("Pen down")
        down.clicked.connect(self._pen_down)
        up = QPushButton("Pen up")
        up.clicked.connect(self._pen_up)
        pen_buttons.addWidget(down)
        pen_buttons.addWidget(up)
        self._manual_widgets += [down, up]
        control.add_layout(pen_buttons)

        motors = QPushButton("Release motors (M84)")
        motors.setToolTip("Lets you push the head by hand. The machine forgets where it is.")
        motors.clicked.connect(self._release_motors)
        control.add(motors)
        self._manual_widgets.append(motors)
        outer.addWidget(control)

        # ---- live control while drawing ----
        live = Card("LIVE CONTROL")
        live.add(hint_label(
            "These work while the drawing is running - dial in the pen while you watch it."
        ))

        self.speed_slider = SliderSpin(25, 300, decimals=0, step=5, suffix="%")
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_speed)
        live.add(FieldRow("Speed", self.speed_slider, "Feedrate override (M220). 100% is the planned speed."))

        self.pen_z_slider = SliderSpin(-1.5, 1.5, decimals=2, step=0.02, suffix="mm")
        self.pen_z_slider.setValue(0.0)
        self.pen_z_slider.valueChanged.connect(self._on_live_z)
        live.add(FieldRow("Pen pressure", self.pen_z_slider,
                          "Lowers or raises the drawing height. Negative presses harder."))

        self.lift_slider = SliderSpin(-2.0, 8.0, decimals=2, step=0.1, suffix="mm")
        self.lift_slider.setValue(0.0)
        self.lift_slider.valueChanged.connect(self._on_live_z)
        live.add(FieldRow("Extra lift", self.lift_slider,
                          "Adds to the pen hop if the pen is catching on the paper."))

        self.live_summary = QLabel("")
        self.live_summary.setObjectName("Hint")
        live.add(self.live_summary)

        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        reset_live = QPushButton("Reset")
        reset_live.clicked.connect(self._reset_live)
        keep = QPushButton("Keep as default")
        keep.setToolTip("Write the current pen height and lift into the settings")
        keep.clicked.connect(self._keep_live)
        buttons.addWidget(reset_live)
        buttons.addWidget(keep)
        live.add_layout(buttons)
        outer.addWidget(live)

        # ---- pen height ----
        calibration = Card("PEN HEIGHT")
        calibration.add(hint_label(
            "Everything is measured from the height where the tip touches the paper. "
            "Set it before every drawing - it does not survive a power cycle."
        ))
        wizard = QPushButton("Set the pen height…")
        wizard.setObjectName("Primary")
        wizard.clicked.connect(self._open_wizard)
        calibration.add(wizard)
        self._manual_widgets.append(wizard)
        set_zero = QPushButton("Use the height it is at now")
        set_zero.setToolTip("Skip the wizard: take the current Z as the drawing height")
        set_zero.clicked.connect(self._set_zero)
        calibration.add(set_zero)
        self._manual_widgets.append(set_zero)
        self.zero_label = QLabel("")
        self.zero_label.setObjectName("Hint")
        calibration.add(self.zero_label)
        outer.addWidget(calibration)

        # ---- console input ----
        send_card = Card("SEND COMMAND", collapsible=True, expanded=False)
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("M115, G28 X Y, M114 …")
        self.command_input.returnPressed.connect(self._send_command)
        send_card.add(self.command_input)
        outer.addWidget(send_card)

        outer.addStretch(1)

        self.printer.connected.connect(self._on_connected)
        self.printer.disconnected.connect(self._on_disconnected)
        self.printer.state_changed.connect(self._on_state)
        self.refresh_ports()
        self._update_enabled()
        self._update_live_summary()

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(layout) -> QWidget:
        holder = QWidget()
        holder.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return holder

    def _jog_button(self, grid: QGridLayout, text: str, row: int, column: int, handler) -> None:
        button = QPushButton(text)
        button.setMinimumWidth(52)
        button.clicked.connect(handler)
        self._manual_widgets.append(button)
        grid.addWidget(button, row, column)

    def _set_step(self, value: float) -> None:
        self._step = value
        for button, (_, step) in zip(self.step_buttons, STEPS):
            button.setChecked(abs(step - value) < 1e-9)

    # ------------------------------------------------------------------
    def refresh_ports(self) -> None:
        current = self.port_combo.currentData()
        self.port_combo.clear()
        ports = available_ports()
        for info in ports:
            self.port_combo.addItem(info.label, info.device)
        if not ports:
            self.port_combo.addItem("No serial ports found", "")
        target = self.settings.machine.port or current
        if target:
            index = self.port_combo.findData(target)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

    def _on_simple_mode(self, simple: bool) -> None:
        self.settings.machine.use_checksums = not simple
        self.printer.set_protocol(self.settings.machine.use_checksums)
        self.settings_changed.emit()

    def _on_baud(self, index: int) -> None:
        self.settings.machine.baud = int(self.baud_combo.itemData(index) or 115200)
        self.settings_changed.emit()

    def _toggle_connection(self) -> None:
        if self.printer.is_connected:
            self.printer.disconnect_from()
            return
        port = self.port_combo.currentData()
        if not port:
            self.status.emit("No serial port selected. Connect the printer over USB and press ↻.")
            return
        self.settings.machine.port = port
        self.settings_changed.emit()
        self.printer.set_protocol(self.settings.machine.use_checksums)
        self.printer.connect_to(port, self.settings.machine.baud)

    def _on_connected(self, port: str) -> None:
        self.connect_button.setText("Disconnect")
        self.state_label.setText(f"Connected to {port}")
        self.state_label.setObjectName("Success")
        self.state_label.setStyleSheet(f"color: {theme.SUCCESS};")
        self._update_enabled()

    def _on_disconnected(self, reason: str) -> None:
        self.connect_button.setText("Connect")
        self.state_label.setText("Not connected" if not reason else f"Disconnected: {reason}")
        self.state_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self._update_enabled()

    def _on_state(self, state: str) -> None:
        if state == "printing":
            self.state_label.setText("Drawing…")
        elif state == "paused":
            self.state_label.setText("Paused")
        self._printing = state == "printing"
        self._update_enabled()

    def _update_enabled(self) -> None:
        """Jogging mid-job would fight the stream and ruin the drawing."""
        printing = getattr(self, "_printing", False)
        for widget in getattr(self, "_manual_widgets", []):
            widget.setEnabled(not printing)
            if printing:
                widget.setToolTip("Not while the printer is drawing - pause first")

    def _release_motors(self) -> None:
        answer = QMessageBox.question(
            self,
            "Release the motors?",
            "The head can then be pushed by hand, and the machine forgets where it is.\n\n"
            "You will have to home again, and the pen height reference is lost.",
            QMessageBox.Cancel | QMessageBox.Ok,
            QMessageBox.Cancel,
        )
        if answer == QMessageBox.Ok:
            self.printer.send("M84")
            self.zero_label.setText("Pen height lost with the motors - set it again before drawing.")
            self.zero_label.setStyleSheet(f"color: {theme.WARNING};")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _on_speed(self, value: float) -> None:
        self.printer.set_speed(int(value))
        self._update_live_summary()

    def _on_live_z(self, _value: float = 0.0) -> None:
        self.printer.set_live_z(self.pen_z_slider.value(), self.lift_slider.value())
        self._update_live_summary()

    def _reset_live(self) -> None:
        self.speed_slider.setValue(100)
        self.pen_z_slider.setValue(0.0)
        self.lift_slider.setValue(0.0)
        self.printer.set_speed(100)
        self.printer.set_live_z(0.0, 0.0)
        self._update_live_summary()

    def _keep_live(self) -> None:
        pen = self.settings.pen
        pen.draw_z += self.pen_z_slider.value()
        pen.lift = max(pen.lift + self.lift_slider.value(), 0.2)
        self.pen_z_slider.setValue(0.0)
        self.lift_slider.setValue(0.0)
        self.printer.set_live_z(0.0, 0.0)
        self.settings_changed.emit()
        self._update_live_summary()

    def _update_live_summary(self) -> None:
        pen = self.settings.pen
        draw = pen.draw_z + self.pen_z_slider.value()
        lift = max(pen.lift + self.lift_slider.value(), 0.1)
        self.live_summary.setText(
            f"Drawing at Z {draw:+.2f} mm, lifting {lift:.2f} mm, running at {self.speed_slider.value():.0f}%"
        )

    def _jog(self, dx: float, dy: float, dz: float) -> None:
        feed = self.settings.machine.travel_feed if dz == 0 else self.settings.machine.z_feed
        parts = []
        if dx:
            parts.append(f"X{dx:.3f}")
        if dy:
            parts.append(f"Y{dy:.3f}")
        if dz:
            parts.append(f"Z{dz:.3f}")
        if not parts:
            return
        self.printer.send("G91")
        self.printer.send(f"G1 {' '.join(parts)} F{feed:.0f}")
        self.printer.send("G90")

    def _home_xy(self) -> None:
        self.printer.send("G91")
        self.printer.send(f"G1 Z{max(self.settings.pen.lift, 2.0) + 2:.2f} F{self.settings.machine.z_feed:.0f}")
        self.printer.send("G90")
        self.printer.send("G28 X Y")

    def _go_centre(self) -> None:
        machine = self.settings.machine
        self.printer.send(f"G1 X{machine.bed_x/2:.1f} Y{machine.bed_y/2:.1f} F{machine.travel_feed:.0f}")

    def _go_park(self) -> None:
        machine = self.settings.machine
        self.printer.send(f"G1 X{machine.park_x:.1f} Y{machine.park_y:.1f} F{machine.travel_feed:.0f}")

    def _pen_down(self) -> None:
        pen = self.settings.pen
        self.printer.send(f"G1 Z{pen.draw_z:.2f} F{self.settings.machine.z_feed:.0f}")

    def _pen_up(self) -> None:
        pen = self.settings.pen
        self.printer.send(f"G1 Z{pen.draw_z + pen.lift:.2f} F{self.settings.machine.z_feed:.0f}")

    def _open_wizard(self) -> None:
        from ..pen_height_dialog import ask_to_calibrate

        if ask_to_calibrate(self, self.settings, self.printer):
            self._mark_zeroed()

    def _set_zero(self) -> None:
        """Take the height the pen is standing at, as a machine coordinate.

        This used to send `G92 Z0`, and so did the top of every generated file -
        the second one re-zeroed at whatever height the pen had drifted to, and
        the drawing quietly happened that far above the paper.
        """
        self.printer.query_position()
        QTimer.singleShot(600, self._store_measured)
        self.zero_label.setText("Asking the printer where it is…")
        self.zero_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")

    def _store_measured(self) -> None:
        z = self.printer.machine_z
        if z is None:
            self.zero_label.setText(
                "The printer did not answer M114. Use the wizard, or switch on "
                "'Re-zero from where the pen is' in the Printer panel."
            )
            self.zero_label.setStyleSheet(f"color: {theme.WARNING};")
            return
        self.settings.pen.draw_z = round(float(z), 3)
        self.settings.pen.zero_z_at_start = False
        self.printer.pen_zeroed = True
        self._mark_zeroed()

    def _mark_zeroed(self) -> None:
        self.zero_label.setText(
            f"✓ Pen height set: the paper is at machine Z{self.settings.pen.draw_z:.2f}. "
            "Keep the printer powered on and do not release the motors."
        )
        self.zero_label.setStyleSheet(f"color: {theme.SUCCESS};")
        self.settings_changed.emit()

    def _send_command(self) -> None:
        text = self.command_input.text().strip()
        if text:
            self.printer.send(text)
            self.command_input.clear()


class JobPanel(QWidget):
    """Progress, the pen currently in use, and the console."""

    def __init__(self, settings: AppSettings, printer: PrinterLink, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.printer = printer
        self._started_at: float | None = None
        self._total_lines = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        card = Card("JOB")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        card.add(self.progress)

        self.headline = QLabel("Nothing running")
        self.headline.setObjectName("CardTitle")
        card.add(self.headline)
        self.detail = QLabel("")
        self.detail.setObjectName("Hint")
        self.detail.setWordWrap(True)
        card.add(self.detail)
        self.pens_label = QLabel("")
        self.pens_label.setObjectName("Hint")
        self.pens_label.setWordWrap(True)
        self.pens_label.setVisible(False)
        card.add(self.pens_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("Danger")
        self.stop_button.clicked.connect(self._confirm_stop)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.stop_button)
        card.add_layout(buttons)
        outer.addWidget(card)

        console_card = Card("CONSOLE")
        self.console = QPlainTextEdit()
        self.console.setObjectName("Console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(600)
        self.console.setMinimumHeight(220)
        console_card.add(self.console)

        row = QHBoxLayout()
        clear = QPushButton("Clear")
        clear.clicked.connect(self.console.clear)
        self.verbose = QPushButton("Show all traffic")
        self.verbose.setCheckable(True)
        row.addWidget(clear)
        row.addWidget(self.verbose)
        row.addStretch(1)
        console_card.add_layout(row)
        outer.addWidget(console_card, 1)

        printer.log.connect(self.append_log)
        printer.progress.connect(self.on_progress)
        printer.job_finished.connect(self.on_finished)
        printer.state_changed.connect(self.on_state)

    # ------------------------------------------------------------------
    def append_log(self, message: str, kind: str) -> None:
        if kind in ("tx", "rx") and not self.verbose.isChecked():
            return
        colours = {"tx": "#7FB3E8", "rx": "#9AA7B2", "info": "#D7E0E8", "error": "#F08A8A"}
        prefix = {"tx": "→ ", "rx": "← ", "info": "", "error": "!! "}.get(kind, "")
        colour = colours.get(kind, "#D7E0E8")
        self.console.appendHtml(
            f'<span style="color:{colour}">{prefix}{message.replace("<", "&lt;")}</span>'
        )

    def _confirm_stop(self) -> None:
        answer = QMessageBox.question(
            self,
            "Stop the drawing?",
            "The drawing stops where it is and cannot be resumed from that point.\n\n"
            "The pen height reference is lost, so set it again before starting anything new.",
            QMessageBox.Cancel | QMessageBox.Yes,
            QMessageBox.Cancel,
        )
        if answer == QMessageBox.Yes:
            self.printer.cancel(True)

    def job_started(self, total_lines: int, pen_names: list[str], estimate: float) -> None:
        self._started_at = time.monotonic()
        self._total_lines = total_lines
        self.progress.setValue(0)
        self.headline.setText("Drawing…")
        # The pen list has to survive the whole job: at a pen change the operator
        # needs to know which pen goes in, and progress used to overwrite it.
        self.pens_label.setText(f"Pens, in order: {', '.join(pen_names)}")
        self.pens_label.setVisible(bool(pen_names))
        self.detail.setText(f"{total_lines:,} commands · estimated {format_duration(estimate)}")
        self.pause_button.setText("Pause")

    def on_progress(self, line: int, total: int, drawn: float) -> None:
        if not total:
            return
        fraction = line / total
        self.progress.setValue(int(fraction * 1000))
        elapsed = time.monotonic() - (self._started_at or time.monotonic())
        remaining = (elapsed / fraction - elapsed) if fraction > 0.01 else 0.0
        self.detail.setText(
            f"{line:,} / {total:,} · {drawn/1000:.1f} m drawn · {format_duration(remaining)} left"
        )

    def on_finished(self, ok: bool, message: str) -> None:
        self.headline.setText(message)
        self.pause_button.setText("Pause")
        if ok:
            self.progress.setValue(1000)

    def on_state(self, state: str) -> None:
        if state == "paused":
            self.headline.setText("Paused")
            self.pause_button.setText("Resume")
        elif state == "printing":
            self.headline.setText("Drawing…")
            self.pause_button.setText("Pause")

    def _toggle_pause(self) -> None:
        if self.pause_button.text() == "Pause":
            self.printer.pause()
        else:
            self.printer.resume()
