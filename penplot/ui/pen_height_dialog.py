"""The pen height wizard.

Everything the machine draws is measured from one number: the Z at which the
tip touches the paper.  This finds that number and stores it as an absolute
machine coordinate.

It used to be stored as a `G92 Z0` instead, and the job repeated that `G92 Z0`
at its own start - which re-zeroed at whatever height the pen had drifted to
(the lift this dialog applies, or the Z60 the previous job parked at).  The
paper then sat that far below Z0 and the machine lifted and dropped for hours
without ever touching it: "it only goes up in Z and never starts".
"""

from __future__ import annotations

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.printer import PrinterLink
from ..core.settings import AppSettings
from . import theme
from .widgets import Card, hint_label

__all__ = ["PenHeightDialog", "ask_to_calibrate"]

#: how far each Z button travels.  Coarse first, then the tenths that decide
#: whether the pen kisses the paper or digs into it.
Z_STEPS = [5.0, 1.0, 0.25, 0.05]


class PenHeightDialog(QDialog):
    def __init__(self, settings: AppSettings, printer: PrinterLink, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.printer = printer
        self._travelled = 0.0
        #: where the machine said Z was when the wizard opened.  Everything is
        #: measured from this, so the reference is a real machine coordinate
        #: and not "wherever the pen happened to be when you pressed Send".
        self._z_at_open: float | None = None
        printer.measured.connect(self._on_measured)
        if printer.is_connected:
            printer.query_position()

        self.setWindowTitle("Set the pen height")
        self.setModal(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.px(16), theme.px(16), theme.px(16), theme.px(16))
        outer.setSpacing(theme.px(10))

        intro = QLabel(
            "Find the height where the tip touches the paper. It is stored as a machine "
            "coordinate, so every job afterwards goes straight to it.\n"
            "Keep the printer powered on - a power cycle or M84 loses the reference."
        )
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # ---- 1. home -----------------------------------------------------
        step1 = Card("1 · HOME THE CARRIAGE")
        step1.add(hint_label(
            "Lifts the pen a little, then homes X and Y. Z is never homed: on this "
            "machine that would drive the pen straight into the bed."
        ))
        home = QPushButton("Home X and Y")
        home.clicked.connect(self._home)
        step1.add(home)
        outer.addWidget(step1)

        # ---- 2. paper ----------------------------------------------------
        step2 = Card("2 · GO TO THE PAPER")
        step2.add(hint_label("Tape a sheet down first, then send the head to the middle of it."))
        centre = QPushButton("Move to the middle of the bed")
        centre.clicked.connect(self._centre)
        step2.add(centre)
        outer.addWidget(step2)

        # ---- 3. come down ------------------------------------------------
        step3 = Card("3 · LOWER THE PEN")
        step3.add(hint_label(
            "Come down in big steps while there is a gap, then in 0.25 and 0.05 mm steps "
            "until the tip just touches. It should mark the paper without bending."
        ))
        grid = QGridLayout()
        grid.setSpacing(theme.px(4))
        for column, step in enumerate(Z_STEPS):
            down = QPushButton(f"↓ {step:g}")
            down.setToolTip(f"Lower the pen {step:g} mm")
            down.clicked.connect(lambda _c=False, s=step: self._jog(-s))
            grid.addWidget(down, 0, column)
        for column, step in enumerate(Z_STEPS):
            up = QPushButton(f"↑ {step:g}")
            up.setToolTip(f"Raise the pen {step:g} mm")
            up.clicked.connect(lambda _c=False, s=step: self._jog(s))
            grid.addWidget(up, 1, column)
        holder = QWidget()
        holder.setLayout(grid)
        grid.setContentsMargins(0, 0, 0, 0)
        step3.add(holder)

        row = QHBoxLayout()
        row.setSpacing(theme.px(5))
        test = QPushButton("Draw a test line")
        test.setToolTip("A 20 mm stroke at the height you are on right now")
        test.clicked.connect(self._test_line)
        row.addWidget(test)
        self.travel_label = QLabel("Not moved yet")
        self.travel_label.setObjectName("Hint")
        row.addWidget(self.travel_label, 1)
        step3.add_layout(row)
        outer.addWidget(step3)

        # ---- 4. commit ---------------------------------------------------
        step4 = Card("4 · SET IT")
        step4.add(hint_label(
            "Stores this height as the drawing Z, as an absolute machine coordinate. "
            "Every job then travels straight to it - you can send the same drawing again "
            "and again. Only a power cycle or releasing the motors loses it."
        ))
        commit = QPushButton("The pen is touching · use this height")
        commit.setObjectName("Primary")
        commit.clicked.connect(self._commit)
        step4.add(commit)
        outer.addWidget(step4)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        if not printer.is_connected:
            intro.setText("Connect to the printer first - these buttons have nowhere to go.")
            for widget in (home, centre, test, commit):
                widget.setEnabled(False)

    # ------------------------------------------------------------------
    def _z_feed(self) -> str:
        return f"{self.settings.machine.z_feed:.0f}"

    def _home(self) -> None:
        self.printer.send("G91")
        self.printer.send(f"G1 Z{max(self.settings.pen.lift, 2.0) + 2:.2f} F{self._z_feed()}")
        self.printer.send("G90")
        self.printer.send("G28 X Y")
        self._travelled = 0.0
        self._update_travel()

    def _centre(self) -> None:
        machine = self.settings.machine
        self.printer.send(
            f"G1 X{machine.bed_x / 2:.1f} Y{machine.bed_y / 2:.1f} F{machine.travel_feed:.0f}"
        )

    def _on_measured(self, _x: float, _y: float, z: float) -> None:
        if self._z_at_open is None:
            self._z_at_open = z
        self._update_travel()

    def _jog(self, delta: float) -> None:
        self.printer.send("G91")
        self.printer.send(f"G1 Z{delta:.3f} F{self._z_feed()}")
        self.printer.send("G90")
        self.printer.query_position()
        self._travelled += delta
        self._update_travel()

    def _machine_z(self) -> float:
        """Best available machine Z: what the printer last reported, or dead
        reckoning from where it was when the wizard opened."""
        reported = self.printer.machine_z
        if reported is not None:
            return float(reported)
        if self._z_at_open is not None:
            return self._z_at_open + self._travelled
        return 0.0

    def _update_travel(self) -> None:
        if not hasattr(self, "travel_label"):
            return          # a reply that beat the widgets into existence
        if abs(self._travelled) < 1e-9:
            moved = "Not moved yet"
        elif self._travelled < 0:
            moved = f"Lowered {-self._travelled:.2f} mm"
        else:
            moved = f"Raised {self._travelled:.2f} mm"
        if self.printer.machine_z is not None:
            moved += f"   ·   machine Z {self.printer.machine_z:.2f}"
        self.travel_label.setText(moved)

    def _test_line(self) -> None:
        """20 mm out and back at the current height, so the mark is visible."""
        feed = f"{self.settings.machine.draw_feed:.0f}"
        self.printer.send("G91")
        self.printer.send(f"G1 X20 F{feed}")
        self.printer.send(f"G1 X-20 F{feed}")
        self.printer.send("G90")

    def _commit(self) -> None:
        """Store the height as an absolute machine coordinate.

        The old behaviour was `G92 Z0` here and `G92 Z0` again at the top of
        every job.  The second one re-zeroed wherever the pen had ended up -
        the lift this dialog applies, or the Z60 the previous job parked at -
        so the paper silently moved that far below Z0 and the pen drew in the
        air, lifting and dropping for hours without marking anything.  Storing
        a machine coordinate means the job can be sent as many times as you
        like: it always travels to the same absolute height.
        """
        self.printer.query_position()
        self._wait_for_reply()
        z = self._machine_z()
        self.settings.pen.draw_z = round(z, 3)
        self.settings.pen.zero_z_at_start = False   # no G92 - absolute from now on
        self.printer.pen_zeroed = True
        self.printer.send(f"G1 Z{z + max(self.settings.pen.lift, 2.0):.2f} F{self._z_feed()}")
        self.accept()

    def _wait_for_reply(self, timeout_ms: int = 1500) -> None:
        """Give M114 a moment to come back before falling back to dead reckoning."""
        if not self.printer.is_connected:
            return
        loop = QEventLoop()
        self.printer.measured.connect(loop.quit)
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()
        try:
            self.printer.measured.disconnect(loop.quit)
        except (RuntimeError, TypeError):
            pass


def ask_to_calibrate(parent, settings: AppSettings, printer: PrinterLink) -> bool:
    """Open the wizard. True if the pen height was actually set."""
    dialog = PenHeightDialog(settings, printer, parent)
    return dialog.exec() == QDialog.Accepted
