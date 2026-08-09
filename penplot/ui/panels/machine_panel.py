"""Printer geometry, pen mechanics, speeds and path optimisation."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from ...core import profiles
from ...core.settings import AppSettings
from ..widgets import Binder, Card, FieldRow, hint_label


class MachinePanel(QWidget):
    changed = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.binder = Binder(self.changed.emit, self)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        machine = settings.machine
        pen = settings.pen
        optimise = settings.optimize

        pen_card = Card("PEN & Z HEIGHT")
        pen_card.add(self.binder.slider(pen, "draw_z", -5.0, 60.0, label="Drawing Z", decimals=2, step=0.05, suffix="mm"))
        pen_card.add(self.binder.slider(pen, "lift", 0.2, 20.0, label="Lift", decimals=2, step=0.1, suffix="mm",
                                        hint="How far the pen rises for travel moves."))
        pen_card.add(self.binder.check(pen, "zero_z_at_start", "Re-zero from where the pen is (G92)",
                                       hint="Off is better: the drawing Z above is then a machine coordinate "
                                            "the job travels to, so the same file can be sent again and "
                                            "again. On, the job calls the pen's current height Z0 - which "
                                            "silently draws in mid-air if it is not touching the paper."))
        pen_card.add_heading("Pen change position")
        pen_card.add(self.binder.slider(pen, "change_z", 5.0, 200.0, label="Z", decimals=0, step=5, suffix="mm"))
        pen_card.add(self.binder.slider(pen, "change_x", 0.0, 300.0, label="X", decimals=0, step=5, suffix="mm"))
        pen_card.add(self.binder.slider(pen, "change_y", 0.0, 300.0, label="Y", decimals=0, step=5, suffix="mm"))
        pen_card.add_heading("Pen lifts")
        pen_card.add(hint_label(
            "Lifting the pen is most of the drawing time on this machine. A short "
            "hop only has to clear the paper."
        ))
        pen_card.add(self.binder.slider(pen, "short_hop", 0.0, 30.0, label="Short hop up to", decimals=1, step=0.5, suffix="mm",
                                        hint="Moves shorter than this use the small lift. 0 turns it off."))
        pen_card.add(self.binder.slider(pen, "short_lift", 0.05, 5.0, label="Small lift", decimals=2, step=0.05, suffix="mm"))

        pen_card.add_heading("Fine tuning")
        pen_card.add(self.binder.slider(pen, "down_delay", 0, 1000, label="Settle down", decimals=0, step=10, suffix="ms"))
        pen_card.add(self.binder.slider(pen, "up_delay", 0, 1000, label="Settle up", decimals=0, step=10, suffix="ms"))
        pen_card.add(self.binder.slider(pen, "dot_diameter", 0.0, 3.0, label="Dot size", decimals=2, step=0.05, suffix="mm",
                                        hint="0 dabs a single point for stipple dots; larger draws a tiny circle."))
        pen_card.add(self.binder.check(pen, "scale_with_pen_width", "Everything follows the pen width",
                                       hint="Settings are written for a 0.5 mm pen and scale from there."))
        outer.addWidget(pen_card)

        speed_card = Card("SPEEDS", collapsible=True, expanded=True)
        speed_card.add(self.binder.slider(machine, "draw_feed", 100, 9000, label="Drawing", decimals=0, step=100, suffix="mm/min"))
        speed_card.add(self.binder.slider(machine, "travel_feed", 100, 12000, label="Travel", decimals=0, step=100, suffix="mm/min"))
        speed_card.add(self.binder.slider(machine, "z_feed", 60, 3000, label="Pen up/down", decimals=0, step=60, suffix="mm/min"))
        speed_card.add(self.binder.slider(machine, "z_max_feed", 60, 3000, label="Z limit in firmware", decimals=0, step=60, suffix="mm/min",
                                          hint="Stock Ender 3 firmware allows 300 mm/min. Asking for more than this "
                                               "does nothing unless you raise the limit below."))
        speed_card.add(self.binder.check(machine, "raise_z_limit", "Raise the Z speed limit for the job",
                                         hint="Sends M203 at the start and puts it back at the end. A bare pen "
                                              "carriage is much lighter than a hot end, so this is usually safe - "
                                              "and pen lifts are most of the drawing time."))
        speed_card.add(self.binder.slider(machine, "z_limit_target", 120, 3000, label="Raise Z to", decimals=0, step=60, suffix="mm/min"))
        speed_card.add(self.binder.slider(machine, "acceleration", 100, 3000, label="Acceleration", decimals=0, step=50, suffix="mm/s²"))
        speed_card.add(self.binder.slider(machine, "travel_acceleration", 100, 5000, label="Travel accel", decimals=0, step=100, suffix="mm/s²"))
        speed_card.add(self.binder.check(machine, "send_acceleration", "Send M204 so the machine matches"))
        speed_card.add(hint_label("Ballpoints like 1500-2500 mm/min. Fineliners and pencils can go faster."))
        outer.addWidget(speed_card)

        printer_card = Card("PRINTER", collapsible=True, expanded=False)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Pick a model…", "")
        for profile_name in profiles.profile_names():
            self.profile_combo.addItem(profile_name, profile_name)
        self.profile_combo.currentIndexChanged.connect(self._apply_profile)
        printer_card.add(FieldRow("Model", self.profile_combo))
        printer_card.add(hint_label(
            "Sets the bed size and, just as importantly, which optional commands your "
            "firmware understands. M203 means mm/s on Marlin but mm/min on a Duet, and "
            "Klipper does not take it at all."
        ))
        printer_card.add(
            self.binder.combo(machine, "firmware", profiles.FIRMWARES, label="Firmware")
        )
        printer_card.add(self.binder.line(machine, "name", label="Name"))
        printer_card.add(self.binder.slider(machine, "bed_x", 50, 500, label="Bed X", decimals=0, step=5, suffix="mm"))
        printer_card.add(self.binder.slider(machine, "bed_y", 50, 500, label="Bed Y", decimals=0, step=5, suffix="mm"))
        printer_card.add(
            self.binder.combo(
                machine, "home_mode",
                {"xy": "Home X and Y only (safe with a pen)", "all": "Home all axes", "none": "Do not home"},
                label="Homing",
            )
        )
        printer_card.add(self.binder.slider(machine, "park_x", 0, 400, label="Park X", decimals=0, step=5, suffix="mm"))
        printer_card.add(self.binder.slider(machine, "park_y", 0, 400, label="Park Y", decimals=0, step=5, suffix="mm"))
        printer_card.add(self.binder.check(machine, "disable_motors_at_end", "Release motors when finished"))
        printer_card.add(self.binder.check(machine, "use_bed_mesh", "Use the stored bed mesh (M420)",
                                           hint="If you have run a bed levelling probe, this keeps the pen pressure "
                                                "even across a warped bed."))
        outer.addWidget(printer_card)

        opt_card = Card("PATH OPTIMISATION", collapsible=True, expanded=False)
        opt_card.add(self.binder.slider(optimise, "simplify", 0.0, 1.0, label="Simplify", decimals=3, step=0.01, suffix="mm",
                                        hint="Drops points that do not change the shape. Keeps files small."))
        opt_card.add(self.binder.slider(optimise, "join", 0.0, 3.0, label="Join gaps", decimals=2, step=0.05, suffix="mm",
                                        hint="Chains strokes that nearly touch, so the pen lifts less often."))
        opt_card.add(self.binder.slider(optimise, "min_length", 0.0, 5.0, label="Drop shorter than", decimals=2, step=0.05, suffix="mm"))
        opt_card.add(self.binder.slider(optimise, "stitch", 0.0, 10.0, label="Connect strokes", decimals=2, step=0.1, suffix="mm",
                                        hint="Draws straight through gaps this small instead of lifting. The single "
                                             "biggest time saver on hatching - measured 33 to 60 % off. Only applies "
                                             "to techniques where it does not change the picture."))
        opt_card.add(self.binder.check(optimise, "reorder", "Reorder to cut travel"))
        opt_card.add(self.binder.check(optimise, "tidy_tour", "Extra pass to tidy the order"))
        opt_card.add(self.binder.check(optimise, "allow_reverse", "Allow drawing backwards"))
        outer.addWidget(opt_card)

        outer.addStretch(1)

    def _apply_profile(self, index: int) -> None:
        name = self.profile_combo.itemData(index)
        if not name:
            return
        profile = profiles.find_profile(name)
        if profile is None:
            return
        profiles.apply_profile(self.settings.machine, profile, self.settings.pen)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(0)
        self.profile_combo.blockSignals(False)
        self.binder.refresh()
        self.changed.emit()

    def refresh(self) -> None:
        self.binder.refresh()
