"""The Z axis must only ever move because of an instruction given right now.

Two ways it used to move on its own, both of which put the gantry into the top
of the frame on a real machine:

  * commands queued while nothing was connected, flushed the moment a port
    opened - twenty taps on the jog button became 210 mm of relative travel
    before the operator had asked for anything;
  * the escape lift on a failed job re-arming itself, so a printer that had
    stopped answering was fed another relative `G1 Z10` every stall cycle.
"""

import os
import re
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv[:1])

from penplot.core import printer as P
from penplot.core import gcode
from penplot.core.drawing import Drawing, Layer
from penplot.core.pipeline import PlotJob
from penplot.core.settings import AppSettings


class FakePort:
    """A Marlin that tracks G90/G91 and can be told to stop answering."""

    def __init__(self, answers: int | None = None):
        self.sent: list[str] = []
        self.rx = b"start\n"
        self.z = 0.0
        self.relative = False
        self.answers = answers

    @property
    def in_waiting(self) -> int:
        return len(self.rx)

    def read(self, n: int) -> bytes:
        out, self.rx = self.rx[:n], self.rx[n:]
        return out

    def write(self, data: bytes) -> int:
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            self.sent.append(line)
            body = re.sub(r"^N\d+ ", "", line).split("*")[0].split(";")[0].strip().upper()
            if body.startswith("G91"):
                self.relative = True
            elif body.startswith("G90"):
                self.relative = False
            elif body.startswith(("G0", "G1")):
                match = re.search(r"Z(-?\d+(?:\.\d+)?)", body)
                if match:
                    value = float(match.group(1))
                    self.z = self.z + value if self.relative else value
            if self.answers is None or self.answers > 0:
                if self.answers is not None:
                    self.answers -= 1
                self.rx += b"ok\n"
        return len(data)

    def close(self) -> None:
        pass


def _worker(port: FakePort) -> "P._Worker":
    P.serial = types.SimpleNamespace(Serial=lambda **kw: port)
    worker = P._Worker()
    worker.log.connect(lambda *a: None)
    worker.open_port("/dev/fake", 115200)
    worker.start_deadline = 0
    for _ in range(50):
        worker._tick()
    return worker


print("\nbuttons pressed with nothing connected")
port = FakePort()
P.serial = types.SimpleNamespace(Serial=lambda **kw: port)
idle = P._Worker()
idle.log.connect(lambda *a: None)
for _ in range(20):                       # twenty taps on Z-up at 10 mm
    for command in ("G91", "G1 Z10.000 F900", "G90"):
        idle.send_manual(command)
idle.open_port("/dev/fake", 115200)
idle.start_deadline = 0
for _ in range(4000):
    idle._tick()
assert abs(port.z) < 1e-6, f"Z moved {port.z:+.1f} mm on connect"
moves = [l for l in port.sent if re.search(r"\b(G0|G1|G28|G92)\b", l.upper())]
assert not moves, f"motion went out on connect: {moves}"
print(f"  ok   only {', '.join(l.split('*')[0] for l in port.sent)} went out; "
      f"Z moved {port.z:+.1f} mm")


print("\na printer that stops answering mid-job")
P.STALL_SECONDS = 0.02
deaf = FakePort(answers=12)
worker = _worker(deaf)
settings = AppSettings.load()
drawing = Drawing(layers=[Layer(pen=0, paths=[np.array([[50, 50], [150, 150]], float)] * 40)])
program = gcode.generate(PlotJob(drawing=drawing), settings, settings.library)
worker.start_job(program.lines, program.drawn_at, program.pause_at, program.z_at, True)
for _ in range(20000):
    worker._tick()
    time.sleep(0.0001)
lifts = [line for line in deaf.sent if "Z10" in line.upper()]
assert len(lifts) <= 1, f"{len(lifts)} escape lifts - the abort path is re-arming"
assert deaf.z <= 15.0, f"Z walked to {deaf.z:+.1f} mm against a silent printer"
print(f"  ok   one 10 mm escape lift and no more; Z ended at {deaf.z:+.1f} mm")


print("\nthe job never redefines where the paper is")
header = "\n".join(program.lines[:20])
if not settings.pen.zero_z_at_start:
    assert "G92" not in header, f"a measured job still re-zeros:\n{header}"
    print("  ok   no G92 in a measured-height job")
else:
    print("  --   touch-off mode is selected; G92 is expected")


print("\nthe escape lift has to survive a numbering desync")


class PickyPort(FakePort):
    """Marlin after a desync: rejects numbered lines until M110 resets it."""

    def __init__(self):
        super().__init__()
        self.synced = True
        self.accepted_lifts = 0

    def write(self, data: bytes) -> int:
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            self.sent.append(line)
            if "M110" in line:          # exempt from the sequence rule, as in firmware
                self.synced = True
                self.rx += b"ok\n"
                continue
            if not self.synced:
                self.rx += b"Error:Line Number is not Last Line Number+1, Last Line: 4\n"
                continue
            body = re.sub(r"^N\d+ ", "", line).split("*")[0].split(";")[0].strip().upper()
            if body.startswith("G91"):
                self.relative = True
            elif body.startswith("G90"):
                self.relative = False
            elif body.startswith(("G0", "G1")):
                match = re.search(r"Z(-?\d+(?:\.\d+)?)", body)
                if match:
                    value = float(match.group(1))
                    self.z = self.z + value if self.relative else value
                    if self.relative and value >= 10:
                        self.accepted_lifts += 1
            self.rx += b"ok\n"
        return len(data)


picky = PickyPort()
worker = _worker(picky)
worker.running = True                    # pretend a drawing is under way
picky.synced = False                     # ...and the counter has drifted
worker._abort_job("lost sync")
for _ in range(400):
    worker._tick()
assert picky.accepted_lifts == 1, (
    f"the pen was left on the paper: {picky.accepted_lifts} lifts accepted"
)
print(f"  ok   the pen came up ({picky.z:+.1f} mm) even though the counter had drifted")


print("\nMarlin's own parser rules")


class StrictMarlin(FakePort):
    """Parses the way queue.cpp does.

    Comment mode is entered at the first ';' and everything after it is thrown
    away, so a checksum written after a comment never arrives.  A line that
    starts with N and has no '*' is refused with STR_ERR_NO_CHECKSUM.
    """

    def __init__(self):
        super().__init__()
        self.refused: list[str] = []
        self.executed: list[str] = []

    def write(self, data: bytes) -> int:
        for raw in data.decode().splitlines():
            if not raw.strip():
                continue
            self.sent.append(raw)
            line = raw.split(";", 1)[0]          # comment mode eats the rest
            if line.lstrip().startswith("N") and "M110" not in line:
                if "*" not in line:
                    self.refused.append(raw)
                    self.rx += b"Error:No Checksum with line number, Last Line: 0\n"
                    continue
                body, _, given = line.rpartition("*")
                checksum = 0
                for char in body:
                    checksum ^= ord(char)
                if int(given) != (checksum & 0xFF):
                    self.refused.append(raw)
                    self.rx += b"Error:checksum mismatch, Last Line: 0\n"
                    continue
                line = re.sub(r"^\s*N\d+\s+", "", body)
            self.executed.append(line.strip())
            body = line.strip().upper()
            if body.startswith("G91"):
                self.relative = True
            elif body.startswith("G90"):
                self.relative = False
            elif body.startswith(("G0", "G1")):
                match = re.search(r"Z(-?\d+(?:\.\d+)?)", body)
                if match:
                    value = float(match.group(1))
                    self.z = self.z + value if self.relative else value
            self.rx += b"ok\n"
        return len(data)


strict = StrictMarlin()
worker = _worker(strict)
worker.relative = True
worker.send_manual("G91")                      # leave it relative, as a jog would
for _ in range(200):
    worker._tick()
worker.start_job(program.lines, program.drawn_at, program.pause_at, program.z_at, True)
for _ in range(60000):
    worker._tick()
assert not strict.refused, (
    f"{len(strict.refused)} lines refused by the firmware, first: {strict.refused[0]!r}"
)
assert "G90" in strict.executed, "the machine never got out of relative mode"
assert strict.z <= settings.machine.max_z, f"Z reached {strict.z:.0f} mm"
print(f"  ok   {len(strict.executed)} lines accepted, 0 refused, Z stayed at {strict.z:+.2f} mm")

print("\nall checks passed")


print("\nthe file must carry the height that was measured, not the one before it")

# A job generated before the pen height was set, then sent afterwards, used to
# go out unchanged: the wizard measured the paper at Z12.40 and the machine
# drew the whole picture at Z0, twelve millimetres above the sheet.
settings.pen.zero_z_at_start = False
settings.pen.draw_z = 0.0
stale = gcode.generate(PlotJob(drawing=drawing), settings, settings.library)
settings.pen.draw_z = 12.4                       # the wizard runs
fresh = gcode.generate(PlotJob(drawing=drawing), settings, settings.library)


def _draw_height(program):
    heights = []
    for line, tag in program.z_at.items():
        if tag[0] == "down":
            heights.append(tag[1])
    return min(heights) if heights else None


assert _draw_height(stale) == 0.0, _draw_height(stale)
assert abs(_draw_height(fresh) - 12.4) < 1e-6, _draw_height(fresh)
print(f"  ok   before {_draw_height(stale):.2f}, after {_draw_height(fresh):.2f} - "
      "send_to_printer builds the file after the wizard, not before")

print("\ntyping a decimal has to work in a comma locale")
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt as _Qt
from penplot.ui import theme as _theme
from penplot.ui.widgets import SliderSpin

_theme.apply_theme(app)
spin = SliderSpin(-1.5, 1.5, decimals=2, step=0.05, suffix="mm")
spin.show()
for _ in range(5):
    app.processEvents()
for text, expected in (("-0.50", -0.5), ("-0,50", -0.5), ("1.25", 1.25)):
    spin.spin.setFocus()
    spin.spin.lineEdit().selectAll()
    QTest.keyClicks(spin.spin, text)
    QTest.keyClick(spin.spin, _Qt.Key_Return)
    app.processEvents()
    assert abs(spin.value() - expected) < 1e-9, f"typed {text!r}, got {spin.value()}"
print("  ok   both '.' and ',' accepted, negatives survive")

print("\nall checks passed")
