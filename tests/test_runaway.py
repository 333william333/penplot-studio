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
assert len(port.sent) == 1, f"{len(port.sent)} lines went out on connect: {port.sent[:5]}"
print(f"  ok   nothing reached the wire but {port.sent[0]}, Z moved {port.z:+.1f} mm")


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

print("\nall checks passed")
