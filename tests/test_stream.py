"""End-to-end tests of the USB streaming protocol against a fake Marlin.

A pseudo terminal stands in for the printer, so the handshake, the resend
recovery, the line-number resync and the stall watchdog are all exercised
without any hardware attached.

Three firmwares are simulated:

* **normal**  - well behaved, asks for one resend on purpose;
* **quirky**  - reports a line-number error mid-job (this is what a real
  Ender 3 did when the M110 handshake was ambiguous);
* **silent**  - stops answering completely.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
import tty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from penplot.core import printer as printer_module  # noqa: E402
from penplot.core.gcode import PAUSE_MARKER, GCodeProgram  # noqa: E402
from penplot.core.printer import PrinterLink  # noqa: E402

NUMBERED = re.compile(r"^N(\d+)\s(.*)\*(\d+)$")
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name} {detail}")
    if not condition:
        FAILURES.append(name)


class FakeMarlin(threading.Thread):
    """Line-numbered G-code receiver with configurable misbehaviour."""

    def __init__(self, master_fd: int, mode: str = "normal"):
        super().__init__(daemon=True)
        self.fd = master_fd
        self.mode = mode
        self.rebooted = False
        self.noise_sent = False
        self.busy_sent = 0
        self.received: list[str] = []
        self.numbers: list[int] = []
        self.bad_checksums: list[str] = []
        self.stop = False
        self.expected = 1
        self.forced_resend_done = False
        self.forced_desync_done = False
        self.went_silent = False
        self._buffer = ""

    def run(self) -> None:
        os.write(self.fd, b"start\n")
        os.write(self.fd, b"echo:Marlin 2.0.9 (fake)\n")
        while not self.stop:
            try:
                chunk = os.read(self.fd, 4096)
            except OSError:
                return
            except Exception as exc:  # surface crashes instead of hanging the test
                print("fake printer crashed:", exc)
                return
            if not chunk:
                time.sleep(0.005)
                continue
            self._buffer += chunk.decode("ascii", errors="replace")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._handle(line.strip())

    def _reply(self, text: bytes) -> None:
        if not self.went_silent:
            os.write(self.fd, text)

    def _handle(self, line: str) -> None:
        if not line:
            return

        match = NUMBERED.match(line)
        if not match:
            # unnumbered line: accepted as-is, and M110 sets the line counter
            inner = re.search(r"M110.*?N(\d+)", line)
            if inner:
                self.expected = int(inner.group(1)) + 1
            else:
                self.received.append(line)
            self._reply(b"ok\n")
            return

        number, body, checksum = int(match.group(1)), match.group(2), int(match.group(3))
        wanted = 0
        for char in f"N{number} {body}":
            wanted ^= ord(char)
        if wanted != checksum:
            self.bad_checksums.append(line)

        # M110 carries its own line number and is exempt from the sequence
        # rule, exactly as in firmware - stock Creality refuses a bare
        # "M110 N0" because it finds the N and then wants a checksum.
        inner = re.search(r"M110.*?N(\d+)", body)
        if inner:
            self.expected = int(inner.group(1)) + 1
            self._reply(b"ok\n")
            return

        if self.mode == "silent" and len(self.received) >= 10:
            self.went_silent = True
            return

        if self.mode == "reboot" and not self.rebooted and len(self.received) >= 24:
            # brown-out: the board restarts and forgets where it is
            self.rebooted = True
            self.expected = 1
            self._reply(b"start\n")
            return

        if self.mode == "noise" and not self.noise_sent and len(self.received) == 9:
            # a burst with no newline gets glued to the next reply
            self.noise_sent = True
            self.expected = number + 1
            self.numbers.append(number)
            self.received.append(body)
            self._reply(b"\xfe\xff garbage")
            self._reply(b"ok\n")
            return

        if self.mode == "oldresend" and not self.forced_resend_done and len(self.received) >= 1400:
            # asks for a line that has long fallen out of the replay buffer
            self.forced_resend_done = True
            self.expected = 5
            self._reply(b"Resend: 5\n")
            return

        if self.mode == "busy" and self.busy_sent < 40 and len(self.received) >= 12:
            # a long move: Marlin keeps saying it is busy, with no ok
            self.busy_sent += 1
            self._reply(b"echo:busy: processing\n")
            return

        if self.mode == "normal" and not self.forced_resend_done and len(self.received) == 12:
            self.forced_resend_done = True
            self.expected = number
            self._reply(b"Error:checksum mismatch\n")
            self._reply(f"Resend: {number}\n".encode())
            self._reply(b"ok\n")
            return

        if self.mode == "quirky" and not self.forced_desync_done and len(self.received) == 8:
            # the firmware is one line further along than we think and only
            # reports it as an error - there is no Resend line to help us
            self.forced_desync_done = True
            self.expected = number
            self._reply(
                f"Error:Line Number is not Last Line Number+1, Last Line: {number - 1}\n".encode()
            )
            self._reply(b"ok\n")
            return

        if number != self.expected:
            self._reply(f"Resend: {self.expected}\n".encode())
            self._reply(b"ok\n")
            return

        self.expected = number + 1
        self.numbers.append(number)
        self.received.append(body)
        self._reply(b"ok\n")


def make_program(repeats: int = 10) -> GCodeProgram:
    program = GCodeProgram()
    lines = ["G21", "G90", "; a comment that must not be sent", ""]
    for i in range(repeats):
        lines += [f"G0 X{i*5} Y10 F6000", "G1 Z0 F900", f"G1 X{i*5+4} Y10 F2400", "G1 Z2 F900"]
    lines.append(f"{PAUSE_MARKER}Insert pen 2: Red")
    lines.append("M0 Insert pen 2: Red")
    for i in range(max(repeats * 6 // 10, 1)):
        lines += [f"G0 X{i*5} Y30 F6000", "G1 Z0 F900", f"G1 X{i*5+4} Y30 F2400", "G1 Z2 F900"]
    lines.append("M84")

    program.lines = lines
    program.drawn_at = [float(i) for i in range(len(lines))]
    program.pause_at = {
        i: text[len(PAUSE_MARKER):] for i, text in enumerate(lines) if text.startswith(PAUSE_MARKER)
    }
    program.z_at = {
        i: ("down" if "Z0" in text else "up", 0.0, 2.0)
        for i, text in enumerate(lines)
        if text.startswith("G1 Z")
    }
    return program


def wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def run_job(mode: str, timeout: float = 12.0, live_z: tuple[float, float] | None = None, repeats: int = 10):
    master_fd, slave_fd = os.openpty()
    tty.setraw(master_fd)  # no line-discipline echo, so the fake never reads its own output
    tty.setraw(slave_fd)
    fake = FakeMarlin(master_fd, mode)
    fake.start()

    link = PrinterLink()
    events: dict[str, object] = {"paused": None, "finished": None, "progress": []}
    link.paused.connect(lambda message: events.__setitem__("paused", message))
    link.job_finished.connect(lambda ok, message: events.__setitem__("finished", (ok, message)))
    link.progress.connect(lambda line, total, drawn: events["progress"].append((line, total)))

    link.connect_to(os.ttyname(slave_fd), 115200)
    wait(600)
    if live_z is not None:
        link.set_live_z(*live_z)
        wait(80)
    program = make_program(repeats)
    link.start(program, skip_m0=True)

    deadline = time.monotonic() + timeout
    while events["finished"] is None and time.monotonic() < deadline:
        if events["paused"] is not None and link.worker.is_paused:
            link.resume()
        wait(40)
    wait(150)

    link.shutdown()
    fake.stop = True
    os.close(slave_fd)
    try:
        os.close(master_fd)
    except OSError:
        pass
    return events, fake, program


def main() -> int:
    QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    expected_moves = 10 * 4 + 6 * 4

    print("\nwell-behaved firmware")
    events, fake, _program = run_job("normal")
    check("pauses for the pen change", events["paused"] == "Insert pen 2: Red", str(events["paused"]))
    check("finishes the job", events["finished"] == (True, "Drawing finished"), str(events["finished"]))
    check("every checksum was valid", not fake.bad_checksums, str(fake.bad_checksums[:2]))
    check(
        "comments and blank lines were skipped",
        not any(line.startswith(";") or not line for line in fake.received),
    )
    check("M0 was skipped in host-pause mode", not any(line.startswith("M0") for line in fake.received))
    check("resend was honoured", fake.forced_resend_done)
    check(
        "accepted line numbers are strictly sequential",
        all(b == a + 1 for a, b in zip(fake.numbers, fake.numbers[1:])),
        str(fake.numbers[:5]),
    )
    moves = [line for line in fake.received if line.startswith(("G0", "G1"))]
    check(f"all {expected_moves} moves arrived", len(moves) == expected_moves, f"got {len(moves)}")
    check("progress was reported", len(events["progress"]) > 10, str(len(events["progress"])))

    print("\nfirmware that reports a line-number error (the real Ender 3 case)")
    events, fake, _program = run_job("quirky")
    check("recovers and finishes", events["finished"] == (True, "Drawing finished"), str(events["finished"]))
    check("the desync really happened", fake.forced_desync_done)
    moves = [line for line in fake.received if line.startswith(("G0", "G1"))]
    check(f"all {expected_moves} moves still arrived", len(moves) == expected_moves, f"got {len(moves)}")

    print("\nlive pen height while streaming")
    events, fake, _program = run_job("normal", live_z=(-0.30, 1.5))
    check("job still finishes", events["finished"] == (True, "Drawing finished"), str(events["finished"]))
    downs = {line for line in fake.received if line.startswith("G1 Z") and "F900" in line}
    z_values = sorted({float(line.split("Z")[1].split()[0]) for line in downs})
    check(
        f"pen height and lift were rewritten live (Z values {z_values})",
        any(abs(v - (-0.30)) < 1e-6 for v in z_values) and any(abs(v - 3.2) < 1e-6 for v in z_values),
        "expected a draw Z of -0.3 and a travel Z of 3.2",
    )

    print("\nthe printer reboots in the middle of the job")
    events, fake, _p = run_job("reboot", timeout=10.0)
    finished = events["finished"]
    check("the job is stopped, not silently continued", bool(finished) and finished[0] is False, str(finished))
    check("and it says the printer reset", "reset" in str(finished[1]).lower(), str(finished))

    print("\nresend for a line far in the past")
    events, fake, program = run_job("oldresend", timeout=45.0, repeats=250)
    finished = events["finished"]
    check("the request really happened", fake.forced_resend_done)
    check(
        "stops rather than redrawing half the picture on top of itself",
        bool(finished) and finished[0] is False and "sync" in str(finished[1]).lower(),
        str(finished),
    )

    print("\nline noise with no newline")
    events, fake, _p = run_job("noise", timeout=14.0)
    check("recovers and finishes", events["finished"] == (True, "Drawing finished"), str(events["finished"]))
    moves = [line for line in fake.received if line.startswith(("G0", "G1"))]
    check(f"all {expected_moves} moves arrived", len(moves) == expected_moves, f"got {len(moves)}")

    print("\na long busy period is not mistaken for a dead printer")
    printer_module.STALL_SECONDS = 0.5
    events, fake, _p = run_job("busy", timeout=20.0)
    printer_module.STALL_SECONDS = 15.0
    check("keeps waiting while the printer says it is busy",
          events["finished"] == (True, "Drawing finished"), str(events["finished"]))

    print("\nfirmware that stops answering")
    printer_module.STALL_SECONDS = 0.6
    events, fake, _program = run_job("silent", timeout=14.0)
    finished = events["finished"]
    check("gives up instead of hanging silently", bool(finished) and finished[0] is False, str(finished))
    check("says why", bool(finished) and "answer" in str(finished[1]).lower(), str(finished))
    printer_module.STALL_SECONDS = 15.0

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing checks: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
