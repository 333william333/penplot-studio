"""USB serial link to the printer.

Marlin is driven the way a normal host does it: numbered lines with checksums,
a small look-ahead window, and one "ok" per accepted command.  Resend requests
are honoured, so a dropped byte on the USB cable does not ruin a drawing.

Pen changes and sharpening stops are handled *host side* - the stream simply
stops and the app asks the user to continue - which works no matter how the
firmware is configured.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot

try:
    import serial
    from serial.tools import list_ports
except Exception:  # pragma: no cover - pyserial is a hard dependency in practice
    serial = None
    list_ports = None

from .gcode import PAUSE_MARKER

__all__ = ["PrinterLink", "available_ports", "PortInfo"]

# how long to wait for an "ok" before assuming something went wrong
STALL_SECONDS = 15.0

#: Past this many resend requests something is wrong with the cable or the baud
#: rate, and replaying strokes on top of each other helps nobody.
MAX_RESENDS = 40

#: A manual command is a live gesture.  If it has not reached the wire within
#: this long something went wrong, and replaying it later moves the machine on
#: behalf of an operator who has moved on.
MANUAL_TTL = 20.0


@dataclass
class PortInfo:
    device: str
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.device}  -  {self.description}" if self.description else self.device


def _num(value: float) -> str:
    """Trim a float for the wire without eating a digit: 10.0 -> "10", not "1"."""
    text = f"{value:.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def available_ports() -> list[PortInfo]:
    """Serial ports, with the likely printer ports first."""
    if list_ports is None:
        return []
    found: list[PortInfo] = []
    for port in list_ports.comports():
        device = port.device
        if "Bluetooth" in device or "debug-console" in device:
            continue
        description = (port.description or "").strip()
        if description.lower() in ("n/a", "unknown", "none"):
            description = ""
        found.append(PortInfo(device=device, description=description))

    def rank(info: PortInfo) -> int:
        needle = f"{info.device} {info.description}".lower()
        for i, hint in enumerate(("usbserial", "wchusb", "ch340", "usbmodem", "cp210", "ftdi")):
            if hint in needle:
                return i
        return 10

    found.sort(key=rank)
    return found


class _Worker(QObject):
    """Lives on the serial thread and does all the blocking work."""

    log = Signal(str, str)              # message, kind: tx/rx/info/error
    connected = Signal(str)
    disconnected = Signal(str)
    progress = Signal(int, int, float)  # acknowledged line, total, drawn mm
    paused = Signal(str)
    job_finished = Signal(bool, str)
    position = Signal(float, float, float)
    measured = Signal(float, float, float)   # a real M114 reply, not a guess
    reference_lost = Signal()                # motors released: Z means nothing now
    state = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.serial = None
        self.timer: QTimer | None = None
        self.buffer = ""
        self.history: dict[int, str] = {}
        self.line_number = 0
        self.pending: list[int] = []
        self.resend_from: int | None = None
        self.window = 3
        self.resends = 0
        #: one escape lift per job, no matter how the job dies
        self._escaped = False
        self.use_checksums = True
        self.last_ok = 0.0
        self.last_rx = 0.0
        self.opened_at = 0.0
        self.saw_any_ok = False
        self.warned_silent = False
        self.stall_nudges = 0

        self.program_lines: list[str] = []
        self.program_drawn: list[float] = []
        self.pause_at: dict[int, str] = {}
        self.z_at: dict[int, tuple[str, float, float]] = {}
        #: sent line number -> index in program_lines, so a resend request for a
        #: line that has fallen out of the history can still be recovered exactly
        self.number_to_cursor: dict[int, int] = {}
        self.live_z_offset = 0.0     # mm added to the drawing height
        self.live_lift_delta = 0.0   # mm added to the pen hop
        self.cursor = 0
        self.total_lines = 0
        self.running = False
        self.is_paused = False
        self.skip_m0 = True
        #: (queued_at, command) - see MANUAL_TTL
        self.manual: list[tuple[float, str]] = []
        self.last_position = [0.0, 0.0, 0.0]
        self.relative = False
        self.waiting_for_start = False
        self.start_deadline = 0.0

    # ---------------- lifecycle ----------------------------------------
    @Slot()
    def setup(self) -> None:
        self.timer = QTimer()
        self.timer.setInterval(4)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    @Slot(str, int)
    def open_port(self, port: str, baud: int) -> None:
        if serial is None:
            self.log.emit("pyserial is not installed", "error")
            return
        self.close_port()
        try:
            self.serial = serial.Serial(port=port, baudrate=int(baud), timeout=0, write_timeout=2)
        except Exception as exc:
            self.log.emit(f"Could not open {port}: {exc}", "error")
            self.disconnected.emit(str(exc))
            return
        self.buffer = ""
        self.history.clear()
        self.pending.clear()
        self.line_number = 0
        self.manual.clear()          # nothing from before this port belongs here
        self.resend_from = None
        self.relative = False
        self.resends = 0
        self._escaped = False
        self.waiting_for_start = True
        self.opened_at = time.monotonic()
        self.last_ok = self.opened_at
        self.last_rx = self.opened_at
        self.saw_any_ok = False
        self.warned_silent = False
        self.stall_nudges = 0
        self.start_deadline = self.opened_at + 5.0
        self.log.emit(f"Opened {port} at {baud} baud", "info")
        self.connected.emit(port)
        self.state.emit("connected")

    @Slot()
    def teardown(self) -> None:
        """Stop the poll timer from *inside* the serial thread before it quits."""
        if self.timer is not None:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None
        self.close_port()

    @Slot()
    def close_port(self) -> None:
        self.running = False
        self.is_paused = False
        self.manual.clear()
        self.resend_from = None
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None
            self.log.emit("Port closed", "info")
            self.disconnected.emit("")
            self.state.emit("idle")

    # ---------------- job control --------------------------------------
    @Slot(float, float)
    def set_live_z(self, offset: float, lift_delta: float) -> None:
        """Pen pressure and hop height, adjustable while the job runs."""
        self.live_z_offset = float(offset)
        self.live_lift_delta = float(lift_delta)

    @Slot(object, object, object, object, bool)
    def start_job(self, lines: list, drawn: list, pause_at: dict, z_at: dict, skip_m0: bool) -> None:
        if self.serial is None:
            self.log.emit("Not connected", "error")
            return
        self.program_lines = list(lines)
        self.program_drawn = list(drawn)
        self.pause_at = {int(k): v for k, v in pause_at.items()}
        self.z_at = {int(k): tuple(v) for k, v in (z_at or {}).items()}
        self.number_to_cursor.clear()
        self.total_lines = len(self.program_lines)
        self.cursor = 0
        self.skip_m0 = skip_m0
        self.resends = 0
        self._escaped = False
        self.running = True
        self.is_paused = False
        self.last_ok = time.monotonic()
        self.stall_nudges = 0
        self._reset_line_numbers()
        self.log.emit(f"Streaming {self.total_lines} lines", "info")
        self.state.emit("printing")

    @Slot()
    def pause_job(self) -> None:
        if self.running and not self.is_paused:
            self.is_paused = True
            self.state.emit("paused")
            self.log.emit("Paused", "info")

    @Slot()
    def resume_job(self) -> None:
        if self.running and self.is_paused:
            self.is_paused = False
            self.state.emit("printing")
            self.log.emit("Resumed", "info")

    @Slot(bool)
    def cancel_job(self, lift: bool) -> None:
        was_running = self.running
        self.running = False
        self.is_paused = False
        self.program_lines = []
        self.cursor = 0
        if lift and self.serial is not None:
            for command in ("G91", "G1 Z10 F900", "G90", "M117 Cancelled"):
                self.send_manual(command)
        if was_running:
            self.job_finished.emit(False, "Cancelled")
        self.state.emit("connected" if self.serial else "idle")

    @Slot(bool)
    def set_protocol(self, use_checksums: bool) -> None:
        self.use_checksums = bool(use_checksums)
        self.log.emit(
            "Line numbers and checksums on" if use_checksums else "Simple mode: no line numbers",
            "info",
        )

    @Slot(str)
    def send_manual(self, command: str) -> None:
        """Queue one line for the next free slot in the send window.

        Nothing is queued while the port is shut.  It used to be, and the list
        survived open_port, so twenty taps on the Z jog button with no printer
        attached fired 210 mm of relative Z travel the instant a port opened -
        the machine ran for the top of the gantry before the operator had asked
        for anything at all.
        """
        command = command.strip()
        if not command:
            return
        if self.serial is None:
            self.log.emit(f"Not connected - {command} was not sent", "error")
            return
        self.manual.append((time.monotonic(), command))

    @Slot()
    def emergency_stop(self) -> None:
        if self.serial is None:
            return
        self.running = False
        try:
            self.serial.write(b"\nM112\n")
            self.log.emit("M112 emergency stop sent - the printer needs a reset", "error")
        except Exception as exc:
            self.log.emit(f"Emergency stop failed: {exc}", "error")

    # ---------------- internals ----------------------------------------
    def _reset_line_numbers(self) -> None:
        """Agree with the firmware on where the line numbering starts.

        This is sent *without* a line number of its own.  Firmwares disagree
        about whether `N1 M110 N0` means "the last line was 1" or "the last
        line was 0", and guessing wrong desynchronises every following line.
        An unnumbered M110 goes straight to the firmware's own M110 handler,
        which simply takes the N value out of the command - no ambiguity.
        """
        self.history.clear()
        self.pending.clear()
        self.resend_from = None
        self.line_number = 0
        self._write_plain("M110 N0")

    def _checksum(self, text: str) -> int:
        checksum = 0
        for char in text:
            checksum ^= ord(char)
        return checksum & 0xFF

    def _write_plain(self, command: str) -> None:
        """Send a bare line with no number and no checksum."""
        if self.serial is None:
            return
        try:
            self.serial.write((command + "\n").encode("ascii", errors="replace"))
        except Exception as exc:
            self.log.emit(f"Write failed: {exc}", "error")
            self.close_port()
            return
        self.pending.append(0)  # 0 = unnumbered: still costs one "ok"
        self._track_position(command)
        self.log.emit(command, "tx")

    def _write_numbered(self, number: int, command: str, track: bool = True) -> None:
        if self.serial is None:
            return
        body = f"N{number} {command}"
        payload = f"{body}*{self._checksum(body)}\n"
        try:
            self.serial.write(payload.encode("ascii", errors="replace"))
        except Exception as exc:
            self.log.emit(f"Write failed: {exc}", "error")
            self.close_port()
            return
        self.pending.append(number)
        if track:
            self._track_position(command)
        self.log.emit(command, "tx")

    def _write_raw(self, command: str) -> None:
        if self.serial is None:
            return
        if not self.use_checksums:
            self._write_plain(command)
            return
        self.line_number += 1
        self.history[self.line_number] = command
        if len(self.history) > 1200:
            for key in sorted(self.history)[:300]:
                self.history.pop(key, None)
        if len(self.number_to_cursor) > 4000:
            for key in sorted(self.number_to_cursor)[:1000]:
                self.number_to_cursor.pop(key, None)
        self._write_numbered(self.line_number, command)

    def _track_position(self, command: str) -> None:
        upper = command.upper()
        if upper.startswith(("M84", "M18", "M112")):
            # Streamed, not typed: the generated footer ends with M84, so the
            # pen height reference dies at the end of every job and the app has
            # to stop believing it is still calibrated.
            self.reference_lost.emit()
        if upper.startswith("G91"):
            self.relative = True
        elif upper.startswith("G90"):
            self.relative = False
        if not (upper.startswith("G0") or upper.startswith("G1")):
            return
        changed = False
        for index, axis in enumerate("XYZ"):
            marker = f" {axis}"
            position = upper.find(marker)
            if position < 0:
                continue
            chunk = upper[position + 2 :].split(" ")[0]
            try:
                value = float(chunk)
            except ValueError:
                continue
            if self.relative:
                self.last_position[index] += value
            else:
                self.last_position[index] = value
            changed = True
        if changed:
            self.position.emit(*self.last_position)

    def _resend_from(self, number: int) -> None:
        """Rewind the stream; the normal send window replays it line by line.

        Blasting the whole backlog in one go would overrun Marlin's 128-byte
        receive buffer, so recovery has to respect the same flow control.
        """
        if number not in self.history:
            # The line has fallen out of the replay buffer, but the program is
            # still in memory: rewind to the matching program line and send it
            # again.  Simply renumbering (the old behaviour) skipped everything
            # in between, including pen-up moves - the pen would be dragged
            # across the paper for hundreds of commands.
            # A legitimate resend is always for a line the firmware just
            # received.  Anything far behind is a glitch, and replaying it would
            # redraw a large part of the picture on top of itself.
            if self.line_number - number > 200:
                self.log.emit(
                    f"Ignoring a resend request for line {number}, {self.line_number - number} "
                    "lines behind - stopping instead of redrawing.",
                    "error",
                )
                self._abort_job("Lost sync with the printer")
                return
            target = self.number_to_cursor.get(number)
            if target is not None:
                self.pending.clear()
                self.resend_from = None
                self.line_number = number - 1
                self.cursor = target
                self.log.emit(f"Rewound to program line {target} to resend {number}", "info")
                return
            self.log.emit(
                f"The printer asked for line {number}, which is no longer available. Stopping.",
                "error",
            )
            self._abort_job("Lost sync with the printer")
            return
        self.resends += 1
        if self.resends > MAX_RESENDS:
            self.log.emit(
                f"The printer has asked for {self.resends} resends. Stopping instead of "
                "replaying the drawing over and over.",
                "error",
            )
            self._abort_job("Too many resend requests")
            return
        self.pending.clear()
        self.resend_from = number
        self.log.emit(f"Resending from line {number}", "info")

    def _resync(self, last_accepted: int) -> None:
        """Adopt the firmware's idea of the last accepted line number.

        Everything after it has to be replayed - dropping it would silently
        lose a move, which on a plotter means a missing stroke.
        """
        last_accepted = max(int(last_accepted), 0)
        self._resend_from(last_accepted + 1)

    def _abort_job(self, reason: str, lift: bool = True) -> None:
        """Stop the drawing and get the pen off the paper.

        Exactly one escape lift per job.  It used to re-arm: the lift itself
        refills `pending`, a printer that has stopped answering then trips the
        stall timer again, and every cycle queued another relative `G1 Z10` -
        the machine walked up its own Z axis 10 mm at a time until it hit the
        top, with nobody having asked for anything.
        """
        was_running = self.running
        self.running = False
        self.is_paused = False
        self.pending.clear()
        self.resend_from = None
        if self._escaped or not was_running:
            # nothing was drawing, so there is no pen on paper to rescue
            lift = False
        if lift and self.serial is not None:
            self._escaped = True
            # An abort almost always follows a numbering desync, and recovery
            # lines sent on the old counter are rejected out of hand - the pen
            # would sit on the paper bleeding a hole while we congratulated
            # ourselves on having lifted it.  Resynchronise first.
            self._reset_line_numbers()
            # Relative on purpose: after a desync the tracked height may be a
            # fiction, and an absolute guess could drive the pen down instead.
            for command in ("G91", "G1 Z10 F900", "G90"):
                self.send_manual(command)
        if was_running:
            self.job_finished.emit(False, reason)
        self.state.emit("connected" if self.serial else "idle")

    def _handle_response(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        lowered = stripped.lower()
        self.last_rx = time.monotonic()

        # A burst of line noise with no newline gets glued to the front of the
        # next reply ("<garbage>ok").  Accepting a trailing ok keeps one bad
        # byte from stalling the whole stream.
        if lowered.startswith("ok") or lowered.endswith("ok"):
            self.last_ok = time.monotonic()
            self.saw_any_ok = True
            self.stall_nudges = 0
            if self.pending:
                self.pending.pop(0)
                self._emit_progress()
            return
        if lowered.startswith("resend") or lowered.startswith("rs"):
            digits = "".join(c if c.isdigit() else " " for c in stripped).split()
            if digits:
                self._resend_from(int(digits[0]))
            return
        if lowered.startswith("start"):
            self.waiting_for_start = False
            if self.running:
                # The board rebooted: X/Y are unknown and the G92 paper zero is
                # gone.  Carrying on would drive the pen to an absolute position
                # the machine no longer understands.
                self.log.emit("The printer reset in the middle of the drawing - stopping.", "error")
                self._reset_line_numbers()
                self._abort_job("The printer reset - set the pen height again before restarting", lift=False)
                return
            self.log.emit("Printer ready", "info")
            self._reset_line_numbers()
            return
        if "x:" in lowered and "z:" in lowered:
            # M114: "X:110.00 Y:110.00 Z:4.35 E:0.00 Count: ..."  The machine is
            # the only honest source for the pen height reference, so take it
            # from here rather than from what we think we sent.
            found = dict(re.findall(r"([XYZ]):\s*(-?\d+(?:\.\d+)?)", stripped.upper()))
            if len(found) >= 3:
                self.last_position = [float(found["X"]), float(found["Y"]), float(found["Z"])]
                self.position.emit(*self.last_position)
                self.measured.emit(*self.last_position)
            return
        if "busy" in lowered:
            # proof the machine is alive and working: homing, a long move or an
            # M0 prompt can legitimately take minutes
            self.last_ok = time.monotonic()
            self.stall_nudges = 0
            return
        if lowered.startswith("error") or "!!" in lowered:
            self.log.emit(stripped, "error")
            # "Line Number is not Last Line Number+1, Last Line: 42"
            match = re.search(r"last line:?\s*(\d+)", lowered)
            if match:
                self._resync(int(match.group(1)))
            return
        self.log.emit(stripped, "rx")

    def _emit_progress(self) -> None:
        if not self.running or not self.total_lines:
            return
        acknowledged = max(self.cursor - len(self.pending), 0)
        drawn = self.program_drawn[min(acknowledged, len(self.program_drawn) - 1)] if self.program_drawn else 0.0
        self.progress.emit(acknowledged, self.total_lines, drawn)

    def _apply_live_z(self, index: int, line: str) -> str:
        """Rewrite a tagged Z move with the values the user is dialling right now."""
        tag = self.z_at.get(index)
        if not tag:
            return line
        role, draw_z, lift = tag
        target = draw_z + self.live_z_offset
        if role == "up":
            target += max(lift + self.live_lift_delta, 0.1)
        # role "draw" is a modulated drawing move: it already carries its own
        # height, and only needs the live offset added
        return re.sub(r"Z-?\d+(?:\.\d+)?", f"Z{_num(target)}", line, count=1)

    def _should_send(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            return False
        if self.skip_m0 and stripped.upper().startswith("M0"):
            return False
        return True

    def _tick(self) -> None:
        if self.serial is None:
            return
        # ---- read ----
        try:
            waiting = self.serial.in_waiting
            if waiting:
                self.buffer += self.serial.read(waiting).decode("utf-8", errors="replace")
        except Exception as exc:
            self.log.emit(f"Read failed: {exc}", "error")
            self.close_port()
            return
        if len(self.buffer) > 4096:
            self.log.emit("Discarding line noise from the serial port", "error")
            self.buffer = self.buffer[-256:]
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._handle_response(line)

        now = time.monotonic()
        if self.waiting_for_start:
            if now > self.start_deadline:
                self.waiting_for_start = False
                self.log.emit("No boot message - continuing anyway", "info")
                self._reset_line_numbers()
            return

        # ---- is the printer answering at all? ----
        if not self.saw_any_ok and not self.warned_silent and now - self.opened_at > 8.0:
            self.warned_silent = True
            self.log.emit(
                "Connected, but the printer has not answered once. The baud rate is "
                "probably wrong - try 250000, or check the USB cable.",
                "error",
            )

        if self.pending and now - self.last_ok > STALL_SECONDS:
            self.stall_nudges += 1
            self.last_ok = now
            if self.stall_nudges > 3:
                self.log.emit(
                    "The printer stopped answering. Check the cable, the baud rate, "
                    "and whether the display is waiting for a button press.",
                    "error",
                )
                self._abort_job("No answer from the printer")
                return
            self.log.emit(f"No answer for {STALL_SECONDS:.0f} s - nudging the printer", "error")
            numbered = [n for n in self.pending if n > 0]
            if numbered and self.running:
                self._resend_from(min(numbered))
            else:
                # Nothing is streaming, so there is no stream to recover.
                # Replaying here would re-run whatever was last written, and
                # the last thing written after an abort is a relative lift.
                self.pending.clear()

        # ---- catch up after a resend request before anything new goes out ----
        while self.resend_from is not None and len(self.pending) < self.window:
            if self.resend_from > self.line_number:
                self.resend_from = None
                break
            command = self.history.get(self.resend_from)
            if command is None:
                self.resend_from = None
                break
            self._write_numbered(self.resend_from, command, track=False)
            self.resend_from += 1
        if self.resend_from is not None:
            return

        # ---- write ----
        while self.manual and len(self.pending) < self.window:
            queued_at, command = self.manual.pop(0)
            if now - queued_at > MANUAL_TTL:
                self.log.emit(f"Dropped a stale command ({command})", "error")
                continue
            self._write_raw(command)

        if not self.running or self.is_paused:
            return

        while self.cursor < self.total_lines and len(self.pending) < self.window:
            index = self.cursor
            line = self.program_lines[index]
            if index in self.pause_at:
                if self.pending:
                    return  # let the queued moves finish before stopping
                self.cursor += 1
                self.is_paused = True
                self.state.emit("paused")
                self.paused.emit(self.pause_at[index])
                return
            self.cursor += 1
            if self._should_send(line):
                self._write_raw(self._apply_live_z(index, line))
                self.number_to_cursor[self.line_number] = index

        if self.running and self.serial is not None and self.cursor >= self.total_lines and not self.pending:
            self.running = False
            self.progress.emit(self.total_lines, self.total_lines, self.program_drawn[-1] if self.program_drawn else 0.0)
            self.job_finished.emit(True, "Drawing finished")
            self.state.emit("connected")


class PrinterLink(QObject):
    """Main-thread facade around the serial worker."""

    log = Signal(str, str)
    connected = Signal(str)
    disconnected = Signal(str)
    progress = Signal(int, int, float)
    paused = Signal(str)
    job_finished = Signal(bool, str)
    position = Signal(float, float, float)
    measured = Signal(float, float, float)
    reference_lost = Signal()
    state_changed = Signal(str)

    _open = Signal(str, int)
    _close = Signal()
    _start = Signal(object, object, object, object, bool)  # object, not dict: QVariantMap needs string keys
    _pause = Signal()
    _resume = Signal()
    _cancel = Signal(bool)
    _manual = Signal(str)
    _estop = Signal()
    _protocol = Signal(bool)
    _live_z = Signal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.thread = QThread()
        self.thread.setObjectName("serial")
        self.worker = _Worker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.setup)
        self.worker.log.connect(self.log)
        self.worker.connected.connect(self._on_connected)
        self.worker.disconnected.connect(self._on_disconnected)
        self.worker.progress.connect(self.progress)
        self.worker.paused.connect(self.paused)
        self.worker.job_finished.connect(self.job_finished)
        self.worker.position.connect(self.position)
        self.worker.measured.connect(self._on_measured)
        self.worker.reference_lost.connect(self._on_reference_lost)
        self.worker.state.connect(self.state_changed)

        self._open.connect(self.worker.open_port)
        self._close.connect(self.worker.close_port)
        self._start.connect(self.worker.start_job)
        self._pause.connect(self.worker.pause_job)
        self._resume.connect(self.worker.resume_job)
        self._cancel.connect(self.worker.cancel_job)
        self._manual.connect(self.worker.send_manual)
        self._estop.connect(self.worker.emergency_stop)
        self._protocol.connect(self.worker.set_protocol)
        self._live_z.connect(self.worker.set_live_z)

        self._is_connected = False
        self._port = ""
        #: the machine's own Z from the last M114, or None if it has never said
        self.machine_z: float | None = None
        #: has the operator zeroed the pen on the paper since connecting?
        #: The G92 reference does not survive a power cycle or an M84, so this
        #: is deliberately not persisted.
        self.pen_zeroed = False
        self.thread.start()

    # ---------------- API ------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def port(self) -> str:
        return self._port

    def connect_to(self, port: str, baud: int) -> None:
        self._open.emit(port, int(baud))

    def disconnect_from(self) -> None:
        self._close.emit()

    def start(self, program, skip_m0: bool = True) -> None:
        self._start.emit(
            list(program.lines),
            list(program.drawn_at),
            dict(program.pause_at),
            dict(getattr(program, "z_at", {})),
            bool(skip_m0),
        )

    def set_live_z(self, offset: float, lift_delta: float) -> None:
        self._live_z.emit(float(offset), float(lift_delta))

    def set_speed(self, percent: int) -> None:
        """Marlin's feedrate override - takes effect immediately."""
        self.send(f"M220 S{int(max(10, min(percent, 300)))}")

    def pause(self) -> None:
        self._pause.emit()

    def resume(self) -> None:
        self._resume.emit()

    def cancel(self, lift: bool = True) -> None:
        self._cancel.emit(bool(lift))

    def send(self, command: str) -> None:
        head = command.strip().upper()
        if head.startswith("M84") or head.startswith("M18") or head.startswith("M112"):
            self.pen_zeroed = False   # the motors let go; the reference is gone
        self._manual.emit(command)

    def query_position(self) -> None:
        """Ask the machine where it actually is."""
        self.send("M114")

    def _on_reference_lost(self) -> None:
        self.pen_zeroed = False
        self.machine_z = None
        self.reference_lost.emit()

    def _on_measured(self, x: float, y: float, z: float) -> None:
        self.machine_z = z
        self.measured.emit(x, y, z)

    def set_protocol(self, use_checksums: bool) -> None:
        self._protocol.emit(bool(use_checksums))

    def emergency_stop(self) -> None:
        self._estop.emit()

    def shutdown(self) -> None:
        if self.thread.isRunning():
            QMetaObject.invokeMethod(self.worker, "teardown", Qt.BlockingQueuedConnection)
        self.thread.quit()
        self.thread.wait(1500)

    # ---------------- internals -----------------------------------------
    def _on_connected(self, port: str) -> None:
        self._is_connected = True
        self._port = port
        self.pen_zeroed = False
        self.machine_z = None
        self.connected.emit(port)

    def _on_disconnected(self, reason: str) -> None:
        self._is_connected = False
        self.pen_zeroed = False
        self._port = ""
        self.disconnected.emit(reason)
