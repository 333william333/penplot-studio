"""Printer profiles.

Two things vary between machines and both bite silently if you get them wrong:
the bed size, and what the firmware understands.  The G-code that makes an
Ender 3 lift its pen faster (`M203 Z15`) means *fifteen millimetres per minute*
on a Duet, which would make the job take a day; Klipper does not implement
`M203` at all and will simply error; GRBL has never heard of `M117`.

So a profile carries both: the geometry, and a firmware flavour that decides
which of the optional commands are safe to send.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FIRMWARES", "MachineProfile", "PROFILES", "profile_names", "find_profile", "apply_profile"]


FIRMWARES: dict[str, str] = {
    "marlin": "Marlin / Marlin-compatible",
    "klipper": "Klipper",
    "rrf": "RepRapFirmware (Duet)",
    "grbl": "GRBL",
}


@dataclass(frozen=True)
class Firmware:
    """Which optional commands this firmware understands."""

    key: str
    label: str
    line_numbers: bool = True      # numbered lines with checksums
    status_message: bool = True    # M117
    pause_command: str = "M0"      # what stops and waits for the user
    acceleration: str = "PT"       # "PT" = M204 P/T, "S" = M204 S, "" = none
    z_speed_limit: str = "mm/s"    # units for M203 Z, or "" when unsupported
    axis_acceleration: bool = True # M201 per-axis acceleration
    junction_deviation: bool = True # M205 J
    bed_mesh: str = "M420 S1"      # command to switch on stored levelling, or ""
    motors_off: str = "M84"
    notes: str = ""


FIRMWARE_RULES: dict[str, Firmware] = {
    "marlin": Firmware("marlin", FIRMWARES["marlin"]),
    "klipper": Firmware(
        "klipper",
        FIRMWARES["klipper"],
        pause_command="PAUSE",
        acceleration="S",
        z_speed_limit="",          # Klipper uses SET_VELOCITY_LIMIT, not M203
        axis_acceleration=False,   # nor M201
        junction_deviation=False,  # square corner velocity lives in printer.cfg
        bed_mesh="BED_MESH_PROFILE LOAD=default",
        motors_off="M84",
        notes="Klipper ignores M203; set the Z limit in printer.cfg instead.",
    ),
    "rrf": Firmware(
        "rrf",
        FIRMWARES["rrf"],
        z_speed_limit="mm/min",    # the units differ from Marlin - a 60x mistake
        bed_mesh="G29 S1",
        notes="RepRapFirmware takes M203 in mm/min, not mm/s.",
    ),
    "grbl": Firmware(
        "grbl",
        FIRMWARES["grbl"],
        line_numbers=False,
        status_message=False,
        acceleration="",
        z_speed_limit="",
        axis_acceleration=False,
        junction_deviation=False,
        bed_mesh="",
        motors_off="",
        notes="GRBL understands only motion commands; everything optional is skipped.",
    ),
}


@dataclass(frozen=True)
class MachineProfile:
    name: str
    bed_x: float
    bed_y: float
    max_z: float = 250.0
    firmware: str = "marlin"
    baud: int = 115200
    z_max_feed: float = 300.0       # mm/min the stock firmware allows
    z_acceleration: float = 100.0   # mm/s^2 the stock firmware allows
    #: Three times stock, not five.  The Ender 3's single Z screw lifts the
    #: whole X gantry, and a skipped Z step does not show up as a jam - it
    #: quietly changes the pen pressure for the rest of the drawing.
    z_acceleration_target: float = 300.0
    junction_deviation: float = 0.08
    travel_feed: float = 6000.0
    draw_feed: float = 2400.0
    acceleration: float = 500.0
    travel_acceleration: float = 1200.0
    park_x: float = 5.0
    park_y: float = 200.0
    note: str = ""

    @property
    def firmware_rules(self) -> Firmware:
        return FIRMWARE_RULES.get(self.firmware, FIRMWARE_RULES["marlin"])


#: Bed sizes are the usable area, which is what matters for drawing.
PROFILES: list[MachineProfile] = [
    MachineProfile("Ender 3 / Pro / V2", 220, 220, 250, park_y=200),
    MachineProfile("Ender 3 S1 / S1 Pro", 220, 220, 270, park_y=200),
    MachineProfile("Ender 5 / Plus", 220, 220, 300, park_y=200),
    MachineProfile("Ender 7", 250, 250, 300, park_y=230),
    MachineProfile("CR-10 / CR-10S", 300, 300, 400, park_y=280),
    MachineProfile("CR-10 S5", 500, 500, 500, park_y=470),
    MachineProfile("CR-6 SE", 235, 235, 250, park_y=215),
    MachineProfile("Prusa i3 MK3S / MK4", 250, 210, 210, park_y=190,
                   note="Prusa firmware is Marlin-based; the stock Z limit is higher than Creality's."),
    MachineProfile("Prusa MINI", 180, 180, 180, park_y=165),
    MachineProfile("Artillery Sidewinder X1/X2", 300, 300, 400, park_y=280),
    MachineProfile("Anycubic i3 Mega", 210, 210, 205, park_y=190),
    MachineProfile("Anycubic Kobra 2", 220, 220, 250, park_y=200),
    MachineProfile("Sovol SV06", 220, 220, 250, park_y=200),
    MachineProfile("Elegoo Neptune 4", 225, 225, 265, firmware="klipper", park_y=205),
    MachineProfile("Voron 2.4 (350)", 350, 350, 330, firmware="klipper", park_y=330),
    MachineProfile("Voron Trident (300)", 300, 300, 250, firmware="klipper", park_y=280),
    MachineProfile("Duet-based printer", 300, 300, 300, firmware="rrf", park_y=280),
    MachineProfile("GRBL pen plotter", 300, 200, 40, firmware="grbl", baud=115200,
                   z_max_feed=1000.0, park_y=190,
                   note="A dedicated plotter: Z is usually a servo or a short lead screw."),
    MachineProfile("Custom", 220, 220, 250, park_y=200),
]


def profile_names() -> list[str]:
    return [p.name for p in PROFILES]


def find_profile(name: str) -> MachineProfile | None:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    return None


def apply_profile(machine, profile: MachineProfile, pen_setup=None) -> None:
    """Copy a profile onto a MachineSettings, leaving the port alone.

    `pen_setup` matters more than it looks: the pen-change position and the
    height used while swapping pens are stored per pen setup, not per machine,
    and the Ender 3 defaults (X110 Y200 Z60) are off the bed of a Prusa MINI and
    above the whole Z range of a plotter.  Passing it in is how those follow the
    machine you actually chose.
    """
    machine.name = profile.name
    machine.bed_x = profile.bed_x
    machine.bed_y = profile.bed_y
    machine.max_z = profile.max_z
    machine.firmware = profile.firmware
    machine.baud = profile.baud
    machine.z_max_feed = profile.z_max_feed
    machine.z_acceleration = profile.z_acceleration
    machine.z_acceleration_target = profile.z_acceleration_target
    machine.junction_deviation = profile.junction_deviation
    machine.travel_feed = profile.travel_feed
    machine.draw_feed = profile.draw_feed
    machine.acceleration = profile.acceleration
    machine.travel_acceleration = profile.travel_acceleration
    machine.park_x = min(profile.park_x, profile.bed_x)
    machine.park_y = min(profile.park_y, profile.bed_y)
    if pen_setup is not None:
        # a pen change in the middle of the front edge is reachable on every bed
        pen_setup.change_x = round(profile.bed_x / 2.0, 1)
        pen_setup.change_y = round(max(profile.bed_y - 20.0, 0.0), 1)
        pen_setup.change_z = round(min(pen_setup.change_z, max(profile.max_z - 5.0, 5.0)), 1)


def firmware_of(machine) -> Firmware:
    return FIRMWARE_RULES.get(getattr(machine, "firmware", "marlin"), FIRMWARE_RULES["marlin"])
