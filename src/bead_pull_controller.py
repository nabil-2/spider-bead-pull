"""Bead-position control for the MADMAX spider-booster bead-pull measurement.

One dielectric bead is glued onto a single thread strung back and forth through
the booster, so the thread passes through it at several axial (z) positions.
Each pass is a *sub-thread*.  The thread is wound on a wheel driven by one
stepper motor, so the motor's absolute step count fixes the bead's position; the
controller turns a (sub-thread, position) request into an absolute step target.

Pieces:

* ``Calibration`` / ``SubThreadCalibration`` -- the persisted anchors.  Each
  sub-thread stores the motor step count *and* the 3-D booster position at its
  *start* (the position zero) and its *end*; direction, scan length and the
  step<->metre conversion all follow from those numbers.  The conversion is
  therefore computed *per sub-thread* (steps span / Euclidean distance between the
  two 3-D endpoints) -- there is no single global ``steps_per_meter``.
* ``Stepper`` / ``SimulatedMotor`` -- the motor.  Any object with ``home()``,
  ``get_position()``, ``move_by(delta_steps)`` and ``shutdown()`` works, so you
  can drop in your own.  ``Stepper`` drives the L6470 ASCII controller over
  serial (see ``stepper motor instruction set.pdf``).
* ``BeadPullController`` / ``run_scan`` -- walk the bead through every sub-thread
  in fixed steps and measure at each point.  A *measurement* is any object with a
  ``measure(point)`` method (and optional ``setup()`` / ``teardown()``).
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

CALIBRATION_SCHEMA_VERSION = 2
DEFAULT_CALIBRATION_PATH = Path("config/bead_pull_calibration.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_int(reply: Sequence[str]) -> int:
    """Pull the last integer token out of a controller reply (e.g. ``getpos``)."""
    last: int | None = None
    for line in reply:
        for token in str(line).replace(",", " ").split():
            try:
                last = int(token)
            except ValueError:
                continue
    if last is None:
        raise ValueError(f"Could not parse an integer position from reply: {reply!r}")
    return last


# The physical direction the motor turns for a *positive* raw ``rotate`` command
# is fixed by the wiring; by convention here a positive raw rotation is
# clockwise (as seen from the motor shaft).  ``unwind_direction`` in the config
# names the physical rotation that lets thread *out* and advances the bead, so
# the controller can always drive that sense regardless of how the winding wheel
# is mounted.  See :meth:`Stepper.direction_sign`.
_CLOCKWISE_ALIASES = {"clockwise", "cw"}
_COUNTERCLOCKWISE_ALIASES = {"counterclockwise", "anticlockwise", "anti-clockwise", "ccw"}


def unwind_direction_sign(unwind_direction: str | int) -> int:
    """Map an ``unwind_direction`` setting to the raw-rotation sign (+1/-1) that
    unwinds the thread.

    Accepts ``"clockwise"``/``"cw"`` (-> +1) or
    ``"counterclockwise"``/``"anticlockwise"``/``"ccw"`` (-> -1); ``+1``/``-1``
    are passed through.  The convention is that a *positive* raw ``rotate``
    command turns the motor clockwise, so a clockwise unwind needs +1 and a
    counter-clockwise unwind needs -1.
    """
    if isinstance(unwind_direction, (int, float)) and not isinstance(unwind_direction, bool):
        sign = int(unwind_direction)
        if sign in (1, -1):
            return sign
        raise ValueError(f"unwind_direction sign must be +1 or -1, got {unwind_direction!r}")
    key = str(unwind_direction).strip().lower()
    if key in _CLOCKWISE_ALIASES:
        return 1
    if key in _COUNTERCLOCKWISE_ALIASES:
        return -1
    raise ValueError(
        f"unwind_direction must be 'clockwise' or 'counterclockwise' "
        f"(or +1/-1), got {unwind_direction!r}"
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
Xyz = tuple[float, float, float]


def _as_xyz(value: Any, which: str) -> Xyz:
    """Parse a required (x, y, z) triple; raise if it is missing or malformed.

    The 3-D endpoints are no longer optional: the step<->metre conversion of a
    sub-thread is derived from the distance between them, so both are mandatory.
    """
    if value is None:
        raise ValueError(f"{which}_xyz_m is required (used to derive the step<->metre scale)")
    xyz = tuple(float(c) for c in value)
    if len(xyz) != 3:
        raise ValueError(f"expected an (x, y, z) triple for {which}_xyz_m, got {value!r}")
    return xyz


@dataclass(frozen=True)
class SubThreadCalibration:
    """One pass of the thread through the booster.

    Each pass is anchored at both ends by two measured quantities: the motor step
    count (``start_steps`` at the bead-position zero, ``end_steps`` at the far end)
    and the 3-D booster position (``start_xyz_m`` / ``end_xyz_m``, the (x, y, z) of
    those same two points in the booster coordinate system, metres).  Everything
    else is derived from these four numbers:

    * ``direction`` and ``length_steps`` from the step anchors,
    * ``length_m`` (the physical length of the pass) from the distance between the
      two 3-D endpoints, and
    * ``steps_per_meter``, the step<->metre conversion *for this pass*, from the two
      together (step span / physical length).  There is no global conversion.

    ``margin_start_m`` and ``margin_end_m`` are non-measurement margins (metres):
    the bead still travels through them, but no measurement is taken within
    ``margin_start_m`` of the start or ``margin_end_m`` of the end.
    """

    index: int
    start_steps: int
    end_steps: int
    start_xyz_m: Xyz
    end_xyz_m: Xyz
    name: str | None = None
    margin_start_m: float = 0.0
    margin_end_m: float = 0.0

    @property
    def direction(self) -> int:
        """+1 if moving into the booster increases the step count, else -1."""
        return 1 if self.end_steps >= self.start_steps else -1

    @property
    def length_steps(self) -> int:
        return abs(self.end_steps - self.start_steps)

    @property
    def length_m(self) -> float:
        """Physical length of the pass: the Euclidean distance between its two 3-D
        endpoints, in metres (the pass is modelled as the straight, taut segment
        between them)."""
        return math.dist(self.start_xyz_m, self.end_xyz_m)

    @property
    def steps_per_meter(self) -> float:
        """Step<->metre conversion for *this* sub-thread, derived from its own
        anchors: the motor step span divided by the physical length between the two
        3-D endpoints (``steps = metres * steps_per_meter``)."""
        return self.length_steps / self.length_m

    @property
    def label(self) -> str:
        return self.name if self.name else f"sub_thread_{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "margin_start_m": self.margin_start_m,
            "margin_end_m": self.margin_end_m,
            "start_xyz_m": list(self.start_xyz_m),
            "end_xyz_m": list(self.end_xyz_m),
            "start_steps": self.start_steps,
            "end_steps": self.end_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubThreadCalibration":
        return cls(
            index=int(data["index"]),
            start_steps=int(data["start_steps"]),
            end_steps=int(data["end_steps"]),
            start_xyz_m=_as_xyz(data.get("start_xyz_m"), "start"),
            end_xyz_m=_as_xyz(data.get("end_xyz_m"), "end"),
            name=data.get("name"),
            margin_start_m=float(data.get("margin_start_m", 0.0)),
            margin_end_m=float(data.get("margin_end_m", 0.0)),
        )


@dataclass
class Calibration:
    """The measured sub-thread anchors.

    The calibration file holds, per sub-thread, the ``index``/``name``, the
    measured ``start_steps``/``end_steps`` and the ``start_xyz_m``/``end_xyz_m``
    booster positions (all established by the calibration notebook).  The
    step<->metre conversion is *not* a single global number: each sub-thread
    carries its own, derived from its step span and the distance between its 3-D
    endpoints (see :attr:`SubThreadCalibration.steps_per_meter`).
    """

    sub_threads: list[SubThreadCalibration]
    created_utc: str = field(default_factory=_utcnow)
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    @property
    def n_sub_threads(self) -> int:
        return len(self.sub_threads)

    def indices(self) -> list[int]:
        return [s.index for s in self.sub_threads]

    def get(self, index: int) -> SubThreadCalibration:
        for sub in self.sub_threads:
            if sub.index == index:
                return sub
        raise KeyError(f"No sub-thread with index {index}; have {self.indices()}")

    def length_m(self, index: int) -> float:
        return self.get(index).length_m

    def steps_for_position(self, index: int, x_m: float) -> int:
        """Absolute motor step count for bead position ``x_m`` (metres from zero)."""
        sub = self.get(index)
        return sub.start_steps + sub.direction * int(round(x_m * sub.steps_per_meter))

    def position_for_steps(self, index: int, steps: int) -> float:
        """Bead position (metres from zero) for an absolute step count."""
        sub = self.get(index)
        return (steps - sub.start_steps) * sub.direction / sub.steps_per_meter

    def booster_xyz(self, index: int, position_m: float) -> "Xyz":
        """3-D booster-coordinate position (metres) of a bead at ``position_m``
        along a sub-thread.

        The sub-thread is the straight line from ``start_xyz_m`` (at position 0)
        to ``end_xyz_m`` (at the calibrated length); the 1-D position is mapped
        onto it by linear interpolation.
        """
        sub = self.get(index)
        length_m = sub.length_m
        frac = 0.0 if length_m == 0 else position_m / length_m
        s, e = sub.start_xyz_m, sub.end_xyz_m
        return (
            s[0] + frac * (e[0] - s[0]),
            s[1] + frac * (e[1] - s[1]),
            s[2] + frac * (e[2] - s[2]),
        )

    def scan_points_m(
        self,
        index: int,
        step_size_m: float,
        length_m: float | None = None,
        include_endpoint: bool = True,
    ) -> np.ndarray:
        """Bead positions (metres from zero) to measure within a sub-thread:
        ``margin_start_m``, +step, +2*step, ... up to ``length - margin_end_m``
        (calibrated length unless overridden).  The non-measurement margins are
        skipped at both ends."""
        if step_size_m <= 0:
            raise ValueError("step_size_m must be positive")
        sub = self.get(index)
        span = self.length_m(index) if length_m is None else float(length_m)
        if span < 0:
            raise ValueError("length_m must be non-negative")
        lo = sub.margin_start_m
        hi = span - sub.margin_end_m
        if hi < lo:
            raise ValueError(
                f"Sub-thread {sub.label}: margins "
                f"({sub.margin_start_m} + {sub.margin_end_m} m) leave no room in the "
                f"{span:.4f} m scan."
            )
        eps = step_size_m * 1e-9
        # round away arange's float noise (sub-nm precision is irrelevant here)
        points = list(np.round(np.arange(lo, hi + eps, step_size_m), 9))
        if include_endpoint and (not points or abs(points[-1] - hi) > eps):
            points.append(round(hi, 9))
        return np.asarray(points, dtype=float)

    def validate(self) -> None:
        seen: set[int] = set()
        for sub in self.sub_threads:
            if sub.index in seen:
                raise ValueError(f"Duplicate sub-thread index {sub.index}")
            seen.add(sub.index)
            if sub.length_steps == 0:
                raise ValueError(
                    f"Sub-thread {sub.label} has start_steps == end_steps "
                    "(zero step span); re-calibrate its end point."
                )
            if sub.margin_start_m < 0 or sub.margin_end_m < 0:
                raise ValueError(f"Sub-thread {sub.label} has a negative margin.")
            for xyz, which in ((sub.start_xyz_m, "start"), (sub.end_xyz_m, "end")):
                if len(xyz) != 3:
                    raise ValueError(
                        f"Sub-thread {sub.label}: {which}_xyz_m must have 3 components."
                    )
            if sub.length_m == 0:
                raise ValueError(
                    f"Sub-thread {sub.label} has coincident start_xyz_m and end_xyz_m "
                    "(zero physical length); the step<->metre scale is undefined."
                )

    def to_dict(self) -> dict[str, Any]:
        # Calibration file = measured data only (step anchors) plus provenance.
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "sub_threads": [s.to_dict() for s in self.sub_threads],
        }

    def save(self, path: str | Path = DEFAULT_CALIBRATION_PATH) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_calibration_file(cls, path: str | Path) -> "Calibration":
        """Build a calibration from the sub-thread anchors in ``path`` (index,
        name, start/end steps and start/end 3-D positions).  The step<->metre
        conversion is derived per sub-thread from those anchors, so nothing else is
        needed."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        subs = sorted(
            (SubThreadCalibration.from_dict(entry) for entry in data["sub_threads"]),
            key=lambda s: s.index,
        )
        cal = cls(
            sub_threads=subs,
            created_utc=data.get("created_utc", _utcnow()),
            schema_version=int(data.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
        )
        cal.validate()
        return cal

    def summary(self) -> str:
        lines = [
            f"Calibration ({self.n_sub_threads} sub-threads; "
            "steps_per_meter derived per sub-thread)",
            f"{'idx':>3}  {'name':<12} {'start':>9} {'end':>9} "
            f"{'dir':>3} {'len_m':>8} {'steps/m':>10} {'mrg_s':>7} {'mrg_e':>7}",
        ]
        for sub in sorted(self.sub_threads, key=lambda s: s.index):
            lines.append(
                f"{sub.index:>3}  {sub.label:<12} {sub.start_steps:>9} "
                f"{sub.end_steps:>9} {sub.direction:>3} {sub.length_m:>8.4f} "
                f"{sub.steps_per_meter:>10.6g} "
                f"{sub.margin_start_m:>7.4f} {sub.margin_end_m:>7.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Motors (any object with home/get_position/move_by/shutdown works)
# ---------------------------------------------------------------------------
class SimulatedMotor:
    """In-memory motor for dry runs and tests; tracks an integer step counter."""

    def __init__(self, start_position: int = 0, verbose: bool = False) -> None:
        self._position = int(start_position)
        self.verbose = verbose

    def home(self) -> None:
        self._position = 0
        if self.verbose:
            print("[SimulatedMotor] home -> 0")

    def get_position(self) -> int:
        return self._position

    def move_by(self, delta_steps: int) -> None:
        self._position += int(delta_steps)
        if self.verbose:
            print(f"[SimulatedMotor] move_by {int(delta_steps):+d} -> {self._position}")

    def shutdown(self) -> None:
        pass


class Stepper:
    """Stepper motor on the L6470 ASCII controller over serial.

    Commands are newline-terminated (see ``stepper motor instruction set.pdf``).
    Exposes the high-level methods the controller needs -- ``home()``,
    ``get_position()``, ``move_by()``, ``shutdown()`` -- plus the raw command set.
    ``pyserial`` is imported lazily so the rest of the module works without it.
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0,
                 motor: int = 1, reset_delay: float = 2.0,
                 unwind_direction: str | int = "clockwise") -> None:
        import serial  # lazy: only needed for real hardware

        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.motor = motor
        # +1/-1 mapping a logical "advance the bead / unwind" move onto the raw
        # rotation sense; the high-level interface (move_by/get_position) works in
        # the logical frame so the calibration step anchors are unaffected.
        self.direction_sign = unwind_direction_sign(unwind_direction)
        time.sleep(reset_delay)  # USB-serial bridges often reset on open
        self.ser.reset_input_buffer()

    @classmethod
    def open(cls, port: str, baud: int = 115200, motor: int = 1, timeout: float = 1.0,
             microsteps: int = 8, acceleration: int = 1000, speed: int = 500,
             driving_voltage: float = 6.8, holding_voltage: float = 2.0,
             configure: bool = True, unwind_direction: str | int = "clockwise") -> "Stepper":
        dev = cls(port, baud=baud, timeout=timeout, motor=motor,
                  unwind_direction=unwind_direction)
        if configure:
            dev.reset()
            dev.setprofile(microsteps, acceleration, speed)
            dev.setvoltage(driving_voltage, holding_voltage)
        return dev

    # -- raw I/O / command set --------------------------------------------
    def cmd(self, cmd: str, expect_reply: bool = False) -> list[str]:
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()
        if not expect_reply:
            time.sleep(0.05)  # let any echo arrive, then drain
        lines: list[str] = []
        deadline = time.time() + (self.ser.timeout or 1.0)
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                break
            lines.append(raw.decode("ascii", errors="replace").strip())
        return lines

    def reset(self):                 return self.cmd("reset")
    def gotoswitch(self):            return self.cmd(f"gotoswitch {self.motor}")
    def wait(self):                  return self.cmd(f"wait {self.motor}")
    def rotate(self, steps):         return self.cmd(f"rotate {self.motor} {int(steps)}")
    def getpos(self):                return self.cmd(f"getpos {self.motor}", expect_reply=True)
    def clearpos(self):              return self.cmd(f"clearpos {self.motor}")
    def hiz(self):                   return self.cmd(f"hiz {self.motor}")
    def hardstop(self):             return self.cmd(f"hardstop {self.motor}")
    def setprofile(self, microsteps=8, acceleration=1000, speed=500):
        return self.cmd(f"setprofile {self.motor} {int(microsteps)} {int(acceleration)} {int(speed)}")
    def setvoltage(self, driving=6.8, holding=2.0):
        return self.cmd(f"setvoltage {self.motor} {driving} {holding}")

    # -- high-level interface ---------------------------------------------
    def home(self) -> None:
        self.gotoswitch()
        self.wait()
        self.clearpos()

    def get_position(self) -> int:
        # controller register is in the raw-rotation frame; report it in the
        # logical frame so it round-trips with ``move_by``.
        return self.direction_sign * _parse_int(self.getpos())

    def move_by(self, delta_steps: int) -> None:
        if int(delta_steps) == 0:
            return
        # translate a logical (unwind-positive) step delta into the raw rotation
        # sense the configured unwind direction demands.
        self.rotate(self.direction_sign * int(delta_steps))
        self.wait()

    def shutdown(self) -> None:
        try:
            self.hiz()
        finally:
            if self.ser.is_open:
                self.ser.close()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
@dataclass
class ScanPoint:
    """One bead position the controller stops at and measures.

    ``position_m`` is the *true* bead position in metres along the sub-thread,
    obtained by converting the motor's actual step count back to metres -- i.e.
    the position really reached (on the step lattice), not the requested grid
    value.  ``position_xyz_m`` is that same point in the 3-D booster coordinate
    system, obtained by interpolating between the sub-thread's calibrated
    endpoints.
    """

    sub_thread_index: int
    sub_thread_name: str | None
    point_index: int
    n_points: int
    position_m: float
    position_xyz_m: "Xyz"
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_thread_index": self.sub_thread_index,
            "sub_thread_name": self.sub_thread_name,
            "point_index": self.point_index,
            "n_points": self.n_points,
            "position_m": self.position_m,
            "position_xyz_m": list(self.position_xyz_m),
            "timestamp_utc": self.timestamp_utc,
        }


def scan_targets(
    calibration: Calibration,
    step_size_m: float,
    sub_thread_indices: Sequence[int] | None = None,
    lengths_m: dict[int, float] | None = None,
    include_endpoint: bool = True,
) -> Iterator[tuple[SubThreadCalibration, int, int, float, int]]:
    """Yield ``(sub, point_index, n_points, x_m, target_steps)`` for every planned
    bead position, without moving anything."""
    indices = (
        list(sub_thread_indices)
        if sub_thread_indices is not None
        else sorted(calibration.indices())
    )
    lengths_m = lengths_m or {}
    for index in indices:
        sub = calibration.get(index)
        points = calibration.scan_points_m(
            index, step_size_m, lengths_m.get(index), include_endpoint
        )
        for i, x_m in enumerate(points):
            yield sub, i, len(points), float(x_m), calibration.steps_for_position(index, float(x_m))


class BeadPullController:
    """Drives the bead to calibrated positions on a motor."""

    def __init__(self, motor, calibration: Calibration, settle_s: float = 0.0,
                 position_tolerance_steps: int | None = 5, logger=print) -> None:
        calibration.validate()
        self.motor = motor
        self.calibration = calibration
        self.settle_s = settle_s
        self.position_tolerance_steps = position_tolerance_steps
        self.log = logger

    def home(self) -> None:
        self.log("Homing to limit switch ...")
        self.motor.home()
        self.log(f"Homed; position = {self.motor.get_position()} steps")

    def move_to(self, sub_thread_index: int, x_m: float) -> int:
        """Move the bead to ``x_m`` metres from a sub-thread's zero; return the
        actual motor position afterwards."""
        target = self.calibration.steps_for_position(sub_thread_index, x_m)
        self.motor.move_by(target - self.motor.get_position())
        if self.settle_s:
            time.sleep(self.settle_s)
        actual = self.motor.get_position()
        tol = self.position_tolerance_steps
        if tol is not None and abs(actual - target) > tol:
            self.log(
                f"WARNING: sub-thread {sub_thread_index} x={x_m:.4f} m: "
                f"target {target} steps but motor at {actual} (off by {actual - target})"
            )
        return actual

    def iter_points(
        self,
        step_size_m: float,
        sub_thread_indices: Sequence[int] | None = None,
        lengths_m: dict[int, float] | None = None,
        include_endpoint: bool = True,
    ) -> Iterator[ScanPoint]:
        """Walk every requested sub-thread point by point, yielding a
        :class:`ScanPoint` once the bead is parked at each position."""
        for sub, i, n, x_m, _target_steps in scan_targets(
            self.calibration, step_size_m, sub_thread_indices, lengths_m, include_endpoint
        ):
            if i == 0:
                self.log(f"Sub-thread {sub.index} ({sub.label}): {n} points")
            actual_steps = self.move_to(sub.index, x_m)
            # record the true position actually reached (steps -> metres), and the
            # same point mapped into the 3-D booster coordinate system
            position_m = self.calibration.position_for_steps(sub.index, actual_steps)
            position_xyz_m = self.calibration.booster_xyz(sub.index, position_m)
            yield ScanPoint(sub.index, sub.name, i, n, position_m, position_xyz_m, _utcnow())


_LOG_COLUMNS = [
    "sub_thread_index", "sub_thread_name", "point_index", "n_points",
    "position_m", "position_x_m", "position_y_m", "position_z_m",
    "timestamp_utc", "measurement_json",
]


def run_scan(
    controller: BeadPullController,
    measurement,
    step_size_m: float,
    output_dir: str | Path,
    sub_thread_indices: Sequence[int] | None = None,
    lengths_m: dict[int, float] | None = None,
    include_endpoint: bool = True,
    home_first: bool = True,
    metadata: dict[str, Any] | None = None,
    progress: bool = True,
) -> Path:
    """Walk every requested sub-thread in ``step_size_m`` increments, calling
    ``measurement.measure(point)`` at each parked position.  Writes
    ``scan_log.csv``, ``scan_log.jsonl`` and ``scan_manifest.json`` into
    ``output_dir`` and returns it.

    ``measurement`` is any object with ``measure(point) -> dict | None`` and
    optional ``setup()`` / ``teardown()``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = sum(1 for _ in scan_targets(
        controller.calibration, step_size_m, sub_thread_indices, lengths_m, include_endpoint))
    manifest = {
        "started_utc": _utcnow(),
        "step_size_m": step_size_m,
        "include_endpoint": include_endpoint,
        "home_first": home_first,
        "n_points_planned": n_total,
        "sub_thread_indices": (
            list(sub_thread_indices) if sub_thread_indices is not None
            else sorted(controller.calibration.indices())
        ),
        "lengths_m_override": lengths_m or {},
        "calibration": {
            **controller.calibration.to_dict(),
            # step<->metre scale is derived per sub-thread (step span / 3-D length),
            # recorded here so the run is a self-contained record.
            "derived": [
                {
                    "index": sub.index,
                    "length_m": sub.length_m,
                    "steps_per_meter": sub.steps_per_meter,
                }
                for sub in controller.calibration.sub_threads
            ],
        },
        "metadata": metadata or {},
    }
    manifest_path = output_dir / "scan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    points = controller.iter_points(
        step_size_m, sub_thread_indices, lengths_m, include_endpoint)
    if progress:
        try:
            from tqdm import tqdm
            points = tqdm(points, total=n_total, desc="bead-pull scan", unit="pt")
        except ImportError:
            pass

    setup = getattr(measurement, "setup", None)
    teardown = getattr(measurement, "teardown", None)
    csv_path = output_dir / "scan_log.csv"
    jsonl_path = output_dir / "scan_log.jsonl"

    if callable(setup):
        setup()
    try:
        if home_first:
            controller.home()
        with csv_path.open("w", newline="", encoding="utf-8") as csv_fh, \
                jsonl_path.open("w", encoding="utf-8") as jsonl_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=_LOG_COLUMNS)
            writer.writeheader()
            for point in points:
                result = measurement.measure(point)
                record = point.to_dict()
                # jsonl keeps the xyz triple as a list next to the full result
                jsonl_fh.write(json.dumps({**record, "measurement": result}) + "\n")
                jsonl_fh.flush()
                # csv flattens the xyz triple into three scalar columns
                x, y, z = record.pop("position_xyz_m") or (None, None, None)
                writer.writerow({
                    **record,
                    "position_x_m": x, "position_y_m": y, "position_z_m": z,
                    "measurement_json": json.dumps(result),
                })
                csv_fh.flush()
    finally:
        if callable(teardown):
            teardown()

    manifest["finished_utc"] = _utcnow()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


__all__ = [
    "Calibration",
    "SubThreadCalibration",
    "SimulatedMotor",
    "Stepper",
    "unwind_direction_sign",
    "ScanPoint",
    "scan_targets",
    "BeadPullController",
    "run_scan",
]
