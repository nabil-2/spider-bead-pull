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
import os
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


def _opt_int(value: Any) -> int | None:
    """Coerce a value to ``int``, returning ``None`` for missing/unparsable input."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Persistent motor state (position + home), shared across programs
# ---------------------------------------------------------------------------
# This rig has no limit switch, so the absolute motor position cannot be
# recovered from the hardware after a power cycle (the L6470 step register resets
# to zero).  Instead the *logical* motor position and an operator-chosen *home*
# position are persisted to a small JSON file at a fixed absolute path in the
# user's home directory, so every program on the same computer (this package, the
# calibration notebook, any other script) shares one motor position and one home
# reference.  Both are stored in the logical (unwind-positive) frame -- the same
# frame ``get_position``/``move_by`` and the calibration step anchors use.
# Override the location with the ``MADMAX_BEAD_PULL_STATE`` environment variable.
MOTOR_STATE_ENV_VAR = "MADMAX_BEAD_PULL_STATE"
DEFAULT_MOTOR_STATE_PATH = Path.home() / ".madmax_bead_pull" / "motor_state.json"


def motor_state_path(path: str | Path | None = None) -> Path:
    """Resolve the motor-state file location.

    Precedence: explicit ``path`` argument, then the ``MADMAX_BEAD_PULL_STATE``
    environment variable, then the default
    ``~/.madmax_bead_pull/motor_state.json``.
    """
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(MOTOR_STATE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_MOTOR_STATE_PATH


@dataclass
class MotorState:
    """The persisted motor position and home position (logical frame, steps).

    Backed by a JSON file at an absolute path (see :func:`motor_state_path`) so it
    is shared by every program driving the motor on this computer.
    ``position_steps`` is the last known motor position; ``home_steps`` is the
    operator-set home reference the motor returns to on :meth:`Stepper.home`.
    Either may be ``None`` until it has first been established.  Writes are atomic
    (temp file + ``os.replace``) so a crash mid-write cannot corrupt the file.
    """

    path: Path
    position_steps: int | None = None
    home_steps: int | None = None
    updated_utc: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MotorState":
        """Load the state file, or return an empty state if it is absent/corrupt."""
        resolved = motor_state_path(path)
        state = cls(path=resolved)
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return state  # missing / unreadable / malformed -> establish it later
        if isinstance(data, dict):
            state.position_steps = _opt_int(data.get("position_steps"))
            state.home_steps = _opt_int(data.get("home_steps"))
            updated = data.get("updated_utc")
            state.updated_utc = str(updated) if updated is not None else None
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_steps": self.position_steps,
            "home_steps": self.home_steps,
            "updated_utc": self.updated_utc,
        }

    def _save(self) -> None:
        self.updated_utc = _utcnow()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on the same filesystem

    def set_position(self, steps: int) -> int:
        """Persist ``steps`` as the current motor position (no write if unchanged)."""
        steps = int(steps)
        if steps != self.position_steps:
            self.position_steps = steps
            self._save()
        return steps

    def set_home(self, steps: int) -> int:
        """Persist ``steps`` as the home position (no write if unchanged)."""
        steps = int(steps)
        if steps != self.home_steps:
            self.home_steps = steps
            self._save()
        return steps


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
    """In-memory motor for dry runs and tests; tracks an integer step counter.

    Mirrors the :class:`Stepper` interface, including the settable *home* position
    (``home_position`` defaults to ``0`` so a plain ``home()`` returns to zero).
    It keeps its state in memory only -- it does not touch the shared state file.
    """

    def __init__(self, start_position: int = 0, home_position: int | None = 0,
                 verbose: bool = False) -> None:
        self._position = int(start_position)
        self._home = int(home_position) if home_position is not None else None
        self.verbose = verbose

    def home(self) -> None:
        if self._home is None:
            raise RuntimeError("No home position set; call set_home() first.")
        self.move_by(self._home - self._position)
        if self.verbose:
            print(f"[SimulatedMotor] home -> {self._position}")

    def get_position(self) -> int:
        return self._position

    def move_by(self, delta_steps: int) -> None:
        self._position += int(delta_steps)
        if self.verbose:
            print(f"[SimulatedMotor] move_by {int(delta_steps):+d} -> {self._position}")

    def set_home(self, position: int | None = None) -> int:
        """Record a home position (default: the current position)."""
        self._home = int(position) if position is not None else self._position
        if self.verbose:
            print(f"[SimulatedMotor] set_home -> {self._home}")
        return self._home

    @property
    def home_position(self) -> int | None:
        return self._home

    def set_position(self, steps: int) -> int:
        """Override the tracked position (the simulated analogue of ``setpos``)."""
        self._position = int(steps)
        if self.verbose:
            print(f"[SimulatedMotor] set_position -> {self._position}")
        return self._position

    def shutdown(self) -> None:
        pass


class Stepper:
    """Stepper motor on the L6470 ASCII controller over serial.

    Commands are newline-terminated (see ``stepper motor instruction set.pdf``).
    Exposes the high-level methods the controller needs -- ``home()``,
    ``get_position()``, ``move_by()``, ``shutdown()`` -- plus the raw command set.
    ``pyserial`` is imported lazily so the rest of the module works without it.
    """

    # Refuse any single move larger than this many steps unless overridden; a
    # cheap circuit-breaker against a runaway from a bad target (the physical
    # limit switch that would have stopped one is not present on this rig).
    DEFAULT_MAX_STEP_MAGNITUDE = 500_000

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0,
                 motor: int = 1, reset_delay: float = 2.0,
                 unwind_direction: str | int = "clockwise",
                 state_path: str | Path | None = None,
                 max_step_magnitude: int | None = DEFAULT_MAX_STEP_MAGNITUDE) -> None:
        import serial  # lazy: only needed for real hardware

        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.motor = motor
        # +1/-1 mapping a logical "advance the bead / unwind" move onto the raw
        # rotation sense; the high-level interface (move_by/get_position) works in
        # the logical frame so the calibration step anchors are unaffected.
        self.direction_sign = unwind_direction_sign(unwind_direction)
        self.max_step_magnitude = (
            int(max_step_magnitude) if max_step_magnitude is not None else None
        )
        # Persisted position + home, shared with every other program on this
        # computer.  Loaded now; the controller register is reconciled to it in
        # restore_position() once the port is up and any reset has happened.
        self.state = MotorState.load(state_path)
        time.sleep(reset_delay)  # USB-serial bridges often reset on open
        self.ser.reset_input_buffer()

    @classmethod
    def open(cls, port: str, baud: int = 115200, motor: int = 1, timeout: float = 1.0,
             microsteps: int = 8, acceleration: int = 1000, speed: int = 500,
             driving_voltage: float = 6.8, holding_voltage: float = 2.0,
             configure: bool = True, unwind_direction: str | int = "clockwise",
             state_path: str | Path | None = None,
             max_step_magnitude: int | None = DEFAULT_MAX_STEP_MAGNITUDE) -> "Stepper":
        dev = cls(port, baud=baud, timeout=timeout, motor=motor,
                  unwind_direction=unwind_direction, state_path=state_path,
                  max_step_magnitude=max_step_magnitude)
        if configure:
            dev.reset()
            dev.setprofile(microsteps, acceleration, speed)
            dev.setvoltage(driving_voltage, holding_voltage)
        # After any reset (which zeroes the register), put the controller's step
        # count back in sync with the last saved position.
        dev.restore_position()
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
    def gotoswitch(self):
        # WARNING: rotates until a limit switch triggers.  This rig has no switch,
        # so this never stops on its own -- do NOT use it for homing (that was the
        # original runaway bug).  home() uses a saved position instead.
        return self.cmd(f"gotoswitch {self.motor}")
    def wait(self):                  return self.cmd(f"wait {self.motor}")
    def rotate(self, steps):         return self.cmd(f"rotate {self.motor} {int(steps)}")
    def getpos(self):                return self.cmd(f"getpos {self.motor}", expect_reply=True)
    def setpos(self, step):          return self.cmd(f"setpos {self.motor} {int(step)}")
    def clearpos(self):              return self.cmd(f"clearpos {self.motor}")
    def hiz(self):                   return self.cmd(f"hiz {self.motor}")
    def hardstop(self):             return self.cmd(f"hardstop {self.motor}")
    def setprofile(self, microsteps=8, acceleration=1000, speed=500):
        return self.cmd(f"setprofile {self.motor} {int(microsteps)} {int(acceleration)} {int(speed)}")
    def setvoltage(self, driving=6.8, holding=2.0):
        return self.cmd(f"setvoltage {self.motor} {driving} {holding}")

    # -- high-level interface ---------------------------------------------
    def restore_position(self) -> None:
        """Reconcile the controller's step register with the persisted position.

        With no limit switch the register is meaningless after a power cycle (it
        resets to zero while the bead stays put), so on connect we write the last
        saved *logical* position back into it with ``setpos``.  That keeps the
        absolute step frame -- and therefore the calibration anchors -- stable
        across power cycles and across programs.  If no position has ever been
        saved, adopt whatever the controller currently reports and persist it as
        the baseline.
        """
        saved = self.state.position_steps
        if saved is not None:
            try:
                self.setpos(self.direction_sign * int(saved))
            except Exception as exc:  # pragma: no cover - hardware quirk
                print(f"WARNING: could not restore saved position via setpos: {exc}")
            return
        raw = self._read_raw_position()
        if raw is not None:
            self.state.set_position(self.direction_sign * raw)

    def _read_raw_position(self) -> int | None:
        """Controller step register (raw frame), or ``None`` if it gave no reply.

        Reading the position must never crash a session: after power is cut or a
        cable is pulled, ``getpos`` returns an empty reply (the original crash).
        """
        try:
            return _parse_int(self.getpos())
        except ValueError:
            return None

    def _persist_logical(self, raw: int) -> int:
        """Convert a raw register reading to the logical frame and persist it."""
        return self.state.set_position(self.direction_sign * raw)

    def get_position(self) -> int:
        """Current motor position in the logical frame.

        The controller register is authoritative when it answers (reported in the
        logical frame so it round-trips with ``move_by``, and persisted).  If the
        controller is silent -- power removed, cable pulled -- fall back to the
        last saved position with a warning instead of raising.
        """
        raw = self._read_raw_position()
        if raw is not None:
            return self._persist_logical(raw)
        tracked = self.state.position_steps
        if tracked is None:
            raise RuntimeError(
                "Cannot determine the motor position: the controller returned no "
                "reply and no position has ever been saved. Check the serial "
                "connection and motor power, then set the position with "
                "set_position()."
            )
        print(
            "WARNING: controller returned no position; using the last saved value "
            f"({tracked} steps). Check the serial connection and motor power."
        )
        return tracked

    def _guard_move(self, delta: int) -> None:
        cap = self.max_step_magnitude
        if cap is not None and abs(delta) > cap:
            raise ValueError(
                f"Refusing to move {delta:+d} steps: exceeds the safety limit of "
                f"{cap} steps (max_step_magnitude). If this move really is "
                "intended, raise max_step_magnitude when opening the motor."
            )

    def move_by(self, delta_steps: int) -> None:
        delta = int(delta_steps)
        if delta == 0:
            return
        self._guard_move(delta)
        # translate a logical (unwind-positive) step delta into the raw rotation
        # sense the configured unwind direction demands.
        self.rotate(self.direction_sign * delta)
        self.wait()
        # Keep the persisted position current: prefer the controller's own count,
        # but if it is silent advance the saved position by the commanded delta so
        # we never lose track of where the bead is.
        raw = self._read_raw_position()
        if raw is not None:
            self._persist_logical(raw)
        else:
            base = self.state.position_steps
            if base is not None:
                self.state.set_position(base + delta)
            print(
                "WARNING: controller returned no position after the move; advanced "
                f"the saved position by {delta:+d} steps instead."
            )

    def home(self) -> None:
        """Drive the bead to the saved home position.

        This rig has no limit switch, so 'home' is an operator-set position (see
        :meth:`set_home`), not a physical switch -- this is an ordinary bounded
        move to that step count, never the old runaway ``gotoswitch``.
        """
        target = self.state.home_steps
        if target is None:
            raise RuntimeError(
                "No home position has been set. Set one with set_home() (e.g. in "
                "the calibration notebook) before calling home()."
            )
        self.move_by(int(target) - self.get_position())

    def set_home(self, position: int | None = None) -> int:
        """Record a home position (default: the current position) and persist it.

        Saved to the shared state file so every program on this computer homes to
        the same place.  Returns the home step count.
        """
        pos = int(position) if position is not None else self.get_position()
        return self.state.set_home(pos)

    @property
    def home_position(self) -> int | None:
        """The saved home position (logical steps), or ``None`` if unset."""
        return self.state.home_steps

    def set_position(self, steps: int) -> int:
        """Teach the controller its current position (raw ``setpos`` + persist).

        Use this to re-establish the absolute frame after physically placing the
        bead at a known reference -- the manual stand-in for the missing limit
        switch.  Returns the position in the logical frame.
        """
        steps = int(steps)
        self.setpos(self.direction_sign * steps)
        return self.state.set_position(steps)

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
        self.log("Homing to the saved home position ...")
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
    "MotorState",
    "motor_state_path",
    "SimulatedMotor",
    "Stepper",
    "unwind_direction_sign",
    "ScanPoint",
    "scan_targets",
    "BeadPullController",
    "run_scan",
]
