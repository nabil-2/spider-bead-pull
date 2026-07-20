#!/usr/bin/env python
"""Run a bead-pull measurement: walk the bead through every calibrated
sub-thread and read a VNA S-parameter at each step.

All measurement settings (stepper motor, VNA, scan) live in a single JSON config
file (default ``config/measurement_config.json``); the *calibration* lives in its
own file referenced from there.  The command line only selects the run mode.

Typical use
-----------
Dry run (no hardware, prints the planned positions only)::

    python run_bead_pull_measurement.py --dry-run

Simulated motor + emulated VNA (motion + logging + HDF5, no instruments)::

    python run_bead_pull_measurement.py --simulate --vna-testmode

Real run (settings come from the config file)::

    python run_bead_pull_measurement.py --config config/measurement_config.json

At each bead position the VNA sweep is read and the complex S<port_a><port_b>
trace is appended to ``s11_traces.h5`` in the run directory; ``scan_log.csv`` /
``scan_log.jsonl`` record the position plus a ``trace_index`` into that file and a
few scalar summaries.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Make ``src`` importable when running this script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bead_pull_controller import (  # noqa: E402
    BeadPullController,
    Calibration,
    ScanPoint,
    SimulatedMotor,
    Stepper,
    run_scan,
    scan_targets,
    unwind_direction_sign,
)

DEFAULT_CONFIG = "config/measurement_config.json"


# ===========================================================================
# Configuration model (mirrors config/measurement_config.json)
# ===========================================================================
@dataclass
class StepperConfig:
    port: str | None = None        # serial port, e.g. "/dev/ttyUSB0" or "COM5"
    baud: int = 9600
    motor: int = 1
    configure: bool = True         # push profile/voltage on connect
    microsteps: int = 8
    acceleration: int = 1000
    speed: int = 500
    driving_voltage: float = 6.8
    holding_voltage: float = 2.0
    # physical rotation that unwinds the thread and advances the bead:
    # "clockwise" or "counterclockwise" (aliases "cw"/"ccw"; +1/-1 also accepted).
    unwind_direction: str = "clockwise"

    def __post_init__(self) -> None:
        # fail fast on a bad value rather than at motor-open time
        unwind_direction_sign(self.unwind_direction)


@dataclass
class VnaConfig:
    ip_address: str = "169.254.127.120"
    channel: int = 1
    port_a: int = 2                # first index of S<port_a><port_b>
    port_b: int = 2                # second index of S<port_a><port_b>
    start_freq_Hz: float = 18e9
    stop_freq_Hz: float = 24e9
    points: int = 1001
    averages: int = 1


@dataclass
class ScanConfig:
    step_size_m: float = 0.01
    settle_s: float = 0.5
    sub_thread_indices: list[int] | None = None  # None => all calibrated sub-threads
    length_m: float | None = None                # None => calibrated length per sub-thread
    include_endpoint: bool = True
    home_first: bool = True


@dataclass
class MeasurementConfig:
    calibration_file: str = "config/bead_pull_calibration.json"
    output_root: str = "data/bead_pull_runs"
    output_dir: str | None = None           # None => timestamped under output_root
    scan: ScanConfig = field(default_factory=ScanConfig)
    stepper: StepperConfig = field(default_factory=StepperConfig)
    vna: VnaConfig = field(default_factory=VnaConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "MeasurementConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeasurementConfig":
        output = data.get("output") or {}
        return cls(
            calibration_file=data.get("calibration_file", cls.calibration_file),
            output_root=output.get("root", cls.output_root),
            output_dir=output.get("directory"),
            scan=_build(ScanConfig, data.get("scan")),
            stepper=_build(StepperConfig, data.get("stepper")),
            vna=_build(VnaConfig, data.get("vna")),
        )

    def resolve_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return Path(self.output_root) / f"run_{stamp}"


def _build(dc_cls, data: dict[str, Any] | None):
    """Build a dataclass from a config sub-dict, ignoring (and warning about)
    unknown keys so a typo cannot silently change a setting elsewhere."""
    data = data or {}
    known = {f.name for f in dataclasses.fields(dc_cls)}
    unknown = set(data) - known
    if unknown:
        print(f"WARNING: ignoring unknown {dc_cls.__name__} keys: {sorted(unknown)}")
    return dc_cls(**{k: v for k, v in data.items() if k in known})


# ===========================================================================
# VNA measurement backend
# ===========================================================================
class VnaMeasurement:
    """Rohde & Schwarz VNA measurement of S<port_a><port_b> at each bead position.

    Ported from the old ``lib/vna.py``: connects over TCP, sets the frequency
    band and number of averages/sweeps, and reads the complex S-parameter
    between ``port_a`` and ``port_b``.  Traces are appended to a resizable HDF5
    dataset ``s_parameter`` (shape ``(n_points, n_freq)``) in the run directory,
    alongside the frequency axis; ``measure`` returns the row index plus scalar
    summaries for the human-readable scan log.

    ``testmode=True`` skips the instrument and stores zero traces, so the full
    file-writing pipeline can be exercised without hardware.
    """

    def __init__(
        self,
        output_dir: str | Path,
        ip_address: str = "169.254.218.2",
        channel: int = 1,
        port_a: int = 3,
        port_b: int = 3,
        start_freq: float = 18e9,
        stop_freq: float = 24e9,
        points: int = 1001,
        averages: int = 1,
        testmode: bool = False,
        trace_filename: str = "s11_traces.h5",
        log_filename: str = "vna_log.txt",
        max_retries: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.ip_address = ip_address
        self.channel = int(channel)
        self.port_a = int(port_a)
        self.port_b = int(port_b)
        self.points = int(points)
        self.averages = int(averages)
        self.testmode = testmode
        self.trace_filename = trace_filename
        self.log_filename = log_filename
        self.max_retries = max_retries
        self.freqs = np.linspace(start_freq, stop_freq, self.points)
        self._vna = None
        self._h5 = None
        self._dset = None
        self._n = 0

    @property
    def s_param(self) -> str:
        return f"S{self.port_a}{self.port_b}"

    # -- lifecycle ---------------------------------------------------------
    def setup(self) -> None:
        import h5py

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.testmode:
            print("VNA testmode: storing zero traces (no instrument connected).")
        else:
            from rohdeschwarz.instruments.vna import Vna

            self._vna = Vna()
            self._vna.open_tcp(self.ip_address)
            self._vna.open_log(str(self.output_dir / self.log_filename))
            self._vna.settings.output_power_on = True
            channel = self._vna.channel(self.channel)
            channel.start_frequency_Hz = self.freqs[0]
            channel.stop_frequency_Hz = self.freqs[-1]
            channel.points = len(self.freqs)
            print("Connected to VNA:", self._vna.query("*IDN?").strip())

        path = self.output_dir / self.trace_filename
        self._h5 = h5py.File(path, "w")
        self._h5.create_dataset("frequencies_Hz", data=self.freqs)
        self._dset = self._h5.create_dataset(
            "s_parameter",
            shape=(0, self.points),
            maxshape=(None, self.points),
            dtype=np.complex128,
            chunks=(1, self.points),
        )
        self._dset.attrs["s_param"] = self.s_param
        self._dset.attrs["port_a"] = self.port_a
        self._dset.attrs["port_b"] = self.port_b
        self._dset.attrs["channel"] = self.channel
        self._dset.attrs["averages"] = self.averages
        self._dset.attrs["testmode"] = self.testmode
        self._h5.flush()
        print(
            f"Writing complex {self.s_param} traces to {path} "
            f"({self.points} freq points, "
            f"{self.freqs[0]/1e9:.3f}-{self.freqs[-1]/1e9:.3f} GHz)."
        )

    def teardown(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if self._vna is not None:
            try:
                self._vna.settings.output_power_on = False
            except Exception:
                pass
            close = getattr(self._vna, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._vna = None

    # -- acquisition -------------------------------------------------------
    def _read_trace(self, attempt: int = 0):
        """Read the complex S<port_a><port_b> trace from the VNA (with retries)."""
        if self.testmode:
            return np.zeros(self.points, dtype=np.complex128)
        channel = self._vna.channel(self.channel)
        try:
            if channel.averages != self.averages or channel.sweep_count != self.averages:
                channel.averages = self.averages
                channel.sweep_count = self.averages
            # measure([p, q]) returns the S-matrix for those ports, shape (P, P, F),
            # with result[i, j] = S_{ports[i], ports[j]}. Pick out S_{port_a,port_b}.
            ports = [self.port_a] if self.port_a == self.port_b else [self.port_a, self.port_b]
            result = np.asarray(channel.measure(ports)).reshape(len(ports), len(ports), -1)
            ia = ports.index(self.port_a)
            ib = ports.index(self.port_b)
            return result[ia, ib, :].flatten()
        except KeyboardInterrupt:
            raise
        except Exception:
            is_error = getattr(self._vna, "is_error", None)
            if callable(is_error) and is_error():
                raise RuntimeError("VNA reported an error during measurement")
            print("VNA measurement glitch:", getattr(self._vna, "errors", "?"))
            if attempt < self.max_retries:
                clear = getattr(self._vna, "clear_status", None)
                if callable(clear):
                    clear()
                return self._read_trace(attempt + 1)
            raise RuntimeError("VNA measurement failed after retries")

    def measure(self, point: ScanPoint) -> dict[str, Any]:
        trace = self._read_trace()
        if trace.shape[0] != self.points:
            raise ValueError(
                f"VNA returned {trace.shape[0]} points, expected {self.points}"
            )
        row = self._n
        self._dset.resize(row + 1, axis=0)
        self._dset[row, :] = trace
        self._h5.flush()
        self._n += 1

        magnitude = np.abs(trace)
        peak = int(np.argmax(magnitude))
        return {
            "trace_index": row,
            "trace_file": self.trace_filename,
            "s_param": self.s_param,
            "n_freq": int(self.points),
            "abs_mean": float(magnitude.mean()),
            "abs_max": float(magnitude.max()),
            "peak_freq_Hz": float(self.freqs[peak]),
        }


# ===========================================================================
# Driver
# ===========================================================================
def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to the measurement config JSON")
    p.add_argument("--dry-run", action="store_true", help="print the planned positions and exit (no motion, no measurement)")
    p.add_argument("--simulate", action="store_true", help="use a simulated motor (no serial port)")
    p.add_argument("--vna-testmode", action="store_true", help="store zero VNA traces (no instrument; pairs with --simulate for a full no-hardware run)")
    return p


def build_motor(cfg: MeasurementConfig, simulate: bool):
    if simulate:
        return SimulatedMotor(verbose=True)
    st = cfg.stepper
    if not st.port:
        raise SystemExit("ERROR: stepper.port is not set in the config (or use --simulate / --dry-run).")
    return Stepper.open(
        st.port,
        baud=st.baud,
        motor=st.motor,
        configure=st.configure,
        microsteps=st.microsteps,
        acceleration=st.acceleration,
        speed=st.speed,
        driving_voltage=st.driving_voltage,
        holding_voltage=st.holding_voltage,
        unwind_direction=st.unwind_direction,
    )


def build_measurement(cfg: MeasurementConfig, output_dir: Path, testmode: bool) -> VnaMeasurement:
    v = cfg.vna
    return VnaMeasurement(
        output_dir=output_dir,
        ip_address=v.ip_address,
        channel=v.channel,
        port_a=v.port_a,
        port_b=v.port_b,
        start_freq=v.start_freq_Hz,
        stop_freq=v.stop_freq_Hz,
        points=v.points,
        averages=v.averages,
        testmode=testmode,
    )


def load_calibration(cfg: MeasurementConfig) -> Calibration:
    """Load the sub-thread anchors (calibration file).  The step<->metre
    conversion is derived per sub-thread from those anchors, so nothing else is
    needed."""
    return Calibration.from_calibration_file(cfg.calibration_file)


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    cfg = MeasurementConfig.from_file(args.config)
    calibration = load_calibration(cfg)
    print(calibration.summary())
    print()

    scan = cfg.scan
    lengths_m = None
    if scan.length_m is not None:
        lengths_m = {idx: scan.length_m for idx in calibration.indices()}

    # --- dry run: just show the plan -------------------------------------
    if args.dry_run:
        targets = list(scan_targets(
            calibration, scan.step_size_m, scan.sub_thread_indices, lengths_m, scan.include_endpoint))
        print(f"Planned scan: {len(targets)} points, step_size={scan.step_size_m} m\n")
        print(f"{'st':>3} {'pt':>4} {'target_m':>10} {'target_steps':>13}")
        for sub, point_index, _n, x_m, target_steps in targets:
            print(f"{sub.index:>3} {point_index:>4} {x_m:>10.4f} {target_steps:>13}")
        return 0

    output_dir = cfg.resolve_output_dir()
    motor = build_motor(cfg, args.simulate)
    measurement = build_measurement(cfg, output_dir, args.vna_testmode)
    controller = BeadPullController(motor, calibration, settle_s=scan.settle_s)

    try:
        out = run_scan(
            controller,
            measurement,
            step_size_m=scan.step_size_m,
            output_dir=output_dir,
            sub_thread_indices=scan.sub_thread_indices,
            lengths_m=lengths_m,
            include_endpoint=scan.include_endpoint,
            home_first=scan.home_first,
            metadata={
                "config_file": args.config,
                "config": dataclasses.asdict(cfg),
                "simulate": args.simulate,
                "vna_testmode": args.vna_testmode,
            },
        )
    finally:
        motor.shutdown()

    print(f"\nDone. Results written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
