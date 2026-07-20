# spider-bead-pull
Data evaluation and analysis of the bead-pull method for electric field measurements in the MADMAX spider-booster setup with bead threaded like a spider web.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). To create the
environment and install dependencies:

```bash
uv sync                              # offline analysis only
uv sync --extra hardware --extra notebook   # + real hardware backends and Jupyter
```

Run anything inside the environment with `uv run`, e.g.
`uv run python run_bead_pull_measurement.py` or `uv run jupyter lab`.

Optional dependency groups:
- `hardware` — `pyserial` (stepper motor) and `rohdeschwarz` (VNA); imported
  lazily, only needed when driving real instruments.
- `notebook` — Jupyter/ipykernel for the calibration and analysis notebooks.

## Bead-position control (measurement acquisition)

Drives the single-motor bead through every pass of the thread inside the booster
(each pass = a *sub-thread*) and triggers a measurement at each step.

- **Core logic:** [src/bead_pull_controller.py](src/bead_pull_controller.py) —
  hardware-agnostic. The motor is any object with `home()`/`get_position()`/
  `move_by()`/`shutdown()` and the measurement any object with `measure(point)`
  (plus optional `setup()`/`teardown()`), so you can drop in your own. A working
  `Stepper` (L6470 over serial) and a `SimulatedMotor` are included.
- **Calibration:** run [calibrate_bead_pull.ipynb](calibrate_bead_pull.ipynb).
  You set a **home position** (this rig has no limit switch — see *Homing and the
  motor-state file* below), then for each sub-thread you jog the bead to the
  **start** (the position zero) and to the **end** and capture both motor step
  counts; you also give each sub-thread its `name`, its start/end (x,y,z) in the
  booster frame, and optional non-measurement margins. Direction, scan length and
  the **step↔metre conversion** are all derived per sub-thread from these anchors
  (the conversion is the step span divided by the distance between the two 3-D
  endpoints). The result is written to `config/bead_pull_calibration.json` (see
  [config/bead_pull_calibration.example.json](config/bead_pull_calibration.example.json)).
  The notebook also has an **emergency stop** (Section 2b): a red button and/or
  `Kernel ▸ Interrupt` halt the motor immediately (`hardstop` + coils off) at any
  time, even mid-move, via `motor.emergency_stop()`. A hard stop loses steps, so
  re-establish the position with `motor.set_position(...)` afterwards.
- **Measurement:** [run_bead_pull_measurement.py](run_bead_pull_measurement.py).
  The `VnaMeasurement` backend (Rohde & Schwarz `rohdeschwarz` over TCP, ported
  from the old `lab-control` `lib/vna.py`) reads the complex **S<port_a><port_b>**
  trace per bead position into `s11_traces.h5`.

There are **two config files** (both under `config/`): the **measurement config**
holds the data-taking settings (instruments + scan); the **calibration file** is
the per-sub-thread record produced by the calibration process (step anchors,
`name`, 3-D endpoints, margins). The step↔metre conversion is **not** a setting —
each sub-thread derives its own from its step anchors and 3-D endpoints.

### Homing and the motor-state file

This rig has **no limit switch**, so there is no automatic hardware home (an
earlier version drove `gotoswitch` and the motor spun forever). Instead:

- **Home is a position you set.** In the calibration notebook, jog the bead to a
  repeatable physical reference and call `motor.set_home()`. `motor.home()` then
  performs a normal, bounded move back to that saved step count.
- **A shared state file remembers where the motor is.** The current motor
  position *and* the home position are persisted to a small JSON file at a fixed
  absolute path — `~/.madmax_bead_pull/motor_state.json` by default (override with
  the `MADMAX_BEAD_PULL_STATE` environment variable or the `stepper.state_file`
  config key). Because the L6470's step register resets to zero on a power cycle,
  on connect the controller register is written back to the saved position with
  `setpos`, keeping the absolute step frame — and therefore the calibration
  anchors — stable across power cycles. Every program on the computer reads and
  writes the same file, so they all agree on the position and home.
- **Safety.** Reading the position no longer crashes if the controller gives an
  empty reply (power/cable loss) — it falls back to the saved position with a
  warning. Single moves larger than `stepper.max_step_magnitude` steps are refused
  as a runaway circuit-breaker.

If the bead is moved by hand while the controller is off, the saved position is
stale; re-establish it with `motor.set_position(<steps>)` before homing.

### Configuration reference

Conventions: a `null` value means "use the built-in default / auto" as noted
below; the "Default" column is the value used when the key is omitted (the
shipped [config/measurement_config.json](config/measurement_config.json) may set
a different concrete value). Unknown keys are ignored with a warning, so a typo
never silently changes another setting.

#### `config/measurement_config.json`

Top level:

| Key | Default | Meaning |
| --- | --- | --- |
| `calibration_file` | `"config/bead_pull_calibration.json"` | Path (relative to the repo root) of the calibration-data file the run reads. |
| `disk_spacings_mm` | `[]` | Array of disk spacings (mm) describing the setup under test. Not used by the scan; recorded verbatim in the run metadata so each run is self-describing. |
| `comment` | `""` | Free-text note about the measurement. Not used by the scan; recorded verbatim in the run metadata. |

`output` — where results are written:

| Key | Default | Meaning |
| --- | --- | --- |
| `root` | `"data/bead_pull_runs"` | Parent folder for run directories, used when `directory` is `null`. |
| `directory` | `null` | Explicit output directory. `null` → a fresh `root/run_<UTC-timestamp>/`. |

`scan` — how the bead is stepped and when it measures:

| Key | Default | Meaning |
| --- | --- | --- |
| `step_size_m` | `0.01` | Spacing between measured bead positions within a sub-thread, in metres. |
| `settle_s` | `0.5` | Pause after each move before measuring, in seconds (lets the bead stop swinging). |
| `sub_thread_indices` | `null` | List of sub-thread indices to scan. `null` → all calibrated sub-threads. |
| `length_m` | `null` | Override the scan length for **every** sub-thread, in metres. `null` → each sub-thread's calibrated length (the distance between its 3-D endpoints). The per-sub-thread margins are still applied on top. |
| `include_endpoint` | `true` | Also visit the exact end of each sub-thread (after the end margin) when the stepping doesn't land on it. |
| `home_first` | `true` | Move to the saved home position before the scan starts. Requires a home to have been set (see *Homing and the motor-state file*); otherwise the run stops with a clear error. |

`stepper` — L6470 motor controller (keys match `Stepper.open(...)`):

| Key | Default | Meaning |
| --- | --- | --- |
| `port` | `null` | Serial port, e.g. `"/dev/ttyUSB0"` or `"COM5"`. Required for a real run. |
| `baud` | `9600` | Serial baud rate. |
| `motor` | `1` | Motor number (the `[n]` in the controller's command set). |
| `configure` | `true` | On connect, push the profile + voltages below to the controller. |
| `microsteps` | `8` | Microsteps per full step. |
| `acceleration` | `1000` | Acceleration profile value. |
| `speed` | `500` | Speed profile value. |
| `driving_voltage` | `6.8` | Driving voltage, V. |
| `holding_voltage` | `2.0` | Holding voltage, V. |
| `unwind_direction` | `"counterclockwise"` | Physical rotation that unwinds the thread and advances the bead: `"clockwise"` or `"counterclockwise"` (aliases `"cw"`/`"ccw"`; `+1`/`-1` also accepted). Sets the travel sign. |
| `state_file` | `null` | Path to the shared motor-state file (current position + home). `null` → the default absolute path `~/.madmax_bead_pull/motor_state.json` (also overridable via the `MADMAX_BEAD_PULL_STATE` env var), so every program on the computer shares one file. See *Homing and the motor-state file*. |
| `max_step_magnitude` | `500000` | Refuse any single move larger than this many steps — a runaway circuit-breaker, since this rig has no limit switch. `null` disables the check. |

`vna` — Rohde & Schwarz analyser; one complex **S<port_a><port_b>** trace per
bead position:

| Key | Default | Meaning |
| --- | --- | --- |
| `ip_address` | `"169.254.127.120"` | VNA TCP/IP address. |
| `channel` | `1` | VNA channel. |
| `port_a` | `2` | First index of the measured S-parameter S<port_a><port_b>. |
| `port_b` | `2` | Second index. `port_a == port_b` → a reflection; the shipped config uses `port_a=2, port_b=2` → **S22**. |
| `start_freq_Hz` | `18e9` | Sweep start frequency, Hz. |
| `stop_freq_Hz` | `24e9` | Sweep stop frequency, Hz. |
| `points` | `1001` | Number of frequency points per trace. |
| `averages` | `1` | Sweep count / averages per measurement. |

#### `config/bead_pull_calibration.json`

Written by the calibration notebook; one entry per pass of the thread, with the
measured step anchors, the 3-D endpoints and per-sub-thread settings. The
step↔metre conversion is **not** stored — each sub-thread derives its own from
its step span (`|end_steps − start_steps|`) and the distance between its 3-D
endpoints. Example:
[config/bead_pull_calibration.example.json](config/bead_pull_calibration.example.json).

| Key | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Calibration-format version (currently `2`); set automatically. |
| `created_utc` | str | UTC timestamp when the file was saved; set automatically. |
| `sub_threads[].index` | int | Sub-thread id (which `scan.sub_thread_indices` can select). |
| `sub_threads[].name` | str / null | Label shown in logs (falls back to `sub_thread_<index>`). |
| `sub_threads[].margin_start_m` | float | Non-measurement margin after the start, in metres: no point is measured within this distance of the **zero**. `0` → measure from the start. |
| `sub_threads[].margin_end_m` | float | Non-measurement margin before the end, in metres: no point is measured within this distance of the **far end**. `0` → measure to the end. |
| `sub_threads[].start_xyz_m` | [x,y,z] | **Required.** Position of the pass's **start** (the zero) in the booster coordinate system, in metres. |
| `sub_threads[].end_xyz_m` | [x,y,z] | **Required.** Position of the pass's **end** in the booster coordinate system, in metres. Together with `start_xyz_m` this defines the sub-thread line, onto which each `position_m` is linearly mapped to give the 3-D booster position; the **distance** between the two is also the physical length that sets this sub-thread's step↔metre scale, so enter both accurately. |
| `sub_threads[].start_steps` | int | Absolute motor step count at the bead-position **zero** of the pass. |
| `sub_threads[].end_steps` | int | Absolute motor step count at the **far end**. The sign of `end − start` gives the travel direction; `|end − start|` the step span. The step span ÷ the endpoint distance is this sub-thread's step↔metre conversion. |

### Running

The command line only selects the run mode (everything else is in the config):

```bash
python run_bead_pull_measurement.py --dry-run                         # print planned positions only
python run_bead_pull_measurement.py --simulate --vna-testmode         # full no-hardware run (writes HDF5, zero traces)
python run_bead_pull_measurement.py --config config/measurement_config.json   # real run
```

| Flag | Effect |
| --- | --- |
| `--config <path>` | Measurement config to load (default `config/measurement_config.json`). |
| `--dry-run` | Print the planned positions/steps and exit; no motion, no measurement. |
| `--simulate` | Use the in-memory `SimulatedMotor` instead of the serial stepper. |
| `--vna-testmode` | Run the VNA backend but store zero traces (no instrument); pair with `--simulate` for a full no-hardware run. |

### Output

Each run writes `data/bead_pull_runs/<timestamp>/` (the `root`/`directory` from
the `output` block):

- `s11_traces.h5` — complex `s_parameter` dataset of shape `(n_points, n_freq)`
  (resizable; one row appended per bead position) plus the `frequencies_Hz` axis.
  Dataset attrs: `s_param`, `port_a`, `port_b`, `channel`, `averages`, `testmode`.
  The filename is fixed regardless of which S-parameter is measured.
- `scan_log.csv` / `scan_log.jsonl` — one row per bead position: sub-thread,
  point index, `position_m` (the **true** bead position in metres along the
  sub-thread, converted from the motor's actual step count), the same point in the
  3-D booster frame (`position_x_m`/`position_y_m`/`position_z_m` in the CSV, a
  `position_xyz_m` triple in the JSONL), timestamp, and the measurement result
  (e.g. a `trace_index` into the HDF5).
- `scan_manifest.json` — a self-contained record of the run, written at start and
  updated at finish. Top level: `started_utc` / `finished_utc`, the scan settings
  (`step_size_m`, `include_endpoint`, `home_first`, `n_points_planned`,
  `sub_thread_indices`, `lengths_m_override`), `calibration` (the full calibration
  used, plus a `derived[]` list giving each sub-thread's `length_m` and
  `steps_per_meter`), and a `metadata` block:
  - `config_file` — path of the config that was loaded.
  - `config` — the full resolved config (every value from the tables above,
    including `disk_spacings_mm` and `comment`).
  - `disk_spacings_mm` / `comment` — also surfaced at the top of `metadata` for
    convenience.
  - `simulate` / `vna_testmode` — which no-hardware modes were active.
- `vna_log.txt` — Rohde & Schwarz instrument log; written only on real runs (not
  in `--vna-testmode`).

Requirements for a real run: `pyserial` (stepper) and `rohdeschwarz` (VNA), both
imported lazily so simulated/test runs work without them.
