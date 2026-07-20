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
  It homes to the limit switch, then for each sub-thread you jog the bead to the
  **start** (the position zero) and to the **end** and capture both motor step
  counts; you also give each sub-thread its `name`, its start/end (x,y,z) in the
  booster frame, and optional non-measurement margins. Direction, scan length and
  the **step↔metre conversion** are all derived per sub-thread from these anchors
  (the conversion is the step span divided by the distance between the two 3-D
  endpoints). The result is written to `config/bead_pull_calibration.json` (see
  [config/bead_pull_calibration.example.json](config/bead_pull_calibration.example.json)).
- **Measurement:** [run_bead_pull_measurement.py](run_bead_pull_measurement.py).
  The `VnaMeasurement` backend (Rohde & Schwarz `rohdeschwarz` over TCP, ported
  from the old `lab-control` `lib/vna.py`) reads the complex **S<port_a><port_b>**
  trace per bead position into `s11_traces.h5`.

There are **two config files** (both under `config/`): the **measurement config**
holds the data-taking settings (instruments + scan); the **calibration file** is
the per-sub-thread record produced by the calibration process (step anchors,
`name`, 3-D endpoints, margins). The step↔metre conversion is **not** a setting —
each sub-thread derives its own from its step anchors and 3-D endpoints.

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
| `home_first` | `true` | Home to the limit switch before the scan starts. |

`stepper` — L6470 motor controller (keys match `Stepper.open(...)`):

| Key | Default | Meaning |
| --- | --- | --- |
| `port` | `null` | Serial port, e.g. `"/dev/ttyUSB0"` or `"COM5"`. Required for a real run. |
| `baud` | `115200` | Serial baud rate. |
| `motor` | `1` | Motor number (the `[n]` in the controller's command set). |
| `configure` | `true` | On connect, push the profile + voltages below to the controller. |
| `microsteps` | `8` | Microsteps per full step. |
| `acceleration` | `1000` | Acceleration profile value. |
| `speed` | `500` | Speed profile value. |
| `driving_voltage` | `6.8` | Driving voltage, V. |
| `holding_voltage` | `2.0` | Holding voltage, V. |

`vna` — Rohde & Schwarz analyser; one complex **S<port_a><port_b>** trace per
bead position:

| Key | Default | Meaning |
| --- | --- | --- |
| `ip_address` | `"169.254.218.2"` | VNA TCP/IP address. |
| `channel` | `1` | VNA channel. |
| `port_a` | `3` | First index of the measured S-parameter S<port_a><port_b>. |
| `port_b` | `3` | Second index. `port_a == port_b` → a reflection (e.g. S33); the shipped config uses `port_a=2, port_b=1` → **S21**. |
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

Each run writes `data/bead_pull_runs/<timestamp>/`:

- `s11_traces.h5` — complex `s_parameter` of shape `(n_points, n_freq)` (attrs
  `s_param`/`port_a`/`port_b`/`averages`) plus `frequencies_Hz`.
- `scan_log.csv` / `scan_log.jsonl` — one row per bead position: sub-thread,
  point index, `position_m` (the **true** bead position in metres along the
  sub-thread, converted from the motor's actual step count), the same point in the
  3-D booster frame (`position_x_m`/`position_y_m`/`position_z_m` in the CSV, a
  `position_xyz_m` triple in the JSONL), timestamp, and the measurement result
  (e.g. a `trace_index` into the HDF5).
- `scan_manifest.json` — the full resolved config, the calibration used, and
  start/finish timestamps (self-contained record of the run).

Requirements for a real run: `pyserial` (stepper) and `rohdeschwarz` (VNA), both
imported lazily so simulated/test runs work without them.
