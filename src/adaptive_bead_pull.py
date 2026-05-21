from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.integrate import trapezoid as scipy_trapezoid
from scipy.optimize import minimize
from tqdm import tqdm

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_LINES = (
    (0.15, 80e-3),
    (-0.15, 40e-3),
    (0.15, 0.0),
    (-0.15, -40e-3),
)

DEFAULT_LINE_COUNTS = (1, 2, 3, 4)
DEFAULT_Z_IX_OPTIONS = ((0,), (0, 1), (0, 1, 2))
DEFAULT_GAP_COUNTS = (1, 2, 3, 4)
DEFAULT_GAP_Z_SLICE_STARTS = {
    1: "smallest",
    2: "largest",
    3: "smallest",
    4: 4,
}

DEFAULT_N_BASELINE_MARGIN = 2
DEFAULT_T_WIDTH = 180e-9
DEFAULT_T_OFFSET = 55e-9
DEFAULT_F_IX = 250
DEFAULT_MAX_ET_FOR_SCALING = 500.0
TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz", None) or scipy_trapezoid


@dataclass(frozen=True)
class GapInfo:
    rank: int
    path: Path
    z_min_m: float
    z_max_m: float
    z_positions_m: tuple[float, ...]
    shape: tuple[int, ...]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def z_min_mm(self) -> float:
        return self.z_min_m * 1e3

    @property
    def z_max_mm(self) -> float:
        return self.z_max_m * 1e3


def find_measurements_directory() -> Path:
    candidates = [
        Path("../datasets/OB300 fixed disks/"),
        Path("../madmax-datasets/OB300 fixed disks/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the OB300 fixed disks measurement directory.")


def _read_file_z_positions(file_path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with h5py.File(file_path, "r") as file:
        shape = tuple(file["s11_map"].shape)
        metadata = json.loads(file.attrs["measurement_config"])
    z_pos = np.linspace(metadata["space"][2][0], metadata["space"][2][1], shape[2]) * 1e-3
    return z_pos, shape


def discover_gap_files(measurements_directory: str | Path | None = None) -> list[GapInfo]:
    root = Path(measurements_directory) if measurements_directory is not None else find_measurements_directory()
    files = sorted(root.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in: {root}")

    gaps = []
    for path in files:
        z_pos, shape = _read_file_z_positions(path)
        gaps.append(
            GapInfo(
                rank=0,
                path=path,
                z_min_m=float(np.min(z_pos)),
                z_max_m=float(np.max(z_pos)),
                z_positions_m=tuple(float(v) for v in z_pos),
                shape=shape,
            )
        )

    gaps = sorted(gaps, key=lambda item: (item.z_min_m, item.z_max_m))
    return [
        GapInfo(
            rank=i + 1,
            path=gap.path,
            z_min_m=gap.z_min_m,
            z_max_m=gap.z_max_m,
            z_positions_m=gap.z_positions_m,
            shape=gap.shape,
        )
        for i, gap in enumerate(gaps)
    ]


def print_gap_mapping(gaps: list[GapInfo]) -> None:
    for gap in gaps:
        print(
            f"gap {gap.rank}: {gap.z_min_mm:.1f}-{gap.z_max_mm:.1f} mm, "
            f"{gap.name}, shape={gap.shape}"
        )


def normalize_z_slice_count(z_slice_option) -> int:
    if np.isscalar(z_slice_option):
        z_slice_count = int(z_slice_option)
    else:
        z_slice_count = len(tuple(int(v) for v in z_slice_option))
    if z_slice_count <= 0:
        raise ValueError(f"z_slice_count must be positive, got {z_slice_count}.")
    return z_slice_count


def max_requested_z_slice_count(z_ix_options=DEFAULT_Z_IX_OPTIONS) -> int:
    counts = [normalize_z_slice_count(option) for option in z_ix_options]
    if not counts:
        raise ValueError("At least one z-slice option is required.")
    return max(counts)


def gap_z_index_order(gap: GapInfo) -> tuple[int, ...]:
    # Starts are expressed in absolute z (z_abs = z_mirror - z_rel). Since gap.z_positions_m
    # is stored ascending in z_rel, ascending z_abs corresponds to descending z_rel-index.
    # Numeric `k` means "k-th smallest z_abs" (0-indexed); selection proceeds by adding the
    # next-larger z_abs slice, except for the "largest" rule which proceeds by descending z_abs.
    n_z = len(gap.z_positions_m)
    z_abs_ascending = tuple(range(n_z - 1, -1, -1))
    start_rule = DEFAULT_GAP_Z_SLICE_STARTS.get(int(gap.rank), "smallest")

    if isinstance(start_rule, str):
        if start_rule == "smallest":
            return z_abs_ascending
        if start_rule == "largest":
            return tuple(range(n_z))
        raise ValueError(f"Unknown start_rule {start_rule!r} for gap {gap.rank}.")

    start_k = int(start_rule)
    if start_k < 0 or start_k >= n_z:
        raise ValueError(
            f"Gap {gap.rank} has {n_z} z-slices, but its start index is configured as {start_k}."
        )
    return z_abs_ascending[start_k:]


def select_gap_z_ixs(gap: GapInfo, z_slice_option) -> tuple[int, ...]:
    z_slice_count = normalize_z_slice_count(z_slice_option)
    order = gap_z_index_order(gap)
    if z_slice_count > len(order):
        raise ValueError(
            f"Gap {gap.rank} only supports {len(order)} z-slices from its configured start, "
            f"but {z_slice_count} were requested."
        )
    return tuple(int(ix) for ix in order[:z_slice_count])


def select_gap_fit_z_ixs(gap: GapInfo, z_ix_options=DEFAULT_Z_IX_OPTIONS) -> tuple[int, ...]:
    return select_gap_z_ixs(gap, max_requested_z_slice_count(z_ix_options))


def apply_time_gating(
    s11,
    frequencies,
    x_pos,
    y_pos,
    z_pos,
    t_width=DEFAULT_T_WIDTH,
    t_offset=DEFAULT_T_OFFSET,
    c_const=299792458.0,
    edge_fraction=0.2,
):
    s11 = np.asarray(s11)
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    x_pos = np.asarray(x_pos, dtype=float).ravel()
    y_pos = np.asarray(y_pos, dtype=float).ravel()
    z_pos = np.asarray(z_pos, dtype=float).ravel()

    Nx, Ny, Nz, Nf = s11.shape
    if frequencies.size != Nf:
        raise ValueError("frequencies length must match last axis of s11")
    if Nf < 2:
        raise ValueError("Need at least 2 frequency points for time-domain transform")

    df = float(np.mean(np.diff(frequencies)))
    if not np.isfinite(df) or df <= 0:
        raise ValueError("frequencies must be strictly increasing and finite")

    taxis = np.fft.fftshift(np.fft.fftfreq(Nf, d=df))
    s11_t = np.fft.fftshift(np.fft.ifft(s11, axis=-1), axes=-1)

    mid_x = Nx // 2
    mid_y = Ny // 2
    t_bead_z = np.empty(Nz, dtype=float)
    for z in range(Nz):
        idx = np.argmax(np.abs(s11_t[mid_x, mid_y, z, :]))
        t_bead_z[z] = taxis[idx] + float(t_offset)

    if Nz >= 2:
        p = np.polyfit(z_pos, t_bead_z, 1)
        t_bead_fit_z = np.polyval(p, z_pos)
    else:
        t_bead_fit_z = np.full_like(z_pos, t_bead_z[0], dtype=float)

    r_axis = np.maximum((t_bead_fit_z * c_const / 2.0) ** 2, 0.0)
    t_bead_xyz = (
        2.0
        / c_const
        * np.sqrt(
            r_axis[None, None, :]
            + x_pos[:, None, None] ** 2
            + y_pos[None, :, None] ** 2
        )
    )

    t = taxis[None, None, None, :]
    center = t_bead_xyz[..., None]
    start = center - t_width / 2.0
    end = center + t_width / 2.0
    edge = max(float(t_width) * float(edge_fraction), np.finfo(float).eps)

    in_core = (t >= start) & (t <= end)
    in_rise = (t >= (start - edge)) & (t < start)
    in_fall = (t > end) & (t <= (end + edge))

    rise = 0.5 * (1.0 - np.cos(np.pi * (t - (start - edge)) / edge))
    fall = 0.5 * (1.0 + np.cos(np.pi * (t - end) / edge))
    gate = np.where(in_core, 1.0, np.where(in_rise, rise, np.where(in_fall, fall, 0.0)))

    s11_t_gated = s11_t * gate
    s11_gated = np.fft.fft(np.fft.ifftshift(s11_t_gated, axes=-1), axis=-1)
    return s11_gated, gate, taxis, t_bead_xyz


def _unwrap_with_period(phi, axis, period):
    try:
        return np.unwrap(phi, axis=axis, period=period)
    except TypeError:
        scale = 2.0 * np.pi / float(period)
        return np.unwrap(phi * scale, axis=axis) / scale


def calculate_sqrt_dgamma(delta_gamma, unwrap_xy=True):
    delta_gamma = np.asarray(delta_gamma)
    sqrt_abs = np.sqrt(np.abs(delta_gamma))
    arg_wrapped = np.angle(np.sqrt(delta_gamma))

    if not unwrap_xy:
        return sqrt_abs * np.exp(1j * arg_wrapped)

    arg_unwrapped = np.array(arg_wrapped, copy=True)
    if arg_unwrapped.ndim >= 4:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=1, period=np.pi)
    elif arg_unwrapped.ndim == 3:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
    elif arg_unwrapped.ndim == 2:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=1, period=np.pi)
    elif arg_unwrapped.ndim == 1:
        arg_unwrapped = _unwrap_with_period(arg_unwrapped, axis=0, period=np.pi)
    return sqrt_abs * np.exp(1j * arg_unwrapped)


def load_file(
    file_path,
    z_ixs_used=(0,),
    time_gating=True,
    t_width=DEFAULT_T_WIDTH,
    t_offset=DEFAULT_T_OFFSET,
    frequency_indices=None,
):
    file_path = Path(file_path)
    z_ixs_used = tuple(int(v) for v in z_ixs_used)
    with h5py.File(file_path, "r") as file:
        s11 = np.array(file["s11_map"])
        freq_info = json.loads(file.attrs["vna_config"])
        metadata = json.loads(file.attrs["measurement_config"])

    frequencies = np.linspace(freq_info["start"], freq_info["stop"], freq_info["points"])
    x_pos, y_pos, z_pos = [
        np.linspace(metadata["space"][ax][0], metadata["space"][ax][1], s11.shape[ax]) * 1e-3
        for ax in range(3)
    ]

    if time_gating:
        s11, _, _, _ = apply_time_gating(
            s11,
            frequencies,
            x_pos,
            y_pos,
            z_pos,
            t_width=t_width,
            t_offset=t_offset,
        )

    if frequency_indices is not None:
        frequency_indices = np.asarray(frequency_indices, dtype=int)
        s11 = s11[..., frequency_indices]
        frequencies = frequencies[frequency_indices]

    x_plt, y_plt = np.meshgrid(x_pos, y_pos, indexing="ij")
    x_pts, y_pts = x_plt.flatten(), y_plt.flatten()
    s11_z = s11[:, :, z_ixs_used, :]
    s11_flat = s11.reshape(-1, s11.shape[2], s11.shape[3])
    s11_z_flat = s11_z.reshape(-1, s11_z.shape[2], s11_z.shape[3])

    return {
        "s11": s11,
        "s11_flat": s11_flat,
        "s11_z": s11_z,
        "s11_z_flat": s11_z_flat,
        "frequencies": frequencies,
        "x_pos": x_pos,
        "x_plt": x_plt,
        "x_pts": x_pts,
        "y_pos": y_pos,
        "y_plt": y_plt,
        "y_pts": y_pts,
        "z_pos": z_pos,
        "z_ixs_used": z_ixs_used,
    }


def plot_point_selection(
    s11,
    frequencies,
    ixs,
    lines,
    x_pos,
    x_pts,
    y_pos,
    y_pts,
    z_pos,
    z_ix=0,
    f_ix=DEFAULT_F_IX,
    show_point_ix=False,
    show_line_ix=False,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    cf = ax.contourf(x_pos, y_pos, np.abs(s11[:, :, z_ix, f_ix]).T, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="|S11|")

    ax.scatter(x_pts, y_pts, s=10, c="grey", alpha=0.6, linewidths=0, label="grid points", zorder=3)
    x_line = np.linspace(x_pos.min(), x_pos.max(), 400)
    for i, (a, b) in enumerate(lines):
        y_line = a * x_line + b
        mask = (y_line >= y_pos.min()) & (y_line <= y_pos.max())
        ax.plot(
            x_line[mask],
            y_line[mask],
            color="orange",
            linewidth=1,
            zorder=5,
            label="selection lines" if i == 0 else None,
        )
        if show_line_ix and np.any(mask):
            x_seg = x_line[mask]
            y_seg = y_line[mask]
            mid = len(x_seg) // 2
            ax.text(
                x_seg[mid],
                y_seg[mid] - 5.0,
                f"Line {i}",
                ha="center",
                va="top",
                fontsize=8,
                color="orange",
                zorder=6,
                clip_on=True,
            )

    if show_point_ix:
        ixs_arr = np.asarray(ixs)
        sel_idx = np.flatnonzero(ixs_arr) if ixs_arr.dtype == bool else ixs_arr.astype(int)
        for idx in sel_idx:
            ax.text(
                x_pts[idx],
                y_pts[idx] - 2.0,
                str(idx),
                ha="center",
                va="top",
                fontsize=7,
                color="red",
                zorder=6,
                clip_on=True,
            )

    ax.scatter(x_pts[ixs], y_pts[ixs], s=20, c="red", alpha=0.9, linewidths=0, label="selected points", zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"|S11| at z={z_pos[z_ix]} m, nu={frequencies[f_ix] * 1e-9:.2f} GHz\n"
        "with grid, selection lines and selected points"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_sqrt_Dgamma_line(
    lines,
    idxs_sorted_lines,
    s_sorted_lines,
    baseline_lines,
    sqrt_Dgamma_lines,
    line_index,
    n_baseline_margin,
    s11_flat,
    z_pos,
    frequencies,
    z_ix=0,
    f_ix=DEFAULT_F_IX,
):
    if not (0 <= line_index < len(lines)):
        raise ValueError(f"Line index i={line_index} out of range (0..{len(lines) - 1}).")
    if z_ix < 0 or z_ix >= s11_flat.shape[1]:
        raise ValueError(f"z_ix out of range (0..{s11_flat.shape[1] - 1}).")
    if f_ix < 0 or f_ix >= s11_flat.shape[2]:
        raise ValueError(f"f_ix out of range (0..{s11_flat.shape[2] - 1}).")
    if (
        idxs_sorted_lines[line_index] is None
        or s_sorted_lines[line_index] is None
        or baseline_lines[line_index] is None
        or sqrt_Dgamma_lines[line_index] is None
    ):
        raise ValueError("Precomputed data for this line is missing.")

    idxs_sorted = idxs_sorted_lines[line_index]
    s_sorted = s_sorted_lines[line_index]
    baseline = baseline_lines[line_index][:, z_ix, f_ix]
    sqrt_dg = sqrt_Dgamma_lines[line_index][:, z_ix, f_ix]
    sqrt_dg_abs = np.abs(sqrt_dg)

    m_points = idxs_sorted.size
    n = min(n_baseline_margin, m_points // 2)
    if n == 0:
        raise ValueError("Baseline margin results in empty baseline set.")
    base_idx = np.r_[np.arange(n), np.arange(m_points - n, m_points)]
    y_line = s11_flat[idxs_sorted, z_ix, f_ix]

    a, b = lines[line_index]
    u = np.array([1.0, a], dtype=float)
    u /= np.linalg.norm(u)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax0.plot(s_sorted, np.abs(y_line), "o", ms=4, alpha=0.7, label="|Gamma| (points)")
    ax0.plot(s_sorted, np.abs(baseline), "-", lw=2, color="tab:orange", label="|baseline| (fit)")
    ax0.plot(s_sorted[base_idx], np.abs(y_line[base_idx]), "o", ms=6, color="red", label="baseline points")
    ax0.set_ylabel("|Gamma|")
    ax0.legend(loc="best")
    ax0.set_title(
        f"Line {line_index}: y={a:.3f}x + {b:.3f}, "
        f"z={z_pos[z_ix]} m, nu={frequencies[f_ix] * 1e-9:.2f} GHz"
    )
    ax0.grid()

    ax1.plot(s_sorted, sqrt_dg_abs, "o-", ms=4, label="|sqrt(DeltaGamma)|")
    ax1.plot(s_sorted[base_idx], sqrt_dg_abs[base_idx], "o", ms=6, color="red", label="baseline points")
    ax1.set_xlabel(f"s={u[0]:.3f}x + {u[1]:.3f}y (line parameterization)")
    ax1.set_ylabel("|sqrt(DeltaGamma)| proportional to |E|")
    ax1.legend(loc="best")
    ax1.grid()
    fig.tight_layout()
    return fig, (ax0, ax1)


def plot_fitted_efield(
    params,
    x_plt,
    y_plt,
    x_reduced,
    y_reduced,
    e_field_reduced,
    e_field_full,
    beam_model_func,
    z_pos,
    frequencies,
    costs=None,
    z_ix=0,
    f_ix=DEFAULT_F_IX,
    overlay_points=True,
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    p = params[z_ix, f_ix]
    x0, y0, A, s = p[:4]
    kappa = float(p[4]) if p.size >= 5 else 0.0
    phi0 = float(p[5]) if p.size >= 6 else 0.0
    w = p[6:] if p.size > 6 else (1.0, 0.0, 0.0, 0.0)

    e_field_fit = beam_model_func(x_plt, y_plt, x0, y0, A, s, kappa, phi0, *w)
    e_fit_mag = np.abs(e_field_fit)
    e_fit_real = np.real(e_field_fit)
    e_fit_imag = np.imag(e_field_fit)

    e_field_data = e_field_full[:, :, z_ix, f_ix]
    e_data_mag = np.abs(e_field_data)
    e_data_real = np.real(e_field_data)
    e_data_imag = np.imag(e_field_data)

    overlay_mag = overlay_real = overlay_imag = None
    if overlay_points and e_field_reduced is not None:
        e_field_samples = np.asarray(e_field_reduced[:, z_ix, f_ix])
        overlay_mag = np.abs(e_field_samples)
        overlay_real = np.real(e_field_samples)
        overlay_imag = np.imag(e_field_samples)

    def _finite_limits(values, symmetric=False, percent=95):
        values = np.concatenate([np.asarray(v).ravel() for v in values])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)
        if symmetric:
            lim = np.nanpercentile(np.abs(values), percent)
            if not np.isfinite(lim) or lim == 0:
                lim = np.nanmax(np.abs(values))
            if not np.isfinite(lim) or lim == 0:
                lim = 1.0
            return -lim, lim
        lo = np.nanpercentile(values, 5)
        hi = np.nanpercentile(values, percent)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        return lo, hi

    mag_vals = [e_fit_mag, e_data_mag]
    real_vals = [e_fit_real, e_data_real]
    imag_vals = [e_fit_imag, e_data_imag]
    if overlay_mag is not None:
        mag_vals.append(overlay_mag)
        real_vals.append(overlay_real)
        imag_vals.append(overlay_imag)
    mag_vmin, mag_vmax = _finite_limits(mag_vals, symmetric=False)
    real_vmin, real_vmax = _finite_limits(real_vals, symmetric=True)
    imag_vmin, imag_vmax = _finite_limits(imag_vals, symmetric=True)

    denom = np.where(e_data_mag > np.finfo(float).eps, e_data_mag, np.nan)
    rel_residual = 100.0 * (e_data_mag - e_fit_mag) / denom
    rmaxp = np.nanpercentile(np.abs(rel_residual), 95)
    if not np.isfinite(rmaxp) or rmaxp == 0:
        rmaxp = np.nanmax(np.abs(rel_residual))
    if not np.isfinite(rmaxp) or rmaxp == 0:
        rmaxp = 1.0

    fig_top, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=False)
    fig_top.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.93, wspace=0.35, hspace=0)
    ax00, ax01, ax02 = ax[0, 0], ax[0, 1], ax[0, 2]
    ax10, ax11, ax12 = ax[1, 0], ax[1, 1], ax[1, 2]

    c0_fit = ax00.contourf(x_plt, y_plt, e_fit_mag, levels=30, cmap="viridis", vmin=mag_vmin, vmax=mag_vmax)
    c0_data = ax10.contourf(x_plt, y_plt, e_data_mag, levels=30, cmap="viridis", vmin=mag_vmin, vmax=mag_vmax)
    cax00 = make_axes_locatable(ax00).append_axes("right", size="4%", pad=0.08)
    cax10 = make_axes_locatable(ax10).append_axes("right", size="4%", pad=0.08)
    fig_top.colorbar(c0_fit, cax=cax00, label="|E|")
    fig_top.colorbar(c0_data, cax=cax10, label="|E|")
    if overlay_mag is not None and x_reduced is not None and y_reduced is not None:
        ax00.scatter(
            x_reduced,
            y_reduced,
            c=overlay_mag,
            cmap="viridis",
            vmin=mag_vmin,
            vmax=mag_vmax,
            s=25,
            edgecolors="k",
            linewidths=0.2,
            alpha=0.9,
            label="data points",
        )
        ax00.legend(loc="upper right")

    c1_fit = ax01.contourf(x_plt, y_plt, e_fit_real, levels=30, cmap="RdBu_r", vmin=real_vmin, vmax=real_vmax)
    c1_data = ax11.contourf(x_plt, y_plt, e_data_real, levels=30, cmap="RdBu_r", vmin=real_vmin, vmax=real_vmax)
    cax01 = make_axes_locatable(ax01).append_axes("right", size="4%", pad=0.08)
    cax11 = make_axes_locatable(ax11).append_axes("right", size="4%", pad=0.08)
    fig_top.colorbar(c1_fit, cax=cax01, label="Re(E)")
    fig_top.colorbar(c1_data, cax=cax11, label="Re(E)")
    if overlay_real is not None and x_reduced is not None and y_reduced is not None:
        ax01.scatter(
            x_reduced,
            y_reduced,
            c=overlay_real,
            cmap="RdBu_r",
            vmin=real_vmin,
            vmax=real_vmax,
            s=25,
            edgecolors="k",
            linewidths=0.2,
            alpha=0.9,
            label="data points",
        )
        ax01.legend(loc="upper right")

    c2_fit = ax02.contourf(x_plt, y_plt, e_fit_imag, levels=30, cmap="RdBu_r", vmin=imag_vmin, vmax=imag_vmax)
    c2_data = ax12.contourf(x_plt, y_plt, e_data_imag, levels=30, cmap="RdBu_r", vmin=imag_vmin, vmax=imag_vmax)
    cax02 = make_axes_locatable(ax02).append_axes("right", size="4%", pad=0.08)
    cax12 = make_axes_locatable(ax12).append_axes("right", size="4%", pad=0.08)
    fig_top.colorbar(c2_fit, cax=cax02, label="Im(E)")
    fig_top.colorbar(c2_data, cax=cax12, label="Im(E)")
    if overlay_imag is not None and x_reduced is not None and y_reduced is not None:
        ax02.scatter(
            x_reduced,
            y_reduced,
            c=overlay_imag,
            cmap="RdBu_r",
            vmin=imag_vmin,
            vmax=imag_vmax,
            s=25,
            edgecolors="k",
            linewidths=0.2,
            alpha=0.9,
            label="data points",
        )
        ax02.legend(loc="upper right")

    title_fit = f"Fitted E-field at z={z_pos[z_ix]} m, nu={frequencies[f_ix] * 1e-9:.2f} GHz"
    if costs is not None:
        title_fit += f", MSE cost={float(costs[z_ix, f_ix]):.3e}"
    ax00.set_title(title_fit)
    ax01.set_title("Fitted Re(E)")
    ax02.set_title("Fitted Im(E)")
    ax10.set_title("Data |E| (all points)")
    ax11.set_title("Data Re(E) (all points)")
    ax12.set_title("Data Im(E) (all points)")
    for axi in (ax00, ax01, ax02, ax10, ax11, ax12):
        axi.set_xlabel("x [m]")
        axi.set_ylabel("y [m]")
        axi.set_aspect("equal", adjustable="box")
        axi.grid(True, alpha=0.2)

    fig_res = plt.figure(figsize=(10, 5), constrained_layout=True)
    axr = fig_res.add_subplot(111)
    cfr = axr.contourf(x_plt, y_plt, rel_residual, levels=30, cmap="coolwarm", vmin=-rmaxp, vmax=rmaxp)
    fig_res.colorbar(cfr, ax=axr, label="Residual (% of data)")
    axr.set_title("Residual (data - fit) [%]")
    axr.set_xlabel("x [m]")
    axr.set_ylabel("y [m]")
    axr.set_aspect("equal", adjustable="box")
    axr.grid(True, alpha=0.2)
    return fig_top, (ax00, ax01, ax02, ax10, ax11, ax12), fig_res, axr


def plot_sqrt_dgamma_before_after_correction(
    sqrt_dgamma_full,
    sqrt_dgamma_reduced_before,
    sqrt_dgamma_reduced_after,
    all_ixs,
    reduced_z_ix=0,
    full_z_ix=None,
    f_ix=DEFAULT_F_IX,
    colors=("tab:orange", "tab:blue"),
    labels=("before correction", "after correction"),
    title_prefix="",
):
    if full_z_ix is None:
        full_z_ix = reduced_z_ix
    full_line = np.asarray(sqrt_dgamma_full).reshape(-1, sqrt_dgamma_full.shape[2], sqrt_dgamma_full.shape[3])[
        all_ixs, full_z_ix, f_ix
    ]
    red_before = np.asarray(sqrt_dgamma_reduced_before)[:, reduced_z_ix, f_ix]
    red_after = np.asarray(sqrt_dgamma_reduced_after)[:, reduced_z_ix, f_ix]
    valid = np.isfinite(full_line) & np.isfinite(red_before) & np.isfinite(red_after)
    full_line = full_line[valid]
    red_before = red_before[valid]
    red_after = red_after[valid]
    if full_line.size == 0:
        raise ValueError("No valid overlapping line-point samples for plotting.")

    def _lims(xv, yv1, yv2):
        lo = np.nanmin(np.r_[xv, yv1, yv2])
        hi = np.nanmax(np.r_[xv, yv1, yv2])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -1.0, 1.0
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    comps = [
        (np.abs(full_line), np.abs(red_before), np.abs(red_after), "|sqrt(DeltaGamma)|"),
        (np.real(full_line), np.real(red_before), np.real(red_after), "Re(sqrt(DeltaGamma))"),
        (np.imag(full_line), np.imag(red_before), np.imag(red_after), "Im(sqrt(DeltaGamma))"),
    ]
    for ax, (xv, yb, ya, name) in zip(axes, comps):
        ax.scatter(xv, yb, s=22, label=labels[0], marker="o", edgecolors=colors[0], linewidths=0.8, facecolors="none")
        ax.scatter(xv, ya, s=22, label=labels[1], marker="o", edgecolors=colors[1], linewidths=0.8, facecolors="none")
        lo, hi = _lims(xv, yb, ya)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full {name}")
        ax.set_ylabel(f"reduced {name}")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(
        f"{title_prefix} sqrt(DeltaGamma) at line points, reduced_z_ix={reduced_z_ix}, full_z_ix={full_z_ix}, f_ix={f_ix}",
        y=1.03,
    )
    return fig, axes


def plot_beta_overlay(beta_reduced, beta_full, frequencies, colors=("tab:blue", "tab:red")):
    freq_ghz = np.asarray(frequencies) * 1e-9
    n = freq_ghz.size
    step = max(1, n // 40)
    fig, ax = plt.subplots(figsize=(9, 4))
    l1 = ax.plot(
        freq_ghz,
        beta_reduced,
        color=colors[0],
        lw=1.0,
        linestyle="--",
        marker="o",
        ms=3,
        markevery=step,
        alpha=0.9,
        zorder=3,
        label="beta from reduced BP",
    )
    ax.set_xlim(19, 20)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("beta (a.u.)")
    ax.grid(True, which="both", alpha=0.3)
    l2 = ax.plot(
        freq_ghz,
        beta_full,
        color=colors[1],
        lw=1.0,
        linestyle="-",
        marker="s",
        ms=3,
        markevery=(step // 2, step),
        alpha=0.9,
        zorder=2,
        label="beta from full BP",
    )
    l2[0].set_dashes((6, 2))
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_ET_complex_plane(ETs_full, ETs_reduced, z_full, z_reduced, f_ix=DEFAULT_F_IX, z_decimals=2):
    ETs_full = np.asarray(ETs_full)
    ETs_reduced = np.asarray(ETs_reduced)
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()
    ix_z_full_matched = [int(np.argmin(np.abs(z_full - zr))) for zr in z_reduced]
    full_vals = ETs_full[ix_z_full_matched, f_ix]
    reduced_vals = ETs_reduced[:, f_ix]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.real(full_vals), np.imag(full_vals), label="Full Set (at z_reduced)", marker="o", s=100, alpha=0.7)
    ax.scatter(np.real(reduced_vals), np.imag(reduced_vals), label="Reduced Set", marker="x", s=100, alpha=0.8)
    for i, zr in enumerate(z_reduced):
        z_mm = zr * 1e3
        z_label = f"z={z_mm:.{z_decimals}f} mm"
        ax.annotate(z_label, (np.real(full_vals[i]), np.imag(full_vals[i])), xytext=(5, 5), textcoords="offset points", color="tab:blue")
        ax.annotate(
            z_label,
            (np.real(reduced_vals[i]), np.imag(reduced_vals[i])),
            xytext=(5, -15),
            textcoords="offset points",
            color="tab:orange",
        )
    ax.set_xlabel("Re(ET)")
    ax.set_ylabel("Im(ET)")
    ax.set_title(f"ET values in complex plane at f_ix={f_ix}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_abs_ET_over_z(ETs_full, ETs_reduced, z_full, z_reduced, f_ix=DEFAULT_F_IX, z_decimals=2, atol=1e-9):
    ETs_full = np.asarray(ETs_full)
    ETs_reduced = np.asarray(ETs_reduced)
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()

    pairs = []
    for i_red, zr in enumerate(z_reduced):
        is_match = np.isclose(z_full, zr, atol=atol, rtol=0.0)
        if np.any(is_match):
            i_full = int(np.flatnonzero(is_match)[0])
            pairs.append((i_full, i_red))
    if len(pairs) == 0:
        raise ValueError("No overlapping z values found between z_full and z_reduced.")

    full_idx = np.array([p[0] for p in pairs], dtype=int)
    red_idx = np.array([p[1] for p in pairs], dtype=int)
    z_plot = z_reduced[red_idx]
    full_abs = np.abs(ETs_full[full_idx, f_ix])
    red_abs = np.abs(ETs_reduced[red_idx, f_ix])
    order = np.argsort(z_plot)
    z_plot = z_plot[order]
    full_abs = full_abs[order]
    red_abs = red_abs[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_plot * 1e3, full_abs, "o-", label="|ET| Full (z overlap)", alpha=0.8)
    ax.plot(z_plot * 1e3, red_abs, "x--", label="|ET| Reduced", alpha=0.8)
    if z_decimals is not None:
        for z_mm, y in zip(z_plot * 1e3, red_abs):
            ax.annotate(
                f"{z_mm:.{z_decimals}f} mm",
                (z_mm, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="tab:orange",
            )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|ET|")
    ax.set_title(f"|ET| over z at f_ix={f_ix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_abs_ET_grid_over_frequency(
    ETs_full,
    ETs_reduced,
    z_full,
    z_reduced,
    frequencies,
    z_decimals=2,
    atol=1e-9,
    max_et_for_scaling=DEFAULT_MAX_ET_FOR_SCALING,
    colors=("tab:blue", "tab:red"),
    colors_real=("tab:green", "tab:orange"),
    colors_imag=("tab:purple", "tab:brown"),
    reduced_linewidth=1.0,
    full_linewidth=1.0,
    reduced_alpha=0.75,
    full_alpha=0.65,
    reduced_linestyle="-",
    full_linestyle=(0, (4, 2)),
    return_all=False,
):
    ETs_full = np.asarray(ETs_full)
    ETs_reduced = np.asarray(ETs_reduced)
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    n = z_reduced.size
    if n == 0:
        raise ValueError("z_reduced is empty; nothing to plot.")

    ix_z_full_matched = np.empty(n, dtype=int)
    for i, zr in enumerate(z_reduced):
        matches = np.flatnonzero(np.isclose(z_full, zr, atol=atol, rtol=0.0))
        ix_z_full_matched[i] = int(matches[0]) if matches.size > 0 else int(np.argmin(np.abs(z_full - zr)))

    freq_ghz = frequencies * 1e-9
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    def _component_ylim(y_reduced, y_full, symmetric):
        vals = np.r_[
            y_reduced[np.isfinite(y_reduced) & (np.abs(y_reduced) <= max_et_for_scaling)],
            y_full[np.isfinite(y_full) & (np.abs(y_full) <= max_et_for_scaling)],
        ]
        if vals.size == 0:
            lim = max_et_for_scaling
        elif symmetric:
            lim = float(np.nanmax(np.abs(vals)))
            lim = max(lim, np.finfo(float).eps)
        else:
            lim = float(np.nanmax(vals))
            lim = max(lim, np.finfo(float).eps)
        if symmetric:
            return -1.05 * lim, 1.05 * lim
        return 0.0, 1.05 * lim

    def _plot_grid(component_fn, ylabel, title, palette, symmetric=False):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows), sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for i in range(n):
            ax = axes[i]
            y_reduced = component_fn(ETs_reduced[i, :])
            y_full = component_fn(ETs_full[ix_z_full_matched[i], :])
            reduced_line = ax.plot(
                freq_ghz,
                y_reduced,
                color=palette[0],
                lw=reduced_linewidth,
                alpha=reduced_alpha,
                linestyle=reduced_linestyle,
                zorder=2,
                label=f"{ylabel} reduced",
            )[0]
            full_line = ax.plot(
                freq_ghz,
                y_full,
                color=palette[1],
                lw=full_linewidth,
                alpha=full_alpha,
                linestyle=full_linestyle,
                zorder=3,
                label=f"{ylabel} full",
            )[0]
            ax.set_ylim(*_component_ylim(y_reduced, y_full, symmetric=symmetric))
            ax.set_title(f"z={z_reduced[i] * 1e3:.{z_decimals}f} mm")
            ax.set_xlabel("Frequency [GHz]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[reduced_line, full_line], loc="best")
        for j in range(n, axes.size):
            axes[j].set_visible(False)
        fig.suptitle(title, y=1.02)
        return fig, axes[:n]

    fig_abs, axes_abs = _plot_grid(
        lambda x: np.abs(x),
        "|ET|",
        "|ET| over frequency per reduced z-slice (reduced vs full)",
        colors,
        symmetric=False,
    )
    fig_real, axes_real = _plot_grid(
        lambda x: np.real(x),
        "Re(ET)",
        "Re(ET) over frequency per reduced z-slice (reduced vs full)",
        colors_real,
        symmetric=True,
    )
    fig_imag, axes_imag = _plot_grid(
        lambda x: np.imag(x),
        "Im(ET)",
        "Im(ET) over frequency per reduced z-slice (reduced vs full)",
        colors_imag,
        symmetric=True,
    )
    if return_all:
        return (fig_abs, axes_abs), (fig_real, axes_real), (fig_imag, axes_imag), ix_z_full_matched
    return fig_abs, axes_abs, ix_z_full_matched


def compute_sqrt_Dgamma_full(s11, x_pos, y_pos, z_pos=None, n_baseline_margin=DEFAULT_N_BASELINE_MARGIN):
    Nx, Ny, Nz, Nf = s11.shape
    n = min(int(n_baseline_margin), Nx // 2)
    if n <= 0:
        raise ValueError("n_baseline_margin must be >= 1 and less than half the number of x points.")

    baseline_ix = np.r_[np.arange(n), np.arange(Nx - n, Nx)]
    m = baseline_ix.size
    s_base = x_pos[baseline_ix]
    sum_s = s_base.sum()
    sum_s2 = (s_base**2).sum()
    den = m * sum_s2 - sum_s**2
    delta_gamma_full = np.empty(s11.shape, dtype=np.complex128)

    for j in range(Ny):
        y_base = s11[baseline_ix, j, :, :]
        sum_y = y_base.sum(axis=0)
        sum_sy = (s_base[:, None, None] * y_base).sum(axis=0)
        if den == 0:
            beta0 = y_base.mean(axis=0)
            beta1 = np.zeros_like(beta0)
        else:
            beta0 = (sum_s2 * sum_y - sum_s * sum_sy) / den
            beta1 = (m * sum_sy - sum_s * sum_y) / den
        base_row = beta0[None, :, :] + beta1[None, :, :] * x_pos[:, None, None]
        delta_gamma_full[:, j, :, :] = s11[:, j, :, :] - base_row
    return calculate_sqrt_dgamma(delta_gamma_full, unwrap_xy=True)


def sqrt_dgamma_to_E(sqrt_dgamma, frequencies):
    eps_b = 9.23
    delta_eps = eps_b - 1.0
    r_b = (2.93e-3) / 2.0
    V_b = 4.0 / 3.0 * np.pi * r_b**3
    alpha_0 = 3.0 * delta_eps * V_b / (eps_b + 2.0)
    P_in = 1.0
    epsilon0 = 8.85418782e-12
    nu_hz = np.asarray(frequencies, dtype=float).ravel()
    conv = np.sqrt(4.0 * P_in / (epsilon0 * alpha_0 * 1j * 2.0 * np.pi * nu_hz))
    conv_shape = (1,) * (sqrt_dgamma.ndim - 1) + (conv.size,)
    return sqrt_dgamma * conv.reshape(conv_shape)


def compute_ET_full(field_full, x_pos, y_pos, frequencies=None):
    return np.nansum(field_full, axis=(0, 1)) * np.mean(np.diff(x_pos)) * np.mean(np.diff(y_pos))


def reduce_points_to_lines(s11_z_flat, x_pts, y_pts, n_lines=4, lines=DEFAULT_LINES, tol=3e-3):
    lines_all = list(lines)
    n_lines = int(n_lines)
    if n_lines <= 0:
        lines = []
    elif n_lines == 1 and len(lines_all) >= 3:
        # For a single-line reduced set, use the central line rather than the outermost one.
        lines = [lines_all[2]]
    elif n_lines == 2 and len(lines_all) >= 3:
        # For two lines, use the two middle lines (smaller |y|) instead of the outer lines.
        lines = [lines_all[1], lines_all[2]]
    else:
        lines = list(lines_all[:n_lines])
    if not lines:
        raise ValueError("n_lines must select at least one line.")
    line_arr = np.asarray(lines, dtype=float)
    a = line_arr[:, 0]
    b = line_arr[:, 1]
    residuals = np.abs(y_pts[:, None] - (x_pts[:, None] * a[None, :] + b[None, :]))
    per_line_masks = residuals <= float(tol)
    all_ixs = per_line_masks.any(axis=1)
    per_line_ixs = per_line_masks.T
    s11_reduced = s11_z_flat[all_ixs, :, :]
    x_reduced = x_pts[all_ixs]
    y_reduced = y_pts[all_ixs]
    return s11_reduced, x_reduced, y_reduced, lines, all_ixs, per_line_ixs


def compute_sqrt_Dgamma_lines(s11_z_flat, x_pts, y_pts, lines, per_line_ixs, all_ixs, n_baseline_margin):
    sqrt_Dgamma = np.empty(s11_z_flat.shape, dtype=np.complex128)
    sqrt_Dgamma_lines = [None] * len(lines)
    baseline_lines = [None] * len(lines)
    s_sorted_lines = [None] * len(lines)
    idxs_sorted_lines = [None] * len(lines)

    for line_index, (a, b) in enumerate(lines):
        mask = per_line_ixs[line_index]
        idxs = np.flatnonzero(mask)
        if idxs.size < 2:
            continue
        u = np.array([1.0, a], dtype=float)
        u /= np.linalg.norm(u)
        s_all = u[0] * x_pts[idxs] + u[1] * y_pts[idxs]
        order = np.argsort(s_all)
        idxs_sorted = idxs[order]
        s_all_sorted = s_all[order]
        m_points = idxs_sorted.size
        n = min(n_baseline_margin, m_points // 2)
        if n == 0:
            continue

        s_base = np.concatenate([s_all_sorted[:n], s_all_sorted[-n:]])
        y_base = s11_z_flat[np.concatenate([idxs_sorted[:n], idxs_sorted[-n:]]), :, :]
        m = s_base.size
        sum_s = s_base.sum()
        sum_s2 = (s_base**2).sum()
        sum_y = y_base.sum(axis=0)
        sum_sy = (s_base[:, None, None] * y_base).sum(axis=0)
        den = m * sum_s2 - sum_s**2
        if den == 0:
            beta0 = y_base.mean(axis=0)
            beta1 = np.zeros_like(beta0)
        else:
            beta0 = (sum_s2 * sum_y - sum_s * sum_sy) / den
            beta1 = (m * sum_sy - sum_s * sum_y) / den
        base = beta0[None, :, :] + beta1[None, :, :] * s_all_sorted[:, None, None]
        delta_gamma_line = s11_z_flat[idxs_sorted, :, :] - base
        sqrt_Dgamma_line = calculate_sqrt_dgamma(delta_gamma_line, unwrap_xy=True)
        sqrt_Dgamma[idxs_sorted, :, :] = sqrt_Dgamma_line
        sqrt_Dgamma_lines[line_index] = sqrt_Dgamma_line
        baseline_lines[line_index] = base
        s_sorted_lines[line_index] = s_all_sorted
        idxs_sorted_lines[line_index] = idxs_sorted

    sqrt_Dgamma_reduced = sqrt_Dgamma[all_ixs]
    return sqrt_Dgamma_reduced, sqrt_Dgamma_lines, baseline_lines, s_sorted_lines, idxs_sorted_lines


def correct_sqrt_dgamma_reduced_with_full(
    sqrt_dgamma_full,
    sqrt_Dgamma_lines,
    idxs_sorted_lines,
    all_ixs,
    z_ixs_used,
):
    full_flat = np.asarray(sqrt_dgamma_full).reshape(-1, sqrt_dgamma_full.shape[2], sqrt_dgamma_full.shape[3])
    first_line = next((arr for arr in sqrt_Dgamma_lines if arr is not None), None)
    if first_line is None:
        raise ValueError("No line data available in sqrt_Dgamma_lines.")
    Zred, F = first_line.shape[1], first_line.shape[2]
    z_map = np.asarray(z_ixs_used, dtype=int).ravel()
    if z_map.size != Zred:
        raise ValueError(f"z_ixs_used has length {z_map.size}, expected {Zred}.")

    sqrt_Dgamma_lines_corr = [None] * len(sqrt_Dgamma_lines)
    for line_ix, idxs in enumerate(idxs_sorted_lines):
        if idxs is None or sqrt_Dgamma_lines[line_ix] is None:
            continue
        idxs = np.asarray(idxs, dtype=int)
        full_line = full_flat[idxs[:, None], z_map[None, :], :]
        red_line = np.asarray(sqrt_Dgamma_lines[line_ix])
        corr_line = np.array(red_line, copy=True)
        for z_r in range(Zred):
            for f_ix in range(F):
                fvals = full_line[:, z_r, f_ix]
                rvals = red_line[:, z_r, f_ix]
                valid = np.isfinite(fvals) & np.isfinite(rvals)
                if not np.any(valid):
                    continue
                sign = np.where(np.real(rvals[valid] * np.conj(fvals[valid])) >= 0.0, 1.0, -1.0)
                r_signed = np.array(rvals, copy=True)
                r_signed[valid] = rvals[valid] * sign
                den = np.vdot(r_signed[valid], r_signed[valid]).real
                alpha = np.vdot(r_signed[valid], fvals[valid]) / den if den > np.finfo(float).eps else 1.0 + 0.0j
                corr_line[:, z_r, f_ix] = alpha * r_signed
        sqrt_Dgamma_lines_corr[line_ix] = corr_line

    n_points = full_flat.shape[0]
    sqrt_Dgamma_corr_pts = np.full((n_points, Zred, F), np.nan + 0j, dtype=np.complex128)
    for line_ix, idxs in enumerate(idxs_sorted_lines):
        if idxs is None or sqrt_Dgamma_lines_corr[line_ix] is None:
            continue
        sqrt_Dgamma_corr_pts[np.asarray(idxs, dtype=int), :, :] = sqrt_Dgamma_lines_corr[line_ix]
    sqrt_Dgamma_reduced_corr = sqrt_Dgamma_corr_pts[all_ixs, :, :]
    return sqrt_Dgamma_reduced_corr, sqrt_Dgamma_lines_corr


def hermite_0(_, __):
    return 1.0


def hermite_1(x, s):
    return 2.0 * x / s


def hermite_2(x, s):
    return 4.0 * (x / s) ** 2 - 2.0


H_POLYN = (hermite_0, hermite_1, hermite_2)


def beam_model(x, y, x0, y0, A, s, kappa, phi0, *weights):
    degree = 1
    s_eff = max(abs(float(s)), 1e-12)
    kappa = float(kappa)
    phi0 = float(phi0)
    x_shift = x - x0
    y_shift = y - y0
    envelope = np.exp(-(x_shift**2 + y_shift**2) / (s_eff**2))

    n_modes = (degree + 1) ** 2
    if len(weights) == 0:
        w = np.zeros(n_modes, dtype=float)
        w[0] = 1.0
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size < n_modes:
            w = np.pad(w, (0, n_modes - w.size), mode="constant")
        elif w.size > n_modes:
            w = w[:n_modes]
        norm = np.linalg.norm(w)
        if not np.isfinite(norm) or norm <= 0:
            w = np.zeros(n_modes, dtype=float)
            w[0] = 1.0
        else:
            w = w / norm

    hg_sum = np.zeros_like(x_shift, dtype=float)
    k = 0
    for i in range(degree + 1):
        hx = H_POLYN[i](x_shift, s_eff)
        for j in range(degree + 1):
            hy = H_POLYN[j](y_shift, s_eff)
            hg_sum += w[k] * hx * hy
            k += 1
    phase = np.exp(1j * (kappa * (x_shift**2 + y_shift**2) + phi0))
    return A * envelope * hg_sum * phase


class BeamFit:
    def __init__(
        self,
        sqrt_Dgamma_reduced,
        frequencies,
        x_reduced,
        y_reduced,
        kappa_max=800.0,
        kappa_reg=5e-4,
        center_reg=3e-3,
        s_bounds=(8e-3, 0.12),
        alpha_max=1e4,
        model_norm_floor=1e-9,
        alpha_reg=1e-4,
        s_reg=1e-4,
    ):
        self.sqrt_Dgamma_reduced = sqrt_Dgamma_reduced
        self.frequencies = frequencies
        self.x_reduced = np.asarray(x_reduced, dtype=float).ravel()
        self.y_reduced = np.asarray(y_reduced, dtype=float).ravel()
        self.kappa_max = float(kappa_max)
        self.kappa_reg = float(kappa_reg)
        self.center_reg = float(center_reg)
        self.alpha_max = float(alpha_max)
        self.model_norm_floor = float(model_norm_floor)
        self.alpha_reg = float(alpha_reg)
        self.s_reg = float(s_reg)

        self.x_min = float(np.nanmin(self.x_reduced))
        self.x_max = float(np.nanmax(self.x_reduced))
        self.y_min = float(np.nanmin(self.y_reduced))
        self.y_max = float(np.nanmax(self.y_reduced))
        self.x_center = 0.5 * (self.x_min + self.x_max)
        self.y_center = 0.5 * (self.y_min + self.y_max)
        self.x_scale = max(0.5 * (self.x_max - self.x_min), 1e-12)
        self.y_scale = max(0.5 * (self.y_max - self.y_min), 1e-12)

        if s_bounds is None:
            dx = np.diff(np.unique(np.sort(self.x_reduced)))
            dy = np.diff(np.unique(np.sort(self.y_reduced)))
            dx_min = float(np.nanmin(dx)) if dx.size else (self.x_max - self.x_min) / 20.0
            dy_min = float(np.nanmin(dy)) if dy.size else (self.y_max - self.y_min) / 20.0
            s_min = max(0.5 * min(abs(dx_min), abs(dy_min)), 1e-6)
            s_max = max(0.75 * np.hypot(self.x_max - self.x_min, self.y_max - self.y_min), s_min * 1.1)
        else:
            s_min, s_max = map(float, s_bounds)
            if not np.isfinite(s_min) or not np.isfinite(s_max) or s_min <= 0 or s_max <= s_min:
                raise ValueError("Invalid s_bounds. Expected (s_min, s_max) with 0 < s_min < s_max.")
        self.s_min = float(s_min)
        self.s_max = float(s_max)

    def _build_bounds(self, n_params):
        bounds = [
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (0.0, np.inf),
            (self.s_min, self.s_max),
        ]
        if n_params >= 5:
            bounds.append((-self.kappa_max, self.kappa_max))
        if n_params >= 6:
            bounds.append((-np.pi, np.pi))
        if n_params > 6:
            bounds.extend([(-np.inf, np.inf)] * (n_params - 6))
        return bounds

    @staticmethod
    def _best_complex_scale(model, data, alpha_max=1e4):
        den = np.vdot(model, model).real
        if (not np.isfinite(den)) or (den <= np.finfo(float).eps):
            return 1.0 + 0.0j
        alpha = np.vdot(model, data) / den
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
            return 1.0 + 0.0j
        aabs = np.abs(alpha)
        amax = float(max(alpha_max, 1.0))
        if aabs > amax:
            alpha = alpha * (amax / aabs)
        return alpha

    @staticmethod
    def _wrap_phase_pi(phi):
        return (phi + np.pi) % (2.0 * np.pi) - np.pi

    def _absorb_alpha_into_params(self, fit_param, z_ix, f_ix):
        p = np.asarray(fit_param, dtype=float).copy()
        x0, y0, A, s = p[:4]
        kappa = float(p[4]) if p.size >= 5 else 0.0
        phi0 = float(p[5]) if p.size >= 6 else 0.0
        model_base = beam_model(self.x_reduced, self.y_reduced, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
        data = self.sqrt_Dgamma_reduced[:, z_ix, f_ix]
        mask = np.isfinite(data) & np.isfinite(model_base)
        alpha = self._best_complex_scale(model_base[mask], data[mask], alpha_max=self.alpha_max) if np.any(mask) else 1.0 + 0.0j
        p[2] = p[2] * np.abs(alpha)
        if p.size >= 6:
            p[5] = self._wrap_phase_pi(p[5] + float(np.angle(alpha)))
        return p, alpha

    def cost(self, fit_param, z_ix, f_ix):
        x0, y0, A, s = fit_param[:4]
        kappa = float(fit_param[4]) if len(fit_param) >= 5 else 0.0
        phi0 = float(fit_param[5]) if len(fit_param) >= 6 else 0.0
        model_base = beam_model(self.x_reduced, self.y_reduced, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
        data = self.sqrt_Dgamma_reduced[:, z_ix, f_ix]
        mask = np.isfinite(data) & np.isfinite(model_base)
        if not np.any(mask):
            return np.inf

        model_rms = np.sqrt(np.nanmean(np.abs(model_base[mask]) ** 2))
        if (not np.isfinite(model_rms)) or (model_rms < self.model_norm_floor):
            return np.inf
        alpha = self._best_complex_scale(model_base[mask], data[mask], alpha_max=self.alpha_max)
        model_output = alpha * model_base
        mse = np.nanmean(np.abs(data - model_output) ** 2)
        reg_kappa = self.kappa_reg * (kappa / max(self.kappa_max, 1e-12)) ** 2
        reg_center = self.center_reg * (
            ((x0 - self.x_center) / self.x_scale) ** 2
            + ((y0 - self.y_center) / self.y_scale) ** 2
        )
        s_mid = 0.5 * (self.s_min + self.s_max)
        s_span = max(0.5 * (self.s_max - self.s_min), 1e-12)
        reg_s = self.s_reg * ((s - s_mid) / s_span) ** 2
        reg_alpha = self.alpha_reg * (np.abs(alpha) / max(self.alpha_max, 1.0)) ** 2
        return float(mse + reg_kappa + reg_center + reg_s + reg_alpha)

    def fit_slice(self, start, z_ix, f_ix, disp=False):
        bounds = self._build_bounds(len(start))
        start = np.asarray(start, dtype=float).copy()
        start[0] = np.clip(start[0], self.x_min, self.x_max)
        start[1] = np.clip(start[1], self.y_min, self.y_max)
        start[3] = np.clip(start[3], self.s_min, self.s_max)
        if start.size >= 5:
            start[4] = np.clip(start[4], -self.kappa_max, self.kappa_max)
        if start.size >= 6:
            start[5] = np.clip(start[5], -np.pi, np.pi)
        res = minimize(
            self.cost,
            start,
            method="L-BFGS-B",
            args=(z_ix, f_ix),
            bounds=bounds,
            options={"maxiter": 500},
        )
        res.x_raw = np.asarray(res.x, dtype=float).copy()
        p_absorbed, alpha = self._absorb_alpha_into_params(res.x, z_ix, f_ix)
        res.alpha = alpha
        res.x = p_absorbed
        if disp:
            print(f"z={z_ix}, f={f_ix}, fit={res.x}, cost={res.fun}, success={res.success}, alpha={res.alpha}")
        return res

    def fit_all(self, start=None, disp=False, n_workers=None, backend="thread"):
        if start is None:
            s0 = float(np.clip(0.05, self.s_min, self.s_max))
            start = np.array([self.x_center, self.y_center, 4e3, s0, 0.0, -np.pi / 2.0], dtype=float)
        else:
            start = np.asarray(start, dtype=float)
        Z = self.sqrt_Dgamma_reduced.shape[1]
        F = self.sqrt_Dgamma_reduced.shape[2]
        self.params = np.full((Z, F, start.size), np.nan)
        self.costs = np.full((Z, F), np.nan)
        self.success = np.zeros((Z, F), dtype=bool)
        n_tasks = Z * F
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        n_workers = int(max(1, min(n_workers, n_tasks)))
        if backend not in ("thread", "sequential"):
            raise ValueError("backend must be one of {'thread', 'sequential'}")

        print(f"Using backend='{backend}' with n_workers={n_workers} for fitting {n_tasks} tasks.")
        progress_bar = tqdm(total=n_tasks, desc="E-field fit", unit="fit")
        if backend == "thread" and n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.fit_slice, start, z, f, False): (z, f)
                    for z in range(Z)
                    for f in range(F)
                }
                for future in as_completed(futures):
                    z, f = futures[future]
                    res = future.result()
                    self.params[z, f, :] = res.x
                    self.costs[z, f] = res.fun
                    self.success[z, f] = res.success
                    progress_bar.set_postfix(z=z, f=f, cost=float(res.fun))
                    progress_bar.update(1)
        else:
            for z in range(Z):
                for f in range(F):
                    res = self.fit_slice(start, z, f, False)
                    self.params[z, f, :] = res.x
                    self.costs[z, f] = res.fun
                    self.success[z, f] = res.success
                    progress_bar.set_postfix(z=z, f=f, cost=float(res.fun))
                    progress_bar.update(1)
        progress_bar.close()
        if disp and F > DEFAULT_F_IX:
            for z in range(Z):
                tqdm.write(
                    f"z={z}, f={self.frequencies[DEFAULT_F_IX] * 1e-9:.2f} GHz, "
                    f"fit={self.params[z, DEFAULT_F_IX, :]}, "
                    f"cost={self.costs[z, DEFAULT_F_IX]}, success={self.success[z, DEFAULT_F_IX]}"
                )
        return self.params, self.costs, self.success


def compute_ET_reduced(
    params,
    x_pos,
    y_pos,
    beam_model_func,
    frequencies,
    fit_costs=None,
    fit_success=None,
    cost_quantile=None,
    max_abs_A=None,
    max_abs_kappa=None,
):
    n1 = 50
    n2 = 50
    x_lin = np.linspace(np.min(x_pos), np.max(x_pos), n1)
    y_lin = np.linspace(np.min(y_pos), np.max(y_pos), n2)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    Z, Nf = params.shape[0], params.shape[1]
    ET_reduced = np.full((Z, Nf), np.nan + 0j, dtype=np.complex128)
    good_mask = np.ones((Z, Nf), dtype=bool)
    if fit_success is not None:
        fit_success = np.asarray(fit_success, dtype=bool)
        if fit_success.shape == (Z, Nf):
            good_mask &= fit_success
    if (fit_costs is not None) and (cost_quantile is not None):
        fit_costs = np.asarray(fit_costs, dtype=float)
        if fit_costs.shape == (Z, Nf):
            finite_costs = fit_costs[np.isfinite(fit_costs)]
            if finite_costs.size > 0:
                cost_thr = np.nanquantile(finite_costs, float(cost_quantile))
                good_mask &= np.isfinite(fit_costs) & (fit_costs <= cost_thr)

    for f_ix in range(Nf):
        for z_ix in range(Z):
            if not good_mask[z_ix, f_ix]:
                continue
            p = params[z_ix, f_ix, :]
            if p.size < 4 or not np.all(np.isfinite(p[:4])):
                continue
            x0, y0, A, s = map(float, p[:4])
            kappa = float(p[4]) if p.size >= 5 and np.isfinite(p[4]) else 0.0
            phi0 = float(p[5]) if p.size >= 6 and np.isfinite(p[5]) else 0.0
            if (s <= 0.0) or (not np.isfinite(s)):
                continue
            if (max_abs_A is not None) and (np.abs(A) > float(max_abs_A)):
                continue
            if (max_abs_kappa is not None) and (np.abs(kappa) > float(max_abs_kappa)):
                continue
            E_field = beam_model_func(X, Y, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
            if not np.all(np.isfinite(E_field)):
                continue
            ET_reduced[z_ix, f_ix] = TRAPEZOID(TRAPEZOID(E_field, y_lin, axis=1), x_lin, axis=0)
    return ET_reduced


def compute_reduced_products(
    file_path,
    line_count,
    z_ixs_used,
    n_baseline_margin=DEFAULT_N_BASELINE_MARGIN,
    time_gating=True,
    t_width=DEFAULT_T_WIDTH,
    t_offset=DEFAULT_T_OFFSET,
    frequency_indices=None,
):
    data = load_file(
        file_path,
        z_ixs_used=z_ixs_used,
        time_gating=time_gating,
        t_width=t_width,
        t_offset=t_offset,
        frequency_indices=frequency_indices,
    )
    s11_reduced, x_reduced, y_reduced, lines, all_ixs, per_line_ixs = reduce_points_to_lines(
        data["s11_z_flat"], data["x_pts"], data["y_pts"], n_lines=line_count
    )
    sqrt_Dgamma_reduced, sqrt_Dgamma_lines, baseline_lines, s_sorted_lines, idxs_sorted_lines = compute_sqrt_Dgamma_lines(
        data["s11_z_flat"],
        data["x_pts"],
        data["y_pts"],
        lines,
        per_line_ixs,
        all_ixs,
        n_baseline_margin,
    )
    sqrt_dgamma_full = compute_sqrt_Dgamma_full(data["s11"], data["x_pos"], data["y_pos"], data["z_pos"], n_baseline_margin)
    sqrt_Dgamma_reduced_corr, sqrt_Dgamma_lines_corr = correct_sqrt_dgamma_reduced_with_full(
        sqrt_dgamma_full,
        sqrt_Dgamma_lines,
        idxs_sorted_lines,
        all_ixs,
        data["z_ixs_used"],
    )
    E_reduced = sqrt_dgamma_to_E(sqrt_Dgamma_reduced_corr, data["frequencies"])
    E_full = sqrt_dgamma_to_E(sqrt_dgamma_full, data["frequencies"])
    data.update(
        {
            "s11_reduced": s11_reduced,
            "x_reduced": x_reduced,
            "y_reduced": y_reduced,
            "lines": lines,
            "all_ixs": all_ixs,
            "per_line_ixs": per_line_ixs,
            "sqrt_Dgamma_reduced": sqrt_Dgamma_reduced,
            "sqrt_Dgamma_lines": sqrt_Dgamma_lines,
            "sqrt_Dgamma_lines_corr": sqrt_Dgamma_lines_corr,
            "baseline_lines": baseline_lines,
            "s_sorted_lines": s_sorted_lines,
            "idxs_sorted_lines": idxs_sorted_lines,
            "sqrt_dgamma_full": sqrt_dgamma_full,
            "sqrt_Dgamma_reduced_corr": sqrt_Dgamma_reduced_corr,
            "E_reduced": E_reduced,
            "E_full": E_full,
        }
    )
    return data


def fit_cache_path(fit_cache_dir, line_count, gap_rank):
    return Path(fit_cache_dir) / f"line_{int(line_count):02d}_gap_{int(gap_rank):02d}.pkl"


def _normalized_path_string(path):
    return str(Path(path).resolve())


def _normalized_int_tuple(values):
    if values is None:
        return tuple()
    return tuple(int(v) for v in np.asarray(values, dtype=int).ravel())


def _fit_result_matches_expected(cached, expected_file_path, expected_z_ixs=None):
    actual = cached.get("file_path")
    if actual is None or _normalized_path_string(actual) != _normalized_path_string(expected_file_path):
        return False
    if expected_z_ixs is None:
        return True
    return _normalized_int_tuple(cached.get("z_ixs_used")) == _normalized_int_tuple(expected_z_ixs)


def _fit_cache_matches_expected(cache_path, expected_file_path, expected_z_ixs=None):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        with open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
    except Exception:
        return False
    return _fit_result_matches_expected(cached, expected_file_path, expected_z_ixs=expected_z_ixs)


def _npz_string_value(npz_file, key):
    value = npz_file[key]
    return str(value.item()) if np.ndim(value) == 0 else str(value)


def _full_cache_matches_expected(cache_path, expected_file_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path) as cached:
            for key in ("source_file", "file_path", "file"):
                if key in cached.files:
                    return _normalized_path_string(_npz_string_value(cached, key)) == _normalized_path_string(
                        expected_file_path
                    )
    except Exception:
        return False
    return False


def load_fit_result(fit_cache_dir, line_count, gap_rank, expected_file_path=None, expected_z_ixs=None):
    path = fit_cache_path(fit_cache_dir, line_count, gap_rank)
    with open(path, "rb") as handle:
        cached = pickle.load(handle)
    if expected_file_path is not None and not _fit_result_matches_expected(
        cached, expected_file_path, expected_z_ixs=expected_z_ixs
    ):
        raise ValueError(
            f"Fit cache {path} does not match the expected source file/z-slice selection "
            f"({expected_file_path}, {expected_z_ixs})."
        )
    return cached


def fit_gap_line_count(
    file_path,
    line_count,
    gap_rank,
    fit_cache_dir,
    z_ixs_used=(0, 1, 2),
    n_baseline_margin=DEFAULT_N_BASELINE_MARGIN,
    n_workers=None,
    backend="thread",
    force=False,
    frequency_indices=None,
):
    fit_cache_dir = Path(fit_cache_dir)
    fit_cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = fit_cache_path(fit_cache_dir, line_count, gap_rank)
    if output_path.exists() and not force:
        if _fit_cache_matches_expected(output_path, file_path, expected_z_ixs=z_ixs_used):
            print(f"Fit cache exists, skipping: {output_path}")
            return output_path
        print(f"Fit cache is stale for {output_path}, recomputing.")

    products = compute_reduced_products(
        file_path,
        line_count=line_count,
        z_ixs_used=z_ixs_used,
        n_baseline_margin=n_baseline_margin,
        frequency_indices=frequency_indices,
    )
    beam_fit = BeamFit(
        products["E_reduced"],
        products["frequencies"],
        products["x_reduced"],
        products["y_reduced"],
        kappa_max=800.0,
        kappa_reg=5e-4,
        center_reg=3e-3,
        s_bounds=(8e-3, 0.12),
        alpha_max=1e4,
        model_norm_floor=1e-9,
        alpha_reg=1e-4,
        s_reg=1e-4,
    )
    params, costs, success = beam_fit.fit_all(backend=backend, n_workers=n_workers)
    result = {
        "line_count": int(line_count),
        "gap_rank": int(gap_rank),
        "file_path": str(Path(file_path)),
        "z_ixs_used": tuple(int(v) for v in z_ixs_used),
        "frequencies": products["frequencies"],
        "params": params,
        "costs": costs,
        "success": success,
        "n_reduced_points": int(products["x_reduced"].size),
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(result, handle)
    tmp_path.replace(output_path)
    print(f"Wrote fit cache: {output_path}")
    return output_path


def all_fit_caches_exist(
    gaps,
    line_counts=DEFAULT_LINE_COUNTS,
    fit_cache_dir="data/adaptive_bead_pull_variations/fits",
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
):
    missing = []
    for line_count in line_counts:
        for gap in gaps:
            path = fit_cache_path(fit_cache_dir, line_count, gap.rank)
            expected_z_ixs = select_gap_fit_z_ixs(gap, z_ix_options=z_ix_options)
            if not _fit_cache_matches_expected(path, gap.path, expected_z_ixs=expected_z_ixs):
                missing.append(path)
    return len(missing) == 0, missing


def _json_task_records(gaps, line_counts, fit_cache_dir, n_baseline_margin, z_ix_options):
    records = []
    for line_count in line_counts:
        for gap in gaps:
            records.append(
                {
                    "line_count": int(line_count),
                    "gap_rank": int(gap.rank),
                    "file_path": str(gap.path.resolve()),
                    "fit_cache_dir": str(Path(fit_cache_dir).resolve()),
                    "n_baseline_margin": int(n_baseline_margin),
                    "z_ixs_used": list(map(int, select_gap_fit_z_ixs(gap, z_ix_options=z_ix_options))),
                }
            )
    return records


def write_slurm_fit_files(
    repo_root,
    gaps,
    fit_cache_dir,
    slurm_dir,
    line_counts=DEFAULT_LINE_COUNTS,
    partition="maxcpu",
    cpus_per_task=40,
    time_limit="04:00:00",
    mem="16G",
    python_executable=None,
    n_baseline_margin=DEFAULT_N_BASELINE_MARGIN,
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
):
    repo_root = Path(repo_root).resolve()
    slurm_dir = Path(slurm_dir).resolve()
    slurm_dir.mkdir(parents=True, exist_ok=True)
    fit_cache_dir = Path(fit_cache_dir).resolve()
    fit_cache_dir.mkdir(parents=True, exist_ok=True)
    python_executable = python_executable or sys.executable

    records = _json_task_records(gaps, line_counts, fit_cache_dir, n_baseline_margin, z_ix_options)
    task_config = slurm_dir / "fit_tasks.json"
    task_config.write_text(json.dumps(records, indent=2), encoding="utf-8")

    script_path = slurm_dir / "fit_array.sbatch"
    array_max = len(records) - 1
    script = f"""#!/bin/bash
#SBATCH --job-name=abp-fit
#SBATCH --partition={partition}
#SBATCH --array=0-{array_max}
#SBATCH --cpus-per-task={int(cpus_per_task)}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={slurm_dir}/abp-fit-%A_%a.out
#SBATCH --error={slurm_dir}/abp-fit-%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd {repo_root}
{python_executable} -m src.adaptive_bead_pull fit-job \\
  --task-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --task-config {task_config} \\
  --n-workers "${{SLURM_CPUS_PER_TASK}}" \\
  --backend thread
"""
    script_path.write_text(script, encoding="utf-8")
    return {"task_config": task_config, "script": script_path, "n_tasks": len(records)}


def submit_slurm_fit_array(
    repo_root,
    gaps,
    fit_cache_dir,
    slurm_dir,
    line_counts=DEFAULT_LINE_COUNTS,
    partition="maxcpu",
    cpus_per_task=40,
    time_limit="04:00:00",
    mem="16G",
    python_executable=None,
    force_submit=False,
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
):
    complete, missing = all_fit_caches_exist(
        gaps,
        line_counts=line_counts,
        fit_cache_dir=fit_cache_dir,
        z_ix_options=z_ix_options,
    )
    if complete and not force_submit:
        return {"submitted": False, "reason": "all caches already exist", "missing": []}
    files = write_slurm_fit_files(
        repo_root=repo_root,
        gaps=gaps,
        fit_cache_dir=fit_cache_dir,
        slurm_dir=slurm_dir,
        line_counts=line_counts,
        partition=partition,
        cpus_per_task=cpus_per_task,
        time_limit=time_limit,
        mem=mem,
        python_executable=python_executable,
        z_ix_options=z_ix_options,
    )
    proc = subprocess.run(["sbatch", str(files["script"])], check=True, text=True, capture_output=True)
    stdout = proc.stdout.strip()
    job_id = stdout.split()[-1] if stdout else None
    return {
        "submitted": True,
        "job_id": job_id,
        "stdout": stdout,
        "missing_before_submit": [str(path) for path in missing],
        **files,
    }


def wait_for_fit_caches(
    gaps,
    fit_cache_dir,
    line_counts=DEFAULT_LINE_COUNTS,
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
    poll_seconds=60,
    timeout_seconds=None,
):
    start = time.time()
    while True:
        complete, missing = all_fit_caches_exist(
            gaps,
            line_counts=line_counts,
            fit_cache_dir=fit_cache_dir,
            z_ix_options=z_ix_options,
        )
        if complete:
            print("All fit caches are present.")
            return True
        elapsed = time.time() - start
        print(f"Waiting for {len(missing)} fit caches after {elapsed / 60:.1f} min...")
        if timeout_seconds is not None and elapsed > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for fit caches. Missing examples: {missing[:5]}")
        time.sleep(float(poll_seconds))


def compute_full_reference(
    gaps,
    cache_dir,
    n_baseline_margin=DEFAULT_N_BASELINE_MARGIN,
    frequency_indices=None,
    force=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ETs_full = []
    z_full = []
    frequencies = None
    for gap in gaps:
        cache_path = cache_dir / f"full_gap_{gap.rank:02d}.npz"
        if cache_path.exists() and not force and _full_cache_matches_expected(cache_path, gap.path):
            with np.load(cache_path) as cached:
                ET_gap = cached["ET_full"].copy()
                z_gap = cached["z_full"].copy()
                freq_gap = cached["frequencies"].copy()
        else:
            if cache_path.exists() and not force:
                print(f"Full-reference cache is stale for {cache_path}, recomputing.")
            data = load_file(gap.path, z_ixs_used=(0,), frequency_indices=frequency_indices)
            sqrt_full = compute_sqrt_Dgamma_full(data["s11"], data["x_pos"], data["y_pos"], data["z_pos"], n_baseline_margin)
            E_full = sqrt_dgamma_to_E(sqrt_full, data["frequencies"])
            ET_gap = compute_ET_full(E_full, data["x_pos"], data["y_pos"], data["frequencies"])
            z_gap = data["z_pos"]
            freq_gap = data["frequencies"]
            np.savez(cache_path, ET_full=ET_gap, z_full=z_gap, frequencies=freq_gap, source_file=str(gap.path.resolve()))
        frequencies = freq_gap if frequencies is None else frequencies
        ETs_full.extend(list(ET_gap))
        z_full.extend(list(z_gap))

    ETs_full = np.asarray(ETs_full)
    z_full = np.asarray(z_full).flatten()
    order = np.argsort(z_full)
    return {
        "ETs_full": ETs_full[order, ...],
        "z_full": z_full[order],
        "frequencies": np.asarray(frequencies),
    }


def reset_generated_outputs(result_root="data/adaptive_bead_pull_variations", figs_root="figs", keep_diagnostics=True):
    result_root = Path(result_root)
    figs_root = Path(figs_root)
    removed = []
    targets = [
        result_root / "results",
        result_root / "metrics.csv",
        figs_root / "configs",
        figs_root / "comparisons",
    ]
    if not keep_diagnostics:
        targets.append(figs_root / "diagnostics")

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
        elif target.exists():
            target.unlink()
            removed.append(target)
    return removed


def prune_stale_gap_caches(
    gaps,
    fit_cache_dir="data/adaptive_bead_pull_variations/fits",
    full_cache_dir="data/adaptive_bead_pull_variations/full_reference",
    line_counts=DEFAULT_LINE_COUNTS,
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
):
    fit_cache_dir = Path(fit_cache_dir)
    full_cache_dir = Path(full_cache_dir)
    removed = []

    for line_count in line_counts:
        for gap in gaps:
            cache_path = fit_cache_path(fit_cache_dir, line_count, gap.rank)
            expected_z_ixs = select_gap_fit_z_ixs(gap, z_ix_options=z_ix_options)
            if cache_path.exists() and not _fit_cache_matches_expected(
                cache_path, gap.path, expected_z_ixs=expected_z_ixs
            ):
                cache_path.unlink()
                removed.append(cache_path)

    for gap in gaps:
        cache_path = full_cache_dir / f"full_gap_{gap.rank:02d}.npz"
        if cache_path.exists() and not _full_cache_matches_expected(cache_path, gap.path):
            cache_path.unlink()
            removed.append(cache_path)

    return removed


def _slice_fit_result(fit_result, z_slice_count):
    return (
        np.asarray(fit_result["params"])[:z_slice_count, :, :],
        np.asarray(fit_result["costs"])[:z_slice_count, :],
        np.asarray(fit_result["success"])[:z_slice_count, :],
    )


def match_full_to_reduced(ETs_full, z_full, z_reduced, atol=1e-9):
    z_full = np.asarray(z_full).ravel()
    z_reduced = np.asarray(z_reduced).ravel()
    ix = np.empty(z_reduced.size, dtype=int)
    for i, zr in enumerate(z_reduced):
        matches = np.flatnonzero(np.isclose(z_full, zr, atol=atol, rtol=0.0))
        ix[i] = int(matches[0]) if matches.size > 0 else int(np.argmin(np.abs(z_full - zr)))
    return np.asarray(ETs_full)[ix, :], ix


def config_slug(line_count, z_slice_count, gap_count):
    return f"lines_{int(line_count):02d}/zslices_{int(z_slice_count):02d}/gaps_{int(gap_count):02d}"


def save_figure(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_beam_model_components(params, x_plt, y_plt, z_ix=0, f_ix=DEFAULT_F_IX):
    init_params = np.asarray(params[z_ix, f_ix, :], dtype=float)
    x0, y0, A, s, kappa, phi0 = init_params[:6]
    field = beam_model(x_plt, y_plt, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].contourf(x_plt, y_plt, np.abs(field), levels=30, cmap="viridis")
    axes[0].set_title("|Beam model| (fit params)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].contourf(x_plt, y_plt, np.real(field), levels=30, cmap="RdBu_r")
    axes[1].set_title("Re(Beam model)")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].contourf(x_plt, y_plt, np.imag(field), levels=30, cmap="RdBu_r")
    axes[2].set_title("Im(Beam model)")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, axes


def create_hg_sanity_figures():
    x_lin = np.linspace(-0.15, 0.15, 200)
    y_lin = np.linspace(-0.15, 0.15, 200)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")
    modes = [
        ("HG00", (1.0, 0.0, 0.0, 0.0)),
        ("HG10", (0.0, 0.0, 1.0, 0.0)),
        ("HG01", (0.0, 1.0, 0.0, 0.0)),
        ("HG11", (0.0, 0.0, 0.0, 1.0)),
    ]
    fields = [(name, beam_model(X, Y, 0.0, 0.0, 1.0, 0.05, 400.0, 0.0, *w)) for name, w in modes]
    figures = []
    for component, title, cmap in [
        (lambda z: np.abs(z), "Absolute values of Hermite-Gaussian modes up to degree 1", "viridis"),
        (lambda z: np.real(z), "Real parts of Hermite-Gaussian modes up to degree 1", "coolwarm"),
        (lambda z: np.imag(z), "Imaginary parts of Hermite-Gaussian modes up to degree 1", "coolwarm"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, (name, field) in zip(axes.ravel(), fields):
            value = component(field)
            if cmap == "coolwarm":
                vmax = np.nanmax(np.abs(value))
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
                im = ax.contourf(X, Y, value, levels=40, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                im = ax.contourf(X, Y, value, levels=40, cmap=cmap)
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(im, ax=ax)
        fig.suptitle(title)
        figures.append(fig)
    return figures


def _metric_percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.nanmedian(values)), float(np.nanpercentile(values, 90))


def compute_metrics_record(
    line_count,
    z_slice_count,
    gap_count,
    n_reduced_points,
    ETs_reduced,
    ETs_full,
    z_full,
    z_reduced,
    frequencies,
    fit_costs,
    fit_success,
):
    full_overlap, _ = match_full_to_reduced(ETs_full, z_full, z_reduced)
    reduced = np.asarray(ETs_reduced)
    valid = np.isfinite(reduced) & np.isfinite(full_overlap)
    denom = np.linalg.norm(full_overlap[valid]) if np.any(valid) else np.nan
    complex_rmse = np.linalg.norm((reduced - full_overlap)[valid]) / denom if np.isfinite(denom) and denom > 0 else np.nan

    full_abs = np.abs(full_overlap)
    red_abs = np.abs(reduced)
    rel_abs = np.full_like(full_abs, np.nan, dtype=float)
    valid_abs = valid & (full_abs > np.finfo(float).eps)
    rel_abs[valid_abs] = np.abs(red_abs[valid_abs] - full_abs[valid_abs]) / full_abs[valid_abs]
    med_rel, p90_rel = _metric_percentiles(rel_abs)

    freq = np.asarray(frequencies)
    focus = (freq >= 19e9) & (freq <= 20e9)
    if np.any(focus):
        focus_valid = valid[:, focus]
        focus_full = full_overlap[:, focus]
        focus_red = reduced[:, focus]
        focus_denom = np.linalg.norm(focus_full[focus_valid]) if np.any(focus_valid) else np.nan
        complex_rmse_focus = (
            np.linalg.norm((focus_red - focus_full)[focus_valid]) / focus_denom
            if np.isfinite(focus_denom) and focus_denom > 0
            else np.nan
        )
        focus_rel = rel_abs[:, focus]
        med_rel_focus, p90_rel_focus = _metric_percentiles(focus_rel)
    else:
        complex_rmse_focus = np.nan
        med_rel_focus = np.nan
        p90_rel_focus = np.nan

    fit_costs = np.asarray(fit_costs, dtype=float)
    fit_success = np.asarray(fit_success, dtype=bool)
    finite_costs = fit_costs[np.isfinite(fit_costs)]
    return {
        "line_count": int(line_count),
        "z_slice_count": int(z_slice_count),
        "gap_count": int(gap_count),
        "n_reduced_points": int(n_reduced_points),
        "n_reduced_z": int(np.asarray(z_reduced).size),
        "fit_success_fraction": float(np.nanmean(fit_success)) if fit_success.size else np.nan,
        "fit_cost_median": float(np.nanmedian(finite_costs)) if finite_costs.size else np.nan,
        "fit_cost_p90": float(np.nanpercentile(finite_costs, 90)) if finite_costs.size else np.nan,
        "et_complex_nrmse": float(complex_rmse),
        "rel_abs_et_median": float(med_rel),
        "rel_abs_et_p90": float(p90_rel),
        "et_complex_nrmse_19_20ghz": float(complex_rmse_focus),
        "rel_abs_et_median_19_20ghz": float(med_rel_focus),
        "rel_abs_et_p90_19_20ghz": float(p90_rel_focus),
    }


def run_configuration(
    gaps,
    line_count,
    z_ixs_used,
    gap_count,
    fit_cache_dir,
    full_reference,
    result_dir,
    figs_root,
    fit_z_ix_options=DEFAULT_Z_IX_OPTIONS,
    n_baseline_margin=DEFAULT_N_BASELINE_MARGIN,
    f_ix=DEFAULT_F_IX,
    make_figures=True,
    frequency_indices=None,
):
    z_slice_count = normalize_z_slice_count(z_ixs_used)
    selected_gaps = list(gaps[: int(gap_count)])
    ETs_reduced = []
    z_reduced = []
    all_costs = []
    all_success = []
    n_reduced_points = None
    representative = None
    representative_min_z = None
    z_ixs_used_by_gap = []
    selected_gap_ranks = []

    for gap in selected_gaps:
        gap_z_ixs = select_gap_z_ixs(gap, z_slice_count)
        products = compute_reduced_products(
            gap.path,
            line_count=line_count,
            z_ixs_used=gap_z_ixs,
            n_baseline_margin=n_baseline_margin,
            frequency_indices=frequency_indices,
        )
        fit_result = load_fit_result(
            fit_cache_dir,
            line_count,
            gap.rank,
            expected_file_path=gap.path,
            expected_z_ixs=select_gap_fit_z_ixs(gap, z_ix_options=fit_z_ix_options),
        )
        fit_params, fit_costs, fit_success = _slice_fit_result(fit_result, z_slice_count)
        ET_gap_reduced = compute_ET_reduced(
            fit_params,
            products["x_pos"],
            products["y_pos"],
            beam_model,
            products["frequencies"],
            fit_costs=fit_costs,
            fit_success=fit_success,
            cost_quantile=None,
            max_abs_A=None,
            max_abs_kappa=None,
        )
        ETs_reduced.extend(list(ET_gap_reduced))
        z_reduced.extend(list(products["z_pos"][list(gap_z_ixs)]))
        all_costs.append(fit_costs)
        all_success.append(fit_success)
        z_ixs_used_by_gap.append(np.asarray(gap_z_ixs, dtype=int))
        selected_gap_ranks.append(int(gap.rank))
        if n_reduced_points is None:
            n_reduced_points = int(products["x_reduced"].size)
        gap_min_selected_z = float(np.min(products["z_pos"][list(gap_z_ixs)]))
        if representative is None or gap_min_selected_z < representative_min_z:
            representative = (products, fit_params, fit_costs, fit_success, gap, gap_z_ixs)
            representative_min_z = gap_min_selected_z

    ETs_reduced = np.asarray(ETs_reduced)
    z_reduced = np.asarray(z_reduced).flatten()
    order = np.argsort(z_reduced)
    z_reduced = z_reduced[order]
    ETs_reduced = ETs_reduced[order, ...]

    ETs_full = np.asarray(full_reference["ETs_full"])
    z_full = np.asarray(full_reference["z_full"])
    frequencies = np.asarray(full_reference["frequencies"])
    costs_concat = np.concatenate([arr.reshape(-1) for arr in all_costs]) if all_costs else np.array([])
    success_concat = np.concatenate([arr.reshape(-1) for arr in all_success]) if all_success else np.array([], dtype=bool)

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"result_lines_{line_count:02d}_zslices_{z_slice_count:02d}_gaps_{gap_count:02d}.npz"
    np.savez(
        result_path,
        ETs_reduced=ETs_reduced,
        ETs_full=ETs_full,
        z_reduced=z_reduced,
        z_full=z_full,
        frequencies=frequencies,
        line_count=int(line_count),
        z_slice_count=int(z_slice_count),
        z_ixs_used=np.asarray(z_ixs_used_by_gap, dtype=int),
        selected_gap_ranks=np.asarray(selected_gap_ranks, dtype=int),
        gap_count=int(gap_count),
    )

    if make_figures:
        cfg_fig_dir = Path(figs_root) / "configs" / config_slug(line_count, z_slice_count, gap_count)
        cfg_fig_dir.mkdir(parents=True, exist_ok=True)
        products, fit_params, fit_costs, fit_success, gap, representative_gap_z_ixs = representative
        f_ix_plot = min(int(f_ix), products["frequencies"].size - 1)
        representative_z_values = np.asarray(products["z_pos"])[list(representative_gap_z_ixs)]
        representative_local_z_ix = int(np.argmin(representative_z_values))
        representative_full_z_ix = int(np.asarray(representative_gap_z_ixs, dtype=int)[representative_local_z_ix])

        fig, _ = plot_point_selection(
            products["s11"],
            products["frequencies"],
            products["all_ixs"],
            products["lines"],
            products["x_pos"],
            products["x_pts"],
            products["y_pos"],
            products["y_pts"],
            products["z_pos"],
            z_ix=representative_full_z_ix,
            f_ix=f_ix_plot,
            show_point_ix=False,
            show_line_ix=True,
        )
        save_figure(fig, cfg_fig_dir / "point_selection.png")

        for line_index in range(line_count):
            fig, _ = plot_sqrt_Dgamma_line(
                products["lines"],
                products["idxs_sorted_lines"],
                products["s_sorted_lines"],
                products["baseline_lines"],
                products["sqrt_Dgamma_lines"],
                line_index=line_index,
                n_baseline_margin=n_baseline_margin,
                s11_flat=products["s11_z_flat"],
                z_pos=products["z_pos"][list(representative_gap_z_ixs)],
                frequencies=products["frequencies"],
                z_ix=representative_local_z_ix,
                f_ix=f_ix_plot,
            )
            save_figure(fig, cfg_fig_dir / f"sqrt_dgamma_line_{line_index:02d}.png")

        fig_fit, _, fig_res, _ = plot_fitted_efield(
            fit_params,
            products["x_plt"],
            products["y_plt"],
            products["x_reduced"],
            products["y_reduced"],
            products["E_reduced"],
            products["E_full"][:, :, list(representative_gap_z_ixs), :],
            beam_model,
            products["z_pos"][list(representative_gap_z_ixs)],
            products["frequencies"],
            costs=fit_costs,
            z_ix=representative_local_z_ix,
            f_ix=f_ix_plot,
            overlay_points=True,
        )
        save_figure(fig_fit, cfg_fig_dir / "fitted_efield_comparison.png")
        save_figure(fig_res, cfg_fig_dir / "fitted_efield_residual.png")

        fig, _ = plot_sqrt_dgamma_before_after_correction(
            products["sqrt_dgamma_full"],
            products["sqrt_Dgamma_reduced"],
            products["sqrt_Dgamma_reduced_corr"],
            products["all_ixs"],
            reduced_z_ix=representative_local_z_ix,
            full_z_ix=representative_full_z_ix,
            f_ix=f_ix_plot,
            title_prefix=f"{gap.name}:",
        )
        save_figure(fig, cfg_fig_dir / "sqrt_dgamma_correction.png")

        fig, _ = plot_beam_model_components(
            fit_params,
            products["x_plt"],
            products["y_plt"],
            z_ix=representative_local_z_ix,
            f_ix=f_ix_plot,
        )
        save_figure(fig, cfg_fig_dir / "beam_model_components.png")

        fig, _ = plot_ET_complex_plane(ETs_full, ETs_reduced, z_full, z_reduced, f_ix=f_ix_plot)
        save_figure(fig, cfg_fig_dir / "et_complex_plane.png")
        fig, _ = plot_abs_ET_over_z(ETs_full, ETs_reduced, z_full, z_reduced, f_ix=f_ix_plot)
        save_figure(fig, cfg_fig_dir / "abs_et_over_z.png")
        (fig_abs, _), (fig_real, _), (fig_imag, _), _ = plot_abs_ET_grid_over_frequency(
            ETs_full,
            ETs_reduced,
            z_full,
            z_reduced,
            frequencies,
            max_et_for_scaling=DEFAULT_MAX_ET_FOR_SCALING,
            return_all=True,
        )
        save_figure(fig_abs, cfg_fig_dir / "abs_et_over_frequency.png")
        save_figure(fig_real, cfg_fig_dir / "real_et_over_frequency.png")
        save_figure(fig_imag, cfg_fig_dir / "imag_et_over_frequency.png")

    metrics = compute_metrics_record(
        line_count,
        z_slice_count,
        gap_count,
        n_reduced_points,
        ETs_reduced,
        ETs_full,
        z_full,
        z_reduced,
        frequencies,
        costs_concat,
        success_concat,
    )
    return {"metrics": metrics, "result_path": result_path}


def write_metrics_csv(metrics_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(metrics_rows)
    if not rows:
        raise ValueError("No metrics rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


ET_VARIATION_LOOKUP_COLUMNS = ("configuration_index", "line_count", "z_slice_count", "gap_count")


def _configuration_result_path(results_dir, line_count, z_slice_count, gap_count):
    return Path(results_dir) / f"result_lines_{line_count:02d}_zslices_{z_slice_count:02d}_gaps_{gap_count:02d}.npz"


def _arrays_equal(left, right):
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def _pad_variation_arrays(
    et_reduced_by_config,
    z_reduced_by_config,
    z_ixs_by_config,
    gap_ranks_by_config,
    n_frequency,
):
    n_config = len(et_reduced_by_config)
    max_z = max(np.asarray(values).shape[0] for values in et_reduced_by_config)
    max_gap_count = max((np.asarray(values).shape[0] for values in z_ixs_by_config), default=0)
    max_z_ix = max((np.asarray(values).shape[1] for values in z_ixs_by_config), default=0)

    et_padded = np.full((n_config, max_z, n_frequency), np.nan + 1j * np.nan, dtype=np.complex128)
    z_padded = np.full((n_config, max_z), np.nan, dtype=np.float64)
    z_count = np.zeros(n_config, dtype=np.int64)
    z_ixs_padded = np.full((n_config, max_gap_count, max_z_ix), -1, dtype=np.int64)
    z_ixs_count = np.zeros((n_config, max_gap_count), dtype=np.int64)
    gap_ranks_padded = np.full((n_config, max_gap_count), -1, dtype=np.int64)
    gap_ranks_count = np.zeros(n_config, dtype=np.int64)

    for config_index, (et_values, z_values, z_ixs, gap_ranks) in enumerate(
        zip(et_reduced_by_config, z_reduced_by_config, z_ixs_by_config, gap_ranks_by_config)
    ):
        et_values = np.asarray(et_values, dtype=np.complex128)
        z_values = np.asarray(z_values, dtype=np.float64)
        z_ixs = np.asarray(z_ixs, dtype=np.int64)
        gap_ranks = np.asarray(gap_ranks, dtype=np.int64)

        if et_values.ndim != 2:
            raise ValueError(f"ETs_reduced for configuration {config_index} must be 2D.")
        if et_values.shape[1] != n_frequency:
            raise ValueError(
                f"ETs_reduced for configuration {config_index} has {et_values.shape[1]} frequencies, "
                f"expected {n_frequency}."
            )
        if et_values.shape[0] != z_values.size:
            raise ValueError(
                f"ETs_reduced and z_reduced length mismatch for configuration {config_index}: "
                f"{et_values.shape[0]} vs {z_values.size}."
            )
        if z_ixs.ndim != 2:
            raise ValueError(f"z_ixs_used for configuration {config_index} must be 2D, got {z_ixs.ndim}D.")
        if gap_ranks.ndim != 1:
            raise ValueError(
                f"selected_gap_ranks for configuration {config_index} must be 1D, got {gap_ranks.ndim}D."
            )
        if z_ixs.shape[0] != gap_ranks.size:
            raise ValueError(
                f"z_ixs_used gap count mismatch for configuration {config_index}: "
                f"{z_ixs.shape[0]} vs {gap_ranks.size}."
            )

        n_z = z_values.size
        et_padded[config_index, :n_z, :] = et_values
        z_padded[config_index, :n_z] = z_values
        z_count[config_index] = n_z

        gap_count = gap_ranks.size
        if gap_count:
            z_ixs_padded[config_index, :gap_count, : z_ixs.shape[1]] = z_ixs
            z_ixs_count[config_index, :gap_count] = z_ixs.shape[1]
            gap_ranks_padded[config_index, :gap_count] = gap_ranks
        gap_ranks_count[config_index] = gap_count

    return et_padded, z_padded, z_count, z_ixs_padded, z_ixs_count, gap_ranks_padded, gap_ranks_count


def save_et_results_variations(
    result_root="data/adaptive_bead_pull_variations",
    output_path="data/ET_results_variations.npz",
    line_counts=DEFAULT_LINE_COUNTS,
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
    gap_counts=DEFAULT_GAP_COUNTS,
    gaps=None,
):
    result_root = Path(result_root)
    results_dir = result_root / "results"
    output_path = Path(output_path)
    line_counts = tuple(int(value) for value in line_counts)
    z_ix_options = tuple(z_ix_options)
    gap_counts = tuple(int(value) for value in gap_counts)
    gaps = list(discover_gap_files() if gaps is None else gaps)

    et_reduced_by_config = []
    z_reduced_by_config = []
    z_ixs_by_config = []
    gap_ranks_by_config = []
    lookup_rows = []
    shared = {}
    configuration_index = 0

    for line_count in line_counts:
        for z_slice_option in z_ix_options:
            z_slice_count = normalize_z_slice_count(z_slice_option)
            for gap_count in gap_counts:
                result_path = _configuration_result_path(results_dir, line_count, z_slice_count, gap_count)
                if not result_path.exists():
                    raise FileNotFoundError(f"Missing configuration result file: {result_path}")

                with np.load(result_path, allow_pickle=True) as data:
                    source_line_count = int(np.asarray(data["line_count"]).item())
                    source_z_slice_count = int(np.asarray(data["z_slice_count"]).item())
                    source_z_ixs = np.asarray(data["z_ixs_used"], dtype=int).copy()
                    source_gap_ranks = np.asarray(data["selected_gap_ranks"], dtype=int).copy()
                    source_gap_count = int(np.asarray(data["gap_count"]).item())

                    if source_line_count != line_count:
                        raise ValueError(f"{result_path} has line_count={source_line_count}, expected {line_count}.")
                    if source_z_slice_count != z_slice_count:
                        raise ValueError(
                            f"{result_path} has z_slice_count={source_z_slice_count}, expected {z_slice_count}."
                        )
                    if source_gap_count != gap_count:
                        raise ValueError(f"{result_path} has gap_count={source_gap_count}, expected {gap_count}.")

                    expected_gap_ranks = np.asarray([gap.rank for gap in gaps[:gap_count]], dtype=int)
                    expected_z_ixs = np.asarray(
                        [select_gap_z_ixs(gap, z_slice_count) for gap in gaps[:gap_count]], dtype=int
                    )
                    if not np.array_equal(source_gap_ranks, expected_gap_ranks):
                        raise ValueError(
                            f"{result_path} has selected_gap_ranks={source_gap_ranks}, "
                            f"expected {expected_gap_ranks}."
                        )
                    if not np.array_equal(source_z_ixs, expected_z_ixs):
                        raise ValueError(f"{result_path} has z_ixs_used={source_z_ixs}, expected {expected_z_ixs}.")

                    for key in ("ETs_full", "z_full", "frequencies"):
                        value = np.asarray(data[key]).copy()
                        if key not in shared:
                            shared[key] = value
                        elif not _arrays_equal(shared[key], value):
                            raise ValueError(f"{key} in {result_path} does not match the shared reference arrays.")

                    et_reduced_by_config.append(np.asarray(data["ETs_reduced"]).copy())
                    z_reduced_by_config.append(np.asarray(data["z_reduced"]).copy())
                    z_ixs_by_config.append(source_z_ixs)
                    gap_ranks_by_config.append(source_gap_ranks)

                lookup_rows.append((configuration_index, line_count, z_slice_count, gap_count))
                configuration_index += 1

    if not lookup_rows:
        raise ValueError("No configuration results were found to save.")

    et_padded, z_padded, z_count, z_ixs_padded, z_ixs_count, gap_ranks_padded, gap_ranks_count = (
        _pad_variation_arrays(
        et_reduced_by_config,
        z_reduced_by_config,
        z_ixs_by_config,
        gap_ranks_by_config,
        n_frequency=np.asarray(shared["frequencies"]).size,
    )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        schema_version=np.asarray([3], dtype=np.int64),
        ETs_reduced=et_padded,
        z_reduced=z_padded,
        z_count=z_count,
        ETs_full=np.asarray(shared["ETs_full"], dtype=np.complex128),
        z_full=np.asarray(shared["z_full"], dtype=np.float64),
        frequencies=np.asarray(shared["frequencies"], dtype=np.float64),
        configuration_indices=np.arange(len(lookup_rows), dtype=int),
        configuration_lookup=np.asarray(lookup_rows, dtype=int),
        z_ixs_used=z_ixs_padded,
        z_ixs_used_count=z_ixs_count,
        selected_gap_ranks=gap_ranks_padded,
        selected_gap_ranks_count=gap_ranks_count,
    )
    return output_path


def assemble_all_configurations(
    gaps,
    fit_cache_dir,
    full_reference,
    result_root="data/adaptive_bead_pull_variations",
    figs_root="figs",
    line_counts=DEFAULT_LINE_COUNTS,
    z_ix_options=DEFAULT_Z_IX_OPTIONS,
    gap_counts=DEFAULT_GAP_COUNTS,
    make_figures=True,
    frequency_indices=None,
):
    result_root = Path(result_root)
    results_dir = result_root / "results"
    metrics_rows = []
    total = len(line_counts) * len(z_ix_options) * len(gap_counts)
    with tqdm(total=total, desc="Configurations", unit="config") as bar:
        for line_count in line_counts:
            for z_slice_option in z_ix_options:
                for gap_count in gap_counts:
                    out = run_configuration(
                        gaps,
                        int(line_count),
                        z_slice_option,
                        int(gap_count),
                        fit_cache_dir=fit_cache_dir,
                        full_reference=full_reference,
                        result_dir=results_dir,
                        figs_root=figs_root,
                        fit_z_ix_options=z_ix_options,
                        make_figures=make_figures,
                        frequency_indices=frequency_indices,
                    )
                    metrics_rows.append(out["metrics"])
                    bar.update(1)
    metrics_path = write_metrics_csv(metrics_rows, result_root / "metrics.csv")
    plot_comparison_metrics(metrics_rows, Path(figs_root) / "comparisons")
    return metrics_rows, metrics_path


def _metric_grid(metrics_rows, metric_key, gap_count):
    line_counts = sorted({int(row["line_count"]) for row in metrics_rows})
    z_counts = sorted({int(row["z_slice_count"]) for row in metrics_rows})
    grid = np.full((len(line_counts), len(z_counts)), np.nan)
    for row in metrics_rows:
        if int(row["gap_count"]) != int(gap_count):
            continue
        i = line_counts.index(int(row["line_count"]))
        j = z_counts.index(int(row["z_slice_count"]))
        grid[i, j] = float(row[metric_key])
    return line_counts, z_counts, grid


def plot_metric_heatmaps(metrics_rows, metric_key, title, output_path):
    gap_counts = sorted({int(row["gap_count"]) for row in metrics_rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()
    finite_vals = [float(row[metric_key]) for row in metrics_rows if np.isfinite(float(row[metric_key]))]
    vmin = min(finite_vals) if finite_vals else None
    vmax = max(finite_vals) if finite_vals else None
    last_im = None
    for ax, gap_count in zip(axes, gap_counts):
        line_counts, z_counts, grid = _metric_grid(metrics_rows, metric_key, gap_count)
        last_im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(z_counts)), labels=z_counts)
        ax.set_yticks(range(len(line_counts)), labels=line_counts)
        ax.set_xlabel("z-slices per gap")
        ax.set_ylabel("line count")
        ax.set_title(f"gaps={gap_count}")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center", color="white", fontsize=8)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85)
    fig.suptitle(title)
    save_figure(fig, output_path)


def plot_metric_vs_gap(metrics_rows, metric_key, title, output_path):
    groups = sorted({(int(row["line_count"]), int(row["z_slice_count"])) for row in metrics_rows})
    fig, ax = plt.subplots(figsize=(10, 6))
    for line_count, z_count in groups:
        rows = [
            row
            for row in metrics_rows
            if int(row["line_count"]) == line_count and int(row["z_slice_count"]) == z_count
        ]
        rows = sorted(rows, key=lambda row: int(row["gap_count"]))
        ax.plot(
            [int(row["gap_count"]) for row in rows],
            [float(row[metric_key]) for row in rows],
            "o-",
            label=f"lines={line_count}, z={z_count}",
        )
    ax.set_xlabel("gap count")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_fit_summary(metrics_rows, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for metric_key, ax, title in [
        ("fit_success_fraction", axes[0], "Fit success fraction"),
        ("fit_cost_median", axes[1], "Median fit cost"),
    ]:
        for z_count in sorted({int(row["z_slice_count"]) for row in metrics_rows}):
            rows = [row for row in metrics_rows if int(row["z_slice_count"]) == z_count]
            rows = sorted(rows, key=lambda row: (int(row["gap_count"]), int(row["line_count"])))
            x = np.arange(len(rows))
            ax.plot(x, [float(row[metric_key]) for row in rows], "o-", label=f"z={z_count}")
        ax.set_title(title)
        ax.set_xlabel("configuration index")
        ax.set_ylabel(metric_key)
        ax.grid(True, alpha=0.3)
        ax.legend()
    save_figure(fig, output_path)


def plot_error_vs_points(metrics_rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    z_counts = sorted({int(row["z_slice_count"]) for row in metrics_rows})
    for z_count in z_counts:
        rows = [row for row in metrics_rows if int(row["z_slice_count"]) == z_count]
        ax.scatter(
            [int(row["n_reduced_points"]) * int(row["n_reduced_z"]) for row in rows],
            [float(row["et_complex_nrmse"]) for row in rows],
            s=[35 + 15 * int(row["gap_count"]) for row in rows],
            alpha=0.75,
            label=f"z-slices={z_count}",
        )
    ax.set_xlabel("reduced points x reduced z positions")
    ax.set_ylabel("ET complex NRMSE")
    ax.set_title("ET error vs reduced dataset size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_path)


def plot_comparison_metrics(metrics_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_heatmaps(
        metrics_rows,
        "et_complex_nrmse",
        "ET complex NRMSE by configuration",
        output_dir / "heatmap_et_complex_nrmse.png",
    )
    plot_metric_heatmaps(
        metrics_rows,
        "rel_abs_et_median",
        "Median relative |ET| error by configuration",
        output_dir / "heatmap_median_relative_abs_et_error.png",
    )
    plot_metric_vs_gap(
        metrics_rows,
        "et_complex_nrmse",
        "ET complex NRMSE vs gap count",
        output_dir / "et_complex_nrmse_vs_gap_count.png",
    )
    plot_metric_vs_gap(
        metrics_rows,
        "rel_abs_et_median",
        "Median relative |ET| error vs gap count",
        output_dir / "median_relative_abs_et_error_vs_gap_count.png",
    )
    plot_fit_summary(metrics_rows, output_dir / "fit_success_cost_summary.png")
    plot_error_vs_points(metrics_rows, output_dir / "error_vs_reduced_dataset_size.png")


def save_hg_sanity_figures(figs_root="figs"):
    diag_dir = Path(figs_root) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    names = ["hg_modes_abs.png", "hg_modes_real.png", "hg_modes_imag.png"]
    for fig, name in zip(create_hg_sanity_figures(), names):
        save_figure(fig, diag_dir / name)
    return [diag_dir / name for name in names]


def run_fit_task(task_index, task_config, n_workers=None, backend="thread", force=False):
    records = json.loads(Path(task_config).read_text(encoding="utf-8"))
    task = records[int(task_index)]
    return fit_gap_line_count(
        task["file_path"],
        line_count=int(task["line_count"]),
        gap_rank=int(task["gap_rank"]),
        fit_cache_dir=task["fit_cache_dir"],
        z_ixs_used=tuple(task.get("z_ixs_used", [0, 1, 2])),
        n_baseline_margin=int(task.get("n_baseline_margin", DEFAULT_N_BASELINE_MARGIN)),
        n_workers=n_workers,
        backend=backend,
        force=force,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Adaptive bead-pull variation helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_job = subparsers.add_parser("fit-job", help="Run one Slurm array fit task.")
    fit_job.add_argument("--task-index", type=int, required=True)
    fit_job.add_argument("--task-config", type=Path, required=True)
    fit_job.add_argument("--n-workers", type=int, default=None)
    fit_job.add_argument("--backend", choices=("thread", "sequential"), default="thread")
    fit_job.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "fit-job":
        run_fit_task(
            task_index=args.task_index,
            task_config=args.task_config,
            n_workers=args.n_workers,
            backend=args.backend,
            force=args.force,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
