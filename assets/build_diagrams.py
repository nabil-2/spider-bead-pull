#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import importlib
import json
import pickle
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ASSETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASSETS_DIR.parent
SOURCES_DIR = ASSETS_DIR / "sources"
NOTEBOOK_PATH = REPO_ROOT / "adaptive_bead_pull_variations.ipynb"
DATA_ROOT = REPO_ROOT / "data"
FIGS_ROOT = REPO_ROOT / "figs"
RESULTS_ROOT = DATA_ROOT / "adaptive_bead_pull_variations"

SLIDE_W_PT = 960
SLIDE_H_PT = 540
SLIDE_W_IN = SLIDE_W_PT / 72.0
SLIDE_H_IN = SLIDE_H_PT / 72.0

EXPECTED_METRICS_COLUMNS = [
    "line_count",
    "z_slice_count",
    "gap_count",
    "n_reduced_points",
    "n_reduced_z",
    "fit_success_fraction",
    "fit_cost_median",
    "fit_cost_p90",
    "et_complex_nrmse",
    "rel_abs_et_median",
    "rel_abs_et_p90",
    "et_complex_nrmse_19_20ghz",
    "rel_abs_et_median_19_20ghz",
    "rel_abs_et_p90_19_20ghz",
]
EXPECTED_FIT_KEYS = {
    "costs",
    "file_path",
    "frequencies",
    "gap_rank",
    "line_count",
    "n_reduced_points",
    "params",
    "success",
    "z_ixs_used",
}
EXPECTED_FULL_KEYS = {"ET_full", "frequencies", "source_file", "z_full"}
EXPECTED_RESULT_KEYS = {
    "ETs_full",
    "ETs_reduced",
    "frequencies",
    "gap_count",
    "line_count",
    "z_full",
    "z_ixs_used",
    "z_reduced",
}
EXPECTED_COMPARISON_FIGS = {
    "error_vs_reduced_dataset_size.png",
    "et_complex_nrmse_vs_gap_count.png",
    "fit_success_cost_summary.png",
    "heatmap_et_complex_nrmse.png",
    "heatmap_median_relative_abs_et_error.png",
    "median_relative_abs_et_error_vs_gap_count.png",
}
EXPECTED_DIAGNOSTIC_FIGS = {
    "hg_modes_abs.png",
    "hg_modes_imag.png",
    "hg_modes_real.png",
}
EXPECTED_COMMON_CONFIG_FIGS = {
    "abs_et_over_frequency.png",
    "abs_et_over_z.png",
    "beam_model_components.png",
    "et_complex_plane.png",
    "fitted_efield_comparison.png",
    "fitted_efield_residual.png",
    "imag_et_over_frequency.png",
    "point_selection.png",
    "real_et_over_frequency.png",
    "sqrt_dgamma_correction.png",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def import_abp():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("src.adaptive_bead_pull")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def gv_line(text: str) -> str:
    return esc(text).replace("\n", "<BR ALIGN=\"LEFT\"/>")


def mm(value_m: float) -> str:
    return f"{value_m * 1e3:.1f} mm"


def read_npz_dict(path: Path) -> dict:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def format_shape(value) -> str:
    if hasattr(value, "shape"):
        return "x".join(str(int(v)) for v in value.shape) if value.shape else "scalar"
    if isinstance(value, tuple):
        return repr(value)
    return type(value).__name__


def parse_notebook() -> dict:
    notebook = read_json(NOTEBOOK_PATH)
    headings = []
    code_cells = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    headings.append(stripped.lstrip("#").strip())
        elif cell["cell_type"] == "code":
            code_cells.append(source)

    constants_text = code_cells[1] if len(code_cells) > 1 else ""
    constants = {}
    pattern = re.compile(r"^([A-Z_]+)\s*=\s*(.+)$", re.MULTILINE)
    for name, value in pattern.findall(constants_text):
        constants[name] = value.strip()

    return {"headings": headings, "constants": constants}


def metric_int(row: dict, key: str) -> int:
    return int(row[key])


def metric_float(row: dict, key: str) -> float:
    return float(row[key])


def config_slug(line_count: int, z_slice_count: int, gap_count: int) -> str:
    return f"lines_{int(line_count):02d}/zslices_{int(z_slice_count):02d}/gaps_{int(gap_count):02d}"


def result_path_for_row(row: dict) -> Path:
    return (
        RESULTS_ROOT
        / "results"
        / f"result_lines_{metric_int(row, 'line_count'):02d}_zslices_{metric_int(row, 'z_slice_count'):02d}_gaps_{metric_int(row, 'gap_count'):02d}.npz"
    )


def figure_dir_for_row(row: dict) -> Path:
    return FIGS_ROOT / "configs" / config_slug(
        metric_int(row, "line_count"),
        metric_int(row, "z_slice_count"),
        metric_int(row, "gap_count"),
    )


def row_label(row: dict) -> str:
    return (
        f"lines={metric_int(row, 'line_count')}, "
        f"z-slices={metric_int(row, 'z_slice_count')}, "
        f"gaps={metric_int(row, 'gap_count')}"
    )


def representative_sort_key(row: dict) -> tuple[float, float, int, int, int]:
    return (
        metric_float(row, "et_complex_nrmse"),
        metric_float(row, "rel_abs_et_median"),
        -metric_int(row, "line_count"),
        -metric_int(row, "z_slice_count"),
        metric_int(row, "gap_count"),
    )


def choose_representatives(metrics_rows: list[dict]) -> dict:
    best_overall = min(metrics_rows, key=representative_sort_key)
    best_multi_gap = min(
        [row for row in metrics_rows if metric_int(row, "gap_count") >= 2],
        key=representative_sort_key,
    )
    max_coverage = max(
        metrics_rows,
        key=lambda row: (
            metric_int(row, "line_count"),
            metric_int(row, "z_slice_count"),
            metric_int(row, "gap_count"),
        ),
    )
    return {
        "best_overall": best_overall,
        "best_multi_gap": best_multi_gap,
        "max_coverage": max_coverage,
    }


def gather_repo_facts() -> dict:
    abp = import_abp()
    notebook = parse_notebook()
    measurement_dir = abp.find_measurements_directory()
    gaps = abp.discover_gap_files(measurement_dir)

    line_counts = tuple(int(v) for v in abp.DEFAULT_LINE_COUNTS)
    z_ix_options = tuple(tuple(int(v) for v in option) for option in abp.DEFAULT_Z_IX_OPTIONS)
    gap_counts = tuple(int(v) for v in abp.DEFAULT_GAP_COUNTS)
    default_lines = tuple((float(a), float(b)) for a, b in abp.DEFAULT_LINES)

    comparison_figs = sorted((FIGS_ROOT / "comparisons").glob("*.png"))
    diagnostic_figs = sorted((FIGS_ROOT / "diagnostics").glob("*.png"))
    config_dirs = sorted(path for path in (FIGS_ROOT / "configs").glob("lines_*/zslices_*/gaps_*") if path.is_dir())

    sample_result_path = max(
        sorted((RESULTS_ROOT / "results").glob("result_lines_*_zslices_*_gaps_*.npz")),
        key=lambda path: tuple(int(part) for part in re.findall(r"(\d+)", path.stem)),
    )
    sample_result = read_npz_dict(sample_result_path)

    sample_full_path = sorted((RESULTS_ROOT / "full_reference").glob("full_gap_*.npz"))[0]
    sample_full = read_npz_dict(sample_full_path)

    sample_fit_path = max(
        sorted((RESULTS_ROOT / "fits").glob("line_*_gap_*.pkl")),
        key=lambda path: tuple(int(part) for part in re.findall(r"(\d+)", path.stem)),
    )
    with open(sample_fit_path, "rb") as handle:
        sample_fit = pickle.load(handle)

    metrics_path = RESULTS_ROOT / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        metrics_rows = list(reader)
        metrics_columns = list(reader.fieldnames or [])

    points_by_line = {}
    for row in metrics_rows:
        line_count = metric_int(row, "line_count")
        points_by_line.setdefault(line_count, set()).add(metric_int(row, "n_reduced_points"))
    points_by_line = {
        line_count: sorted(values)[0]
        for line_count, values in sorted(points_by_line.items())
    }

    fig_count_by_config = Counter()
    for path in config_dirs:
        relative = path.relative_to(FIGS_ROOT / "configs")
        line_count = int(relative.parts[0].split("_")[1])
        fig_count_by_config[line_count] += 1

    legacy_top_level = sorted(path.name for path in DATA_ROOT.iterdir() if path.is_file())
    slurm_dir = RESULTS_ROOT / "slurm"
    slurm_files = sorted(path.name for path in slurm_dir.iterdir() if path.is_file())
    slurm_logs = [name for name in slurm_files if name.endswith(".out") or name.endswith(".err")]
    fit_tasks = json.loads((slurm_dir / "fit_tasks.json").read_text(encoding="utf-8"))

    notebook_headings = notebook["headings"]
    notebook_title = notebook_headings[0] if notebook_headings else "Notebook"
    notebook_sections = [heading for heading in notebook_headings if heading != notebook_title]

    representatives = choose_representatives(metrics_rows)
    representative_data = {}
    for name, row in representatives.items():
        result = read_npz_dict(result_path_for_row(row))
        matched_full, _ = abp.match_full_to_reduced(result["ETs_full"], result["z_full"], result["z_reduced"])
        representative_data[name] = {
            "row": row,
            "result": result,
            "matched_full": matched_full,
            "figure_dir": figure_dir_for_row(row),
        }

    grid_data = abp.load_file(gaps[0].path, z_ixs_used=(0,), frequency_indices=[abp.DEFAULT_F_IX])
    line_masks = {}
    for line_count in line_counts:
        _, _, _, _, all_ixs, _ = abp.reduce_points_to_lines(
            grid_data["s11_z_flat"],
            grid_data["x_pts"],
            grid_data["y_pts"],
            n_lines=line_count,
        )
        line_masks[line_count] = np.asarray(all_ixs, dtype=bool)

    raw_data = abp.load_file(gaps[0].path, z_ixs_used=(0,), time_gating=False)
    _, gate, taxis, t_bead_xyz = abp.apply_time_gating(
        raw_data["s11"],
        raw_data["frequencies"],
        raw_data["x_pos"],
        raw_data["y_pos"],
        raw_data["z_pos"],
    )
    s11_t = np.fft.fftshift(np.fft.ifft(raw_data["s11"], axis=-1), axes=-1)
    mid_x = raw_data["s11"].shape[0] // 2
    mid_y = raw_data["s11"].shape[1] // 2
    gate_trace = np.asarray(gate[mid_x, mid_y, 0, :], dtype=float)
    signal_trace = np.abs(s11_t[mid_x, mid_y, 0, :])
    signal_trace = signal_trace / np.nanmax(signal_trace)

    best_by_line_gap = {}
    for line_count in line_counts:
        for gap_count in gap_counts:
            subset = [
                row
                for row in metrics_rows
                if metric_int(row, "line_count") == line_count and metric_int(row, "gap_count") == gap_count
            ]
            best = min(
                subset,
                key=lambda row: (
                    metric_float(row, "et_complex_nrmse"),
                    metric_float(row, "rel_abs_et_median"),
                    metric_int(row, "z_slice_count"),
                ),
            )
            best_by_line_gap[(line_count, gap_count)] = best

    z_selection_examples = {}
    for z_choice in z_ix_options:
        z_values = []
        for gap in gaps:
            z_positions = np.asarray(gap.z_positions_m)
            z_values.extend(list(z_positions[list(z_choice)] * 1e3))
        z_selection_examples[len(z_choice)] = sorted(float(v) for v in z_values)

    return {
        "abp": abp,
        "measurement_dir": str(measurement_dir),
        "gaps": gaps,
        "line_counts": line_counts,
        "z_ix_options": z_ix_options,
        "gap_counts": gap_counts,
        "default_lines": default_lines,
        "default_f_ix": int(abp.DEFAULT_F_IX),
        "configuration_total": len(line_counts) * len(z_ix_options) * len(gap_counts),
        "notebook_title": notebook_title,
        "notebook_sections": notebook_sections,
        "notebook_constants": notebook["constants"],
        "comparison_figs": comparison_figs,
        "diagnostic_figs": diagnostic_figs,
        "config_dirs": config_dirs,
        "fig_count_by_config": fig_count_by_config,
        "sample_result_path": sample_result_path,
        "sample_result": sample_result,
        "sample_full_path": sample_full_path,
        "sample_full": sample_full,
        "sample_fit_path": sample_fit_path,
        "sample_fit": sample_fit,
        "metrics_path": metrics_path,
        "metrics_rows": metrics_rows,
        "metrics_columns": metrics_columns,
        "points_by_line": points_by_line,
        "legacy_top_level": legacy_top_level,
        "slurm_files": slurm_files,
        "slurm_log_count": len(slurm_logs),
        "fit_task_count": len(fit_tasks),
        "representatives": representative_data,
        "grid_data": grid_data,
        "line_masks": line_masks,
        "gate_example": {
            "taxis_ns": taxis * 1e9,
            "gate_trace": gate_trace,
            "signal_trace": signal_trace,
            "t_bead_ns": float(t_bead_xyz[mid_x, mid_y, 0] * 1e9),
            "t_width_ns": float(abp.DEFAULT_T_WIDTH * 1e9),
            "t_offset_ns": float(abp.DEFAULT_T_OFFSET * 1e9),
        },
        "best_by_line_gap": best_by_line_gap,
        "z_selection_examples": z_selection_examples,
        "full_z_mm": sorted(float(v * 1e3) for gap in gaps for v in gap.z_positions_m),
    }


def validate_facts(facts: dict) -> None:
    if facts["metrics_columns"] != EXPECTED_METRICS_COLUMNS:
        raise RuntimeError(
            "metrics.csv schema drifted.\n"
            f"Expected: {EXPECTED_METRICS_COLUMNS}\n"
            f"Found:    {facts['metrics_columns']}"
        )

    if set(facts["sample_fit"].keys()) != EXPECTED_FIT_KEYS:
        raise RuntimeError(f"Fit-cache schema drifted: {sorted(facts['sample_fit'].keys())}")
    if set(facts["sample_full"].keys()) != EXPECTED_FULL_KEYS:
        raise RuntimeError(f"Full-reference schema drifted: {sorted(facts['sample_full'].keys())}")
    if set(facts["sample_result"].keys()) != EXPECTED_RESULT_KEYS:
        raise RuntimeError(f"Result-bundle schema drifted: {sorted(facts['sample_result'].keys())}")

    gaps = facts["gaps"]
    fit_cache_count = len(sorted((RESULTS_ROOT / "fits").glob("line_*_gap_*.pkl")))
    if fit_cache_count != len(facts["line_counts"]) * len(gaps):
        raise RuntimeError("Unexpected number of fit-cache files.")
    if len(sorted((RESULTS_ROOT / "full_reference").glob("full_gap_*.npz"))) != len(gaps):
        raise RuntimeError("Unexpected number of full-reference bundles.")
    if len(sorted((RESULTS_ROOT / "results").glob("result_lines_*_zslices_*_gaps_*.npz"))) != facts["configuration_total"]:
        raise RuntimeError("Unexpected number of result bundles.")
    if len(facts["metrics_rows"]) != facts["configuration_total"]:
        raise RuntimeError("Unexpected number of metrics rows.")
    if len(facts["config_dirs"]) != facts["configuration_total"]:
        raise RuntimeError("Unexpected number of config figure folders.")

    if {path.name for path in facts["comparison_figs"]} != EXPECTED_COMPARISON_FIGS:
        raise RuntimeError("Comparison figure set drifted.")
    if {path.name for path in facts["diagnostic_figs"]} != EXPECTED_DIAGNOSTIC_FIGS:
        raise RuntimeError("Diagnostic figure set drifted.")

    for config_dir in facts["config_dirs"]:
        rel = config_dir.relative_to(FIGS_ROOT / "configs")
        line_count = int(rel.parts[0].split("_")[1])
        names = {file.name for file in config_dir.glob("*.png")}
        expected_names = set(EXPECTED_COMMON_CONFIG_FIGS)
        expected_names.update(f"sqrt_dgamma_line_{idx:02d}.png" for idx in range(line_count))
        if names != expected_names:
            raise RuntimeError(f"Unexpected figure set in {config_dir}")

    if facts["points_by_line"] != {1: 19, 2: 34, 3: 51, 4: 66}:
        raise RuntimeError(f"Unexpected reduced-point counts: {facts['points_by_line']}")

    expected_sections = [
        "Gap Ordering",
        "One-Time Diagnostics",
        "Slurm Fitting Cache",
        "Full-Grid Reference",
        "Assemble All 48 Configurations",
        "Quick Metric Preview",
    ]
    if facts["notebook_sections"] != expected_sections:
        raise RuntimeError("Notebook section headings drifted.")


def setup_matplotlib(theme: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": theme["font"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": theme["colors"]["bg"],
            "savefig.facecolor": theme["colors"]["bg"],
            "text.color": theme["colors"]["text"],
            "axes.edgecolor": theme["colors"]["border_strong"],
        }
    )


def resolve_color(theme: dict, color_key: str) -> str:
    return theme["colors"].get(color_key, color_key)


def wrap_line(text: str, width: int) -> str:
    if "$" in text:
        return text
    return textwrap.fill(text, width=width)


def new_slide(title: str, subtitle: str, theme: dict):
    fig = plt.figure(figsize=(SLIDE_W_IN, SLIDE_H_IN), facecolor=resolve_color(theme, "bg"))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.text(0.05, 0.93, title, fontsize=24, fontweight="bold", ha="left", va="top")
    fig.text(
        0.05,
        0.875,
        textwrap.fill(subtitle, width=110),
        fontsize=11,
        ha="left",
        va="top",
        color=resolve_color(theme, "muted"),
        linespacing=1.3,
    )
    fig.add_artist(
        patches.FancyBboxPatch(
            (0.05, 0.835),
            0.90,
            0.004,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            transform=fig.transFigure,
            facecolor=resolve_color(theme, "border"),
            edgecolor="none",
        )
    )
    return fig


def add_card(
    fig,
    rect: list[float],
    title: str,
    lines: list[str],
    theme: dict,
    *,
    header_color: str = "accent",
    panel_color: str = "panel",
    font_size: float = 10.8,
) -> None:
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    body = patches.FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        transform=ax.transAxes,
        linewidth=1.2,
        edgecolor=resolve_color(theme, "border"),
        facecolor=resolve_color(theme, panel_color),
        clip_on=False,
    )
    ax.add_patch(body)
    ax.add_patch(
        patches.FancyBboxPatch(
            (0, 0.84),
            1,
            0.16,
            boxstyle="round,pad=0.012,rounding_size=0.045",
            transform=ax.transAxes,
            linewidth=0,
            facecolor=resolve_color(theme, header_color),
            clip_on=False,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (0, 0.84),
            1,
            0.08,
            transform=ax.transAxes,
            linewidth=0,
            facecolor=resolve_color(theme, header_color),
            clip_on=False,
        )
    )
    ax.text(0.04, 0.92, title, transform=ax.transAxes, ha="left", va="center", fontsize=13, fontweight="bold", color="white")
    y = 0.78
    width = max(26, int(rect[2] * 115))
    for line in lines:
        wrapped = wrap_line(line, width)
        ax.text(
            0.05,
            y,
            wrapped,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=font_size,
            color=resolve_color(theme, "text"),
            linespacing=1.25,
        )
        y -= 0.10 * (wrapped.count("\n") + 1)


def add_panel(
    fig,
    rect: list[float],
    title: str,
    theme: dict,
    *,
    header_color: str = "accent",
    panel_color: str = "panel_alt",
) -> tuple:
    outer = fig.add_axes(rect)
    outer.set_axis_off()
    body = patches.FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        transform=outer.transAxes,
        linewidth=1.2,
        edgecolor=resolve_color(theme, "border"),
        facecolor=resolve_color(theme, panel_color),
        clip_on=False,
    )
    outer.add_patch(body)
    outer.add_patch(
        patches.FancyBboxPatch(
            (0, 0.86),
            1,
            0.14,
            boxstyle="round,pad=0.012,rounding_size=0.045",
            transform=outer.transAxes,
            linewidth=0,
            facecolor=resolve_color(theme, header_color),
            clip_on=False,
        )
    )
    outer.add_patch(
        patches.Rectangle(
            (0, 0.86),
            1,
            0.07,
            transform=outer.transAxes,
            linewidth=0,
            facecolor=resolve_color(theme, header_color),
            clip_on=False,
        )
    )
    outer.text(0.04, 0.93, title, transform=outer.transAxes, ha="left", va="center", fontsize=13, fontweight="bold", color="white")
    inset = [rect[0] + 0.04 * rect[2], rect[1] + 0.07 * rect[3], rect[2] * 0.92, rect[3] * 0.74]
    inner = fig.add_axes(inset, facecolor=resolve_color(theme, panel_color))
    return outer, inner


def add_text_panel(
    fig,
    rect: list[float],
    title: str,
    lines: list[str],
    theme: dict,
    *,
    header_color: str = "accent",
    panel_color: str = "panel",
    font_size: float = 11.0,
    equation_size: float = 15.0,
    line_gap: float = 0.115,
) -> tuple:
    outer, inner = add_panel(
        fig,
        rect,
        title,
        theme,
        header_color=header_color,
        panel_color=panel_color,
    )
    inner.set_axis_off()
    width = max(24, int(rect[2] * 100))
    y = 0.98
    for line in lines:
        stripped = line.strip()
        is_display_math = stripped.startswith("$") and stripped.endswith("$") and stripped.count("$") >= 2
        content = stripped if (is_display_math or "\n" in stripped) else textwrap.fill(stripped, width=width)
        inner.text(
            0.0,
            y,
            content,
            transform=inner.transAxes,
            ha="left",
            va="top",
            fontsize=equation_size if is_display_math else font_size,
            color=resolve_color(theme, "text"),
            linespacing=1.25,
            clip_on=True,
        )
        step = (0.16 if is_display_math else line_gap) * (content.count("\n") + 1)
        y -= step
    return outer, inner


def style_plot_axis(ax, theme: dict, *, grid: bool = True) -> None:
    ax.tick_params(colors=resolve_color(theme, "text"))
    for spine in ax.spines.values():
        spine.set_color(resolve_color(theme, "border_strong"))
        spine.set_linewidth(1.0)
    if grid:
        ax.grid(True, color=resolve_color(theme, "border"), alpha=0.55, linewidth=0.8)
    else:
        ax.grid(False)


def save_slide(fig, output_name: str) -> Path:
    output_path = ASSETS_DIR / output_name
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def format_z_mm_values(values_m, *, max_items: int = 6) -> str:
    values_mm = [float(v) * 1e3 for v in np.asarray(values_m).ravel()]
    if len(values_mm) <= max_items:
        return ", ".join(f"{value:.0f}" for value in values_mm) + " mm"
    shown = ", ".join(f"{value:.0f}" for value in values_mm[:max_items])
    return f"{shown}, ... mm"


def hex_to_rgb01(color: str) -> np.ndarray:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, found {color!r}")
    return np.array([int(value[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4)], dtype=float)


def crop_graphviz_image(image: np.ndarray, bg_color: str) -> np.ndarray:
    rgb = image[..., :3]
    bg_rgb = hex_to_rgb01(bg_color)
    white_rgb = np.ones(3, dtype=float)
    diff_bg = np.abs(rgb - bg_rgb).max(axis=2)
    diff_white = np.abs(rgb - white_rgb).max(axis=2)
    ink_mask = (diff_bg > 0.03) & (diff_white > 0.03)
    if image.shape[-1] == 4:
        ink_mask &= image[..., 3] > 0.02

    ys, xs = np.where(ink_mask)
    if ys.size == 0 or xs.size == 0:
        return image

    row_pad = max(10, int(image.shape[0] * 0.018))
    col_pad = max(10, int(image.shape[1] * 0.014))
    y0 = max(0, int(np.floor(np.percentile(ys, 0.2))) - row_pad)
    y1 = min(image.shape[0], int(np.ceil(np.percentile(ys, 99.8))) + row_pad + 1)
    x0 = max(0, int(np.floor(np.percentile(xs, 0.2))) - col_pad)
    x1 = min(image.shape[1], int(np.ceil(np.percentile(xs, 99.8))) + col_pad + 1)
    return image[y0:y1, x0:x1]


def table_label(
    title: str,
    rows: list[str],
    *,
    theme: dict,
    header_color: str,
    panel_color: str,
) -> str:
    row_html = "".join(
        f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" BGCOLOR="{panel_color}">'
        f'<FONT FACE="{theme["font"]}" POINT-SIZE="10" COLOR="{theme["colors"]["text"]}">{row}</FONT>'
        f"</TD></TR>"
        for row in rows
    )
    return (
        f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6" COLOR="{theme["colors"]["border"]}" BGCOLOR="{panel_color}">'
        f'<TR><TD ALIGN="LEFT" BGCOLOR="{header_color}">'
        f'<FONT FACE="{theme["font"]}" POINT-SIZE="12" COLOR="white"><B>{esc(title)}</B></FONT>'
        f"</TD></TR>"
        f"{row_html}</TABLE>"
    )


def fill_template(template_name: str, replacements: dict[str, str]) -> str:
    template = (SOURCES_DIR / template_name).read_text(encoding="utf-8")
    for token, value in replacements.items():
        template = template.replace(token, value)
    missing = sorted(set(re.findall(r"__[A-Z0-9_]+__", template)))
    if missing:
        raise RuntimeError(f"Unreplaced tokens in {template_name}: {missing}")
    return template


def render_graphviz_slide(
    template_name: str,
    output_name: str,
    replacements: dict[str, str],
    *,
    title: str,
    subtitle: str,
    theme: dict,
) -> Path:
    graph_replacements = dict(replacements)
    graph_replacements["__TITLE__"] = "&#8203;"
    graph_replacements["__SUBTITLE__"] = "&#8203;"
    dot_text = fill_template(template_name, graph_replacements)
    output_path = ASSETS_DIR / output_name
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        dot_path = temp_dir / "slide.dot"
        svg_path = temp_dir / "slide.svg"
        png_path = temp_dir / "slide.png"
        dot_path.write_text(dot_text, encoding="utf-8")
        subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
        subprocess.run(
            [
                "rsvg-convert",
                "-f",
                "png",
                "-a",
                "-w",
                str(SLIDE_W_PT * 2),
                "-h",
                str(SLIDE_H_PT * 2),
                str(svg_path),
                "-o",
                str(png_path),
            ],
            check=True,
        )
        image = crop_graphviz_image(plt.imread(png_path), replacements["__BG__"])
        fig = new_slide(title, subtitle, theme)
        ax = fig.add_axes([0.028, 0.055, 0.944, 0.74])
        ax.imshow(image, interpolation="lanczos")
        ax.axis("off")
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
    return output_path


def base_replacements(title: str, subtitle: str, theme: dict) -> dict[str, str]:
    return {
        "__FONT__": theme["font"],
        "__BG__": resolve_color(theme, "bg"),
        "__TEXT__": resolve_color(theme, "text"),
        "__MUTED__": resolve_color(theme, "muted"),
        "__EDGE__": resolve_color(theme, "edge"),
        "__TITLE__": esc(title),
        "__SUBTITLE__": esc(subtitle),
    }


def build_repo_purpose_slide(facts: dict, text_meta: dict, theme: dict) -> Path:
    gap_min = min(gap.z_min_m for gap in facts["gaps"])
    gap_max = max(gap.z_max_m for gap in facts["gaps"])
    replacements = base_replacements(text_meta["title"], text_meta["subtitle"], theme)
    replacements.update(
        {
            "__RAW_LABEL__": table_label(
                "External Measurements",
                [
                    gv_line("HDF5 gap scans live outside the repo."),
                    gv_line(f"{len(facts['gaps'])} discovered gaps span {mm(gap_min)} to {mm(gap_max)}."),
                ],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__NOTEBOOK_LABEL__": table_label(
                "Notebook Driver",
                [
                    gv_line(f"{facts['notebook_title']} runs 6 stages."),
                    gv_line("discover -> cache -> reference -> sweep"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "panel"),
            ),
            "__MODULE_LABEL__": table_label(
                "Analysis Core",
                [
                    gv_line("src/adaptive_bead_pull.py"),
                    gv_line("time gating, sqrt(Dgamma), fit, ET"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "panel_alt"),
            ),
            "__FULLREF_LABEL__": table_label(
                "Full Reference",
                [
                    gv_line(f"{len(facts['gaps'])} full_gap_## bundles"),
                    gv_line("full-grid ET(z, nu) for each gap"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
            "__FITCACHE_LABEL__": table_label(
                "Fit Caches",
                [
                    gv_line(f"{len(facts['line_counts']) * len(facts['gaps'])} line_count x gap caches"),
                    gv_line("params, costs, success, frequencies"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "physics_soft"),
            ),
            "__ASSEMBLE_LABEL__": table_label(
                "Configuration Sweep",
                [
                    gv_line(f"{facts['configuration_total']} reduced configurations"),
                    gv_line("line_count x z-slices x gap_count"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "accent_soft"),
            ),
            "__RESULTS_LABEL__": table_label(
                "Results",
                [
                    gv_line(f"{facts['configuration_total']} result bundles"),
                    gv_line("reduced and full ET arrays together"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "data_soft"),
            ),
            "__FIGS_LABEL__": table_label(
                "Figures",
                [
                    gv_line(f"{len(facts['diagnostic_figs'])} diagnostics + {len(facts['comparison_figs'])} comparisons"),
                    gv_line(f"{len(facts['config_dirs'])} config folders"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__METRICS_LABEL__": table_label(
                "Metrics",
                [
                    gv_line(f"{len(facts['metrics_rows'])} rows x {len(facts['metrics_columns'])} columns"),
                    gv_line("ET NRMSE and relative |ET| errors"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
            "__NOTE_LABEL__": table_label(
                "Boundary",
                [
                    gv_line("The repo stores derived products, not raw scans."),
                    gv_line("It is an analysis layer for spider-web bead-pull studies."),
                ],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "panel"),
            ),
        }
    )
    return render_graphviz_slide(
        "01_repo_purpose_and_inputs.dot",
        "01_repo_purpose_and_inputs.pdf",
        replacements,
        title=text_meta["title"],
        subtitle=text_meta["subtitle"],
        theme=theme,
    )


def build_notebook_flow_slide(facts: dict, text_meta: dict, theme: dict) -> Path:
    stage_rows = {
        "Gap Ordering": ["rank gaps by descending z_max", "establish the physical scan order"],
        "One-Time Diagnostics": ["write HG sanity figures", "check the model vocabulary visually"],
        "Slurm Fitting Cache": ["fit once per line_count x gap", "reuse 3-slice caches later"],
        "Full-Grid Reference": ["integrate ET on the full aperture", "cache one bundle per gap"],
        "Assemble All 48 Configurations": ["slice caches and build reduced ET", "compare every configuration to full ET"],
        "Quick Metric Preview": ["print sample rows and output paths", "sanity-check the completed sweep"],
    }
    constants_rows = [
        gv_line(f"MAKE_FIGURES = {facts['notebook_constants'].get('MAKE_FIGURES', 'unknown')}"),
        gv_line(f"SUBMIT_SLURM = {facts['notebook_constants'].get('SUBMIT_SLURM', 'unknown')}"),
        gv_line(f"WAIT_FOR_SLURM = {facts['notebook_constants'].get('WAIT_FOR_SLURM', 'unknown')}"),
    ]
    reuse_rows = [
        gv_line("Each cache uses z_ixs_used = (0, 1, 2)."),
        gv_line("Configurations with fewer z slices reuse those fits."),
    ]
    output_rows = [
        gv_line(f"results = {facts['configuration_total']}"),
        gv_line(f"comparisons = {len(facts['comparison_figs'])}; diagnostics = {len(facts['diagnostic_figs'])}"),
    ]
    replacements = base_replacements(text_meta["title"], text_meta["subtitle"], theme)
    stage_cards = []
    header_keys = ["warning", "accent", "physics", "success", "data", "accent"]
    panel_keys = ["panel_warm", "panel", "physics_soft", "panel_green", "data_soft", "panel"]
    for heading, header_key, panel_key in zip(facts["notebook_sections"], header_keys, panel_keys):
        stage_cards.append(
            table_label(
                heading,
                [gv_line(row) for row in stage_rows[heading]],
                theme=theme,
                header_color=resolve_color(theme, header_key),
                panel_color=resolve_color(theme, panel_key),
            )
        )
    replacements.update(
        {
            "__STAGE1_LABEL__": stage_cards[0],
            "__STAGE2_LABEL__": stage_cards[1],
            "__STAGE3_LABEL__": stage_cards[2],
            "__STAGE4_LABEL__": stage_cards[3],
            "__STAGE5_LABEL__": stage_cards[4],
            "__STAGE6_LABEL__": stage_cards[5],
            "__CONSTANTS_LABEL__": table_label(
                "Notebook Toggles",
                constants_rows,
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "panel_alt"),
            ),
            "__REUSE_LABEL__": table_label(
                "Why Cache First",
                reuse_rows,
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__OUTPUTS_LABEL__": table_label(
                "Observed Outputs",
                output_rows,
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
        }
    )
    return render_graphviz_slide(
        "02_notebook_execution_flow.dot",
        "02_notebook_execution_flow.pdf",
        replacements,
        title=text_meta["title"],
        subtitle=text_meta["subtitle"],
        theme=theme,
    )


def build_single_configuration_slide(facts: dict, text_meta: dict, theme: dict) -> Path:
    fit = facts["sample_fit"]
    result = facts["sample_result"]
    replacements = base_replacements(text_meta["title"], text_meta["subtitle"], theme)
    replacements.update(
        {
            "__LOAD_LABEL__": table_label(
                "1. Load",
                [gv_line("S11(x, y, z, nu) for one gap"), gv_line(f"raw cube example: {'x'.join(str(v) for v in facts['gaps'][-1].shape)}")],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__GATE_LABEL__": table_label(
                "2. Time Gate",
                [gv_line("isolate the bead response in time"), gv_line("return to frequency space after gating")],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "panel"),
            ),
            "__REDUCE_LABEL__": table_label(
                "3. Select Lines",
                [gv_line("keep points close to the chosen spider lines"), gv_line("line_count controls how many lines survive")],
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "data_soft"),
            ),
            "__BASELINE_LABEL__": table_label(
                "4. Remove Baseline",
                [gv_line("subtract beta0 + beta1 s along each line"), gv_line("form sqrt(Dgamma)")],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "physics_soft"),
            ),
            "__CORRECT_LABEL__": table_label(
                "5. Match Full Grid",
                [gv_line("fix phase and sign against the full reference"), gv_line("make reduced and full products consistent")],
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
            "__CONVERT_LABEL__": table_label(
                "6. Convert To E",
                [gv_line("apply the bead-pull conversion factor"), gv_line("obtain a field proxy on the retained points")],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "accent_soft"),
            ),
            "__FIT_LABEL__": table_label(
                "7. Beam Fit",
                [gv_line(f"cached params shape: {fit['params'].shape}"), gv_line("fit x0, y0, A, s, kappa, phi0 over (z, nu)")],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__INTEGRATE_LABEL__": table_label(
                "8. Integrate ET",
                [gv_line("integrate fitted E over the aperture"), gv_line("produce reduced ET(z, nu)")],
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
            "__COMPARE_LABEL__": table_label(
                "9. Compare",
                [gv_line("match reduced z positions to full-grid z"), gv_line("compute ET error metrics")],
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "panel_alt"),
            ),
            "__SAVE_LABEL__": table_label(
                "10. Save",
                [gv_line(f"sample result: {facts['sample_result_path'].name}"), gv_line(f"sample ETs_reduced shape: {result['ETs_reduced'].shape}")],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "panel"),
            ),
            "__NOTE_LABEL__": table_label(
                "Physical Question",
                [gv_line("How much of the full-grid ET survives when only spider-web samples are kept?")],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
        }
    )
    return render_graphviz_slide(
        "03_single_configuration_reconstruction.dot",
        "03_single_configuration_reconstruction.pdf",
        replacements,
        title=text_meta["title"],
        subtitle=text_meta["subtitle"],
        theme=theme,
    )


def build_configuration_space_slide(facts: dict, text_meta: dict, theme: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    add_card(
        fig,
        [0.05, 0.49, 0.28, 0.28],
        "Sweep Axes",
        [
            f"line_count = {facts['line_counts']}",
            f"z-slice choices = {facts['z_ix_options']}",
            f"gap_count = {facts['gap_counts']}",
            f"total = {facts['configuration_total']} configurations",
        ],
        theme,
        header_color="accent",
        panel_color="panel",
    )
    add_card(
        fig,
        [0.05, 0.17, 0.28, 0.25],
        "Cache Reuse",
        [
            "Fit caches are built once for z_ixs_used = (0, 1, 2).",
            "Later 1-slice and 2-slice configurations reuse those fits instead of refitting.",
            "n_reduced_z = gap_count x z_slice_count.",
        ],
        theme,
        header_color="physics",
        panel_color="panel_warm",
    )

    _, gap_ax = add_panel(fig, [0.37, 0.49, 0.58, 0.28], "Gap Ordering In z", theme, header_color="data", panel_color="panel_alt")
    style_plot_axis(gap_ax, theme, grid=False)
    gap_colors = [
        resolve_color(theme, "accent"),
        resolve_color(theme, "data"),
        resolve_color(theme, "physics"),
        resolve_color(theme, "success"),
    ]
    for idx, gap in enumerate(facts["gaps"]):
        y = len(facts["gaps"]) - idx
        z_mm = np.asarray(gap.z_positions_m) * 1e3
        gap_ax.plot([z_mm.min(), z_mm.max()], [y, y], color=gap_colors[idx], linewidth=10, alpha=0.25, solid_capstyle="round")
        gap_ax.scatter(z_mm, np.full_like(z_mm, y, dtype=float), color=gap_colors[idx], s=42, zorder=3)
        gap_ax.text(z_mm.max() + 0.8, y, f"gap {gap.rank}", va="center", fontsize=10)
    gap_ax.set_xlim(0, 44)
    gap_ax.set_ylim(0.5, len(facts["gaps"]) + 0.5)
    gap_ax.set_yticks([])
    gap_ax.set_xlabel("z [mm]")

    _, bar_ax = add_panel(fig, [0.37, 0.17, 0.58, 0.25], "Reduced Points Per Gap", theme, header_color="success", panel_color="panel_green")
    style_plot_axis(bar_ax, theme)
    line_counts = list(facts["line_counts"])
    point_counts = [facts["points_by_line"][line_count] for line_count in line_counts]
    bar_colors = [gap_colors[idx - 1] for idx in line_counts]
    bar_ax.bar(line_counts, point_counts, color=bar_colors, width=0.58)
    for line_count, point_count in zip(line_counts, point_counts):
        bar_ax.text(line_count, point_count + 1.8, str(point_count), ha="center", va="bottom", fontsize=10)
    bar_ax.set_xticks(line_counts)
    bar_ax.set_xlabel("line_count")
    bar_ax.set_ylabel("points per gap")
    bar_ax.set_ylim(0, max(point_counts) * 1.25)

    return save_slide(fig, "04_configuration_space.pdf")


def build_figs_guide_slide(facts: dict, text_meta: dict, theme: dict, labels_meta: dict) -> Path:
    fig_labels = labels_meta["figs"]
    replacements = base_replacements(text_meta["title"], text_meta["subtitle"], theme)
    replacements.update(
        {
            "__TREE_LABEL__": table_label(
                "Tree",
                [
                    gv_line("figs/"),
                    gv_line(f"diagnostics/{len(facts['diagnostic_figs'])}"),
                    gv_line(f"comparisons/{len(facts['comparison_figs'])}"),
                    gv_line(f"configs/{len(facts['config_dirs'])}"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__DIAGNOSTICS_LABEL__": table_label(
                "diagnostics/",
                [gv_line("HG sanity plots"), gv_line("abs, real, imag of low-order modes")],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "panel"),
            ),
            "__COMPARISONS_LABEL__": table_label(
                "comparisons/",
                [gv_line("global ET error summaries"), gv_line("heatmaps, gap scans, fit-cost overview")],
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "panel_alt"),
            ),
            "__CONFIGS_LABEL__": table_label(
                "configs/",
                [
                    gv_line("lines_## / zslices_## / gaps_##"),
                    gv_line("the figure tree mirrors the sweep"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
            "__COMMON_LABEL__": table_label(
                "Common PNGs",
                [
                    gv_line("point selection, field fit, ET comparisons"),
                    gv_line("the same core set appears in every folder"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "physics_soft"),
            ),
            "__LINES_LABEL__": table_label(
                "Line-Indexed PNGs",
                [
                    gv_line(f"{fig_labels['config_line_scaled']['pattern']}"),
                    gv_line("11 / 12 / 13 / 14 files for 1 / 2 / 3 / 4 lines"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel"),
            ),
        }
    )
    return render_graphviz_slide(
        "05_figs_folder_guide.dot",
        "05_figs_folder_guide.pdf",
        replacements,
        title=text_meta["title"],
        subtitle=text_meta["subtitle"],
        theme=theme,
    )


def build_data_guide_slide(facts: dict, text_meta: dict, theme: dict, labels_meta: dict) -> Path:
    data_labels = labels_meta["data"]["adaptive_bead_pull_variations"]
    replacements = base_replacements(text_meta["title"], text_meta["subtitle"], theme)
    replacements.update(
        {
            "__TREE_LABEL__": table_label(
                "Tree",
                [
                    gv_line("data/"),
                    gv_line("ET_results.npz + legacy beam-fit caches"),
                    gv_line("adaptive_bead_pull_variations/"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
            "__LEGACY_LABEL__": table_label(
                "Legacy Files",
                [
                    gv_line("ET_results.npz"),
                    gv_line("beam_fit_results*.pkl"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "panel"),
            ),
            "__FITS_LABEL__": table_label(
                "fits/",
                [
                    gv_line(data_labels["fits"]),
                    gv_line(f"count = {len(sorted((RESULTS_ROOT / 'fits').glob('line_*_gap_*.pkl')))}"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "physics_soft"),
            ),
            "__FULLREF_LABEL__": table_label(
                "full_reference/",
                [
                    gv_line(data_labels["full_reference"]),
                    gv_line(f"count = {len(sorted((RESULTS_ROOT / 'full_reference').glob('full_gap_*.npz')))}"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "success"),
                panel_color=resolve_color(theme, "panel_green"),
            ),
            "__RESULTS_LABEL__": table_label(
                "results/",
                [
                    gv_line(data_labels["results"]),
                    gv_line(f"count = {facts['configuration_total']}"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "data"),
                panel_color=resolve_color(theme, "data_soft"),
            ),
            "__METRICS_LABEL__": table_label(
                "metrics.csv",
                [
                    gv_line(data_labels["metrics.csv"]),
                    gv_line(f"{len(facts['metrics_rows'])} rows x {len(facts['metrics_columns'])} columns"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "accent"),
                panel_color=resolve_color(theme, "panel_alt"),
            ),
            "__SLURM_LABEL__": table_label(
                "slurm/",
                [
                    gv_line(data_labels["slurm"]),
                    gv_line(f"task records = {facts['fit_task_count']}, log files = {facts['slurm_log_count']}"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "warning"),
                panel_color=resolve_color(theme, "panel"),
            ),
            "__EXTERNAL_LABEL__": table_label(
                "Boundary",
                [
                    gv_line("Raw HDF5 measurements live outside the repo."),
                    gv_line(f"discovered path: {facts['measurement_dir']}"),
                ],
                theme=theme,
                header_color=resolve_color(theme, "physics"),
                panel_color=resolve_color(theme, "panel_warm"),
            ),
        }
    )
    return render_graphviz_slide(
        "06_data_folder_guide.dot",
        "06_data_folder_guide.pdf",
        replacements,
        title=text_meta["title"],
        subtitle=text_meta["subtitle"],
        theme=theme,
    )


def build_geometry_and_sampling_slide(facts: dict, text_meta: dict, theme: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    line_colors = [
        resolve_color(theme, "accent"),
        resolve_color(theme, "data"),
        resolve_color(theme, "physics"),
        resolve_color(theme, "success"),
    ]
    x_pts_mm = facts["grid_data"]["x_pts"] * 1e3
    y_pts_mm = facts["grid_data"]["y_pts"] * 1e3
    selected_mask = facts["line_masks"][max(facts["line_counts"])]
    full_point_count = int(x_pts_mm.size)

    _, geom_ax = add_panel(fig, [0.05, 0.16, 0.56, 0.66], "Aperture Grid And Spider-Web Lines", theme, header_color="accent", panel_color="panel_alt")
    style_plot_axis(geom_ax, theme)
    geom_ax.scatter(x_pts_mm, y_pts_mm, s=12, color=resolve_color(theme, "muted"), alpha=0.30, label="full grid")
    geom_ax.scatter(x_pts_mm[selected_mask], y_pts_mm[selected_mask], s=18, color=resolve_color(theme, "warning"), alpha=0.70, label="all reduced points")
    x_line_mm = np.linspace(np.min(x_pts_mm), np.max(x_pts_mm), 300)
    for idx, (a_val, b_val) in enumerate(facts["default_lines"]):
        y_line_mm = (a_val * (x_line_mm / 1e3) + b_val) * 1e3
        geom_ax.plot(x_line_mm, y_line_mm, color=line_colors[idx], linewidth=2.6, label=f"L{idx + 1}")
    geom_ax.set_xlabel("x [mm]")
    geom_ax.set_ylabel("y [mm]")
    geom_ax.set_aspect("equal", adjustable="box")
    geom_ax.legend(loc="upper right", fontsize=9, frameon=True)

    add_text_panel(
        fig,
        [0.64, 0.56, 0.31, 0.22],
        "What The Sampling Knob Does",
        [
            "line_count keeps the first N spider tracks.",
            "|y - (a x + b)| <= 3 mm decides whether a point survives.",
            "More lines mean denser in-plane sampling.",
        ],
        theme,
        header_color="physics",
        panel_color="panel_warm",
        font_size=10.4,
    )

    line_rows = [
        f"L{idx + 1}: y = {a_val:+.2f} x {b_val * 1e3:+.0f} mm"
        for idx, (a_val, b_val) in enumerate(facts["default_lines"])
    ]
    add_text_panel(
        fig,
        [0.64, 0.32, 0.31, 0.18],
        "Default Line Equations",
        [*line_rows, "band: |y - (a x + b)| <= 3 mm"],
        theme,
        header_color="data",
        panel_color="data_soft",
        font_size=10.8,
    )

    _, bar_ax = add_panel(fig, [0.64, 0.08, 0.31, 0.18], "Surviving Points Per Gap", theme, header_color="success", panel_color="panel_green")
    style_plot_axis(bar_ax, theme)
    line_counts = list(facts["line_counts"])
    point_counts = [facts["points_by_line"][line_count] for line_count in line_counts]
    bar_ax.bar(line_counts, point_counts, color=line_colors, width=0.58)
    for line_count, point_count in zip(line_counts, point_counts):
        fraction = 100.0 * point_count / full_point_count
        bar_ax.text(line_count, point_count + 1.4, f"{point_count}\n({fraction:.1f}%)", ha="center", va="bottom", fontsize=8.8)
    bar_ax.set_xticks(line_counts)
    bar_ax.set_xlabel("line_count")
    bar_ax.set_ylabel("points")
    bar_ax.set_ylim(0, max(point_counts) * 1.28)

    return save_slide(fig, "07_configuration_geometry_and_sampling.pdf")


def build_gap_and_z_coverage_slide(facts: dict, text_meta: dict, theme: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    gaps_desc = sorted(facts["gaps"], key=lambda gap: gap.z_max_m, reverse=True)
    gap_colors = [
        resolve_color(theme, "accent"),
        resolve_color(theme, "data"),
        resolve_color(theme, "physics"),
        resolve_color(theme, "success"),
    ]
    _, gap_ax = add_panel(fig, [0.05, 0.47, 0.90, 0.32], "Discovered Gap Regions Along z", theme, header_color="accent", panel_color="panel_alt")
    style_plot_axis(gap_ax, theme, grid=False)
    for idx, gap in enumerate(gaps_desc):
        y = len(gaps_desc) - idx
        z_mm = np.asarray(gap.z_positions_m) * 1e3
        gap_ax.plot([z_mm.min(), z_mm.max()], [y, y], color=gap_colors[idx], linewidth=12, alpha=0.25, solid_capstyle="round")
        gap_ax.scatter(z_mm, np.full_like(z_mm, y, dtype=float), color=gap_colors[idx], s=54, zorder=3)
        gap_ax.text(z_mm.max() + 0.7, y, f"gap {idx + 1}: {z_mm.min():.0f}-{z_mm.max():.0f} mm", va="center", fontsize=10)
    gap_ax.set_xlim(0, 44)
    gap_ax.set_ylim(0.5, len(gaps_desc) + 0.5)
    gap_ax.set_xlabel("z [mm]")
    gap_ax.set_yticks([])

    add_text_panel(
        fig,
        [0.05, 0.13, 0.24, 0.24],
        "What gap_count Means",
        [
            "The configuration sweep keeps the gaps in descending z_max order.",
            "gap_count = 1 keeps only the highest-z region at 35-42 mm.",
            "gap_count = 4 spans the full 2-42 mm coverage.",
        ],
        theme,
        header_color="physics",
        panel_color="panel_warm",
        font_size=10.2,
    )

    _, z_ax = add_panel(fig, [0.33, 0.13, 0.39, 0.24], "Which z Planes Survive?", theme, header_color="data", panel_color="data_soft")
    style_plot_axis(z_ax, theme, grid=False)
    z_examples = {}
    for z_choice in facts["z_ix_options"]:
        z_values = []
        for gap in gaps_desc:
            z_positions = np.asarray(gap.z_positions_m)
            z_values.extend(list(z_positions[list(z_choice)] * 1e3))
        z_examples[len(z_choice)] = sorted(float(v) for v in z_values)
    rows = [
        ("full", facts["full_z_mm"]),
        ("1 slice/gap", z_examples[1]),
        ("2 slices/gap", z_examples[2]),
        ("3 slices/gap", z_examples[3]),
    ]
    y_positions = [4, 3, 2, 1]
    for (label, values), y_pos in zip(rows, y_positions):
        z_ax.scatter(values, [y_pos] * len(values), s=42, color=resolve_color(theme, "accent"))
        z_ax.text(-1.5, y_pos, label, ha="right", va="center", fontsize=10)
    z_ax.set_xlim(0, 44)
    z_ax.set_ylim(0.5, 4.5)
    z_ax.set_xlabel("z [mm]")
    z_ax.set_yticks([])

    add_text_panel(
        fig,
        [0.76, 0.13, 0.19, 0.24],
        "Coverage Formula",
        [
            "n_reduced,z = gap_count * z_slice_count",
            f"full data: {len(facts['full_z_mm'])} z planes",
            f"max reduced case: {max(facts['gap_counts']) * len(facts['z_ix_options'][-1])} retained planes",
            "gap_count chooses regions; z_slice_count chooses depth.",
        ],
        theme,
        header_color="success",
        panel_color="panel_green",
        font_size=8.6,
        equation_size=12.0,
    )
    return save_slide(fig, "08_gap_and_z_coverage.pdf")


def build_bead_pull_physics_chain_slide(facts: dict, text_meta: dict, theme: dict, physics_meta: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    chain_sections = [
        ("Measured $S_{11}$", ["$S_{11}(x,y,z,\\nu)$", "Measured on the full aperture for each retained z plane."], "warning", "panel_warm"),
        ("Time Gate", ["$S_{11}^{\\mathrm{gated}}(t)=G(t-t_{\\mathrm{bead}})S_{11}(t)$", "Keeps the bead return and rejects distant reflections."], "accent", "panel"),
        ("Remove Baseline", ["$\\Delta\\Gamma=S_{11}-[\\beta_0+\\beta_1 s]$", "Subtract the slow line-by-line background before taking the square root."], "physics", "physics_soft"),
        ("Convert To Field", ["$E=\\sqrt{\\Delta\\gamma}\\,C(\\nu)$", "$C(\\nu)=\\sqrt{\\frac{4P_{in}}{\\epsilon_0\\alpha_0 i2\\pi\\nu}}$"], "success", "panel_green"),
    ]
    flow_rects = [
        [0.05, 0.55, 0.20, 0.23],
        [0.28, 0.55, 0.20, 0.23],
        [0.51, 0.55, 0.20, 0.23],
        [0.74, 0.55, 0.21, 0.23],
    ]
    for rect, (title, lines, header_key, panel_key) in zip(flow_rects, chain_sections):
        add_text_panel(
            fig,
            rect,
            title,
            lines,
            theme,
            header_color=header_key,
            panel_color=panel_key,
            font_size=10.4,
            equation_size=14.4,
            line_gap=0.12,
        )

    overlay_ax = fig.add_axes([0, 0, 1, 1])
    overlay_ax.set_axis_off()
    for left_rect, right_rect in zip(flow_rects[:-1], flow_rects[1:]):
        x0 = left_rect[0] + left_rect[2]
        x1 = right_rect[0]
        y = left_rect[1] + 0.11
        overlay_ax.annotate(
            "",
            xy=(x1 - 0.01, y),
            xytext=(x0 + 0.01, y),
            xycoords="figure fraction",
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->", lw=1.8, color=resolve_color(theme, "edge")),
        )

    _, gate_ax = add_panel(fig, [0.05, 0.12, 0.42, 0.30], "Example Time Gate At One Aperture Point", theme, header_color="data", panel_color="panel_alt")
    style_plot_axis(gate_ax, theme)
    gate_ax.plot(facts["gate_example"]["taxis_ns"], facts["gate_example"]["signal_trace"], color=resolve_color(theme, "data"), linewidth=1.8, label="|S11(t)|")
    gate_ax.fill_between(
        facts["gate_example"]["taxis_ns"],
        0,
        facts["gate_example"]["gate_trace"],
        color=resolve_color(theme, "accent"),
        alpha=0.28,
        label="gate",
    )
    gate_ax.axvline(facts["gate_example"]["t_bead_ns"], color=resolve_color(theme, "physics"), linewidth=1.6, linestyle="--", label="t_bead")
    gate_ax.set_xlabel("time [ns]")
    gate_ax.set_ylabel("normalized amplitude")
    gate_ax.legend(loc="upper right", fontsize=9)
    gate_ax.text(
        0.03,
        0.97,
        f"t_bead = {facts['gate_example']['t_bead_ns']:.1f} ns\nwindow width = {facts['gate_example']['t_width_ns']:.1f} ns",
        transform=gate_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=resolve_color(theme, "border")),
    )

    add_text_panel(
        fig,
        [0.52, 0.12, 0.43, 0.30],
        "Exact Relations Used In The Conversion",
        [
            "$t_{\\mathrm{bead}}(x,y,z)=\\frac{2}{c}\\sqrt{r(z)^2+x^2+y^2}$",
            "$G(t)$ centers a cosine-edged window on the predicted bead return.",
            "$\\alpha_0=\\frac{3\\Delta\\epsilon V_b}{\\epsilon_b+2}$",
            "Before the final conversion, the reduced line products are phase and sign matched to the full-grid reference.",
        ],
        theme,
        header_color="physics",
        panel_color="panel_warm",
        font_size=10.2,
        equation_size=13.4,
    )
    return save_slide(fig, "09_bead_pull_physics_chain.pdf")


def build_mode_model_slide(facts: dict, text_meta: dict, theme: dict, physics_meta: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    abp = facts["abp"]
    best_overall = facts["representatives"]["best_overall"]
    fit_path = RESULTS_ROOT / "fits" / f"line_{metric_int(best_overall['row'], 'line_count'):02d}_gap_01.pkl"
    with fit_path.open("rb") as handle:
        fit_cache = pickle.load(handle)
    fit_params = np.asarray(fit_cache["params"])[0, facts["default_f_ix"], :6]
    x0, y0, A, s, kappa, phi0 = [float(value) for value in fit_params]

    x_mm = np.linspace(-150.0, 150.0, 240)
    y_mm = np.linspace(-90.0, 120.0, 220)
    X_m, Y_m = np.meshgrid(x_mm / 1e3, y_mm / 1e3, indexing="xy")

    add_text_panel(
        fig,
        [0.05, 0.52, 0.31, 0.28],
        "Model Form",
        [
            "$E(x,y)=A\\exp\\!\\left[-\\frac{r^2}{s^2}\\right]H(x,y)\\exp(i[\\kappa r^2+\\phi_0])$",
            "$r^2=(x-x_0)^2+(y-y_0)^2$",
            "Current fit: H(x,y)=1, so only the HG00 envelope is used.",
            "The reconstruction therefore solves for center, width, amplitude, curvature, and global phase.",
        ],
        theme,
        header_color="physics",
        panel_color="panel_warm",
        font_size=10.0,
        equation_size=12.8,
    )
    add_text_panel(
        fig,
        [0.05, 0.26, 0.31, 0.20],
        "Fit Parameters",
        [
            "x0, y0: beam center.",
            "A and s: complex scale and transverse width.",
            "kappa and phi0: phase curvature and global phase.",
            "One parameter vector is fitted for each retained (z, nu).",
        ],
        theme,
        header_color="accent",
        panel_color="panel",
        font_size=9.8,
    )
    add_text_panel(
        fig,
        [0.05, 0.07, 0.31, 0.14],
        "Live Fit Example At 19.5 GHz",
        [
            f"x0 = {x0 * 1e3:+.1f} mm, y0 = {y0 * 1e3:+.1f} mm, s = {s * 1e3:.1f} mm",
            f"|A| = {abs(A):.1f}, kappa = {kappa:.1f} 1/m^2, phi0 = {phi0:.2f} rad",
        ],
        theme,
        header_color="success",
        panel_color="panel_green",
        font_size=9.0,
    )

    _, hg_panel = add_panel(fig, [0.40, 0.49, 0.55, 0.31], "Low-Order HG Building Blocks", theme, header_color="data", panel_color="panel_alt")
    hg_panel.set_axis_off()
    mode_defs = [
        ("HG00", (1.0, 0.0, 0.0, 0.0)),
        ("HG10", (0.0, 0.0, 1.0, 0.0)),
        ("HG01", (0.0, 1.0, 0.0, 0.0)),
        ("HG11", (0.0, 0.0, 0.0, 1.0)),
    ]
    mode_positions = [
        [0.03, 0.54, 0.43, 0.40],
        [0.54, 0.54, 0.43, 0.40],
        [0.03, 0.05, 0.43, 0.40],
        [0.54, 0.05, 0.43, 0.40],
    ]
    for (mode_name, weights), pos in zip(mode_defs, mode_positions):
        ax = hg_panel.inset_axes(pos)
        field = abp.beam_model(X_m, Y_m, 0.0, 0.0, 1.0, 0.05, 0.0, 0.0, *weights)
        image = np.real(field)
        vmax = np.nanmax(np.abs(image))
        ax.imshow(
            image,
            origin="lower",
            extent=[x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(mode_name, fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(resolve_color(theme, "border"))
            spine.set_linewidth(0.8)

    _, beam_panel = add_panel(fig, [0.40, 0.07, 0.55, 0.34], "Live HG00 Field From The Cached Fit", theme, header_color="accent", panel_color="panel")
    beam_panel.set_axis_off()
    live_field = abp.beam_model(X_m, Y_m, x0, y0, A, s, kappa, phi0, 1.0, 0.0, 0.0, 0.0)
    component_specs = [
        ("|E(x,y)|", np.abs(live_field), "viridis", None),
        ("phase(E) / pi", np.angle(live_field) / np.pi, "coolwarm", (-1.0, 1.0)),
    ]
    component_positions = [[0.04, 0.10, 0.43, 0.82], [0.53, 0.10, 0.43, 0.82]]
    for (title, image, cmap, limits), pos in zip(component_specs, component_positions):
        ax = beam_panel.inset_axes(pos)
        kwargs = {}
        if limits is not None:
            kwargs["vmin"], kwargs["vmax"] = limits
        ax.imshow(
            image,
            origin="lower",
            extent=[x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()],
            cmap=cmap,
            aspect="auto",
            **kwargs,
        )
        ax.scatter([x0 * 1e3], [y0 * 1e3], color="white", s=22, edgecolors="black", linewidths=0.7)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(resolve_color(theme, "border"))
            spine.set_linewidth(0.8)

    return save_slide(fig, "10_mode_model_and_fit_parameters.pdf")


def representative_card_lines(rep: dict) -> list[str]:
    row = rep["row"]
    return [
        row_label(row),
        f"ET complex NRMSE = {metric_float(row, 'et_complex_nrmse'):.3g}",
        f"median relative |ET| error = {metric_float(row, 'rel_abs_et_median'):.3g}",
        f"retained points = {metric_int(row, 'n_reduced_points')}, retained z = {metric_int(row, 'n_reduced_z')}",
    ]


def build_complex_et_slide(facts: dict, text_meta: dict, theme: dict, physics_meta: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    fig.text(
        0.05,
        0.79,
        physics_meta["et_formula"],
        fontsize=16,
        ha="left",
        va="center",
        color=resolve_color(theme, "physics"),
    )
    fig.text(
        0.05,
        0.735,
        "The comparison below uses normalized ET so the phase-space shape remains visible even for the stressed configuration.",
        fontsize=10.2,
        ha="left",
        va="center",
        color=resolve_color(theme, "muted"),
    )

    strong = facts["representatives"]["best_multi_gap"]
    stressed = facts["representatives"]["max_coverage"]

    def choose_display_z_ix(rep: dict) -> int:
        scores = []
        for idx in range(rep["result"]["ETs_reduced"].shape[0]):
            full_trace = rep["matched_full"][idx]
            reduced_trace = rep["result"]["ETs_reduced"][idx]
            score = np.linalg.norm(reduced_trace - full_trace) / max(np.linalg.norm(full_trace), 1e-12)
            scores.append(score)
        return int(np.argmin(scores))

    def rep_summary(rep: dict, *, display_z_mm: float) -> list[str]:
        row = rep["row"]
        return [
            f"{row_label(row)}; NRMSE = {metric_float(row, 'et_complex_nrmse'):.2f}",
            f"retained z: {format_z_mm_values(rep['result']['z_reduced'], max_items=5)}; display z = {display_z_mm:.0f} mm",
        ]

    def plot_complex_trace(ax, rep: dict, *, z_ix: int, title: str) -> None:
        result = rep["result"]
        freqs_ghz = np.asarray(result["frequencies"], dtype=float) / 1e9
        mask = (freqs_ghz >= 19.0) & (freqs_ghz <= 20.0)
        matched_full = rep["matched_full"][z_ix, mask][::4]
        reduced = result["ETs_reduced"][z_ix, mask][::4]
        scale = max(np.nanmax(np.abs(matched_full)), 1e-12)
        matched_full = matched_full / scale
        reduced = reduced / scale
        style_plot_axis(ax, theme)
        ax.plot(np.real(matched_full), np.imag(matched_full), color=resolve_color(theme, "data"), linewidth=2.2, label="full-grid ET")
        ax.plot(np.real(reduced), np.imag(reduced), color=resolve_color(theme, "physics"), linewidth=2.0, linestyle="--", label="reduced ET")
        ax.scatter([np.real(matched_full[0]), np.real(matched_full[-1])], [np.imag(matched_full[0]), np.imag(matched_full[-1])], color=resolve_color(theme, "data"), s=22)
        ax.scatter([np.real(reduced[0]), np.real(reduced[-1])], [np.imag(reduced[0]), np.imag(reduced[-1])], color=resolve_color(theme, "physics"), s=22)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("Re(E_T) / max|full|")
        ax.set_ylabel("Im(E_T) / max|full|")
        ax.legend(loc="best", fontsize=8.6)

    def plot_frequency_magnitude(ax, rep: dict, *, z_ix: int, title: str) -> None:
        result = rep["result"]
        freqs_ghz = np.asarray(result["frequencies"], dtype=float) / 1e9
        matched_full = rep["matched_full"][z_ix]
        reduced = result["ETs_reduced"][z_ix]
        scale = max(np.nanmax(np.abs(matched_full)), 1e-12)
        style_plot_axis(ax, theme)
        ax.plot(freqs_ghz, np.abs(matched_full) / scale, color=resolve_color(theme, "data"), linewidth=2.2, label="full-grid ET")
        ax.plot(freqs_ghz, np.abs(reduced) / scale, color=resolve_color(theme, "physics"), linewidth=2.0, linestyle="--", label="reduced ET")
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("frequency [GHz]")
        ax.set_ylabel("|E_T| / max|full|")
        ax.set_xlim(freqs_ghz.min(), freqs_ghz.max())
        ax.set_ylim(0, 1.1)
        ax.legend(loc="best", fontsize=8.6)

    strong_z_ix = choose_display_z_ix(strong)
    stressed_z_ix = choose_display_z_ix(stressed)
    strong_z_mm = float(strong["result"]["z_reduced"][strong_z_ix] * 1e3)
    stressed_z_mm = float(stressed["result"]["z_reduced"][stressed_z_ix] * 1e3)

    add_text_panel(
        fig,
        [0.05, 0.55, 0.42, 0.10],
        "Representative Strong Multi-Gap Case",
        rep_summary(strong, display_z_mm=strong_z_mm),
        theme,
        header_color="success",
        panel_color="panel_green",
        font_size=8.8,
    )
    add_text_panel(
        fig,
        [0.53, 0.55, 0.42, 0.10],
        "Representative Stress Test",
        rep_summary(stressed, display_z_mm=stressed_z_mm),
        theme,
        header_color="warning",
        panel_color="panel_warm",
        font_size=8.8,
    )

    _, strong_plane_ax = add_panel(fig, [0.05, 0.30, 0.42, 0.21], f"Argand Trace At z = {strong_z_mm:.0f} mm", theme, header_color="data", panel_color="panel_alt")
    plot_complex_trace(strong_plane_ax, strong, z_ix=strong_z_ix, title="19-20 GHz window")

    _, stressed_plane_ax = add_panel(fig, [0.53, 0.30, 0.42, 0.21], f"Argand Trace At z = {stressed_z_mm:.0f} mm", theme, header_color="data", panel_color="panel_alt")
    plot_complex_trace(stressed_plane_ax, stressed, z_ix=stressed_z_ix, title="19-20 GHz window")

    _, strong_freq_ax = add_panel(fig, [0.05, 0.05, 0.42, 0.20], f"|E_T|(nu) At z = {strong_z_mm:.0f} mm", theme, header_color="accent", panel_color="panel")
    plot_frequency_magnitude(strong_freq_ax, strong, z_ix=strong_z_ix, title="full versus reduced magnitude")

    _, stressed_freq_ax = add_panel(fig, [0.53, 0.05, 0.42, 0.20], f"|E_T|(nu) At z = {stressed_z_mm:.0f} mm", theme, header_color="accent", panel_color="panel")
    plot_frequency_magnitude(stressed_freq_ax, stressed, z_ix=stressed_z_ix, title="full versus reduced magnitude")

    return save_slide(fig, "11_complex_ET_reconstruction.pdf")


def build_tradeoff_slide(facts: dict, text_meta: dict, theme: dict, physics_meta: dict) -> Path:
    fig = new_slide(text_meta["title"], text_meta["subtitle"], theme)
    line_counts = list(facts["line_counts"])
    gap_counts = list(facts["gap_counts"])
    grid = np.full((len(line_counts), len(gap_counts)), np.nan)
    z_choice = np.full_like(grid, np.nan)
    for i, line_count in enumerate(line_counts):
        for j, gap_count in enumerate(gap_counts):
            row = facts["best_by_line_gap"][(line_count, gap_count)]
            grid[i, j] = metric_float(row, "et_complex_nrmse")
            z_choice[i, j] = metric_int(row, "z_slice_count")

    _, heat_ax = add_panel(fig, [0.05, 0.44, 0.43, 0.34], "Best NRMSE At Fixed (line_count, gap_count)", theme, header_color="data", panel_color="panel_alt")
    im = heat_ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    heat_ax.set_xticks(range(len(gap_counts)), labels=gap_counts)
    heat_ax.set_yticks(range(len(line_counts)), labels=line_counts)
    heat_ax.set_xlabel("gap_count")
    heat_ax.set_ylabel("line_count")
    for i, line_count in enumerate(line_counts):
        for j, gap_count in enumerate(gap_counts):
            heat_ax.text(
                j,
                i,
                f"{grid[i, j]:.2g}\nz={int(z_choice[i, j])}",
                ha="center",
                va="center",
                fontsize=8.6,
                color="white",
            )
    cbar = plt.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("ET complex NRMSE", fontsize=9)

    _, line_ax = add_panel(fig, [0.53, 0.44, 0.42, 0.34], "Trend: More Lines Help, More Gaps Hurt", theme, header_color="accent", panel_color="panel")
    style_plot_axis(line_ax, theme)
    colors = [
        resolve_color(theme, "accent"),
        resolve_color(theme, "data"),
        resolve_color(theme, "physics"),
        resolve_color(theme, "success"),
    ]
    for idx, line_count in enumerate(line_counts):
        line_ax.plot(gap_counts, grid[idx, :], marker="o", linewidth=2.0, color=colors[idx], label=f"lines={line_count}")
    line_ax.set_xlabel("gap_count")
    line_ax.set_ylabel("best ET NRMSE")
    line_ax.legend(loc="upper left", fontsize=9)

    reps = facts["representatives"]
    add_text_panel(
        fig,
        [0.05, 0.14, 0.27, 0.20],
        "Best Overall",
        representative_card_lines(reps["best_overall"]),
        theme,
        header_color="success",
        panel_color="panel_green",
        font_size=9.8,
    )
    add_text_panel(
        fig,
        [0.365, 0.14, 0.27, 0.20],
        "Best Multi-Gap",
        representative_card_lines(reps["best_multi_gap"]),
        theme,
        header_color="data",
        panel_color="data_soft",
        font_size=9.8,
    )
    add_text_panel(
        fig,
        [0.68, 0.14, 0.27, 0.20],
        "Max Coverage",
        representative_card_lines(reps["max_coverage"]),
        theme,
        header_color="warning",
        panel_color="panel_warm",
        font_size=9.8,
    )

    add_text_panel(
        fig,
        [0.05, 0.01, 0.90, 0.10],
        "Physics Takeaway",
        [
            "Best fidelity comes from dense in-plane sampling in a single gap. Once multiple gap regions are coupled, the error rises sharply; the 4-gap / 3-slice case is a stress test rather than a precision operating point."
        ],
        theme,
        header_color="physics",
        panel_color="panel_warm",
        font_size=9.8,
    )
    return save_slide(fig, "12_configuration_tradeoffs_for_physics.pdf")


def build_all() -> list[Path]:
    theme = read_json(SOURCES_DIR / "theme.json")
    diagram_text = read_json(SOURCES_DIR / "diagram_text.json")
    folder_labels = read_json(SOURCES_DIR / "folder_labels.json")
    physics_text = read_json(SOURCES_DIR / "physics_text.json")

    setup_matplotlib(theme)
    facts = gather_repo_facts()
    validate_facts(facts)

    outputs = [
        build_repo_purpose_slide(facts, diagram_text["01_repo_purpose_and_inputs"], theme),
        build_notebook_flow_slide(facts, diagram_text["02_notebook_execution_flow"], theme),
        build_single_configuration_slide(facts, diagram_text["03_single_configuration_reconstruction"], theme),
        build_configuration_space_slide(facts, diagram_text["04_configuration_space"], theme),
        build_figs_guide_slide(facts, diagram_text["05_figs_folder_guide"], theme, folder_labels),
        build_data_guide_slide(facts, diagram_text["06_data_folder_guide"], theme, folder_labels),
        build_geometry_and_sampling_slide(facts, physics_text["07_configuration_geometry_and_sampling"], theme),
        build_gap_and_z_coverage_slide(facts, physics_text["08_gap_and_z_coverage"], theme),
        build_bead_pull_physics_chain_slide(facts, physics_text["09_bead_pull_physics_chain"], theme, physics_text["09_bead_pull_physics_chain"]),
        build_mode_model_slide(facts, physics_text["10_mode_model_and_fit_parameters"], theme, physics_text["10_mode_model_and_fit_parameters"]),
        build_complex_et_slide(facts, physics_text["11_complex_ET_reconstruction"], theme, physics_text["11_complex_ET_reconstruction"]),
        build_tradeoff_slide(facts, physics_text["12_configuration_tradeoffs_for_physics"], theme, physics_text["12_configuration_tradeoffs_for_physics"]),
    ]
    return outputs


def main() -> int:
    outputs = build_all()
    print("Generated PDF slides:")
    for path in outputs:
        print(f" - {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
