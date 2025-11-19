#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_vertical_results.py

Reads the CSV outputs from run_experiment/final_runner and generates
comparison plots for LCCV vs IPL:

1. Regret vs cost curves (mean ± std) per dataset, with auto-zoomed x-axis.
2. Run-level summaries in a single PNG (3 panels):
   - total cost
   - fraction early-stopped
   - final regret
3. Single scatter plot: total cost vs final regret for all datasets.
4. Regret vs budget per dataset (one PNG with 3 panels), budgets sampled
   every ~5 cost units.

Usage:
    python plot_vertical_results.py --results_dir vertical_results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Modern, pleasant, reasonably colorblind-friendly palette
METHOD_COLORS = {
    "LCCV": "#2A9D8F",  # teal
    "IPL":  "#E76F51",  # coral
}


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def short_dataset_name(name: str) -> str:
    """
    Convert a filename-like dataset name to a shorter label.

    Example:
        'config_performances_dataset-6' -> 'dataset-6'
    """
    if "dataset-" in name:
        return "dataset-" + name.split("dataset-")[-1]
    return name


# ---------------------------------------------------------------------------
# 1) Regret vs cost (mean ± std across seeds), per dataset
# ---------------------------------------------------------------------------

def plot_regret_vs_cost_grid(regret_df: pd.DataFrame, outdir: Path):
    """
    Plot mean ± std regret vs cost curves for each dataset and method.

    Expects columns: dataset, method, budget, mean_regret, std_regret.

    For datasets where regret quickly goes to (near) 0, the x-axis is zoomed
    to the region where regret actually changes.
    """
    _ensure_dir(outdir)

    for dataset, g_ds in regret_df.groupby("dataset"):
        ds_label = short_dataset_name(dataset)
        fig, ax = plt.subplots(figsize=(6, 4))

        # Build a pivot to determine where regret stabilises
        pivot = g_ds.pivot_table(
            index="budget",
            columns="method",
            values="mean_regret",
            aggfunc="mean",
        )
        budgets_all = np.array(pivot.index.values, dtype=float)
        min_over_methods = pivot.min(axis=1).values
        final_min = float(min_over_methods[-1])
        # Consider "near-optimal" once within a small epsilon of final_min
        eps = max(1e-4, 0.01 * (pivot.values.max() - final_min))
        near_opt_mask = min_over_methods <= final_min + eps
        if np.any(near_opt_mask):
            first_near_idx = np.argmax(near_opt_mask)
            x_max = budgets_all[first_near_idx] * 2.0
        else:
            x_max = budgets_all[-1]

        # Always show at least some part of the full range
        x_max = min(x_max, budgets_all[-1])

        for method, g_m in g_ds.groupby("method"):
            g_m = g_m.sort_values("budget")

            budgets = g_m["budget"].values.astype(float)
            mean_reg = g_m["mean_regret"].values.astype(float)
            std_reg = g_m["std_regret"].values.astype(float)

            # skip budgets where mean_regret is NaN
            mask = ~np.isnan(mean_reg)
            budgets = budgets[mask]
            mean_reg = mean_reg[mask]
            std_reg = std_reg[mask]

            if budgets.size == 0:
                continue

            color = METHOD_COLORS.get(method, None)
            ax.plot(
                budgets,
                mean_reg,
                label=method,
                color=color,
                linewidth=2,
            )
            ax.fill_between(
                budgets,
                mean_reg - std_reg,
                mean_reg + std_reg,
                color=color,
                alpha=0.2,
            )

        ax.axhline(0.0, linestyle="--", color="black", alpha=0.4, linewidth=1)
        ax.set_title(f"Regret vs cost – {ds_label}")
        ax.set_xlabel("Cost (full-size equivalent evaluations)")
        ax.set_ylabel("Regret (best_true_so_far − oracle_best)")
        ax.set_xlim(0.0, x_max)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig_path = outdir / f"regret_vs_cost_{dataset}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2) Run-level summaries in one PNG (3 panels, all datasets)
# ---------------------------------------------------------------------------

def plot_run_summaries_all(runs_df: pd.DataFrame, outdir: Path):
    """
    Create a single PNG with three panels, each showing a cross-dataset
    comparison between methods:

        - total_cost
        - frac_early
        - final_regret

    x-axis = dataset (short label), bars = methods (LCCV, IPL).
    """
    _ensure_dir(outdir)

    metrics = [
        ("total_cost", "Total cost"),
        ("frac_early", "Fraction early-stopped"),
        ("final_regret", "Final regret"),
    ]

    datasets = sorted(runs_df["dataset"].unique())
    methods = sorted(runs_df["method"].unique())

    x = np.arange(len(datasets))
    width = 0.35 if len(methods) == 2 else 0.7 / max(1, len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(datasets) * 2.5), 4), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (metric, ylabel) in zip(axes, metrics):
        for j, method in enumerate(methods):
            means = []
            stds = []
            for ds in datasets:
                g = runs_df[(runs_df["dataset"] == ds) & (runs_df["method"] == method)]
                vals = g[metric].values.astype(float)
                if len(vals) == 0:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
            xj = x + (j - (len(methods) - 1) / 2) * width
            ax.bar(
                xj,
                means,
                width=width,
                label=method if metric == "total_cost" else None,
                yerr=stds,
                capsize=4,
                color=METHOD_COLORS.get(method),
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([short_dataset_name(ds) for ds in datasets],
                           rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[0].legend(title="Method")
    fig.suptitle("Run-level summaries across datasets")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    fig_path = outdir / "run_summaries_across_datasets.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) Cost vs final regret scatter (all datasets in one figure)
# ---------------------------------------------------------------------------

def plot_cost_vs_regret_scatter_all(runs_df: pd.DataFrame, outdir: Path):
    """
    Scatter plot of total_cost vs final_regret for all runs, with:

      - color = method
      - marker shape = dataset

    This shows the global cost–quality trade-off at a glance.
    """
    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6, 5))

    datasets = sorted(runs_df["dataset"].unique())
    methods = sorted(runs_df["method"].unique())
    markers = ["o", "s", "^", "D", "P", "X", "v"]  # cycles if >7 datasets

    for d_idx, ds in enumerate(datasets):
        g_ds = runs_df[runs_df["dataset"] == ds]
        marker = markers[d_idx % len(markers)]
        ds_label = short_dataset_name(ds)

        for method in methods:
            g_m = g_ds[g_ds["method"] == method]
            if g_m.empty:
                continue
            color = METHOD_COLORS.get(method, None)
            ax.scatter(
                g_m["total_cost"].values,
                g_m["final_regret"].values,
                label=f"{method} ({ds_label})",
                color=color,
                marker=marker,
                alpha=0.8,
                edgecolor="k",
                linewidth=0.5,
            )

    ax.set_title("Cost vs final regret (all datasets)")
    ax.set_xlabel("Total cost")
    ax.set_ylabel("Final regret")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Create a compact legend by grouping entries with same label
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=2)

    fig.tight_layout()
    fig_path = outdir / "cost_vs_final_regret_all_datasets.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Regret vs budget per dataset (budgets every ~5 units)
# ---------------------------------------------------------------------------

def plot_regret_vs_budget_per_dataset(regret_df: pd.DataFrame, outdir: Path, budget_step: float = 5.0):
    """
    One PNG with three panels (for 3 datasets), each panel showing
    regret vs budget for that dataset, budgets sampled every ~5 units.

    For each dataset:
      - x-axis: budget (downsampled grid, step ~budget_step)
      - lines: mean regret per method, with ±std bands.
    """
    _ensure_dir(outdir)

    datasets = sorted(regret_df["dataset"].unique())
    methods = sorted(regret_df["method"].unique())

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(4 * len(datasets), 4),
        sharey=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, ds in zip(axes, datasets):
        ds_label = short_dataset_name(ds)
        g_ds = regret_df[regret_df["dataset"] == ds]

        # global budget grid for this dataset
        grid_budgets = np.unique(g_ds["budget"].values.astype(float))
        max_B = grid_budgets.max()

        # Choose budgets approximately every budget_step units
        chosen = []
        B = 0.0
        while B <= max_B + 1e-9:
            idx = np.argmin(np.abs(grid_budgets - B))
            chosen.append(grid_budgets[idx])
            B += budget_step
        chosen = np.array(sorted(set(chosen)))

        for method in methods:
            g_m = g_ds[g_ds["method"] == method]
            if g_m.empty:
                continue
            # Snap to chosen grid
            mean_vals = []
            std_vals = []
            for B in chosen:
                row = g_m[np.isclose(g_m["budget"], B)]
                if row.empty:
                    mean_vals.append(np.nan)
                    std_vals.append(0.0)
                else:
                    mean_vals.append(float(row["mean_regret"].iloc[0]))
                    std_vals.append(float(row["std_regret"].iloc[0]))
            mean_vals = np.array(mean_vals, dtype=float)
            std_vals = np.array(std_vals, dtype=float)
            mask = ~np.isnan(mean_vals)
            chosen_masked = chosen[mask]
            mean_vals = mean_vals[mask]
            std_vals = std_vals[mask]

            color = METHOD_COLORS.get(method, None)
            ax.plot(
                chosen_masked,
                mean_vals,
                label=method,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )
            ax.fill_between(
                chosen_masked,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.2,
            )

        ax.set_title(ds_label)
        ax.set_xlabel("Budget (cost)")
        ax.grid(True, axis="both", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Mean regret")
    axes[0].legend(title="Method")
    fig.suptitle("Regret vs budget per dataset (step ≈ 5)", y=1.02)
    fig.tight_layout(rect=[0, 0.0, 1, 0.94])

    fig_path = outdir / "regret_vs_budget_per_dataset.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot LCCV vs IPL results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="vertical_results",
        help="Directory containing vertical_events.csv, "
             "vertical_run_summaries.csv, vertical_regret_vs_cost.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vertical_plots",
        help="Directory to write generated plot PNGs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    outdir = Path(args.output_dir)
    _ensure_dir(outdir)

    events_path = results_dir / "vertical_events.csv"
    runs_path = results_dir / "vertical_run_summaries.csv"
    regret_grid_path = results_dir / "vertical_regret_vs_cost.csv"

    if not events_path.exists():
        print(f"Warning: {events_path} not found (events-level plots skipped).")
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found.")
    if not regret_grid_path.exists():
        raise FileNotFoundError(f"{regret_grid_path} not found.")

    runs_df = pd.read_csv(runs_path)
    regret_df = pd.read_csv(regret_grid_path)

    # 1) Regret vs cost per dataset (auto-zoomed)
    plot_regret_vs_cost_grid(regret_df, outdir / "regret_curves")

    # 2) Run-level summaries (all datasets in one PNG)
    plot_run_summaries_all(runs_df, outdir / "run_summaries")

    # 3) Single scatter: cost vs final regret for all datasets
    plot_cost_vs_regret_scatter_all(runs_df, outdir / "cost_vs_regret")

    # 4) Regret vs budget per dataset (one PNG)
    plot_regret_vs_budget_per_dataset(regret_df, outdir / "regret_vs_budget", budget_step=5.0)


if __name__ == "__main__":
    main()
