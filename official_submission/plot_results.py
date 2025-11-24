import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

METHOD_COLORS = {
    "LCCV": "#2A9D8F",  # teal
    "IPL":  "#E76F51",  # coral
}


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def short_dataset_name(name: str) -> str:
    if "dataset-" in name:
        return "dataset-" + name.split("dataset-")[-1]
    return name

def plot_run_summaries_all(runs_df: pd.DataFrame, outdir: Path):
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

    fig, axes = plt.subplots(
        1, 3,
        figsize=(max(10, len(datasets) * 2.5), 3.8),
        sharex=False
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (metric, ylabel) in zip(axes, metrics):
        for j, method in enumerate(methods):
            means = []
            stds = []
            for ds in datasets:
                g = runs_df[(runs_df["dataset"] == ds) &
                            (runs_df["method"] == method)]
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
        ax.set_xticklabels(
            [short_dataset_name(ds) for ds in datasets],
            rotation=25,
            ha="right"
        )
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    fig.suptitle("Run-level summaries across datasets", y=0.98)

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.18,
        top=0.80,
        wspace=0.25,
    )

    fig_path = outdir / "run_summaries_across_datasets.png"
    fig.savefig(fig_path)
    plt.close(fig)

def plot_regret_vs_budget_per_dataset(
    regret_df: pd.DataFrame,
    outdir: Path,
    budget_step: float = 5.0,
):
    _ensure_dir(outdir)

    datasets = sorted(regret_df["dataset"].unique())
    methods = sorted(regret_df["method"].unique())

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(4 * len(datasets), 3.8),
        sharey=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, ds in zip(axes, datasets):
        ds_label = short_dataset_name(ds)
        g_ds = regret_df[regret_df["dataset"] == ds]

        grid_budgets = np.unique(g_ds["budget"].values.astype(float))
        max_B = float(grid_budgets.max())

        chosen = []
        B = 0.0
        while B <= max_B + 1e-9:
            idx = int(np.argmin(np.abs(grid_budgets - B)))
            chosen.append(float(grid_budgets[idx]))
            B += budget_step
        chosen = np.array(sorted(set(chosen)), dtype=float)

        for method in methods:
            g_m = g_ds[g_ds["method"] == method]
            if g_m.empty:
                continue

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

    axes[0].set_ylabel("Mean regret")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    fig.suptitle("Regret vs budget per dataset (step ≈ 5)", y=0.98)

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.18,
        top=0.80,
        wspace=0.25,
    )

    fig_path = outdir / "regret_vs_budget_per_dataset.png"
    fig.savefig(fig_path)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot LCCV vs IPL results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing vertical_run_summaries.csv and "
             "vertical_regret_vs_cost.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to write generated plot PNGs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    outdir = Path(args.output_dir)
    _ensure_dir(outdir)

    runs_path = results_dir / "vertical_run_summaries.csv"
    regret_grid_path = results_dir / "vertical_regret_vs_cost.csv"

    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found.")
    if not regret_grid_path.exists():
        raise FileNotFoundError(f"{regret_grid_path} not found.")

    runs_df = pd.read_csv(runs_path)
    regret_df = pd.read_csv(regret_grid_path)

    plot_run_summaries_all(runs_df, outdir)
    plot_regret_vs_budget_per_dataset(
        regret_df,
        outdir,
        budget_step=5.0,
    )


if __name__ == "__main__":
    main()
