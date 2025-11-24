import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
CONFIG_COL_CANDIDATES = ["config_id", "config", "configuration_id", "configuration"]

def get_config_col(df: pd.DataFrame) -> str:
    for col in CONFIG_COL_CANDIDATES:
        if col in df.columns:
            return col

    hyper_cols = [c for c in df.columns if c not in ("anchor_size", "score")]
    if not hyper_cols:
        raise ValueError(
            "Could not infer config column: no hyperparameter columns found. "
            f"Columns are: {list(df.columns)}"
        )

    df["config_id"] = (
        df[hyper_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )
    return "config_id"


DATASET_FILES = {
    "dataset-11": "config_performances_dataset-11.csv",
    "dataset-1457": "config_performances_dataset-1457.csv",
    "dataset-6": "config_performances_dataset-6.csv",
}

CONFIG_COL = "config_id"
ANCHOR_COL = "anchor_size"
SCORE_COL = "score"


def compute_monotonicity_stats(df: pd.DataFrame):
    config_col = get_config_col(df)
    anchors = np.sort(df["anchor_size"].unique())
    n_configs = df[config_col].nunique()

    curves = df.pivot_table(
        index=config_col,
        columns="anchor_size",
        values="score"
    ).sort_index(axis=1)

    scores = curves.values
    diffs = np.diff(scores, axis=1)
    delta_anchors = np.diff(anchors).astype(float)[None, :]
    slopes = diffs / delta_anchors

    any_pos = (slopes > 0).any(axis=1)
    pos_first = slopes[:, 0] > 0
    pos_first_two = (slopes[:, :2] > 0).any(axis=1) if slopes.shape[1] >= 2 else pos_first
    monotone_decr = (slopes <= 0).all(axis=1)

    stats = {
        "n_configs": n_configs,
        "frac_any_positive_slope": float(any_pos.mean()),
        "frac_positive_first_step": float(pos_first.mean()),
        "frac_positive_first_two": float(pos_first_two.mean()),
        "frac_monotone_decreasing_all": float(monotone_decr.mean()),
    }
    return stats


def plot_monotonicity_summary(dataset_files: dict, out_path: str = "plots/monotonicity_across_datasets.png"):
    dataset_names = []
    any_pos_list = []
    pos_first_list = []
    pos_first_two_list = []
    mono_decr_list = []

    for short_name, csv_path in dataset_files.items():
        df = pd.read_csv(csv_path)
        stats = compute_monotonicity_stats(df)  # uses get_config_col internally

        dataset_names.append(short_name)
        any_pos_list.append(stats["frac_any_positive_slope"])
        pos_first_list.append(stats["frac_positive_first_step"])
        pos_first_two_list.append(stats["frac_positive_first_two"])
        mono_decr_list.append(stats["frac_monotone_decreasing_all"])

    x = np.arange(len(dataset_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(x - 1.5 * width, any_pos_list,       width, label="Any positive slope")
    ax.bar(x - 0.5 * width, pos_first_list,     width, label="Positive first step")
    ax.bar(x + 0.5 * width, pos_first_two_list, width, label="Positive first two")
    ax.bar(x + 1.5 * width, mono_decr_list,     width, label="Monotone decreasing")

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=20, ha="right")
    ax.set_ylabel("Fraction of configs")
    ax.set_title("Monotonicity / non-monotonicity across datasets")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_monotonicity_summary(DATASET_FILES)
