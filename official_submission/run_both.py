import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ConfigSpace import ConfigurationSpace

from lccv import LCCV
from ipl import IPL
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_space_file",
        type=str,
        default="lcdb_config_space_knn.json",
    )
    parser.add_argument(
        "--configurations_performance_file",
        type=str,
        default="config_performances_dataset-6.csv",
    )
    # minimal_anchor should be >= min(anchor_size) in the CSV
    parser.add_argument("--minimal_anchor", type=int, default=256)
    # final_anchor: we’ll use this as the target anchor for both methods
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_iterations", type=int, default=50)
    return parser.parse_args()


def run(args):
    # --- Load config space (no deprecation warnings) ---
    cs_path = Path(args.config_space_file)
    config_space = ConfigurationSpace.from_json(str(cs_path))

    # --- Load LCDB dataset and fit surrogate ---
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    minimal_anchor = args.minimal_anchor
    final_anchor = args.max_anchor_size

    # --- Create vertical evaluators ---
    lccv = LCCV(surrogate_model, minimal_anchor, final_anchor)
    ipl = IPL(surrogate_model, minimal_anchor, final_anchor)

    best_so_far_lccv = None
    best_so_far_ipl = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_lccv, ax_ipl = axes

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())

        # ----- LCCV -----
        result_lccv = lccv.evaluate_model(best_so_far_lccv, theta_new)
        if result_lccv:
            final_lccv = result_lccv[-1][1]
            if best_so_far_lccv is None or final_lccv < best_so_far_lccv:
                best_so_far_lccv = final_lccv

            x_l = [a for (a, _) in result_lccv]
            y_l = [v for (_, v) in result_lccv]
            ax_lccv.plot(x_l, y_l, "-o", alpha=0.6)

        # ----- IPL -----
        result_ipl = ipl.evaluate_model(best_so_far_ipl, theta_new)
        if result_ipl:
            # NOTE: IPL may *not* evaluate the final anchor if it discards the config.
            final_ipl = result_ipl[-1][1]
            if best_so_far_ipl is None or final_ipl < best_so_far_ipl:
                best_so_far_ipl = final_ipl

            x_i = [a for (a, _) in result_ipl]
            y_i = [v for (_, v) in result_ipl]
            ax_ipl.plot(x_i, y_i, "-o", alpha=0.6)

    ax_lccv.set_title("LCCV learning curves")
    ax_lccv.set_xlabel("Anchor size")
    ax_lccv.set_ylabel("Predicted error")

    ax_ipl.set_title("IPL learning curves")
    ax_ipl.set_xlabel("Anchor size")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    run(parse_args())
