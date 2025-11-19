#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
final_runner.py

Vertical model selection experiment harness for Assignment 2.

Runs LCCV and IPL on multiple LCDB datasets and seeds, using a non-linear cost
model c(s) = (s / s_T) ** alpha, and logs:

- Per-evaluation events (anchor, score, incremental & cumulative cost)
- Per-run summaries (total cost, early-stop fraction, final regret, etc.)
- Regret vs cost curves on a fixed budget grid, aggregated across seeds.

The resulting CSVs can be used directly for plotting and for the report.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace

from surrogate_model import SurrogateModel
from lccv import LCCV
from ipl import IPL


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def eval_cost(anchor_size: int, final_anchor: int, alpha: float) -> float:
    """
    Non-linear cost model for evaluating a configuration at a given anchor.

        c(s) = (s / s_T) ** alpha

    where s_T is the final anchor size, alpha >= 1.

    Interpretation:
        - Cost is measured in "full-size equivalent evaluations".
        - When alpha=1, evaluating at full data costs 1, at half data costs 0.5, etc.
        - For alpha>1, large anchors become even more expensive.

    :param anchor_size: s, anchor size of the evaluation.
    :param final_anchor: s_T, final anchor size for this dataset.
    :param alpha: non-linearity exponent.
    :return: cost as float.
    """
    return float((anchor_size / final_anchor) ** alpha)


# ---------------------------------------------------------------------------
# Single-method runner
# ---------------------------------------------------------------------------

def run_single_method_on_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    config_space: ConfigurationSpace,
    method_name: str,
    evaluator_cls,
    seed: int,
    n_configs: int,
    alpha: float,
) -> Tuple[List[Dict], List[Dict], float]:
    """
    Run one vertical method (LCCV or IPL) on one dataset for a single seed.

    Returns:
        - events: list of dicts, one per (config, anchor) evaluation
        - regret_trace: list of dicts, one per "step" with cumulative cost & regret
        - total_cost: final cumulative cost
    """
    # rng = np.random.RandomState(seed)

    # Determine anchors for this dataset
    final_anchor = int(df["anchor_size"].max())

    # Train surrogate model for this dataset
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    # Build vertical evaluator instance
    evaluator = evaluator_cls(
        surrogate_model=surrogate_model,
        minimal_anchor=int(df["anchor_size"].min()),
        final_anchor=final_anchor,
    )

    # Pre-generate a list of candidate configurations
    # configs: List[Dict] = [dict(config_space.sample_configuration(rng)) for _ in range(n_configs)]

    config_space.seed(seed)

    configs: List[Dict] = [
        dict(config_space.sample_configuration())
        for _ in range(n_configs)
    ]

    # Pre-compute oracle final scores at final anchor for all configs
    oracle_scores: List[float] = []
    for cfg in configs:
        theta = dict(cfg)
        theta["anchor_size"] = final_anchor
        score = float(surrogate_model.predict(theta))
        oracle_scores.append(score)

    oracle_best = float(np.min(oracle_scores))

    # Bookkeeping variables
    events: List[Dict] = []
    regret_trace: List[Dict] = []

    cumulative_cost = 0.0
    best_true_so_far = np.inf   # best oracle score among fully evaluated configs
    best_so_far_for_method = None  # best *observed* final performance (method's view)

    n_full = 0
    n_partial = 0

    # Main loop over configs
    step_counter = 0

    for cfg_id, (cfg, oracle_final) in enumerate(zip(configs, oracle_scores)):
        # Evaluate configuration with vertical evaluator
        evaluations = evaluator.evaluate_model(best_so_far_for_method, cfg)

        if not evaluations:
            # Nothing evaluated (should not happen in our implementations).
            continue

        # Determine if this config reached final anchor
        reached_final = (evaluations[-1][0] == final_anchor)
        if reached_final:
            n_full += 1
        else:
            n_partial += 1

        # Update best_so_far_for_method (only uses the final-anchor prediction)
        if reached_final:
            final_pred = evaluations[-1][1]
            if best_so_far_for_method is None or final_pred < best_so_far_for_method:
                best_so_far_for_method = final_pred

        # Log each evaluation event
        for local_idx, (anchor, perf_anchor) in enumerate(evaluations):
            incremental_cost = eval_cost(anchor, final_anchor, alpha)
            cumulative_cost += incremental_cost
            step_counter += 1

            events.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "seed": seed,
                    "config_index": cfg_id,
                    "event_index": step_counter,
                    "local_eval_index": local_idx,
                    "anchor_size": anchor,
                    "predicted_score_anchor": perf_anchor,
                    "oracle_final_score": oracle_final,
                    "incremental_cost": incremental_cost,
                    "cumulative_cost": cumulative_cost,
                    "reached_final": int(reached_final),
                }
            )

        # Update best_true_so_far using oracle, but *only* for configs that were
        # actually taken to the final anchor by this method.
        if reached_final:
            if oracle_final < best_true_so_far:
                best_true_so_far = oracle_final

        # Regret after finishing this config
        regret = best_true_so_far - oracle_best
        regret_trace.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "seed": seed,
                "cumulative_cost": cumulative_cost,
                "regret": regret,
            }
        )

    # Per-run summary
    total_cost = cumulative_cost
    n_total = n_full + n_partial if (n_full + n_partial) > 0 else 1
    frac_early = n_partial / n_total

    logger.info(
        "[%s | %s | seed=%d] total_cost=%.3f, n_full=%d, n_partial=%d, frac_early=%.3f",
        dataset_name,
        method_name,
        seed,
        total_cost,
        n_full,
        n_partial,
        frac_early,
    )

    # Add one summary dict to the "regret_trace" output? No, we return these
    # separately from the calling function.
    run_summary = {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "total_cost": total_cost,
        "n_full": n_full,
        "n_partial": n_partial,
        "frac_early": frac_early,
        "oracle_best_score": oracle_best,
        "best_true_so_far": best_true_so_far,
        "final_regret": (best_true_so_far - oracle_best),
    }

    return events, regret_trace, run_summary


# ---------------------------------------------------------------------------
# Aggregation: regret vs cost on budget grid
# ---------------------------------------------------------------------------

def build_regret_vs_cost_grid(
    all_traces: List[Dict],
    budgets: np.ndarray,
) -> pd.DataFrame:
    """
    Given a list of per-step traces (dataset, method, seed, cumulative_cost, regret),
    build a regret-vs-cost table on a common budget grid and aggregate (mean, std)
    across seeds.

    For each (dataset, method, budget), we find for each seed the last regret
    value <= that budget (carry-forward). If a seed has not spent that much
    cost yet, its regret is treated as NaN for that budget.

    :param all_traces: list of dicts from run_single_method_on_dataset (regret_trace).
    :param budgets: 1D array of budget values.
    :return: DataFrame with columns:
             dataset, method, budget, mean_regret, std_regret, n_seeds_used
    """
    trace_df = pd.DataFrame(all_traces)

    rows = []

    for (dataset, method), group in trace_df.groupby(["dataset", "method"]):
        # group: all seeds for this dataset & method
        seeds = group["seed"].unique()
        for B in budgets:
            regrets_at_B = []
            for seed in seeds:
                g_seed = group[group["seed"] == seed].sort_values("cumulative_cost")
                # last regret with cost <= B
                g_le = g_seed[g_seed["cumulative_cost"] <= B]
                if g_le.empty:
                    regrets_at_B.append(np.nan)
                else:
                    regrets_at_B.append(float(g_le["regret"].iloc[-1]))
            regrets_arr = np.asarray(regrets_at_B, dtype=float)
            mean_reg = float(np.nanmean(regrets_arr)) if np.any(~np.isnan(regrets_arr)) else np.nan
            std_reg = float(np.nanstd(regrets_arr)) if np.any(~np.isnan(regrets_arr)) else np.nan
            n_used = int(np.sum(~np.isnan(regrets_arr)))

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "budget": B,
                    "mean_regret": mean_reg,
                    "std_regret": std_reg,
                    "n_seeds_used": n_used,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Argument parsing and main entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Final vertical evaluation runner (LCCV vs IPL).")
    parser.add_argument(
        "--config_space_file",
        type=str,
        default="lcdb_config_space_knn.json",
        help="JSON file describing the ConfigSpace for KNN.",
    )
    parser.add_argument(
        "--dataset_files",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of LCDB CSV files. "
            "If omitted, all files matching 'config_performances_dataset-*.csv' "
            "in the current directory will be used."
        ),
    )
    parser.add_argument(
        "--n_configs",
        type=int,
        default=100,
        help="Number of candidate configurations per dataset and seed.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds (runs) per dataset and method.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Exponent for the non-linear cost model c(s) = (s / s_T)^alpha.",
    )
    parser.add_argument(
        "--budget_steps",
        type=int,
        default=50,
        help="Number of budget grid steps (0..max_cost).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vertical_results",
        help="Directory in which CSVs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load config space
    cs_path = Path(args.config_space_file)
    config_space = ConfigurationSpace.from_json(str(cs_path))

    all_events: List[Dict] = []
    all_traces: List[Dict] = []
    all_run_summaries: List[Dict] = []

    methods = {
        "LCCV": LCCV,
        "IPL": IPL,
    }

    if args.dataset_files:
        dataset_files = [str(Path(f)) for f in args.dataset_files]
    else:
        # auto-discover all LCDB datasets of the form config_performances_dataset-*.csv
        dataset_files = sorted(
            str(p) for p in Path(".").glob("config_performances_dataset-*.csv")
        )
        if not dataset_files:
            raise FileNotFoundError(
                "No dataset files found matching 'config_performances_dataset-*.csv'. "
                "Use --dataset_files to specify files explicitly."
            )

    # Run over datasets, methods, seeds
    for dataset_file in dataset_files:
        df = pd.read_csv(dataset_file)
        dataset_name = Path(dataset_file).stem

        for method_name, evaluator_cls in methods.items():
            for seed in range(args.n_seeds):
                events, regret_trace, run_summary = run_single_method_on_dataset(
                    dataset_name=dataset_name,
                    df=df,
                    config_space=config_space,
                    method_name=method_name,
                    evaluator_cls=evaluator_cls,
                    seed=seed,
                    n_configs=args.n_configs,
                    alpha=args.alpha,
                )
                all_events.extend(events)
                all_traces.extend(regret_trace)
                all_run_summaries.append(run_summary)

    # Convert to DataFrames and save
    events_df = pd.DataFrame(all_events)
    events_path = outdir / "vertical_events.csv"
    events_df.to_csv(events_path, index=False)
    logger.info("Wrote per-event log to %s", events_path)

    runs_df = pd.DataFrame(all_run_summaries)
    runs_path = outdir / "vertical_run_summaries.csv"
    runs_df.to_csv(runs_path, index=False)
    logger.info("Wrote per-run summaries to %s", runs_path)

    # Build budget grid using maximum observed cost across all runs
    if not all_traces:
        logger.warning("No traces collected; skipping regret-vs-cost grid.")
        return

    trace_df = pd.DataFrame(all_traces)
    max_cost = float(trace_df["cumulative_cost"].max())
    budgets = np.linspace(0.0, max_cost, num=args.budget_steps)

    regret_grid_df = build_regret_vs_cost_grid(all_traces, budgets)
    grid_path = outdir / "vertical_regret_vs_cost.csv"
    regret_grid_df.to_csv(grid_path, index=False)
    logger.info("Wrote regret-vs-cost grid to %s", grid_path)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(handler)

    main()
