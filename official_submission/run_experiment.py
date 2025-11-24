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

def eval_cost(anchor_size: int, final_anchor: int, alpha: float) -> float:
    return float((anchor_size / final_anchor) ** alpha)


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

    final_anchor = int(df["anchor_size"].max())

    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    logger.info(
        "[%s] Surrogate hold-out Spearman=%.3f (n_test=%d)",
        dataset_name,
        surrogate_model.spearman_test_,
        surrogate_model.n_test_,
    )

    evaluator = evaluator_cls(
        surrogate_model=surrogate_model,
        minimal_anchor=int(df["anchor_size"].min()),
        final_anchor=final_anchor,
    )

    config_space.seed(seed)

    configs: List[Dict] = [
        dict(config_space.sample_configuration())
        for _ in range(n_configs)
    ]

    oracle_scores: List[float] = []
    for cfg in configs:
        theta = dict(cfg)
        theta["anchor_size"] = final_anchor
        score = float(surrogate_model.predict(theta))
        oracle_scores.append(score)

    oracle_best = float(np.min(oracle_scores))

    events: List[Dict] = []
    regret_trace: List[Dict] = []

    cumulative_cost = 0.0
    best_true_so_far = np.inf
    best_so_far_for_method = None

    n_full = 0
    n_partial = 0

    step_counter = 0

    for cfg_id, (cfg, oracle_final) in enumerate(zip(configs, oracle_scores)):
        evaluations = evaluator.evaluate_model(best_so_far_for_method, cfg)

        if not evaluations:
            continue

        reached_final = (evaluations[-1][0] == final_anchor)
        if reached_final:
            n_full += 1
        else:
            n_partial += 1

        if reached_final:
            final_pred = evaluations[-1][1]
            if best_so_far_for_method is None or final_pred < best_so_far_for_method:
                best_so_far_for_method = final_pred

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

        if reached_final:
            if oracle_final < best_true_so_far:
                best_true_so_far = oracle_final

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


def build_regret_vs_cost_grid(
    all_traces: List[Dict],
    budgets: np.ndarray,
) -> pd.DataFrame:
    trace_df = pd.DataFrame(all_traces)

    rows = []

    for (dataset, method), group in trace_df.groupby(["dataset", "method"]):
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
        default="results",
        help="Directory in which CSVs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        dataset_files = sorted(
            str(p) for p in Path(".").glob("config_performances_dataset-*.csv")
        )
        if not dataset_files:
            raise FileNotFoundError(
                "No dataset files found matching 'config_performances_dataset-*.csv'. "
                "Use --dataset_files to specify files explicitly."
            )

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

    events_df = pd.DataFrame(all_events)
    events_path = outdir / "vertical_events.csv"
    events_df.to_csv(events_path, index=False)
    logger.info("Wrote per-event log to %s", events_path)

    runs_df = pd.DataFrame(all_run_summaries)
    runs_path = outdir / "vertical_run_summaries.csv"
    runs_df.to_csv(runs_path, index=False)
    logger.info("Wrote per-run summaries to %s", runs_path)

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
