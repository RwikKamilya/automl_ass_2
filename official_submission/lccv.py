import logging
from typing import Dict, List, Optional, Tuple

from vertical_model_evaluator import VerticalModelEvaluator

import numpy as np  # only used in the self-test


logger = logging.getLogger(__name__)


class LCCV(VerticalModelEvaluator):

    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int,
        previous_performance: float,
        current_anchor: int,
        current_performance: float,
        target_anchor: int,
    ) -> float:
        if current_anchor == previous_anchor:
            raise ValueError("previous_anchor and current_anchor must be different.")

        if target_anchor < current_anchor:
            raise ValueError(
                f"target_anchor ({target_anchor}) must be >= current_anchor ({current_anchor})."
            )

        slope = (current_performance - previous_performance) / (
            current_anchor - previous_anchor
        )
        extrapolated = current_performance + slope * (target_anchor - current_anchor)
        return float(extrapolated)

    def _anchor_schedule(self) -> List[int]:
        anchors = [int(self.minimal_anchor)]
        if self.final_anchor <= self.minimal_anchor:
            return [int(self.final_anchor)]

        while anchors[-1] < self.final_anchor:
            next_anchor = min(self.final_anchor, anchors[-1] * 2)
            if next_anchor == anchors[-1]:
                break
            anchors.append(next_anchor)

        if anchors[-1] != self.final_anchor:
            anchors.append(int(self.final_anchor))

        return anchors

    def evaluate_model(
        self, best_so_far: Optional[float], configuration: Dict
    ) -> List[Tuple[int, float]]:
        evaluations: List[Tuple[int, float]] = []

        # Case 1: first configuration – evaluate directly at final anchor.
        if best_so_far is None:
            perf = self._predict_performance(configuration, self.final_anchor)
            evaluations.append((self.final_anchor, perf))
            logger.debug(
                "LCCV: best_so_far is None, evaluated only at final anchor %d -> %.4f",
                self.final_anchor,
                perf,
            )
            return evaluations

        # Case 2: staged evaluation with early stopping.
        anchors = self._anchor_schedule()
        previous_anchor: Optional[int] = None
        previous_perf: Optional[float] = None

        for anchor in anchors:
            perf = self._predict_performance(configuration, anchor)
            evaluations.append((anchor, perf))
            logger.debug("LCCV: evaluated anchor %d -> %.4f", anchor, perf)

            if previous_anchor is not None and previous_perf is not None:
                optimistic = self.optimistic_extrapolation(
                    previous_anchor=previous_anchor,
                    previous_performance=previous_perf,
                    current_anchor=anchor,
                    current_performance=perf,
                    target_anchor=self.final_anchor,
                )
                logger.debug(
                    "LCCV: optimistic extrapolation from (%d, %.4f) and (%d, %.4f) "
                    "to final anchor %d -> %.4f",
                    previous_anchor,
                    previous_perf,
                    anchor,
                    perf,
                    self.final_anchor,
                    optimistic,
                )

                if optimistic >= best_so_far:
                    logger.debug(
                        "LCCV: stopping early, optimistic %.4f >= best_so_far %.4f",
                        optimistic,
                        best_so_far,
                    )
                    break

            previous_anchor, previous_perf = anchor, perf

        return evaluations


if __name__ == "__main__":
    # ---- Unit tests for LCCV ----
    logging.basicConfig(level=logging.INFO)

    # Test optimistic_extrapolation numerically
    prev_a, prev_p = 10, 1.0
    curr_a, curr_p = 20, 0.8
    target_a = 40
    # slope = (0.8 - 1.0) / (20 - 10) = -0.02
    # C_hat(40) = 0.8 + (-0.02) * (40 - 20) = 0.4
    expected_opt = 0.4
    got_opt = LCCV.optimistic_extrapolation(prev_a, prev_p, curr_a, curr_p, target_a)
    assert abs(got_opt - expected_opt) < 1e-8, f"Unexpected optimistic extrapolation {got_opt} vs {expected_opt}"

    # Dummy surrogate that follows a simple linear pattern in anchor_size
    class DummySurrogate:

        def predict(self, theta_new: Dict) -> float:
            anchor = float(theta_new["anchor_size"])
            return 0.1 + 0.001 * anchor

    surrogate = DummySurrogate()
    evaluator = LCCV(surrogate_model=surrogate, minimal_anchor=10, final_anchor=80)

    cfg = {"n_neighbors": 5}  # arbitrary, surrogate ignores it

    # Case A: best_so_far is None → evaluate only at final anchor
    evals_first = evaluator.evaluate_model(best_so_far=None, configuration=cfg)
    assert len(evals_first) == 1, "Expected a single evaluation for first config."
    assert evals_first[0][0] == 80
    expected_final = 0.1 + 0.001 * 80  # 0.18
    assert abs(evals_first[0][1] - expected_final) < 1e-8

    # Case B: staged evaluation with early stopping
    best_so_far = 0.15  # a fairly good value (lower is better)

    evals_second = evaluator.evaluate_model(best_so_far=best_so_far, configuration=cfg)
    # With DummySurrogate:
    # anchor 10 -> 0.11
    # anchor 20 -> 0.12
    # optimistic from (10,0.11) and (20,0.12) to 80:
    # slope = (0.12 - 0.11)/(20-10) = 0.001
    # C_hat(80) = 0.12 + 0.001 * (80-20) = 0.18 >= best_so_far (0.15)
    # => should stop after anchor 20 (2 evaluations total).
    assert len(evals_second) == 2, f"Expected early stopping after 2 anchors, got {len(evals_second)}"
    assert evals_second[0][0] == 10 and abs(evals_second[0][1] - 0.11) < 1e-8
    assert evals_second[1][0] == 20 and abs(evals_second[1][1] - 0.12) < 1e-8

    print("lccv: all basic tests passed.")
