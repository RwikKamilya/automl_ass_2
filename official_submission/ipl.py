import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from vertical_model_evaluator import VerticalModelEvaluator

logger = logging.getLogger(__name__)


class IPL(VerticalModelEvaluator):

    def _partial_anchor_schedule(self) -> List[int]:
        if self.minimal_anchor >= self.final_anchor:
            # No room for partial anchors.
            return []

        anchors = [int(self.minimal_anchor)]
        while True:
            next_anchor = anchors[-1] * 2
            if next_anchor >= self.final_anchor:
                break
            anchors.append(next_anchor)
        return anchors

    @staticmethod
    def _fit_ipl(anchors: List[int], scores: List[float]) -> Tuple[float, float, float]:
        s = np.asarray(anchors, dtype=float)
        y = np.asarray(scores, dtype=float)
        if len(s) < 2:
            a = float(y.mean())
            b = 0.0
            c = 0.0
            return a, b, c

        eps = 1e-8
        y_min = float(y.min())
        y_shift = y - y_min + eps

        X = np.log(s)
        y_log = np.log(y_shift)

        x_mean = X.mean()
        y_mean = y_log.mean()

        var = np.sum((X - x_mean) ** 2)
        if var <= 0:
            alpha = y_mean
            beta = 0.0
        else:
            cov = np.sum((X - x_mean) * (y_log - y_mean))
            beta = cov / var
            alpha = y_mean - beta * x_mean

        k = float(np.exp(alpha))
        c = max(0.0, -float(beta))  # enforce non-negative exponent
        a = y_min - eps

        b = k
        return a, b, c

    @staticmethod
    def _ipl_predict(s: float, a: float, b: float, c: float) -> float:
        if c == 0.0:
            return float(a + b)
        return float(a + b * (s ** (-c)))

    def evaluate_model(
            self, best_so_far: Optional[float], configuration: Dict
    ) -> List[Tuple[int, float]]:
        evaluations: List[Tuple[int, float]] = []

        partial_anchors = self._partial_anchor_schedule()

        partial_scores: List[float] = []
        for anchor in partial_anchors:
            perf = self._predict_performance(configuration, anchor)
            evaluations.append((anchor, perf))
            partial_scores.append(perf)
            logger.debug("IPL: partial anchor %d -> %.4f", anchor, perf)

        if len(partial_anchors) < 2:
            perf_final = self._predict_performance(configuration, self.final_anchor)
            evaluations.append((self.final_anchor, perf_final))
            logger.debug(
                "IPL: insufficient partial points, directly evaluated final anchor %d -> %.4f",
                self.final_anchor,
                perf_final,
            )
            return evaluations

        a, b, c = self._fit_ipl(partial_anchors, partial_scores)
        logger.debug("IPL: fitted params a=%.6f, b=%.6f, c=%.6f", a, b, c)

        pred_final = self._ipl_predict(self.final_anchor, a, b, c)
        logger.debug(
            "IPL: predicted final performance at anchor %d -> %.4f",
            self.final_anchor,
            pred_final,
        )

        if best_so_far is not None and pred_final >= best_so_far:
            logger.debug(
                "IPL: discarding config, predicted %.4f >= best_so_far %.4f",
                pred_final,
                best_so_far,
            )
            return evaluations

        perf_final = self._predict_performance(configuration, self.final_anchor)
        evaluations.append((self.final_anchor, perf_final))
        logger.debug(
            "IPL: evaluated final anchor %d -> %.4f (predicted %.4f)",
            self.final_anchor,
            perf_final,
            pred_final,
        )
        return evaluations


if __name__ == "__main__":
    # ----------------- Basic unit tests for IPL -----------------
    logging.basicConfig(level=logging.INFO)


    def true_curve(s):
        s = float(s)
        return 1.0 + 1.0 / (s ** 0.5)


    anchors = [16, 32, 64, 128]
    scores = [true_curve(s) for s in anchors]

    a, b, c = IPL._fit_ipl(anchors, scores)
    s_test = 128
    pred = IPL._ipl_predict(s_test, a, b, c)
    true_val = true_curve(s_test)
    assert abs(pred - true_val) < 0.02, f"IPL fit too far off at {s_test}: pred={pred}, true={true_val}"


    class DummySurrogate:

        def predict(self, theta_new: Dict) -> float:
            s = float(theta_new["anchor_size"])
            return true_curve(s)


    surrogate = DummySurrogate()
    minimal_anchor = 16
    final_anchor = 128
    evaluator = IPL(surrogate_model=surrogate, minimal_anchor=minimal_anchor, final_anchor=final_anchor)

    cfg = {"n_neighbors": 5}  # arbitrary; surrogate ignores it

    # Case A: best_so_far is None -> should evaluate partial schedule + final anchor.
    evals_first = evaluator.evaluate_model(best_so_far=None, configuration=cfg)
    partial_schedule = evaluator._partial_anchor_schedule()
    # Expect len(partials) + 1 for final anchor
    assert len(evals_first) == len(partial_schedule) + 1, "First config should evaluate all partial anchors + final."
    assert evals_first[-1][0] == final_anchor, "Last evaluation should be at final anchor."
    final_score_first = evals_first[-1][1]
    assert abs(final_score_first - true_curve(final_anchor)) < 1e-8

    # Case B: best_so_far is quite strict -> likely discard (no final evaluation)
    strict_best = 1.05  # better (lower) than the true final performance.
    evals_second = evaluator.evaluate_model(best_so_far=strict_best, configuration=cfg)
    # With such a strong best_so_far, IPL prediction should not beat it, so we discard.
    assert len(evals_second) == len(partial_schedule), (
        "With strict best_so_far, config should be discarded after partial schedule."
    )
    assert all(a < final_anchor for (a, _) in evals_second), "No final-anchor evaluations expected."

    # Case C: very loose best_so_far -> we should evaluate final anchor as well.
    loose_best = 2.0  # much worse than any realistic error here
    evals_third = evaluator.evaluate_model(best_so_far=loose_best, configuration=cfg)
    assert len(evals_third) == len(partial_schedule) + 1, "With loose best_so_far, final anchor should be evaluated."
    assert evals_third[-1][0] == final_anchor
    print("ipl: all basic tests passed.")
