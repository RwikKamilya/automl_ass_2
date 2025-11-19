from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class VerticalModelEvaluator(ABC):
    """
    Base class for vertical model selection methods that evaluate a configuration
    on multiple anchor sizes (learning-curve points) using an external surrogate
    model.

    IMPORTANT: We rely on the API of the Assignment 1 SurrogateModel:

        surrogate_model.predict(theta_new: dict) -> float

    where `theta_new` is a dict containing all hyperparameters AND "anchor_size".
    We deliberately do NOT try to build feature arrays here; that logic
    already exists in SurrogateModel and must be reused to correctly handle
    conditional hyperparameters and ConfigSpace.
    """

    def __init__(self, surrogate_model, minimal_anchor: int, final_anchor: int) -> None:
        """
        Initialise the vertical model evaluator.

        :param surrogate_model:
            An object with a `predict(theta_new: dict) -> float` method.
            In this assignment, this is typically an instance of
            `SurrogateModel` from Assignment 1.
        :param minimal_anchor: Smallest anchor size to be used.
        :param final_anchor: Largest / final anchor size to be used.
        """
        self.surrogate_model = surrogate_model
        self.minimal_anchor = int(minimal_anchor)
        self.final_anchor = int(final_anchor)

    def _predict_performance(self, configuration: Dict, anchor: int) -> float:
        """
        Helper that calls the surrogate model for a single configuration+anchor.

        :param configuration: Dict of hyperparameters (no "anchor_size" key).
        :param anchor: Anchor size at which to evaluate.
        :return: Predicted performance (float), lower is assumed to be better.
        """
        cfg = dict(configuration)
        cfg["anchor_size"] = int(anchor)
        # Delegate to the Assignment 1 SurrogateModel.predict
        y_pred = self.surrogate_model.predict(cfg)
        return float(y_pred)

    @abstractmethod
    def evaluate_model(
        self, best_so_far: Optional[float], configuration: Dict
    ) -> List[Tuple[int, float]]:
        """
        Evaluate a configuration on one or more anchor sizes.

        :param best_so_far:
            Best (lowest) performance seen so far. If None, the evaluator may
            choose to evaluate directly at the final anchor.
        :param configuration:
            Configuration dictionary (hyperparameters only, without "anchor_size").
        :return:
            A list of (anchor_size, estimated_performance) pairs in the order
            they were evaluated.
        """
        raise NotImplementedError()


if __name__ == "__main__":
    # ---- Minimal unit test for VerticalModelEvaluator helper ----

    class DummySurrogate:
        """
        Dummy surrogate mimicking the SurrogateModel API from Assignment 1.

        It expects a dict with at least "anchor_size" and "n_neighbors" keys and
        returns a simple deterministic performance:
            perf = 0.5 + 0.01 * n_neighbors + 0.001 * anchor_size
        """

        def predict(self, theta_new: Dict) -> float:
            anchor = float(theta_new["anchor_size"])
            n_neighbors = float(theta_new.get("n_neighbors", 1.0))
            return 0.5 + 0.01 * n_neighbors + 0.001 * anchor

    class TestEvaluator(VerticalModelEvaluator):
        def evaluate_model(self, best_so_far, configuration):
            # Just evaluate once at minimal_anchor
            perf = self._predict_performance(configuration, self.minimal_anchor)
            return [(self.minimal_anchor, perf)]

    surrogate = DummySurrogate()
    evaluator = TestEvaluator(surrogate_model=surrogate, minimal_anchor=10, final_anchor=80)

    cfg = {"n_neighbors": 3}
    evals = evaluator.evaluate_model(best_so_far=None, configuration=cfg)

    assert len(evals) == 1, "Expected exactly one evaluation."
    anchor, perf = evals[0]
    assert anchor == 10, f"Expected anchor 10, got {anchor}"

    expected_perf = 0.5 + 0.01 * 3 + 0.001 * 10  # 0.5 + 0.03 + 0.01 = 0.54
    assert abs(perf - expected_perf) < 1e-8, f"Unexpected perf {perf} vs {expected_perf}"

    print("vertical_model_evaluator: basic helper test passed.")
