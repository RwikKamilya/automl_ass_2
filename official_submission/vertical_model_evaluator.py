from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class VerticalModelEvaluator(ABC):
    def __init__(self, surrogate_model, minimal_anchor: int, final_anchor: int) -> None:
        self.surrogate_model = surrogate_model
        self.minimal_anchor = int(minimal_anchor)
        self.final_anchor = int(final_anchor)

    def _predict_performance(self, configuration: Dict, anchor: int) -> float:
        cfg = dict(configuration)
        cfg["anchor_size"] = int(anchor)
        y_pred = self.surrogate_model.predict(cfg)
        return float(y_pred)

    @abstractmethod
    def evaluate_model(
        self, best_so_far: Optional[float], configuration: Dict
    ) -> List[Tuple[int, float]]:
        raise NotImplementedError()


if __name__ == "__main__":

    class DummySurrogate:

        def predict(self, theta_new: Dict) -> float:
            anchor = float(theta_new["anchor_size"])
            n_neighbors = float(theta_new.get("n_neighbors", 1.0))
            return 0.5 + 0.01 * n_neighbors + 0.001 * anchor

    class TestEvaluator(VerticalModelEvaluator):
        def evaluate_model(self, best_so_far, configuration):
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
