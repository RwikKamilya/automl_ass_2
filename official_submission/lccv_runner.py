from surrogate_model import SurrogateModel
from lccv import LCCV
import pandas as pd
from pathlib import Path
from ConfigSpace import ConfigurationSpace
from pathlib import Path


def main():

    cs_path = Path("lcdb_config_space_knn.json")
    config_space = ConfigurationSpace.from_json(str(cs_path))

    # Load dataset
    df = pd.read_csv("config_performances_dataset-11.csv")

    # Train surrogate (your completed A1 implementation)
    sm = SurrogateModel(config_space)
    sm.fit(df)

    minimal_anchor = int(df["anchor_size"].min())
    final_anchor = int(df["anchor_size"].max())

    # NOTE: we now pass `sm` (SurrogateModel), not `sm.model`
    lccv = LCCV(
        surrogate_model=sm,
        minimal_anchor=minimal_anchor,
        final_anchor=final_anchor,
    )

    cfg = dict(config_space.sample_configuration())
    evals_first = lccv.evaluate_model(best_so_far=None, configuration=cfg)
    print("First config evaluations:", evals_first)


if __name__ == "__main__":
    main()
