# Hyperparameter Optimization using Learning Curves  
AutoML Assignment 2 – Reproducibility Instructions

This project implements and evaluates two vertical HPO methods (LCCV and IPL) on k-NN learning curves from LCDB for datasets 6, 11, and 1457.

Follow the instructions **in the order given below** to reproduce the results and plots used in the report.

---

## 1. Project structure

The zip contains the following files:

- `analyse_lcdb.py`
- `config_performances_dataset-6.csv`
- `config_performances_dataset-11.csv`
- `config_performances_dataset-1457.csv`
- `example_run_experiment.py`
- `ipl.py`
- `lccv.py`
- `lcdb_config_space_knn.json`
- `plot_results.py`
- `requirements.txt`
- `run_experiment.py`
- `surrogate_model.py`
- `vertical_model_evaluator.py`
- `README.md`

The following folders will be created automatically when you run the pipeline:

- `results/` – CSVs and logs from the experiments  
- `plots/` – PNG figures (including the plots used in the report)

---

## 2. Setup

1. Ensure you have **Python 3** installed.
2. (Optional but recommended) Create and activate a virtual environment.
3. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   
## 3. Run

1. Run the unit tests

   ```bash
   python ipl.py
   python lccv.py
   python vertical_model_evaluator.py
   
2. Then the experiment

    ```bash
   python run_experiment.py

3. After the experiment ends, plot the results
   
    ```bash
   python plot_results.py
   python analyse_lcdb.py

## 4. Results

1. The unit tests should pass.
2. The experiment should create a new folder called results, containing run statistics.
3. The plots then use these statistics from results to plot the graphs in the plots folder.


