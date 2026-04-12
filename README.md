# Hospital Rating Modeling With MCDA and Machine Learning

## Project Description

This project analyzes hospital quality data and predicts overall hospital rating categories using three modeling approaches:

- XGBoost classifier with monotonic constraints.
- Standard feed-forward neural network.
- ANN-UTADIS model inspired by Multiple Criteria Decision Analysis (MCDA), with monotonic preference modeling.

The goal is to compare predictive performance and interpretability while keeping the model logic consistent with decision-analysis assumptions (for example monotonic utility with respect to selected criteria).

## Notebooks

The notebooks directory contains the full experimental workflow:

- `notebooks/ANN-UTADIS.ipynb`: training and evaluating the custom ANN-UTADIS model, including persistence utilities and metric evaluation.
- `notebooks/StandardNN.ipynb`: baseline neural network experiment with standard preprocessing and training pipeline.
- `notebooks/XGBoost.ipynb`: gradient boosted tree model with monotonic constraints and saved artifact export.
- `notebooks/Report.ipynb`: integrated report notebook with data analysis, model comparison, diagnostics, and interpretability experiments (including optional SHAP/DALEX usage).

## Dataset

The dataset is based on publicly available CMS Hospital Compare sources and is assembled from multiple hospital-level CSV files in `data/raw/`.

Raw tables include quality-related indicators such as:

- mortality and complications,
- unplanned hospital visits/readmissions,
- spending per patient,
- general hospital information.

The preprocessing pipeline in `data/preprocess.py` merges these sources into a single modeling table (`data/hospital_data.csv`) and prepares features with a consistent preference direction suitable for MCDA-style learning.

## Project Structure

- `data/`: raw data files and preprocessing script.
- `models/`: saved trained model artifacts (`.pt`, `.joblib`).
- `notebooks/`: exploratory analysis, training workflows, and final report.
- `src/`: reusable Python package code:
	- `src/metrics.py`: shared evaluation metrics.
	- `src/ann_utadis/`: ANN-UTADIS architecture, custom layers/losses, training, and model persistence.
- `requirements.txt`: Python dependencies for reproducing experiments.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run `data/preprocess.py` if you want to regenerate the processed dataset.
4. Open and run notebooks from `notebooks/`.
