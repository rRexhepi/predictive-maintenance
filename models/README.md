# `models/`

Trained model artifacts (pickle files) live here. **This directory is
gitignored except for this README.** Binaries don't belong in git, and
contributors produce their own from `src/train_model.py`.

## Expected layout after training

```
models/
├── rf_model.pkl   # RandomForestRegressor fit by src/train_model.py
└── scaler.pkl    # StandardScaler fit by src/preprocessing.py
```

The API (`src/api/main.py`) reads from `MODEL_PATH` and `SCALER_PATH`
environment variables, defaulting to the paths above.

## How to produce them

```bash
# 1. Drop the raw dataset into ../data/ (see ../data/README.md)
# 2. Preprocess and save the scaler:
python src/preprocessing.py
# 3. Train + save the estimator:
python src/train_model.py
```
