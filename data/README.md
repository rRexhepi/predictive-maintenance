# `data/`

Raw and preprocessed datasets live here. **This directory is gitignored
except for this README.** Contributors produce their own data.

## Expected layout

```
data/
├── train_FD001.txt        # raw NASA C-MAPSS turbofan engine degradation set
├── test_FD001.txt         # (optional) held-out test engines
├── RUL_FD001.txt          # (optional) true RUL labels for the test set
├── train_preprocessed.csv # produced by src/preprocessing.py
└── val_preprocessed.csv   # produced by src/preprocessing.py
```

## Getting the data

The project is wired against the NASA C-MAPSS turbofan engine
degradation dataset, published on NASA's data portal and mirrored on
Kaggle:

- NASA PCoE: https://www.nasa.gov/intelligent-systems-division (search "C-MAPSS")
- Kaggle mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Drop the raw space-separated files (`train_FD001.txt` etc.) into this
directory, then run the preprocessing pipeline:

```bash
python src/preprocessing.py
```

That produces `train_preprocessed.csv` and `val_preprocessed.csv` which
`src/train_model.py` consumes.
