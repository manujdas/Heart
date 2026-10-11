"""Run BEFORE deploying:  python check_env.py
Prints the versions to pin in requirements.txt and checks the model files."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit

print("Python        :", sys.version.split()[0])
print("streamlit     :", streamlit.__version__)
print("pandas        :", pd.__version__)
print("numpy         :", np.__version__)
print("scikit-learn  :", sklearn.__version__)
print("joblib        :", joblib.__version__)
print()

model_dir = Path(__file__).resolve().parent / "Model"
for name in ["RF_heart.pkl", "scaler.pkl", "feature_columns.pkl"]:
    p = model_dir / name
    if p.exists():
        print(f"OK   {name:22s} {p.stat().st_size / 1_048_576:.2f} MB")
    else:
        print(f"MISSING {name}")

cols = joblib.load(model_dir / "feature_columns.pkl")
print("\nFeature columns:", len(cols))
print(list(cols))
print("\nGitHub limit is 100 MB per file.")