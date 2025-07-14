# src/train.py
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLEAN_DATA, MODEL_PATH, RANDOM_SEED, TEST_SIZE

def main():
    df = pd.read_csv(CLEAN_DATA)

    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(int)  # make it 0/1

    cat_cols = ["workclass", "occupation"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handles imbalance
        random_state=RANDOM_SEED
    )

    pipe = Pipeline([("prep", preproc), ("model", clf)])

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    pipe.fit(X_train, y_train)

    # ─────────── metrics ───────────
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)

    print(f"ROC‑AUC: {roc:.3f}  •  F1: {f1:.3f}")

    # cross‑validated ROC‑AUC for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_roc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC‑AUC: {cv_roc.mean():.3f} ± {cv_roc.std():.3f}")

    # Save the whole pipeline
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅  Saved trained model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
