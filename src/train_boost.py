"""
Train a Histogram Gradient‑Boosting model and save it to models/model_boost.pkl.

Usage:
    python -m src.train_boost
"""
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .config import CLEAN_DATA, MODEL_DIR, RANDOM_SEED, TEST_SIZE

# ──────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Load data
    df = pd.read_csv(CLEAN_DATA)
    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(int)   # 0 / 1 target

    # 2. Columns
    cat_cols = ["workclass", "occupation"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 3. Pre‑processing
    preproc = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),

        ]
    )

    # 4. Model: Histogram Gradient‑Boosting
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_iter=400,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    pipe = Pipeline([("prep", preproc), ("model", clf)])

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # 6. Handle imbalance via sample‑weights
    sample_weights = compute_sample_weight("balanced", y_train)

    pipe.fit(X_train, y_train, model__sample_weight=sample_weights)

    # 7. Metrics
    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)[:, 1]
    roc     = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)

    print(f"HistGB  ROC‑AUC: {roc:.3f}  •  F1: {f1:.3f}")

    # Cross‑validated ROC‑AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_roc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"CV ROC‑AUC: {cv_roc.mean():.3f} ± {cv_roc.std():.3f}")

    # 8. Save
    MODEL_DIR.mkdir(exist_ok=True)
    boost_path = MODEL_DIR / "model_boost.pkl"
    joblib.dump(pipe, boost_path)
    print(f"✅  Saved Gradient‑Boosting model to {boost_path}")

if __name__ == "__main__":
    main()
