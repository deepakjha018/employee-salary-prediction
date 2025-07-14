"""Model evaluation script: generates confusion matrix, ROC curve, and PR curve.

Usage:
    python -m src.evaluate
"""
from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .config import CLEAN_DATA, BOOST_MODEL_PATH, RANDOM_SEED, TEST_SIZE

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def main() -> None:
    # Load data and model
    df = pd.read_csv(CLEAN_DATA)
    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(int)

    model = joblib.load(BOOST_MODEL_PATH)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    cm_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.tight_layout()
    roc_path = REPORT_DIR / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
    plt.tight_layout()
    pr_path = REPORT_DIR / "pr_curve.png"
    plt.savefig(pr_path)
    plt.close()

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    print("\nClassification Report:\n", report)
    (REPORT_DIR / "classification_report.txt").write_text(report)

    print("✅ Reports saved to", REPORT_DIR)

if __name__ == "__main__":
    main()
