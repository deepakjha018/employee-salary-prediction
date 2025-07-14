"""
CLI script to score new employee records.

Examples
--------
Predict one record directly from CLI:

    python -m src.predict \\
        --age 40 --workclass Private --educational-num 13 \\
        --occupation "Exec-managerial" --hours-per-week 50 \\
        --capital-gain 0 --capital-loss 0

Score a CSV file and write results:

    python -m src.predict --csv path/to/new_data.csv --out predictions.csv
"""
import argparse, json
from pathlib import Path
import pandas as pd

from .utils import load_model, predict_sample, FEATURE_ORDER

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Employee salary prediction CLI")
    p.add_argument("--model", type=str, help="Path to .pkl model",
                   default=None)

    # Mutually exclusive: a CSV file OR individual fields
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--csv", type=str,
                       help="CSV file with columns matching cleaned dataset")
    group.add_argument("--json", type=str,
                       help="Path to JSON file with a list of records")

    # Individual field inputs (usable if neither --csv nor --json given)
    for col in FEATURE_ORDER:
        p.add_argument(f"--{col.replace('_','-')}", type=str,
                       help=f"Field: {col}")

    p.add_argument("--out", type=str,
                   help="Output CSV for predictions (if using --csv/--json)")
    return p


def _cli_single_record(args):
    record = {c: getattr(args, c.replace('-', '_')) for c in FEATURE_ORDER}
    # Convert numerics
    for k in ["age", "educational-num", "hours-per-week", "capital-gain", "capital-loss"]:
        record[k] = int(record[k])

    pred = predict_sample(record, model_path=args.model)
    print(json.dumps(pred, indent=2))


def _cli_batch_csv(args):
    model = load_model(args.model) if args.model else load_model()
    df = pd.read_csv(args.csv)
    proba = model.predict_proba(df)[:, 1]
    labels = (proba >= 0.5).astype(int).map({0: "<=50K", 1: ">50K"})
    df["pred_probability"] = proba
    df["pred_label"] = labels

    out_path = args.out or Path(args.csv).with_suffix(".pred.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to {out_path}")


def _cli_batch_json(args):
    model = load_model(args.model) if args.model else load_model()
    records = json.loads(Path(args.json).read_text())
    df = pd.DataFrame(records)
    proba = model.predict_proba(df)[:, 1]
    labels = (proba >= 0.5).astype(int).map({0: "<=50K", 1: ">50K"})
    df["pred_probability"] = proba
    df["pred_label"] = labels

    out_path = args.out or Path(args.json).with_suffix(".pred.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to {out_path}")


def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    if args.csv:
        _cli_batch_csv(args)
    elif args.json:
        _cli_batch_json(args)
    elif all(getattr(args, f.replace("-", "_")) is not None for f in FEATURE_ORDER):
        _cli_single_record(args)
    else:
        print("❌ Error: Please provide either --csv, --json or all individual input fields.")



if __name__ == "__main__":
    main()
