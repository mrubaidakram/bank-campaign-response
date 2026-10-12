import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import ensure_data

def build_pipeline(frame):
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    transform = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    return Pipeline([("features", transform), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])

def evaluate(model, x, y):
    prediction = model.predict(x); probabilities = model.predict_proba(x)[:, 1]
    return {"accuracy": accuracy_score(y, prediction), "roc_auc": roc_auc_score(y, probabilities), "average_precision": average_precision_score(y, probabilities),
        "confusion_matrix": confusion_matrix(y, prediction).tolist(), "classification_report": classification_report(y, prediction, output_dict=True, zero_division=0)}

def main():
    parser = argparse.ArgumentParser(description="Train a leakage-aware UCI Bank Marketing baseline.")
    parser.add_argument("--data-dir", default="data"); parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--include-duration", action="store_true", help="Include call duration; only valid for post-call analysis.")
    args = parser.parse_args()
    frame = pd.read_csv(ensure_data(args.data_dir), sep=";")
    if "y" not in frame or len(frame) < 3: raise ValueError("Unexpected UCI Bank Marketing file schema")
    frame["target"] = (frame.pop("y").astype(str).str.strip().str.lower() == "yes").astype("int8")
    if not args.include_duration and "duration" in frame: frame = frame.drop(columns=["duration"])
    # UCI documents the full dataset as ordered by campaign date; preserve chronology.
    test_start = int(len(frame) * 0.8); train_frame, test_frame = frame.iloc[:test_start], frame.iloc[test_start:]
    validation_start = int(len(train_frame) * 0.85); fit_frame, validation_frame = train_frame.iloc[:validation_start], train_frame.iloc[validation_start:]
    target = "target"; xfit, yfit = fit_frame.drop(columns=target), fit_frame[target]
    model = build_pipeline(xfit).fit(xfit, yfit)
    validation = evaluate(model, validation_frame.drop(columns=target), validation_frame[target])
    final_model = build_pipeline(train_frame.drop(columns=target)).fit(train_frame.drop(columns=target), train_frame[target])
    test = evaluate(final_model, test_frame.drop(columns=target), test_frame[target])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); joblib.dump(final_model, output / "bank_campaign_pipeline.joblib")
    report = {"validation": validation, "test": test, "split": "chronological 68/12/20", "duration_included": args.include_duration, "seed": 42}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_average_precision": validation["average_precision"], "test_average_precision": test["average_precision"], "test_roc_auc": test["roc_auc"]}, indent=2))

if __name__ == "__main__": main()
