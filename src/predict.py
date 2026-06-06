from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import load_or_build_feature_table
from .utils import ensure_data_exists, get_data_paths, make_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate assignment 3 Kaggle submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--submission-name", default="submission.csv")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild features instead of using cached tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_data_paths(args.data_dir)
    ensure_data_exists(paths, require_test=True)
    output_dir = make_output_dir(args.output_dir)

    model_path = Path(args.model_path) if args.model_path else output_dir / "model.joblib"
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}\n"
            "Run `.venv/bin/python -m src.train` first."
        )

    artifact = joblib.load(model_path)
    feature_set = artifact["feature_set"]
    feature_columns = artifact["feature_columns"]
    model = artifact["model"]

    print(f"Loading test features: feature_set={feature_set}")
    X_test, meta = load_or_build_feature_table(
        paths.test_dir,
        feature_set=feature_set,
        include_labels=False,
        cache_dir=output_dir / "cache",
        cache_name="test",
        use_cache=not args.no_cache,
    )
    X_test = X_test.reindex(columns=feature_columns, fill_value=0.0)
    pred = model.predict(X_test).astype(int)

    predictions = pd.DataFrame({"Id": meta["file_id"].astype(int), "Label": pred})
    sample = pd.read_csv(paths.sample_submission)
    submission = sample[["Id"]].merge(predictions, on="Id", how="left")
    if submission["Label"].isna().any():
        missing = submission.loc[submission["Label"].isna(), "Id"].head(10).tolist()
        raise ValueError(f"Missing predictions for sample IDs, first examples: {missing}")
    submission["Label"] = submission["Label"].astype(int)

    output_path = output_dir / args.submission_name
    submission.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")
    print(f"Rows: {len(submission)}")


if __name__ == "__main__":
    main()
