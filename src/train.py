from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .features import FEATURE_SETS, load_or_build_feature_table
from .utils import ensure_data_exists, get_data_paths, make_output_dir


LABELS = [0, 1, 2, 3, 4, 5]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def build_model_specs(random_state: int = 42) -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {
        "dummy": ModelSpec("dummy", DummyClassifier(strategy="most_frequent")),
        "logreg": ModelSpec(
            "logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            C=1.0,
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "random_forest": ModelSpec(
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "extra_trees": ModelSpec(
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=600,
                class_weight="balanced",
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "hist_gradient": ModelSpec(
            "hist_gradient",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                l2_regularization=0.05,
                random_state=random_state,
            ),
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = ModelSpec(
            "lightgbm",
            LGBMClassifier(
                objective="multiclass",
                num_class=len(LABELS),
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=15,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=4,
                force_col_wise=True,
                verbosity=-1,
            ),
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = ModelSpec(
            "xgboost",
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(LABELS),
                n_estimators=700,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    except ImportError:
        pass

    return specs


def _splitter(n_splits: int, random_state: int):
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    except TypeError:
        return GroupKFold(n_splits=n_splits)


def _fit_estimator(estimator, X_train, y_train):
    if isinstance(estimator, DummyClassifier):
        estimator.fit(X_train, y_train)
        return estimator

    model_for_params = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    if getattr(model_for_params, "class_weight", None) is not None:
        estimator.fit(X_train, y_train)
        return estimator

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if isinstance(estimator, Pipeline):
        estimator.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        try:
            estimator.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            estimator.fit(X_train, y_train)
    return estimator


def evaluate_model(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    splitter = _splitter(n_splits=n_splits, random_state=random_state)
    fold_rows = []
    per_class_rows = []
    oof_pred = np.full_like(y, fill_value=-1)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = clone(estimator)
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        _fit_estimator(model, X_train, y_train)
        pred = model.predict(X_valid)
        oof_pred[valid_idx] = pred

        report = classification_report(
            y_valid,
            pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = f1_score(y_valid, pred, average="macro", labels=LABELS, zero_division=0)
        fold_rows.append(
            {
                "fold": fold,
                "macro_f1": macro_f1,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "train_users": len(set(groups[train_idx])),
                "valid_users": len(set(groups[valid_idx])),
            }
        )
        for label in LABELS:
            label_report = report[str(label)]
            per_class_rows.append(
                {
                    "fold": fold,
                    "label": label,
                    "precision": label_report["precision"],
                    "recall": label_report["recall"],
                    "f1": label_report["f1-score"],
                    "support": label_report["support"],
                }
            )
        print(f"Fold {fold}: macro_f1={macro_f1:.5f}")

    return pd.DataFrame(fold_rows), pd.DataFrame(per_class_rows), y, oof_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train assignment 3 HAR models.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full")
    parser.add_argument("--models", nargs="+", default=["extra_trees", "hist_gradient"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--best-model", default=None, help="Override final model choice.")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild features instead of using cached tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_data_paths(args.data_dir)
    ensure_data_exists(paths)
    output_dir = make_output_dir(args.output_dir)

    print(f"Loading features: feature_set={args.feature_set}")
    X, meta = load_or_build_feature_table(
        paths.train_dir,
        feature_set=args.feature_set,
        include_labels=True,
        cache_dir=output_dir / "cache",
        cache_name="train",
        use_cache=not args.no_cache,
    )
    y = meta["label"].to_numpy(dtype=int)
    groups = meta["user"].to_numpy()
    print(f"Feature table: rows={len(X)}, features={X.shape[1]}")

    specs = build_model_specs(random_state=args.random_state)
    unknown = [name for name in args.models if name not in specs]
    if unknown:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")

    all_fold_rows = []
    all_class_rows = []
    confusion_by_model = {}
    summary_rows = []

    for model_name in args.models:
        print(f"Validating model={model_name}")
        fold_df, class_df, y_true, oof_pred = evaluate_model(
            specs[model_name].estimator,
            X,
            y,
            groups,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        fold_df.insert(0, "model", model_name)
        fold_df.insert(1, "feature_set", args.feature_set)
        class_df.insert(0, "model", model_name)
        class_df.insert(1, "feature_set", args.feature_set)
        all_fold_rows.append(fold_df)
        all_class_rows.append(class_df)
        confusion_by_model[model_name] = confusion_matrix(y_true, oof_pred, labels=LABELS)
        summary_rows.append(
            {
                "model": model_name,
                "feature_set": args.feature_set,
                "macro_f1_mean": fold_df["macro_f1"].mean(),
                "macro_f1_std": fold_df["macro_f1"].std(ddof=0),
                "feature_count": X.shape[1],
            }
        )

    validation_df = pd.concat(all_fold_rows, ignore_index=True)
    per_class_df = pd.concat(all_class_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)

    validation_df.to_csv(output_dir / "validation_results.csv", index=False)
    per_class_df.to_csv(output_dir / "per_class_f1.csv", index=False)
    summary_df.to_csv(output_dir / "validation_summary.csv", index=False)

    best_model_name = args.best_model or str(summary_df.iloc[0]["model"])
    cm = pd.DataFrame(confusion_by_model[best_model_name], index=LABELS, columns=LABELS)
    cm.index.name = "true_label"
    cm.columns.name = "pred_label"
    cm.to_csv(output_dir / "confusion_matrix.csv")

    final_model = clone(specs[best_model_name].estimator)
    _fit_estimator(final_model, X, y)
    artifact = {
        "model": final_model,
        "model_name": best_model_name,
        "feature_set": args.feature_set,
        "feature_columns": list(X.columns),
        "labels": LABELS,
    }
    joblib.dump(artifact, output_dir / "model.joblib")

    metadata = {
        "best_model": best_model_name,
        "feature_set": args.feature_set,
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(X)),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "available_models": sorted(specs),
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Validation summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved model artifact: {output_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
