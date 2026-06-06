from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .utils import list_window_files, read_window, user_from_path


BASE_COLUMNS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
FULL_FEATURE_SETS = {
    "full",
    "full_correlation",
    "full_peak",
    "full_rolling",
    "full_peak_rolling",
    "full_corr_peak_rolling",
}
FEATURE_SETS = (
    "basic",
    "magnitude",
    "temporal",
    "fft",
    "full",
    "full_correlation",
    "full_peak",
    "full_rolling",
    "full_peak_rolling",
    "full_corr_peak_rolling",
)


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-12)


def _signals(df: pd.DataFrame, feature_set: str) -> dict[str, np.ndarray]:
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    signals = {col: df[col].to_numpy(dtype=float) for col in BASE_COLUMNS}
    if feature_set in {"magnitude", "temporal", "fft", "full"}:
        mean_x = signals["mean_x"]
        mean_y = signals["mean_y"]
        mean_z = signals["mean_z"]
        std_x = signals["std_x"]
        std_y = signals["std_y"]
        std_z = signals["std_z"]
        signals.update(
            {
                "acc_mag": np.sqrt(mean_x**2 + mean_y**2 + mean_z**2),
                "dyn_mag": np.sqrt(std_x**2 + std_y**2 + std_z**2),
                "horiz_mag": np.sqrt(mean_x**2 + mean_y**2),
                "mean_xy_diff": mean_x - mean_y,
                "mean_yz_diff": mean_y - mean_z,
                "mean_xz_diff": mean_x - mean_z,
            }
        )
        if feature_set in {"temporal", "fft", "full"}:
            signals.update(
                {
                    "mean_xy_ratio": _safe_divide(mean_x, mean_y),
                    "mean_yz_ratio": _safe_divide(mean_y, mean_z),
                    "std_xy_ratio": _safe_divide(std_x, std_y),
                    "std_yz_ratio": _safe_divide(std_y, std_z),
                }
            )
    return signals


def _base_stats(name: str, values: np.ndarray, rich: bool) -> dict[str, float]:
    s = pd.Series(values)
    feats = {
        f"{name}__mean": float(np.mean(values)),
        f"{name}__std": float(np.std(values, ddof=0)),
        f"{name}__min": float(np.min(values)),
        f"{name}__max": float(np.max(values)),
        f"{name}__median": float(np.median(values)),
    }
    if rich:
        q05, q25, q75, q95 = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
        feats.update(
            {
                f"{name}__q05": float(q05),
                f"{name}__q25": float(q25),
                f"{name}__q75": float(q75),
                f"{name}__q95": float(q95),
                f"{name}__range": float(np.max(values) - np.min(values)),
                f"{name}__iqr": float(q75 - q25),
                f"{name}__skew": float(s.skew()),
                f"{name}__kurt": float(s.kurt()),
                f"{name}__energy": float(np.mean(values**2)),
            }
        )
    return feats


def _temporal_features(name: str, values: np.ndarray) -> dict[str, float]:
    n = len(values)
    x = np.arange(n, dtype=float)
    first = values[: n // 2]
    second = values[n // 2 :]
    diffs = np.abs(np.diff(values))
    chunks = np.array_split(values, 4)

    feats = {
        f"{name}__half_mean_diff": float(np.mean(second) - np.mean(first)),
        f"{name}__slope": float(np.polyfit(x, values, deg=1)[0]),
        f"{name}__mean_abs_diff": float(np.mean(diffs)),
        f"{name}__max_abs_diff": float(np.max(diffs)),
        f"{name}__diff_q90": float(np.quantile(diffs, 0.90)),
    }
    diff_q75 = np.quantile(diffs, 0.75)
    diff_q90 = np.quantile(diffs, 0.90)
    feats[f"{name}__diff_gt_q75_rate"] = float(np.mean(diffs > diff_q75))
    feats[f"{name}__diff_gt_q90_rate"] = float(np.mean(diffs > diff_q90))

    for i, chunk in enumerate(chunks):
        feats[f"{name}__seg{i}_mean"] = float(np.mean(chunk))
        feats[f"{name}__seg{i}_std"] = float(np.std(chunk, ddof=0))
    return feats


def _fft_features(name: str, values: np.ndarray) -> dict[str, float]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered))
    power = spectrum**2
    if len(power) <= 1:
        return {
            f"{name}__fft_low_energy": 0.0,
            f"{name}__fft_mid_energy": 0.0,
            f"{name}__fft_high_energy": 0.0,
            f"{name}__fft_dominant_idx": 0.0,
        }

    power_no_dc = power[1:]
    bands = np.array_split(power_no_dc, 3)
    total = np.sum(power_no_dc) + 1e-12
    return {
        f"{name}__fft_low_energy": float(np.sum(bands[0]) / total),
        f"{name}__fft_mid_energy": float(np.sum(bands[1]) / total),
        f"{name}__fft_high_energy": float(np.sum(bands[2]) / total),
        f"{name}__fft_dominant_idx": float(np.argmax(power_no_dc) + 1),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _correlation_features(signals: dict[str, np.ndarray]) -> dict[str, float]:
    pairs = [
        ("mean_x", "mean_y"),
        ("mean_x", "mean_z"),
        ("mean_y", "mean_z"),
        ("std_x", "std_y"),
        ("std_x", "std_z"),
        ("std_y", "std_z"),
        ("acc_mag", "dyn_mag"),
        ("acc_mag", "horiz_mag"),
    ]
    feats = {}
    for left, right in pairs:
        if left in signals and right in signals:
            feats[f"corr__{left}__{right}"] = _corr(signals[left], signals[right])
    return feats


def _longest_true_streak(mask: np.ndarray) -> float:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)


def _peak_burst_features(name: str, values: np.ndarray) -> dict[str, float]:
    q90 = np.quantile(values, 0.90)
    q95 = np.quantile(values, 0.95)
    mean = np.mean(values)
    std = np.std(values, ddof=0)
    high_1std = values > mean + std
    high_2std = values > mean + (2.0 * std)
    above_q90 = values > q90
    above_q95 = values > q95
    centered = values - mean

    return {
        f"{name}__gt_q90_rate": float(np.mean(above_q90)),
        f"{name}__gt_q95_rate": float(np.mean(above_q95)),
        f"{name}__gt_mean_1std_rate": float(np.mean(high_1std)),
        f"{name}__gt_mean_2std_rate": float(np.mean(high_2std)),
        f"{name}__longest_gt_q90": _longest_true_streak(above_q90),
        f"{name}__longest_gt_mean_1std": _longest_true_streak(high_1std),
        f"{name}__q90_crossings": float(np.sum(above_q90[1:] != above_q90[:-1])),
        f"{name}__zero_crossings_centered": float(np.sum(centered[1:] * centered[:-1] < 0)),
    }


def _autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    return _corr(values[:-lag], values[lag:])


def _rolling_autocorr_features(name: str, values: np.ndarray) -> dict[str, float]:
    s = pd.Series(values)
    feats = {}
    for window in (10, 30):
        rolling_mean = s.rolling(window=window, min_periods=window).mean().dropna().to_numpy()
        rolling_std = s.rolling(window=window, min_periods=window).std(ddof=0).dropna().to_numpy()
        rolling_max = s.rolling(window=window, min_periods=window).max()
        rolling_min = s.rolling(window=window, min_periods=window).min()
        rolling_range = (rolling_max - rolling_min).dropna().to_numpy()

        for stat_name, rolled in (
            ("rolling_mean", rolling_mean),
            ("rolling_std", rolling_std),
            ("rolling_range", rolling_range),
        ):
            feats[f"{name}__{stat_name}_{window}s_mean"] = float(np.mean(rolled))
            feats[f"{name}__{stat_name}_{window}s_std"] = float(np.std(rolled, ddof=0))
            feats[f"{name}__{stat_name}_{window}s_max"] = float(np.max(rolled))

    for lag in (1, 2, 5, 10):
        feats[f"{name}__autocorr_lag{lag}"] = _autocorr(values, lag)
    return feats


def extract_features(df: pd.DataFrame, feature_set: str = "full") -> dict[str, float]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}; choose from {FEATURE_SETS}")

    normalized_set = "fft" if feature_set in FULL_FEATURE_SETS else feature_set
    signals = _signals(df, normalized_set)
    rich_stats = normalized_set in {"temporal", "fft"}

    features: dict[str, float] = {}
    for name, values in signals.items():
        features.update(_base_stats(name, values, rich=rich_stats))
        if normalized_set in {"temporal", "fft"}:
            features.update(_temporal_features(name, values))

    if normalized_set == "fft":
        for name in ("acc_mag", "dyn_mag", "horiz_mag"):
            if name in signals:
                features.update(_fft_features(name, signals[name]))

    if feature_set in {"full_correlation", "full_corr_peak_rolling"}:
        features.update(_correlation_features(signals))

    if feature_set in {"full_peak", "full_peak_rolling", "full_corr_peak_rolling"}:
        for name in ("acc_mag", "dyn_mag", "horiz_mag"):
            if name in signals:
                features.update(_peak_burst_features(name, signals[name]))

    if feature_set in {"full_rolling", "full_peak_rolling", "full_corr_peak_rolling"}:
        for name in ("acc_mag", "dyn_mag", "horiz_mag"):
            if name in signals:
                features.update(_rolling_autocorr_features(name, signals[name]))

    return features


def load_feature_table(
    root: Path,
    feature_set: str = "full",
    include_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta_rows = []
    files = list_window_files(root)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    for i, path in enumerate(files, start=1):
        df = read_window(path)
        label = None
        if include_labels:
            if "label" not in df.columns:
                raise ValueError(f"Missing label column in training file: {path}")
            labels = df["label"].unique()
            if len(labels) != 1:
                raise ValueError(f"Training label is not constant in {path}: {labels}")
            label = int(labels[0])
        if "file_id" not in df.columns:
            raise ValueError(f"Missing file_id column in {path}")

        rows.append(extract_features(df, feature_set=feature_set))
        meta_rows.append(
            {
                "path": str(path),
                "user": user_from_path(path),
                "file_id": int(df["file_id"].iloc[0]),
                "label": label,
            }
        )
        if i % 1000 == 0:
            print(f"Loaded {i}/{len(files)} windows from {root}")

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta = pd.DataFrame(meta_rows)
    return X, meta


def load_or_build_feature_table(
    root: Path,
    feature_set: str,
    include_labels: bool,
    cache_dir: Path | None = None,
    cache_name: str | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache_dir is None or not use_cache:
        return load_feature_table(root, feature_set=feature_set, include_labels=include_labels)

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "train" if include_labels else "test"
    cache_path = cache_dir / f"{cache_name or suffix}_{feature_set}.joblib"
    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        cached = joblib.load(cache_path)
        return cached["X"], cached["meta"]

    X, meta = load_feature_table(root, feature_set=feature_set, include_labels=include_labels)
    joblib.dump({"X": X, "meta": meta}, cache_path)
    print(f"Saved feature cache: {cache_path}")
    return X, meta
