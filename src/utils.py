from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    root: Path
    train_dir: Path
    test_dir: Path
    sample_submission: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_paths(data_dir: str | Path = "data") -> DataPaths:
    root = Path(data_dir)
    if not root.is_absolute():
        root = project_root() / root
    return DataPaths(
        root=root,
        train_dir=root / "train" / "train",
        test_dir=root / "test" / "test",
        sample_submission=root / "sample_submission.csv",
    )


def ensure_data_exists(paths: DataPaths, require_test: bool = False) -> None:
    missing = []
    if not paths.train_dir.exists():
        missing.append(paths.train_dir)
    if not paths.sample_submission.exists():
        missing.append(paths.sample_submission)
    if require_test and not paths.test_dir.exists():
        missing.append(paths.test_dir)
    if missing:
        lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Expected assignment data files are missing:\n"
            f"{lines}\n"
            "Run `bash download_data.sh` or check the data directory."
        )


def list_window_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return sorted(root.glob("User_*/*.csv"), key=lambda p: (p.parent.name, int(p.stem)))


def read_window(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "index" in df.columns:
        df = df.sort_values("index", kind="stable")
    if len(df) != 300:
        raise ValueError(f"{path} has {len(df)} rows; expected 300")
    return df


def user_from_path(path: Path) -> str:
    return path.parent.name


def make_output_dir(path: str | Path = "outputs") -> Path:
    output_dir = Path(path)
    if not output_dir.is_absolute():
        output_dir = project_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
