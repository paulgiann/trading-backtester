from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_dual(df: pd.DataFrame, csv_path: str | Path, parquet_path: Optional[str | Path] = None, index: bool = False) -> None:
    csv_path = ensure_parent(csv_path)
    df.to_csv(csv_path, index=index)
    if parquet_path is not None:
        parquet_path = ensure_parent(parquet_path)
        df.to_parquet(parquet_path, index=index)


def load_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file extension: {p.suffix}")
