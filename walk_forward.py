from __future__ import annotations

import os
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


FEATURES_PATH = Path(os.getenv("WF_FEATURES_PATH", "data/processed/features.parquet"))
TRAIN_BARS = int(os.getenv("WF_TRAIN_BARS", "2880"))   # about 2 days of 1m bars
TEST_BARS = int(os.getenv("WF_TEST_BARS", "1440"))     # about 1 day of 1m bars
STEP_BARS = int(os.getenv("WF_STEP_BARS", "1440"))     # roll by 1 day

STRATEGY_NAME = os.getenv("WF_STRATEGY_NAME", "regime")
SHORT_W = int(os.getenv("WF_SHORT_W", "10"))
LONG_W = int(os.getenv("WF_LONG_W", "50"))
ENGINE_SEED = int(os.getenv("WF_ENGINE_SEED", "123"))

SPREAD_TH = os.getenv("WF_SPREAD_TH", "0.0010")
BREAKOUT_Z_MIN = os.getenv("WF_BREAKOUT_Z_MIN", "0.5")
VOL_RATIO_MIN = os.getenv("WF_VOL_RATIO_MIN", "1.10")
RANGE_CAP = os.getenv("WF_RANGE_CAP", "0.020")
MAX_HOLD_HOURS = os.getenv("WF_MAX_HOLD_HOURS", "6")
TARGET_FRAC = os.getenv("WF_TARGET_FRAC", "0.35")
COOLDOWN_MIN = os.getenv("WF_COOLDOWN_MIN", "3")


def run_slice(features_path: Path, fold_id: int, train_start, train_end, test_start, test_end):
    env = dict(os.environ)
    env["SKIP_DOWNLOAD"] = "1"
    env["SHOW_PLOTS"] = "0"
    env["MPLBACKEND"] = "Agg"
    env["FEATURES_PATH"] = str(features_path)
    env["STRATEGY_NAME"] = STRATEGY_NAME
    env["SHORT_W"] = str(SHORT_W)
    env["LONG_W"] = str(LONG_W)
    env["ENGINE_SEED"] = str(ENGINE_SEED)
    env["SPREAD_TH"] = str(SPREAD_TH)
    env["BREAKOUT_Z_MIN"] = str(BREAKOUT_Z_MIN)
    env["VOL_RATIO_MIN"] = str(VOL_RATIO_MIN)
    env["RANGE_CAP"] = str(RANGE_CAP)
    env["MAX_HOLD_HOURS"] = str(MAX_HOLD_HOURS)
    env["TARGET_FRAC"] = str(TARGET_FRAC)
    env["COOLDOWN_MIN"] = str(COOLDOWN_MIN)

    try:
        out = subprocess.check_output(
            [sys.executable, "run_all.py"],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print("\n=== run_all.py failed on fold", fold_id, "===")
        print(e.output)
        raise

    def grab(name: str):
        m = re.search(rf"{name}:\s*([-0-9.]+)", out)
        return float(m.group(1)) if m else None

    return {
        "fold": fold_id,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "end_equity": grab("end_equity"),
        "total_return": grab("total_return"),
        "num_trades": grab("num_trades"),
        "max_drawdown": grab("max_drawdown"),
        "final_position_shares": grab("final_position_shares"),
    }


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing processed features file: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    if len(df) < TRAIN_BARS + TEST_BARS:
        raise ValueError(
            f"Not enough rows for walk-forward: have {len(df)}, "
            f"need at least {TRAIN_BARS + TEST_BARS}"
        )

    rows = []
    fold = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        start = 0
        while start + TRAIN_BARS + TEST_BARS <= len(df):
            train = df.iloc[start : start + TRAIN_BARS].copy()
            test = df.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS].copy()

            slice_df = test.copy()
            slice_path = tmpdir / f"wf_fold_{fold}.parquet"
            slice_df.to_parquet(slice_path, index=False)

            row = run_slice(
                slice_path,
                fold_id=fold,
                train_start=train["Datetime"].iloc[0],
                train_end=train["Datetime"].iloc[-1],
                test_start=test["Datetime"].iloc[0],
                test_end=test["Datetime"].iloc[-1],
            )
            rows.append(row)

            print(
                f"fold={row['fold']} "
                f"test_start={row['test_start']} "
                f"test_end={row['test_end']} "
                f"return={row['total_return']} "
                f"trades={row['num_trades']} "
                f"max_dd={row['max_drawdown']} "
                f"final_pos={row['final_position_shares']}"
            )

            fold += 1
            start += STEP_BARS

    if not rows:
        raise SystemExit("No folds were produced.")

    rets = [r["total_return"] for r in rows if r["total_return"] is not None]
    trds = [r["num_trades"] for r in rows if r["num_trades"] is not None]
    dds = [r["max_drawdown"] for r in rows if r["max_drawdown"] is not None]
    flats = sum(1 for r in rows if r["final_position_shares"] == 0.0)

    print()
    print("walk_forward_folds", len(rows))
    print("mean_return", st.mean(rets), "median_return", st.median(rets), "min_return", min(rets), "max_return", max(rets))
    print("mean_trades", st.mean(trds), "median_trades", st.median(trds), "min_trades", min(trds), "max_trades", max(trds))
    print("mean_max_drawdown", st.mean(dds), "worst_max_drawdown", min(dds))
    print("flat_end_folds", flats, "of", len(rows))


if __name__ == "__main__":
    main()
