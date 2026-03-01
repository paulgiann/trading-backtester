import os
import re
import statistics as st
import subprocess
import sys

SHORT_W = int(os.getenv("ROBUST_SHORT_W", "10"))
LONG_W = int(os.getenv("ROBUST_LONG_W", "50"))
SEED_START = int(os.getenv("ROBUST_SEED_START", "1"))
SEED_END = int(os.getenv("ROBUST_SEED_END", "10"))

def run(seed: int):
    env = dict(os.environ)
    env["SKIP_DOWNLOAD"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["SHORT_W"] = str(SHORT_W)
    env["LONG_W"] = str(LONG_W)
    env["ENGINE_SEED"] = str(seed)

    try:
        out = subprocess.check_output(
            [sys.executable, "run_all.py"],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print("\n=== run_all.py failed ===")
        print(e.output)
        raise

    m_eq = re.search(r"end_equity:\s*([0-9.]+)", out)
    m_ret = re.search(r"total_return:\s*([-0-9.]+)", out)
    m_tr = re.search(r"num_trades:\s*([0-9.]+)", out)
    m_pos = re.search(r"final_position_shares:\s*([-0-9.]+)", out)

    if not m_eq or not m_ret or not m_tr:
        print("\n=== Could not parse expected fields from output ===")
        print(out)
        return None

    eq = float(m_eq.group(1))
    ret = float(m_ret.group(1))
    tr = float(m_tr.group(1))
    pos = float(m_pos.group(1)) if m_pos else float("nan")
    return seed, eq, ret, tr, pos

rows = []
for seed in range(SEED_START, SEED_END + 1):
    print(f"Running seed {seed} with SHORT_W={SHORT_W} LONG_W={LONG_W}...", flush=True)
    row = run(seed)
    if row is not None:
        rows.append(row)
        print(
            f"seed={row[0]} end_equity={row[1]} total_return={row[2]} "
            f"num_trades={row[3]} final_position_shares={row[4]}"
        )

if not rows:
    raise SystemExit("No successful runs.")

rets = [r[2] for r in rows]
trds = [r[3] for r in rows]

print("mean_return", st.mean(rets), "median_return", st.median(rets), "min_return", min(rets), "max_return", max(rets))
print("mean_trades", st.mean(trds), "median_trades", st.median(trds), "min_trades", min(trds), "max_trades", max(trds))
