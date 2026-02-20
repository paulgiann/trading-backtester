import subprocess, re, statistics as st, pathlib, os, sys

p = pathlib.Path("run_all.py").read_text(encoding="utf-8")

def run(seed: int) -> float:
    q = re.sub(r"ENGINE_SEED = \d+", f"ENGINE_SEED = {seed}", p)
    pathlib.Path("_tmp_run.py").write_text(q, encoding="utf-8")

    env = dict(os.environ)
    env["SKIP_DOWNLOAD"] = "1"

    try:
        out = subprocess.check_output([sys.executable, "_tmp_run.py"], text=True, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        print("\n=== _tmp_run.py failed ===")
        print(e.output)  # <-- this is the hidden traceback you need
        raise

    m = re.search(r"end_equity:\s*([0-9.]+)", out)
    if not m:
        print("\n=== Could not parse end_equity from output ===")
        print(out)
        return float("nan")
    return float(m.group(1))

vals = []
for s in range(1, 11):
    print(f"Running seed {s}...", flush=True)
    vals.append(run(s))

vals = [v for v in vals if v == v]
print("end_equity seeds 1..10:", vals)
print("mean", st.mean(vals), "median", st.median(vals), "min", min(vals), "max", max(vals))
print("mean_return", st.mean([(v/100000.0-1.0) for v in vals]), "median_return", st.median([(v/100000.0-1.0) for v in vals]))
