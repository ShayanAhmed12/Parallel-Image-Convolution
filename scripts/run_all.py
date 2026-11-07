"""Run every experiment in the report, then draw the graphs and demo images.

usage: python scripts/run_all.py [--quick]

  --quick   tiny images, for checking that the script works (seconds, not an hour)

Writes, relative to the project folder:
  results/bench.csv        all filters, blur 5x5 and blur 15x15, every method
  results/blur_sizes.csv   blur sizes 3..25: how the window size changes speedup
  results/schedules.csv    Rows method with static, dynamic and guided scheduling
  results/batch.csv        many small images: rows-per-thread vs image-per-thread
  results/*.log            the engine's printed tables for each step
  results/plots/*.png      graphs (scripts/plot_results.py)
  demo/*.png               example images and thread maps

Close other programs first: timings need the CPU, and the 12000 x 12000 runs
need about 3.5 GB of free memory.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "pdc_conv.exe"
RESULTS = ROOT / "results"
DEMO = ROOT / "demo"

FULL = dict(big=12000, sweep=8000, threads="1,2,4,8", sweep_threads="1,4,8", reps=3, sweep_reps=2,
            batch_n=3000, batch_size=1024, batch_reps=2, demo=1024, tile="128x256")
QUICK = dict(big=96, sweep=64, threads="1,2", sweep_threads="1,2", reps=1, sweep_reps=1,
             batch_n=16, batch_size=64, batch_reps=1, demo=64, tile="16x32")


def engine(*args, log=None):
    """Run pdc_conv.exe; stop everything if it fails. Its output is appended to `log`."""
    args = [str(a) for a in args]
    result = subprocess.run([str(EXE), *args], capture_output=True, text=True)
    if log:
        with open(RESULTS / log, "a") as f:
            f.write(f"> pdc_conv {' '.join(args)}\n{result.stdout}{result.stderr}\n")
    if result.returncode != 0:
        sys.exit(f"pdc_conv {' '.join(args)} failed ({result.returncode}):\n{result.stderr}")
    return result


def bench_runs(name, runs):
    """Run several `bench` commands and merge their CSVs into results/<name>.csv."""
    (RESULTS / f"{name}.log").unlink(missing_ok=True)
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for args in runs:
            print(f"  bench {' '.join(map(str, args))}", flush=True)
            part = Path(tmp) / "part.csv"
            engine("bench", *args, "--csv", part, log=f"{name}.log")
            rows = part.read_text().splitlines()
            lines += rows if not lines else rows[1:]      # keep only the first header
    (RESULTS / f"{name}.csv").write_text("\n".join(lines) + "\n")


def make_demo(s):
    """Example images (blur 9x9, sharpen, sobel) and thread maps, as PNG."""
    DEMO.mkdir(exist_ok=True)
    common = ("--variant", "omp_tiled", "--threads", 4)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ppm = {}

        def out(name):
            ppm[name] = tmp / f"{name}.ppm"
            return ppm[name]

        engine("gen", "--size", s["demo"], "--out", out("input"))
        engine("filter", "--in", ppm["input"], "--kernel", "blur", "--blur-size", 9, *common, "--out", out("blur"))
        engine("filter", "--in", ppm["input"], "--kernel", "sharpen", *common, "--out", out("sharpen"))
        engine("filter", "--in", ppm["input"], "--kernel", "sobel", *common, "--out", out("sobel"))
        maps = {
            "threadmap_rows_static": ("--variant", "omp_rows", "--schedule", "static"),
            "threadmap_rows_dynamic": ("--variant", "omp_rows", "--schedule", "dynamic"),
            "threadmap_tiled": ("--variant", "omp_tiled", "--tile", s["tile"]),
        }
        for name, args in maps.items():
            engine("filter", "--in", ppm["input"], "--kernel", "sharpen", "--threads", 4, *args,
                   "--threadmap", out(name))
        for name, path in ppm.items():
            Image.open(path).save(DEMO / f"{name}.png")
            print("  wrote", DEMO / f"{name}.png", flush=True)
    # the PPMs were in the temporary folder, which is now deleted


def main():
    quick = "--quick" in sys.argv[1:]
    s = QUICK if quick else FULL
    start = time.time()

    if Path(r"C:\mingw64").exists():
        print("Building pdc_conv.exe ...", flush=True)
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(ROOT / "scripts" / "build.ps1")], check=True)
    if not EXE.exists():
        sys.exit(f"{EXE} not found: build it first with scripts\\build.ps1")
    RESULTS.mkdir(exist_ok=True)

    print("1-2. All filters and methods (blur 5x5 and 15x15)", flush=True)
    bench_runs("bench", [
        ("--size", s["big"], "--kernel", "all", "--threads", s["threads"], "--reps", s["reps"]),
        ("--size", s["big"], "--kernel", "blur", "--blur-size", 15, "--threads", s["threads"], "--reps", s["reps"]),
    ])
    print("3. Blur size sweep", flush=True)
    bench_runs("blur_sizes", [
        ("--size", s["sweep"], "--kernel", "blur", "--blur-size", n, "--methods", "omp_rows,omp_tiled,omp_sep_fused",
         "--threads", s["sweep_threads"], "--reps", s["sweep_reps"])
        for n in (3, 5, 9, 15, 21, 25)
    ])
    print("4. Loop schedules", flush=True)
    bench_runs("schedules", [
        ("--size", s["big"], "--kernel", "blur", "--methods", "omp_rows", "--schedule", schedule,
         "--threads", s["threads"], "--reps", s["reps"])
        for schedule in ("static", "dynamic", "guided")
    ])
    print("5. Batch of small images", flush=True)
    (RESULTS / "batch.log").unlink(missing_ok=True)
    engine("batch", "--n", s["batch_n"], "--bsize", s["batch_size"], "--kernel", "blur",
           "--threads", s["threads"], "--reps", s["batch_reps"], "--csv", RESULTS / "batch.csv",
           log="batch.log")
    print("6. Graphs", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_results.py"), str(RESULTS)], check=True)
    print("7. Demo images", flush=True)
    make_demo(s)
    print(f"Done in {time.time() - start:.0f} s.", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
