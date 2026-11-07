"""Draw the report's graphs from the CSV files in a results folder.

usage: python scripts/plot_results.py [results_dir]      (default: results)

Reads whichever of bench.csv, blur_sizes.csv, schedules.csv and batch.csv
exist and writes PNGs into <results_dir>/plots/. The pictures come from
gui/charts.py, the same code the app uses.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui import charts  # noqa: E402


def main(results_dir):
    results_dir = Path(results_dir)
    plots = results_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(plots / name, dpi=150)
        print("wrote", plots / name)

    if (results_dir / "bench.csv").exists():
        rows = charts.read_csv(results_dir / "bench.csv")
        save(charts.speedup_chart(rows, "speedup"), "speedup.png")
        save(charts.speedup_chart(rows, "efficiency"), "efficiency.png")
    if (results_dir / "blur_sizes.csv").exists():
        rows = charts.read_csv(results_dir / "blur_sizes.csv")
        most_threads = max(int(r["threads"]) for r in rows)
        save(charts.blur_size_chart(rows, most_threads), "blur_sizes.png")
    if (results_dir / "schedules.csv").exists():
        save(charts.schedule_chart(charts.read_csv(results_dir / "schedules.csv")), "schedules.png")
    if (results_dir / "batch.csv").exists():
        save(charts.batch_chart(charts.read_csv(results_dir / "batch.csv")), "batch.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results")
