"""Tests for the Python side: charts, the plot script and run_all.py.

Run from the repository root:  python -m unittest discover -s tests -v
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))

from gui import charts  # noqa: E402

PLOTS = ["speedup.png", "efficiency.png", "blur_sizes.png", "schedules.png", "batch.png"]
DEMO = ["input.png", "blur.png", "sharpen.png", "sobel.png",
        "threadmap_rows_static.png", "threadmap_rows_dynamic.png", "threadmap_tiled.png"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class ChartTests(unittest.TestCase):
    def test_charts_draw_from_fixtures(self):
        bench = charts.read_csv(FIXTURES / "bench.csv")
        self.assertEqual(bench[0]["method"], "seq")
        figures = [
            charts.speedup_chart(bench),
            charts.speedup_chart(bench, metric="efficiency"),
            charts.blur_size_chart(charts.read_csv(FIXTURES / "blur_sizes.csv"), threads=4),
            charts.schedule_chart(charts.read_csv(FIXTURES / "schedules.csv")),
            charts.batch_chart(charts.read_csv(FIXTURES / "batch.csv")),
            charts.batch_chart(charts.read_csv(FIXTURES / "batch.csv"), metric="images_per_s"),
        ]
        for fig in figures:
            self.assertTrue(fig.axes)
            self.assertTrue(any(ax.lines for ax in fig.axes))

    def test_speedup_chart_one_panel_per_filter_and_size(self):
        fig = charts.speedup_chart(charts.read_csv(FIXTURES / "bench.csv"))
        titles = [ax.get_title() for ax in fig.axes]
        self.assertEqual(titles, ["Blur 5x5", "Sharpen", "Sobel", "Blur 15x15"])


class PlotScriptTests(TempDirTest):
    def test_plot_script(self):
        for name in ("bench.csv", "blur_sizes.csv", "schedules.csv", "batch.csv"):
            shutil.copy(FIXTURES / name, self.tmp / name)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_results.py"), str(self.tmp)], check=True)
        for name in PLOTS:
            path = self.tmp / "plots" / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)


class RunAllTests(TempDirTest):
    def test_run_all_quick(self):
        for folder in ("src", "scripts", "gui"):
            shutil.copytree(ROOT / folder, self.tmp / folder,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pdc_conv.exe", self.tmp / "pdc_conv.exe")
        subprocess.run([sys.executable, str(self.tmp / "scripts" / "run_all.py"), "--quick"],
                       check=True, capture_output=True, text=True)
        for name in ("bench.csv", "blur_sizes.csv", "schedules.csv", "batch.csv"):
            self.assertTrue((self.tmp / "results" / name).exists(), name)
        for name in PLOTS:
            self.assertTrue((self.tmp / "results" / "plots" / name).exists(), name)
        for name in DEMO:
            self.assertTrue((self.tmp / "demo" / name).exists(), name)
        self.assertEqual(list((self.tmp / "demo").glob("*.ppm")), [])


if __name__ == "__main__":
    unittest.main()
