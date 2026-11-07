"""Tests for the Python side: charts, the plot script and run_all.py.

Run from the repository root:  python -m unittest discover -s tests -v
"""
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))

from gui import charts, engine  # noqa: E402

PLOTS = ["time.png", "speedup.png", "efficiency.png", "blur_sizes.png", "schedules.png", "batch.png"]
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
            charts.speedup_chart(bench, metric="time"),
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


def pump(root, until, seconds):
    """Process Tk events until until() is true or the time is up; return until()."""
    end = time.time() + seconds
    while time.time() < end and not until():
        root.update()
        time.sleep(0.02)
    return until()


class EngineHelperTests(TempDirTest):
    def test_thread_choices(self):
        self.assertEqual(engine.thread_choices(8), [1, 2, 4, 8])
        self.assertEqual(engine.thread_choices(6), [1, 2, 4, 6])
        self.assertEqual(engine.thread_choices(12), [1, 2, 4, 8, 12])
        self.assertEqual(engine.thread_choices(1), [1])

    def test_default_threads(self):
        # at most 8, so the thread map never repeats a colour by default
        self.assertEqual(engine.default_threads([1, 2, 4, 8, 16]), 8)
        self.assertEqual(engine.default_threads([1, 2, 4, 6]), 6)
        self.assertEqual(engine.default_threads([1]), 1)

    def test_methods_for(self):
        self.assertEqual([code for code, _ in engine.methods_for("blur")],
                         ["seq", "omp_rows", "omp_tiled", "seq_sep", "omp_sep", "omp_sep_fused"])
        self.assertEqual(engine.methods_for("sobel"),
                         [("seq", "Sequential"), ("omp_rows", "Rows"), ("omp_tiled", "Tiled")])

    def test_bench_memory(self):
        self.assertEqual(engine.bench_memory_bytes(12000, 12000, True), 3 * 432_000_000 + 1_728_000_000)
        self.assertEqual(engine.bench_memory_bytes(12000, 12000, False), 3 * 432_000_000)
        self.assertEqual(engine.batch_memory_bytes(1024, 4), (8 * 3 + 4) * 1024 * 1024 * 3)

    def test_to_ppm_modes(self):
        sources = {
            "rgba.png": Image.new("RGBA", (5, 4), (10, 20, 30, 128)),
            "grey.png": Image.new("L", (3, 7), 99),
            "deep.png": Image.new("I;16", (6, 2), 40000),
            "palette.bmp": Image.new("RGB", (4, 4), (200, 0, 0)).convert("P"),
            "one.jpg": Image.new("RGB", (1, 1), (0, 255, 0)),
        }
        for name, image in sources.items():
            with self.subTest(name=name):
                src = self.tmp / name
                image.save(src)
                dst = self.tmp / (name + ".ppm")
                self.assertEqual(engine.to_ppm(src, dst), image.size)
                self.assertTrue(dst.read_bytes().startswith(b"P6"))
                with Image.open(dst) as converted:
                    self.assertEqual((converted.mode, converted.size), ("RGB", image.size))

    def test_system_info(self):
        info = engine.system_info()
        self.assertGreaterEqual(info.logical_cpus, 1)
        self.assertGreater(info.free_ram, 0)
        self.assertTrue(info.cpu_name)


class EngineRunTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.done = None
        self.lines = []

    def tearDown(self):
        self.root.destroy()
        super().tearDown()

    def start(self, args):
        return engine.EngineRun([str(a) for a in args], self.lines.append,
                                lambda code, err: setattr(self, "done", (code, err)), self.root)

    def test_engine_run_path_with_spaces(self):
        folder = self.tmp / "my dir"
        folder.mkdir()
        image = folder / "my photo (1)+x.ppm"
        csv = folder / "out (2).csv"
        subprocess.run([str(engine.engine_path()), "gen", "--size", "40x30", "--out", str(image)], check=True)
        self.start(["filter", "--in", image, "--kernel", "blur", "--variant", "omp_rows", "--threads", 2,
                    "--verify", "--csv", csv])
        self.assertTrue(pump(self.root, lambda: self.done is not None, 20))
        self.assertEqual(self.done[0], 0, self.done[1])
        self.assertTrue(any("identical: yes" in line for line in self.lines))
        self.assertEqual(charts.read_csv(csv)[0]["identical"], "yes")

    def test_engine_run_out_of_memory(self):
        self.start(["bench", "--size", 100000, "--kernel", "blur", "--threads", 1, "--csv", self.tmp / "b.csv"])
        self.assertTrue(pump(self.root, lambda: self.done is not None, 30))
        self.assertNotEqual(self.done[0], 0)
        self.assertIn("out of memory", self.done[1])

    def test_engine_run_stop(self):
        run = self.start(["bench", "--size", 6000, "--kernel", "blur", "--blur-size", 25, "--threads", 1,
                           "--csv", self.tmp / "b.csv"])
        pump(self.root, lambda: False, 1)
        self.assertTrue(run.running)
        stopped_at = time.time()
        run.stop()
        self.assertTrue(pump(self.root, lambda: self.done is not None, 2.5))
        self.assertLess(time.time() - stopped_at, 2.5)
        self.assertEqual(self.done[0], -1)
        self.assertFalse(run.running)


def load_app_module():
    """Import gui/app.pyw (a .pyw file, so it needs an explicit loader)."""
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("app", str(ROOT / "gui" / "app.pyw"))
    spec = importlib.util.spec_from_loader("app", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class AppTests(unittest.TestCase):
    def test_app_filter_apply(self):
        app = load_app_module().App()
        try:
            tab = app.filter_tab
            tab.source.set("generated")
            tab.size.set("256")
            tab.filter.set("sobel")
            tab.on_filter_change()
            tab.set_method("omp_tiled")
            tab.threads.set("2")
            tab.show_map.set(True)
            tab.apply()
            self.assertTrue(pump(app, lambda: "Identical" in tab.identical_label.cget("text"), 10))
            self.assertTrue(tab.result_shown)
        finally:
            app.destroy()


class RunTabTests(unittest.TestCase):
    def setUp(self):
        self.app_module = load_app_module()
        self.real_engine_path = self.app_module.engine.engine_path
        self.app = self.app_module.App()

    def tearDown(self):
        self.app_module.engine.engine_path = self.real_engine_path
        self.app.quit_app()

    def choose_threads(self, tab, wanted):
        for t, var in tab.thread_vars.items():
            var.set(t in wanted)

    def test_bench_tab_run(self):
        tab = self.app.bench_tab
        tab.size.set("256")
        tab.filter.set("blur")
        self.choose_threads(tab, {1, 2})
        tab.reps.set("1")
        tab.run()
        self.assertTrue(pump(self.app, lambda: tab.last_code is not None, 30))
        self.assertEqual(tab.last_code, 0)
        rows = charts.read_csv(self.app.runs / "bench.csv")
        self.assertEqual(len(tab.table.get_children()), len(rows))
        self.assertIsNotNone(tab.canvas)

    def test_batch_tab_run(self):
        tab = self.app.batch_tab
        tab.count.set("16")
        tab.size.set("64")
        self.choose_threads(tab, {1, 2})
        tab.reps.set("1")
        tab.run()
        self.assertTrue(pump(self.app, lambda: tab.last_code is not None, 30))
        self.assertEqual(tab.last_code, 0)
        self.assertEqual(len(tab.table.get_children()), 5)
        self.assertIsNotNone(tab.canvas)

    def test_filter_tab_stop(self):
        tab = self.app.filter_tab
        tab.source.set("generated")
        tab.size.set("6000")
        tab.filter.set("blur")
        tab.on_filter_change()
        tab.blur_size.set("25")
        tab.set_method("seq")
        tab.apply()
        pump(self.app, lambda: False, 1.5)
        self.assertIsNotNone(self.app.current_run)
        self.assertNotIn("disabled", tab.stop_button.state())
        self.assertNotIn("disabled", self.app.bench_tab.stop_button.state())   # any tab can stop it
        self.assertIn("disabled", self.app.bench_tab.run_button.state())
        tab.stop_button.invoke()
        self.assertTrue(pump(self.app, lambda: self.app.current_run is None, 3))
        self.assertNotIn("disabled", tab.apply_button.state())
        self.assertIn("disabled", tab.stop_button.state())
        self.assertEqual(self.app.status.cget("text"), "Stopped")

    def test_engine_that_cannot_start(self):
        shown = []
        self.app_module.messagebox.showerror = lambda *args, **kwargs: shown.append(args)
        self.app_module.engine.engine_path = lambda: ROOT / "missing" / "pdc_conv.exe"
        tab = self.app.filter_tab
        tab.source.set("generated")
        tab.size.set("64")
        tab.apply()
        self.assertTrue(pump(self.app, lambda: shown, 3))
        self.assertIsNone(self.app.current_run)
        self.assertNotIn("disabled", tab.apply_button.state())

    def test_bench_tab_stop(self):
        tab = self.app.bench_tab
        tab.size.set("6000")
        tab.filter.set("blur")
        tab.blur_size.set("25")
        self.choose_threads(tab, {1})
        tab.reps.set("1")
        tab.run()
        pump(self.app, lambda: False, 1)
        tab.stop_button.invoke()
        self.assertTrue(pump(self.app, lambda: tab.last_code is not None, 3))
        self.assertEqual(tab.last_code, -1)
        self.assertIn("Image 6000x6000", tab.log.get("1.0", "end"))
        self.assertEqual(len(tab.table.get_children()), 0)
        self.assertIsNone(tab.canvas)


if __name__ == "__main__":
    unittest.main()
