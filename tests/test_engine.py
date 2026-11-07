"""Tests for the C engine (pdc_conv.exe).

The key promise of this project: every parallel method produces exactly the
same bytes as the sequential version. These tests run the real program and
check that promise, plus the command-line behaviour the GUI relies on.

Run from the repository root after building:  python -m unittest discover -s tests -v
"""
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "pdc_conv.exe"

# Image sizes chosen to hit every border case: single pixel, thinner than the
# filter, single row/column, odd non-square sizes.
AWKWARD_SIZES = ["1x1", "2x3", "3x3", "4x7", "5x5", "17x9", "37x53", "128x1", "1x128", "257x131"]


def run(*args, expect=0):
    """Run the engine with a list of arguments (never through a shell)."""
    result = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True)
    if expect is not None and result.returncode != expect:
        raise AssertionError(
            f"pdc_conv {' '.join(map(str, args))} exited {result.returncode}, expected {expect}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ppm(path):
    """Return (width, height, pixel bytes) of a binary P6 PPM written by the engine."""
    data = Path(path).read_bytes()
    parts = data.split(maxsplit=4)          # b"P6", width, height, maxval, pixels
    assert parts[0] == b"P6" and parts[3] == b"255"
    width, height = int(parts[1]), int(parts[2])
    pixels = parts[4]
    assert len(pixels) == width * height * 3
    return width, height, pixels


def write_ppm(path, width, height, value):
    """Write a P6 PPM where every byte is `value` (a flat test image)."""
    Path(path).write_bytes(b"P6\n%d %d\n255\n" %(width, height) + bytes([value]) * (width * height * 3))


def golden():
    lines = (ROOT / "tests" / "golden_sha256.txt").read_text().split("\n")
    return dict(line.split() for line in lines if line.strip())


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class RegressionTests(TempDirTest):
    """Outputs must stay byte-identical to the original program (recorded in golden_sha256.txt)."""

    def test_golden_outputs(self):
        expected = golden()
        image = self.tmp / "input.ppm"
        run("gen", "--size", "257x131", "--out", image)
        self.assertEqual(sha256(image), expected["input"])
        for kernel in ("blur", "sharpen", "sobel"):
            out = self.tmp / f"{kernel}.ppm"
            run("filter", "--in", image, "--kernel", kernel, "--variant", "seq", "--out", out)
            self.assertEqual(sha256(out), expected[kernel], kernel)

    def test_all_methods_match_sequential(self):
        for size in AWKWARD_SIZES:
            with self.subTest(size=size):
                result = run("bench", "--size", size, "--threads", "1,3,4", "--reps", "1",
                             "--csv", self.tmp / "bench.csv")
                self.assertIn("ALL VARIANTS MATCH", result.stdout)


class BlurSizeTests(TempDirTest):
    """--blur-size N: odd sizes 3..25 with binomial weights, all methods identical."""

    def test_blur_5_matches_golden(self):
        image = self.tmp / "input.ppm"
        out = self.tmp / "blur5.ppm"
        run("gen", "--size", "257x131", "--out", image)
        run("filter", "--in", image, "--kernel", "blur", "--blur-size", 5, "--variant", "seq", "--out", out)
        self.assertEqual(sha256(out), golden()["blur"])

    def test_blur_sizes_all_methods_match(self):
        for blur_size in (3, 7, 15, 25):
            for size in ("1x1", "4x7", "37x53", "257x131"):
                with self.subTest(blur_size=blur_size, size=size):
                    result = run("bench", "--kernel", "blur", "--blur-size", blur_size, "--size", size,
                                 "--threads", "1,3", "--reps", "1", "--csv", self.tmp / "bench.csv")
                    self.assertIn("ALL VARIANTS MATCH", result.stdout)

    def test_blur_extremes(self):
        for value in (255, 0):
            image = self.tmp / f"flat{value}.ppm"
            write_ppm(image, 64, 64, value)
            for method in ("seq", "omp_sep"):
                with self.subTest(value=value, method=method):
                    out = self.tmp / f"out_{value}_{method}.ppm"
                    run("filter", "--in", image, "--kernel", "blur", "--blur-size", 25,
                        "--variant", method, "--threads", 4, "--out", out)
                    self.assertEqual(read_ppm(out)[2], bytes([value]) * (64 * 64 * 3))

    def test_invalid_blur_size(self):
        for bad in ("4", "1", "27", "abc"):
            with self.subTest(bad=bad):
                result = run("filter", "--size", "8", "--kernel", "blur", "--blur-size", bad, expect=1)
                self.assertIn("error: blur size must be an odd number from 3 to 25", result.stderr)


# tab10 colours, the same table as THREAD_COLORS in parallel.c
THREAD_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                 (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def thread_ids(path):
    """Read a thread map PPM; return a list of rows, each a list of thread numbers."""
    width, height, pixels = read_ppm(path)
    ids = []
    for y in range(height):
        row = []
        for x in range(width):
            i = (y * width + x) * 3
            row.append(THREAD_COLORS.index(tuple(pixels[i:i + 3])))
        ids.append(row)
    return ids


class ScheduleAndThreadMapTests(TempDirTest):
    """--schedule, --threadmap, --verify and filter --csv."""

    def threadmap(self, size, *args):
        path = self.tmp / "map.ppm"
        run("filter", "--size", size, "--kernel", "sharpen", "--threadmap", path, *args)
        return thread_ids(path)

    def test_threadmap_static_rows(self):
        ids = self.threadmap("64x40", "--variant", "omp_rows", "--threads", 2)
        for y, row in enumerate(ids):
            self.assertEqual(set(row), {0 if y < 20 else 1}, f"row {y}")

    def test_threadmap_dynamic_interleaves(self):
        ids = self.threadmap("64x400", "--variant", "omp_rows", "--schedule", "dynamic", "--threads", 4)
        row_owner = []
        for row in ids:
            self.assertEqual(len(set(row)), 1)      # a row is written by one thread
            row_owner.append(row[0])
        self.assertGreaterEqual(len(set(row_owner)), 3)
        runs = [owner for i, owner in enumerate(row_owner) if i == 0 or owner != row_owner[i - 1]]
        self.assertGreater(len(runs), len(set(runs)), "no thread came back for a second run of rows")

    def test_threadmap_tiled(self):
        ids = self.threadmap("64x32", "--variant", "omp_tiled", "--tile", "8x16", "--threads", 4)
        for ty in range(0, 32, 8):
            for tx in range(0, 64, 16):
                tile = {ids[y][x] for y in range(ty, ty + 8) for x in range(tx, tx + 16)}
                self.assertEqual(len(tile), 1, f"tile at {tx},{ty}")

    def test_threadmap_sequential(self):
        ids = self.threadmap("37x53", "--variant", "seq", "--threads", 4)
        self.assertEqual({t for row in ids for t in row}, {0})

    def test_schedules_match(self):
        for schedule in ("static", "dynamic", "guided"):
            with self.subTest(schedule=schedule):
                result = run("filter", "--size", "257x131", "--kernel", "blur", "--variant", "omp_rows",
                             "--schedule", schedule, "--threads", 4, "--verify")
                self.assertIn("identical: yes", result.stdout)

    def test_filter_csv(self):
        csv = self.tmp / "filter.csv"
        run("filter", "--size", "257x131", "--kernel", "blur", "--blur-size", 7, "--variant", "omp_tiled",
            "--threads", 3, "--verify", "--csv", csv)
        lines = csv.read_text().splitlines()
        self.assertEqual(lines[0], "filter,blur_size,method,schedule,threads,width,height,"
                                   "time_s,seq_time_s,speedup,identical")
        self.assertEqual(len(lines), 2)
        row = dict(zip(lines[0].split(","), lines[1].split(",")))
        self.assertEqual((row["filter"], row["blur_size"], row["method"], row["threads"]),
                         ("blur", "7", "omp_tiled", "3"))
        self.assertEqual((row["width"], row["height"], row["identical"]), ("257", "131", "yes"))
        self.assertGreater(float(row["time_s"]), 0)
        self.assertGreater(float(row["seq_time_s"]), 0)


def read_csv_rows(path):
    """Return (header line, list of row dicts) of an engine CSV."""
    lines = Path(path).read_text().splitlines()
    names = lines[0].split(",")
    return lines[0], [dict(zip(names, line.split(","))) for line in lines[1:]]


class BenchAndBatchCsvTests(TempDirTest):
    def test_bench_csv(self):
        csv = self.tmp / "bench.csv"
        run("bench", "--size", 64, "--kernel", "blur", "--blur-size", 3, "--methods", "omp_rows",
            "--schedule", "dynamic", "--threads", "1,2", "--reps", 1, "--csv", csv)
        header, rows = read_csv_rows(csv)
        self.assertEqual(header, "filter,blur_size,method,schedule,threads,time_s,speedup_vs_seq,"
                                 "efficiency,speedup_vs_self_1thread,verified")
        self.assertEqual([(r["method"], r["threads"]) for r in rows],
                         [("seq", "1"), ("omp_rows", "1"), ("omp_rows", "2")])
        self.assertEqual([r["schedule"] for r in rows], ["", "dynamic", "dynamic"])
        self.assertEqual([r["verified"] for r in rows], ["ref", "ok", "ok"])
        self.assertEqual({(r["filter"], r["blur_size"]) for r in rows}, {("blur", "3")})

    def test_bench_csv_blur_size_empty_for_sharpen(self):
        csv = self.tmp / "bench.csv"
        run("bench", "--size", 32, "--kernel", "sharpen", "--methods", "omp_tiled",
            "--threads", 2, "--reps", 1, "--csv", csv)
        _, rows = read_csv_rows(csv)
        self.assertEqual([(r["method"], r["blur_size"]) for r in rows], [("seq", ""), ("omp_tiled", "")])

    def test_bench_unknown_method(self):
        result = run("bench", "--size", 32, "--methods", "omp_fast", expect=1)
        self.assertIn("error: unknown method omp_fast", result.stderr)

    def test_batch_csv(self):
        csv = self.tmp / "batch.csv"
        run("batch", "--n", 16, "--bsize", 64, "--threads", "1,2", "--reps", 1, "--csv", csv)
        header, rows = read_csv_rows(csv)
        self.assertEqual(header, "strategy,filter,blur_size,threads,time_s,speedup_vs_seq,efficiency,"
                                 "images_per_s,verified")
        self.assertEqual([(r["strategy"], r["threads"]) for r in rows],
                         [("seq", "1"), ("per_row", "1"), ("per_image", "1"), ("per_row", "2"), ("per_image", "2")])
        self.assertEqual({r["verified"] for r in rows}, {"ok"})
        self.assertEqual({(r["filter"], r["blur_size"]) for r in rows}, {("blur", "5")})


if __name__ == "__main__":
    unittest.main()
