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


if __name__ == "__main__":
    unittest.main()
