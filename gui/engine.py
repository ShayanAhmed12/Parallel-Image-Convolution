"""Everything the app needs to talk to pdc_conv.exe.

The app never filters or times anything itself: it runs the C program with
ordinary command-line options (EngineRun) and reads the CSV and image files
it writes. This module also holds the method names and help texts shown in
the app.
"""
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent

# (code used by the engine, name shown in the app, works for blur only)
METHODS = [
    ("seq", "Sequential", False),
    ("omp_rows", "Rows", False),
    ("omp_tiled", "Tiled", False),
    ("seq_sep", "Separable (sequential)", True),
    ("omp_sep", "Separable", True),
]
METHOD_NAMES = {code: name for code, name, _ in METHODS}

METHOD_HELP = {
    "seq": "One thread filters the whole image, pixel by pixel. "
           "This is the reference: every other method must give exactly the same bytes, "
           "and its time T1 is what speedup is measured against.",
    "omp_rows": "\"#pragma omp parallel for\" over the image rows: each thread gets rows to filter. "
                "Static gives each thread one block of rows; dynamic hands out one row at a time to "
                "whichever thread is free; guided starts with big pieces and ends with small ones.",
    "omp_tiled": "The image is cut into tiles (16 x 2048 pixels) and collapse(2) turns the two tile "
                 "loops into one list that the threads share. Tiles keep the rows a thread reads "
                 "close together in the cache.",
    "seq_sep": "A blur can be done as two 1-D passes: first along each row, then down each column. "
               "That needs 2N instead of N x N multiplications per pixel. One thread, for comparison.",
    "omp_sep": "The two separable passes, each split between threads by rows. The second pass must "
               "wait for the first to finish completely (an implicit barrier), because each output "
               "row needs rows of sums that other threads produced.",
}

BLUR_SIZES = [3, 5, 7, 9, 11, 15, 21, 25]
SCHEDULES = ["static", "dynamic", "guided"]

# the same colour table as THREAD_COLORS in src/parallel.c (matplotlib "tab10")
THREAD_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                 (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]

CPUS = os.cpu_count() or 1


def methods_for(filter_name):
    """The methods that can run a filter, as (code, friendly name)."""
    return [(code, name) for code, name, blur_only in METHODS if filter_name == "blur" or not blur_only]


def thread_choices(logical_cpus=CPUS):
    """1, 2, 4, 8, ... up to the number of logical CPUs, which is always included."""
    choices = [t for t in (1, 2, 4, 8, 16, 32, 64) if t < logical_cpus]
    return choices + [logical_cpus]


def default_threads(choices):
    """The largest choice up to 8: the thread map has 10 colours, and 8 is enough to show scaling."""
    return max(t for t in choices if t <= 8)


def cpu_name():
    """The processor's name as Windows shows it, e.g. "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz"."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        return "unknown CPU"


def to_ppm(src, dst):
    """Convert any image Pillow can open into the 8-bit RGB PPM the engine reads."""
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)          # phone photos: apply their rotation
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            image = image.point(lambda v: v / 256).convert("L")   # 16/32-bit grey -> 0..255
        image = image.convert("RGB")
        image.save(dst, format="PPM")
        return image.size


def engine_path():
    """pdc_conv.exe: next to the app's exe when packaged, else in the project folder."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "pdc_conv.exe"
    return ROOT / "pdc_conv.exe"


def runs_dir():
    """Where the app's runs write their images and CSVs."""
    folder = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "PDC-Convolution" / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def command_text(args):
    """The command line as a person would type it, for showing in the app."""
    return subprocess.list2cmdline(["pdc_conv", *map(str, args)])


class EngineRun:
    """Run pdc_conv.exe in the background without freezing the window.

    A helper thread reads the program's output; the Tk main loop picks the
    lines up every 50 ms and passes them to on_line. When the program ends,
    on_done(exit code, error text) is called, with code -1 if it was stopped.
    Arguments are passed as a list, never through a shell, so paths with
    spaces or brackets need no quoting.
    """

    def __init__(self, args, on_line, on_done, tk_root):
        self.on_line, self.on_done, self.tk_root = on_line, on_done, tk_root
        self.stopped = False
        self.running = True
        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            [str(engine_path()), *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        # the helper thread gets only the process and the queue, never Tk objects
        threading.Thread(target=_read_output, args=(self.process, self.messages), daemon=True).start()
        self.tk_root.after(50, self._poll)

    def _poll(self):                      # runs in the Tk main loop
        try:
            while True:
                message = self.messages.get_nowait()
                if message[0] == "line":
                    self.on_line(message[1])
                else:
                    self.running = False
                    self.on_done(-1 if self.stopped else message[1], message[2])
                    return
        except queue.Empty:
            pass
        self.tk_root.after(50, self._poll)

    def stop(self):
        """Ask the program to end; force it if it is still there after 2 s."""
        if not self.running:
            return
        self.stopped = True
        self.process.terminate()
        threading.Timer(2.0, _kill_if_alive, args=(self.process,)).start()


def _read_output(process, messages):
    """Helper thread: pass each output line, then the exit code and error text, to the queue."""
    with process.stdout, process.stderr:
        for line in process.stdout:
            messages.put(("line", line.rstrip("\n")))
        error = process.stderr.read()
    messages.put(("done", process.wait(), error))


def _kill_if_alive(process):
    if process.poll() is None:
        process.kill()
