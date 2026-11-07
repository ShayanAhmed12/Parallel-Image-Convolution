"""Everything the app needs to talk to pdc_conv.exe and to the machine.

The app never filters or times anything itself: it runs the C program with
ordinary command-line options (EngineRun) and reads the CSV and image files
it writes. This module also holds the method names and help texts, the
memory estimates for the RAM warning, and the machine details shown in the
status bar.
"""
import ctypes
import os
import queue
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# (code used by the engine, name shown in the app, works for blur only)
METHODS = [
    ("seq", "Sequential", False),
    ("omp_rows", "Rows", False),
    ("omp_tiled", "Tiled", False),
    ("seq_sep", "Separable (sequential)", True),
    ("omp_sep", "Separable", True),
    ("omp_sep_fused", "Separable, fused", True),
]

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
               "wait for the first to finish completely (an implicit barrier), and the full-size "
               "intermediate image goes through main memory.",
    "omp_sep_fused": "Each thread takes a block of rows and does both passes on it, using a small "
                     "private buffer, so no barrier is needed and, if the image is not too wide, the "
                     "buffer stays in the cache. A few rows at the block edges are computed twice.",
}

BLUR_SIZES = [3, 5, 7, 9, 11, 15, 21, 25]
SCHEDULES = ["static", "dynamic", "guided"]

# the same colour table as THREAD_COLORS in src/parallel.c (matplotlib "tab10")
THREAD_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                 (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def methods_for(filter_name):
    """The methods that can run a filter, as (code, friendly name)."""
    return [(code, name) for code, name, blur_only in METHODS if filter_name == "blur" or not blur_only]


def thread_choices(logical_cpus):
    """1, 2, 4, 8, ... up to the number of logical CPUs, which is always included."""
    choices = []
    t = 1
    while t < logical_cpus:
        choices.append(t)
        t *= 2
    choices.append(logical_cpus)
    return choices


def bench_memory_bytes(width, height, blur_separable):
    """Memory a bench run needs: input, reference and output images (+ the separable sums)."""
    image = width * height * 3
    return 3 * image + (image * 4 if blur_separable else 0)


def batch_memory_bytes(size, threads):
    """Memory a batch run needs: 8 inputs, 8 references, 8 checks, plus one output per thread."""
    return (8 * 3 + threads) * size * size * 3


def to_ppm(src, dst):
    """Convert any image Pillow can open into the 8-bit RGB PPM the engine reads."""
    with Image.open(src) as image:
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            # 16-bit or 32-bit greyscale: scale down to 0..255 first
            image = image.point(lambda v: v / 256).convert("L")
        image = image.convert("RGB")
        image.save(dst, format="PPM")
        return image.size


@dataclass
class SystemInfo:
    cpu_name: str
    logical_cpus: int
    free_ram: int        # bytes


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def system_info():
    """CPU name, logical CPU count and free RAM of this machine (read every time)."""
    cpu_name = "unknown CPU"
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return SystemInfo(cpu_name, os.cpu_count() or 1, status.ullAvailPhys)


def engine_path():
    """pdc_conv.exe: next to the app's exe when packaged, else in the project folder."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "pdc_conv.exe"
    return ROOT / "pdc_conv.exe"


def runs_dir():
    """Where the app's runs write their images and CSVs."""
    base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    folder = base / "PDC-Convolution" / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class EngineRun:
    """Run pdc_conv.exe in the background without freezing the window.

    A helper thread reads the program's output; the Tk main loop picks the
    lines up every 50 ms and passes them to on_line. When the program ends,
    on_done(exit code, error text) is called, with code -1 if it was stopped.
    Arguments are passed as a list, never through a shell, so paths with
    spaces or brackets need no quoting.
    """

    def __init__(self, args: list[str], on_line: Callable[[str], None],
                 on_done: Callable[[int, str], None], tk_root):
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
                    code = -1 if self.stopped else message[1]
                    self.on_done(code, message[2])
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
