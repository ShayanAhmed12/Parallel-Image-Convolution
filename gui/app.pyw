"""PDC Convolution: the desktop app.

A thin Windows 10 style window around pdc_conv.exe. It never filters or
times anything itself: each button builds an ordinary pdc_conv command line
(shown at the bottom of the window), runs it in the background
(gui/engine.py) and then shows the images and CSV files the C program wrote.

  Filter tab     apply one method to one image; compare with sequential
  Benchmark tab  every method at several thread counts (pdc_conv bench)
  Batch tab      many small images, two ways of sharing them (pdc_conv batch)
"""
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent

# sharp text on scaled displays (125 %, 150 %) instead of a stretched, blurry window
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except (AttributeError, OSError):
    pass
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import charts, engine  # noqa: E402

GREEN, RED, GREY = "#107c10", "#c42b1c", "#5f5f5f"
ABOUT = ("PDC Convolution\n"
         "Parallel Image Convolution / Filtering with OpenMP\n\n"
         "Parallel and Distributed Computing, FAST-NUCES")
FILTERS = [("blur", "Blur"), ("sharpen", "Sharpen"), ("sobel", "Sobel edges")]


def source_folder():
    """The C source code: next to the app's exe when packaged, else the project's src folder."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "src"
    return ROOT / "src"


def note(parent, text, scale):
    """A grey explanatory paragraph under the settings."""
    ttk.Label(parent, text=text, foreground=GREY, wraplength=int(270 * scale), justify="left").pack(
        anchor="w", pady=(8, 0))


class App(tk.Tk):
    """The main window: menu bar, tabs, the command being run, status bar."""

    def __init__(self):
        super().__init__()
        self.title("PDC Convolution - Parallel Image Filtering with OpenMP")
        self.scale = scale = self.winfo_fpixels("1i") / 96   # 1.0 at 100 % display scaling, 1.5 at 150 %
        self.geometry(f"{int(1180 * scale)}x{int(740 * scale)}")
        self.minsize(int(980 * scale), int(640 * scale))
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=int(20 * scale))   # table rows tall enough when scaled
        self.runs = engine.runs_dir()
        self.current_run = None

        self._build_menu()
        self._build_status_bar()
        self.notebook = ttk.Notebook(self)
        self.filter_tab = FilterTab(self.notebook, self)
        self.bench_tab = BenchTab(self.notebook, self)
        self.batch_tab = BatchTab(self.notebook, self)
        self.notebook.add(self.filter_tab, text="Filter")
        self.notebook.add(self.bench_tab, text="Benchmark")
        self.notebook.add(self.batch_tab, text="Batch")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        # Enter = the Apply / Run button of the tab on screen
        self.bind("<Return>", lambda e: self.nametowidget(self.notebook.select()).start_button().invoke())
        if not engine.engine_path().exists():
            self.after(200, lambda: messagebox.showerror(
                "Engine not found",
                f"{engine.engine_path()} is missing.\n\nBuild it first by running scripts\\build.ps1."))

    def _build_menu(self):
        bar = tk.Menu(self)
        file_menu = tk.Menu(bar, tearoff=False)
        file_menu.add_command(label="Open image...", accelerator="Ctrl+O", command=self.open_image)
        file_menu.add_command(label="Save result...", accelerator="Ctrl+S", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)
        bar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(bar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", ABOUT))
        help_menu.add_command(label="Open source folder", command=self.open_source_folder)
        bar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=bar)
        self.bind_all("<Control-o>", lambda e: self.open_image())
        self.bind_all("<Control-s>", lambda e: self.save_result())

    def _build_status_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom", padx=8, pady=(0, 6))
        self.status = ttk.Label(bar, text="Ready")
        self.status.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="determinate", length=int(140 * self.scale))
        self.progress.pack(side="right", padx=(8, 0))
        ttk.Label(bar, text=f"{engine.cpu_name()}  |  {engine.CPUS} threads").pack(side="right", padx=8)

        # the exact command line of the last run: the app is only a front end to pdc_conv
        line = ttk.Frame(self)
        line.pack(fill="x", side="bottom", padx=8, pady=(0, 2))
        ttk.Label(line, text="Command:", foreground=GREY).pack(side="left")
        self.command = tk.StringVar(value="(none yet)")
        ttk.Entry(line, textvariable=self.command, state="readonly", font=("Consolas", 9)).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

    # ---- running the engine --------------------------------------------------

    def run(self, args, status, on_done, on_line=None):
        """Start pdc_conv with `args` in the background; on_done(code, error text) when it ends."""
        def finished(code, error):
            self.current_run = None
            self.progress.stop()
            self.progress.config(mode="determinate", value=0)   # empty bar when idle
            self.set_busy(False)
            on_done(code, error)

        self.command.set(engine.command_text(args))
        self.status.config(text=status)
        self.progress.config(mode="indeterminate")
        self.progress.start(15)
        self.set_busy(True)
        try:
            self.current_run = engine.EngineRun(args, on_line or (lambda line: None), finished, self)
        except OSError as e:          # missing exe, or blocked by antivirus
            finished(1, f"Could not start {engine.engine_path()}:\n{e}")

    def show_progress(self, done, total):
        """Switch the bar from 'busy' to 'done of total' once a run reports progress."""
        self.progress.stop()
        self.progress.config(mode="determinate", maximum=total, value=done)

    def set_busy(self, busy):
        """While a run is going, every tab's Run is off and every tab's Stop is on."""
        for tab in (self.filter_tab, self.bench_tab, self.batch_tab):
            tab.start_button().state(["disabled"] if busy else ["!disabled"])
            tab.stop_button.state(["!disabled"] if busy else ["disabled"])

    def stop_run(self):
        if self.current_run:
            self.current_run.stop()

    def show_engine_error(self, error):
        self.status.config(text="Failed")
        messagebox.showerror("pdc_conv failed", error.strip() or "The engine stopped with an error.")

    # ---- menu commands -------------------------------------------------------

    def open_image(self):
        self.notebook.select(self.filter_tab)
        self.filter_tab.browse()

    def save_result(self):
        self.notebook.select(self.filter_tab)
        self.filter_tab.save_as()

    def open_source_folder(self):
        folder = source_folder()
        if folder.exists():
            os.startfile(folder)
        else:
            messagebox.showinfo("Source code", f"{folder} was not found.")

    def quit_app(self):
        self.stop_run()
        self.destroy()


class FilterTab(ttk.Frame):
    """Apply one method to one image and compare it with the sequential version."""

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.opened_ppm = None          # the chosen file, converted to PPM
        self.input_ppm = None
        self.result_shown = False
        self.photos = []                # Tk needs references to the images it shows

        self.source = tk.StringVar(value="generated")
        self.file_name = tk.StringVar(value="(no file chosen)")
        self.size = tk.StringVar(value="1024")
        self.filter = tk.StringVar(value="blur")
        self.blur_size = tk.StringVar(value="9")
        self.method = tk.StringVar()
        self.schedule = tk.StringVar(value="static")
        threads = engine.thread_choices()
        self.threads = tk.StringVar(value=str(engine.default_threads(threads)))
        self.show_map = tk.BooleanVar(value=True)

        left = ttk.Frame(self, width=300)
        left.pack(side="left", fill="y")
        self.view = ttk.Frame(self)
        self.view.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # input
        box = ttk.LabelFrame(left, text="Input", padding=8)
        box.pack(fill="x")
        ttk.Radiobutton(box, text="Image file", variable=self.source, value="file").grid(row=0, column=0, sticky="w")
        ttk.Button(box, text="Browse...", command=self.browse).grid(row=0, column=1, sticky="e")
        ttk.Label(box, textvariable=self.file_name, foreground=GREY, width=34).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=(20, 0))
        ttk.Radiobutton(box, text="Generated test image", variable=self.source, value="generated").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        size_row = ttk.Frame(box)
        size_row.grid(row=3, column=0, columnspan=2, sticky="w", padx=(20, 0))
        ttk.Combobox(size_row, textvariable=self.size, values=["512", "1024", "2048", "4096"], width=7).pack(side="left")
        ttk.Label(size_row, text="pixels square").pack(side="left", padx=4)
        box.columnconfigure(0, weight=1)

        # filter
        box = ttk.LabelFrame(left, text="Filter", padding=8)
        box.pack(fill="x", pady=(8, 0))
        for i, (value, text) in enumerate(FILTERS):
            ttk.Radiobutton(box, text=text, variable=self.filter, value=value,
                            command=self.on_filter_change).grid(row=0, column=i, sticky="w", padx=(0, 8))
        self.blur_row = ttk.Frame(box)
        self.blur_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(self.blur_row, text="Blur size:").pack(side="left")
        ttk.Combobox(self.blur_row, textvariable=self.blur_size, state="readonly", width=5,
                     values=engine.BLUR_SIZES).pack(side="left", padx=4)
        ttk.Label(self.blur_row, text="(N x N window)").pack(side="left")

        # method
        box = ttk.LabelFrame(left, text="Method", padding=8)
        box.pack(fill="x", pady=(8, 0))
        ttk.Label(box, text="Method:").grid(row=0, column=0, sticky="w")
        self.method_box = ttk.Combobox(box, textvariable=self.method, state="readonly", width=28)
        self.method_box.grid(row=0, column=1, sticky="w", pady=2)
        self.method_box.bind("<<ComboboxSelected>>", lambda e: self.on_method_change())
        self.schedule_label = ttk.Label(box, text="Schedule:")
        self.schedule_label.grid(row=1, column=0, sticky="w")
        self.schedule_box = ttk.Combobox(box, textvariable=self.schedule, state="readonly", width=10,
                                         values=engine.SCHEDULES)
        self.schedule_box.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Threads:").grid(row=2, column=0, sticky="w")
        self.threads_box = ttk.Combobox(box, textvariable=self.threads, state="readonly", width=5, values=threads)
        self.threads_box.grid(row=2, column=1, sticky="w", pady=2)
        self.help = ttk.Label(box, foreground=GREY, wraplength=int(270 * app.scale), justify="left")
        self.help.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Checkbutton(left, text="Show thread map (which thread computed each pixel)",
                        variable=self.show_map).pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8)
        self.apply_button = ttk.Button(buttons, text="Apply", default="active", command=self.apply)
        self.apply_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=app.stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Save As...", command=self.save_as).pack(side="left")

        # result
        box = ttk.LabelFrame(left, text="Result", padding=8)
        box.pack(fill="x")
        self.result_values = {}
        for i, name in enumerate(["Parallel time", "Sequential time", "Speedup"]):
            ttk.Label(box, text=name + ":").grid(row=i, column=0, sticky="w")
            self.result_values[name] = ttk.Label(box, text="-")
            self.result_values[name].grid(row=i, column=1, sticky="w", padx=8)
        self.identical_label = ttk.Label(box, text="")
        self.identical_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.on_filter_change()
        self.show_message("Choose an input and press Apply (or Enter).")

    def start_button(self):
        return self.apply_button

    # ---- settings ------------------------------------------------------------

    def method_code(self):
        """"Rows (omp_rows)" -> "omp_rows"."""
        return self.method.get().rsplit("(", 1)[1].rstrip(")")

    def set_method(self, code):
        self.method.set(f"{engine.METHOD_NAMES[code]} ({code})")
        self.on_method_change()

    def on_filter_change(self):
        choices = [code for code, _ in engine.methods_for(self.filter.get())]
        self.method_box.config(values=[f"{engine.METHOD_NAMES[code]} ({code})" for code in choices])
        current = self.method_code() if self.method.get() else "omp_rows"
        self.set_method(current if current in choices else "omp_rows")   # separable methods are blur only
        if self.filter.get() == "blur":
            self.blur_row.grid()
        else:
            self.blur_row.grid_remove()

    def on_method_change(self):
        code = self.method_code()
        self.help.config(text=engine.METHOD_HELP[code])
        if code == "omp_rows":
            self.schedule_label.grid()
            self.schedule_box.grid()
        else:
            self.schedule_label.grid_remove()
            self.schedule_box.grid_remove()
        parallel = code not in ("seq", "seq_sep")
        self.threads_box.state(["!disabled", "readonly"] if parallel else ["disabled"])

    # ---- input and output files ----------------------------------------------

    def browse(self):
        path = filedialog.askopenfilename(
            title="Open image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.ppm *.tif *.tiff"),
                                           ("All files", "*.*")])
        if not path:
            return
        try:
            target = self.app.runs / "opened.ppm"
            width, height = engine.to_ppm(path, target)
        except Exception as error:                      # not an image Pillow can read
            messagebox.showerror("Cannot open image", f"{path}\n\n{error}")
            return
        self.opened_ppm = target
        self.file_name.set(f"{Path(path).name}  ({width} x {height})")
        self.source.set("file")
        self.show_images([("Original", target)])

    def save_as(self):
        result = self.app.runs / "result.ppm"
        if not self.result_shown or not result.exists():
            messagebox.showinfo("Save result", "There is no result yet. Press Apply first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save result", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("PPM image", "*.ppm")])
        if path:
            with Image.open(result) as image:
                image.save(path)
            self.app.status.config(text=f"Saved {path}")

    # ---- Apply ---------------------------------------------------------------

    def apply(self):
        if self.app.current_run:
            return
        for name in ("result.ppm", "map.ppm", "filter.csv"):
            (self.app.runs / name).unlink(missing_ok=True)
        self.result_shown = False
        if self.source.get() == "file":
            if not self.opened_ppm:
                messagebox.showinfo("No image", "Choose an image file first, or use the generated test image.")
                return
            self.run_filter(self.opened_ppm)
            return
        try:
            size = int(self.size.get())
            if not 1 <= size <= 20000:
                raise ValueError
        except ValueError:
            messagebox.showerror("Image size", "The image size must be a whole number from 1 to 20000.")
            return
        generated = self.app.runs / "generated.ppm"
        self.app.run(["gen", "--size", size, "--out", generated], f"Generating a {size} x {size} test image...",
                     lambda code, error: self.run_filter(generated) if code == 0 else self.finish(code, error))

    def run_filter(self, input_ppm):
        runs = self.app.runs
        code = self.method_code()
        args = ["filter", "--in", input_ppm, "--kernel", self.filter.get(), "--variant", code,
                "--threads", self.threads.get(), "--verify", "--csv", runs / "filter.csv",
                "--out", runs / "result.ppm"]
        if self.filter.get() == "blur":
            args += ["--blur-size", self.blur_size.get()]
        if code == "omp_rows":
            args += ["--schedule", self.schedule.get()]
        if self.show_map.get():
            args += ["--threadmap", runs / "map.ppm"]
        self.input_ppm = input_ppm
        self.app.run(args, "Filtering (the sequential version also runs, for comparison)...", self.finish)

    def finish(self, code, error):
        csv = self.app.runs / "filter.csv"
        if code == -1:
            self.app.status.config(text="Stopped")
            return
        if code not in (0, 2) or not csv.exists():      # 2 = finished, but the output differs
            self.app.show_engine_error(error)
            return
        row = charts.read_csv(csv)[0]
        speedup = float(row["speedup"])
        self.result_values["Parallel time"].config(text=f"{float(row['time_s']):.4f} s")
        self.result_values["Sequential time"].config(text=f"{float(row['seq_time_s']):.4f} s")
        self.result_values["Speedup"].config(text=f"{speedup:.2f} x")
        if row["identical"] == "yes":
            self.identical_label.config(text="✔ Identical to sequential", foreground=GREEN)
        else:
            self.identical_label.config(text="✘ Differs from sequential", foreground=RED)

        panels = [("Original", self.input_ppm), ("Result", self.app.runs / "result.ppm")]
        if (self.app.runs / "map.ppm").exists():
            panels.append(("Thread map", self.app.runs / "map.ppm"))
        self.show_images(panels, threads=int(row["threads"]), schedule=row["schedule"])
        self.result_shown = True
        self.app.status.config(text=f"Done: {self.method.get()}, {row['threads']} thread(s), "
                                    f"speedup {speedup:.2f} x")

    # ---- the picture area ----------------------------------------------------

    def clear_view(self):
        for child in self.view.winfo_children():
            child.destroy()
        self.photos = []

    def show_message(self, text):
        self.clear_view()
        ttk.Label(self.view, text=text, foreground=GREY).pack(expand=True)

    def show_images(self, panels, threads=1, schedule=""):
        """Show (caption, image file) panels side by side, scaled to fit."""
        self.clear_view()
        self.update_idletasks()
        width = max(200, (self.view.winfo_width() - 12 * len(panels)) // len(panels))
        height = max(200, self.view.winfo_height() - 110)
        for caption, path in panels:
            column = ttk.Frame(self.view)
            column.pack(side="left", fill="both", expand=True, padx=6)
            ttk.Label(column, text=caption).pack(anchor="w")
            with Image.open(path) as image:
                full = image.size
                image.thumbnail((width, height))
                photo = ImageTk.PhotoImage(image)
            self.photos.append(photo)
            ttk.Label(column, image=photo, relief="solid", borderwidth=1).pack(anchor="nw")
            ttk.Label(column, text=f"{full[0]} x {full[1]}", foreground=GREY).pack(anchor="w")
            if caption == "Thread map":
                self.show_legend(column, threads, schedule, width)

    def show_legend(self, parent, threads, schedule, width):
        legend = ttk.Frame(parent)
        legend.pack(anchor="w", pady=(4, 0))
        for t in range(threads):
            r, g, b = engine.THREAD_COLORS[t % 10]
            item = ttk.Frame(legend)
            item.grid(row=t // 2, column=t % 2, sticky="w", padx=(0, 10))
            tk.Label(item, bg=f"#{r:02x}{g:02x}{b:02x}", width=2).pack(side="left")
            ttk.Label(item, text=f"Thread {t}").pack(side="left", padx=3)
        if schedule in ("dynamic", "guided"):
            ttk.Label(parent, text=f"Example assignment: {schedule} scheduling can differ between runs",
                      foreground=GREY, wraplength=width).pack(anchor="w", pady=(4, 0))


class RunTab(ttk.Frame):
    """Shared by the Benchmark and Batch tabs: settings and Run / Stop on the left,
    a chart and a table on the right. While the engine runs, the table and the
    chart fill in row by row from the CSV it writes. Subclasses fill in the
    settings, the command line, the chart and the table columns."""

    csv_name = ""            # file the run writes in the runs folder
    metrics = []             # (value for the chart function, label of the switch button)
    columns = []             # (CSV column, table heading)

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.rows = []
        self.canvas = None
        self.last_code = None
        self.metric = tk.StringVar(value=self.metrics[0][0])

        left = ttk.Frame(self, width=300)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.build_settings(left)
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8)
        self.run_button = ttk.Button(buttons, text="Run", default="active", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=app.stop_run, state="disabled")
        self.stop_button.pack(side="left", padx=6)

        # right side: the chart (with its switch) above the table
        switch = ttk.Frame(right)
        switch.pack(fill="x")
        ttk.Label(switch, text="Show:").pack(side="left")
        for value, label in self.metrics:
            ttk.Radiobutton(switch, text=label, variable=self.metric, value=value,
                            command=self.draw_chart).pack(side="left", padx=4)
        table_frame = ttk.Frame(right)
        table_frame.pack(side="bottom", fill="x")     # packed before the chart, so the chart gets the rest
        self.table = ttk.Treeview(table_frame, columns=[c for c, _ in self.columns], show="headings", height=8)
        self.table.tag_configure("FAIL", foreground=RED)          # rows whose output differs
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        for column, heading in self.columns:
            self.table.heading(column, text=heading)
            wide = column in ("method", "strategy")
            self.table.column(column, width=int((170 if wide else 80) * app.scale), anchor="center")
        self.table.pack(side="left", fill="x", expand=True)
        self.chart_area = ttk.Frame(right, relief="solid", borderwidth=1)
        self.chart_area.pack(fill="both", expand=True, pady=(4, 6))
        self.placeholder = ttk.Label(self.chart_area, text="Choose the settings and press Run (or Enter).",
                                     foreground=GREY)
        self.placeholder.pack(expand=True)

    def start_button(self):
        return self.run_button

    # ---- helpers for the settings --------------------------------------------

    def filter_choice(self, parent):
        box = ttk.LabelFrame(parent, text="Filter", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.filter = tk.StringVar(value="blur")
        self.blur_size = tk.StringVar(value="5")
        for i, (value, text) in enumerate(FILTERS):
            ttk.Radiobutton(box, text=text, variable=self.filter, value=value,
                            command=self.on_filter_change).grid(row=0, column=i, sticky="w", padx=(0, 8))
        self.blur_row = ttk.Frame(box)
        self.blur_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(self.blur_row, text="Blur size:").pack(side="left")
        ttk.Combobox(self.blur_row, textvariable=self.blur_size, state="readonly", width=5,
                     values=engine.BLUR_SIZES).pack(side="left", padx=4)

    def on_filter_change(self):
        if self.filter.get() == "blur":
            self.blur_row.grid()
        else:
            self.blur_row.grid_remove()

    def thread_checkboxes(self, parent):
        """One checkbox per thread count (1, 2, 4, ... up to the CPU count), all ticked."""
        box = ttk.LabelFrame(parent, text="Threads", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.thread_vars = {}
        for i, t in enumerate(engine.thread_choices()):
            self.thread_vars[t] = tk.BooleanVar(value=True)
            ttk.Checkbutton(box, text=str(t), variable=self.thread_vars[t]).grid(row=0, column=i, padx=(0, 8))

    def chosen_threads(self):
        return [t for t, var in self.thread_vars.items() if var.get()]

    def reps_choice(self, parent, default):
        self.reps = tk.StringVar(value=str(default))
        ttk.Label(parent, text="Repetitions:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(parent, from_=1, to=10, textvariable=self.reps, width=5).grid(row=0, column=1, sticky="w", pady=2)

    @staticmethod
    def number(var, name, low, high):
        """The whole number in `var`, or None (after a message) if it is not between low and high."""
        try:
            value = int(var.get())
            if low <= value <= high:
                return value
        except ValueError:
            pass
        messagebox.showerror(name, f"{name} must be a whole number from {low} to {high}.")
        return None

    # ---- running -------------------------------------------------------------

    def run(self):
        if self.app.current_run:
            return
        args = self.command()
        if args is None:
            return
        if not self.chosen_threads():
            messagebox.showerror("Threads", "Tick at least one thread count.")
            return
        args += ["--threads", ",".join(map(str, self.chosen_threads()))]
        if self.filter.get() == "blur":
            args += ["--blur-size", self.blur_size.get()]
        csv = self.app.runs / self.csv_name
        csv.unlink(missing_ok=True)
        self.last_code = None
        self.show_rows([])
        self.app.run(args + ["--csv", csv], self.status_text(), self.finished, on_line=lambda line: self.refresh())

    def refresh(self):
        """Called for each line the engine prints: show any new CSV rows at once."""
        csv = self.app.runs / self.csv_name
        rows = charts.read_csv(csv) if csv.exists() else []
        if len(rows) != len(self.rows):
            self.show_rows(rows)
            self.app.show_progress(len(rows), self.expected_rows())

    def finished(self, code, error):
        self.last_code = code
        if code == -1:
            self.app.status.config(text="Stopped")
            return
        if code not in (0, 2):                      # 2 = finished, but some output differs
            self.app.show_engine_error(error)
            return
        self.refresh()
        self.app.status.config(text="Done" if code == 0 else "Done, but some output DIFFERS from sequential")

    def show_rows(self, rows):
        self.rows = rows
        self.table.delete(*self.table.get_children())
        for row in rows:
            self.table.insert("", "end", values=[self.cell(row, column) for column, _ in self.columns],
                              tags=(row["verified"],))
        if rows:
            self.table.see(self.table.get_children()[-1])
        self.draw_chart()

    @staticmethod
    def cell(row, column):
        """Table text for one CSV value: friendly names, times to 4 decimals."""
        value = row[column]
        if column == "method":
            return engine.METHOD_NAMES.get(value, value)
        if column == "strategy":
            return charts.STRATEGY_NAMES.get(value, value)
        if column == "time_s":
            return f"{float(value):.4f}"
        return value

    def draw_chart(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if not self.rows:
            self.placeholder.pack(expand=True)
            return
        self.placeholder.pack_forget()
        self.canvas = FigureCanvasTkAgg(self.make_chart(self.rows, self.metric.get()), master=self.chart_area)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.config(width=1, height=1)    # take the space the frame has, no more
        widget.pack(fill="both", expand=True)


class BenchTab(RunTab):
    """One large image: every method at the chosen thread counts (pdc_conv bench)."""

    csv_name = "bench.csv"
    metrics = [("speedup", "Speedup"), ("efficiency", "Efficiency"), ("time", "Time")]
    columns = [("method", "Method"), ("schedule", "Schedule"), ("threads", "Threads"), ("time_s", "Time (s)"),
               ("speedup_vs_seq", "Speedup"), ("efficiency", "Efficiency"), ("verified", "Verified")]

    def build_settings(self, left):
        box = ttk.LabelFrame(left, text="Image", padding=8)
        box.pack(fill="x")
        self.size = tk.StringVar(value="4096")
        ttk.Label(box, text="Size:").pack(side="left")
        ttk.Combobox(box, textvariable=self.size, values=["2048", "4096", "8000", "12000"], width=7).pack(
            side="left", padx=4)
        ttk.Label(box, text="pixels square (generated)").pack(side="left")
        self.filter_choice(left)
        self.thread_checkboxes(left)
        box = ttk.LabelFrame(left, text="Options", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.reps_choice(box, 3)
        self.schedule = tk.StringVar(value="static")
        ttk.Label(box, text="Schedule for Rows:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.schedule, state="readonly", width=9,
                     values=engine.SCHEDULES).grid(row=1, column=1, sticky="w", pady=2)
        note(left, "Each method runs Repetitions times and the fastest time counts. Every output is "
                   "compared byte by byte with the sequential one (Verified column).", self.app.scale)

    def command(self):
        size = self.number(self.size, "Image size", 1, 20000)
        reps = self.number(self.reps, "Repetitions", 1, 100)
        if size is None or reps is None:
            return None
        return ["bench", "--size", size, "--kernel", self.filter.get(), "--reps", reps,
                "--schedule", self.schedule.get()]

    def expected_rows(self):
        """seq, then Rows and Tiled at every thread count (+ the two separable methods for blur)."""
        n = len(self.chosen_threads())
        return 1 + 2 * n + (1 + n if self.filter.get() == "blur" else 0)

    def status_text(self):
        return f"Benchmarking {self.filter.get()} on a {self.size.get()} x {self.size.get()} image..."

    def make_chart(self, rows, metric):
        return charts.speedup_chart(rows, metric)


class BatchTab(RunTab):
    """Many small images, two ways of sharing them between threads (pdc_conv batch)."""

    csv_name = "batch.csv"
    metrics = [("speedup", "Speedup"), ("images_per_s", "Images per second")]
    columns = [("strategy", "Strategy"), ("threads", "Threads"), ("time_s", "Time (s)"),
               ("speedup_vs_seq", "Speedup"), ("efficiency", "Efficiency"), ("images_per_s", "Images/s"),
               ("verified", "Verified")]

    def build_settings(self, left):
        box = ttk.LabelFrame(left, text="Images", padding=8)
        box.pack(fill="x")
        self.count = tk.StringVar(value="3000")
        self.size = tk.StringVar(value="1024")
        ttk.Label(box, text="Number of images:").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.count, width=8).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(box, text="Size of each:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.size, values=["512", "1024", "2048"], width=7).grid(
            row=1, column=1, sticky="w", pady=2)
        self.filter_choice(left)
        self.thread_checkboxes(left)
        box = ttk.LabelFrame(left, text="Options", padding=8)
        box.pack(fill="x", pady=(8, 0))
        self.reps_choice(box, 2)
        note(left, "There are two ways to use the threads on many small images. \"Threads share each "
                   "image\" runs the Rows method on one image after another, so every image starts and "
                   "ends a parallel loop. \"One image per thread\" hands whole images to whichever "
                   "thread is free (schedule(dynamic, 1)), so threads never wait for each other.",
             self.app.scale)

    def command(self):
        count = self.number(self.count, "Number of images", 1, 100000)
        size = self.number(self.size, "Image size", 1, 8192)
        reps = self.number(self.reps, "Repetitions", 1, 100)
        if count is None or size is None or reps is None:
            return None
        return ["batch", "--n", count, "--bsize", size, "--kernel", self.filter.get(), "--reps", reps]

    def expected_rows(self):
        """seq, then both strategies at every thread count."""
        return 1 + 2 * len(self.chosen_threads())

    def status_text(self):
        return f"Filtering {self.count.get()} images of {self.size.get()} x {self.size.get()}..."

    def make_chart(self, rows, metric):
        return charts.batch_chart(rows, metric)


if __name__ == "__main__":
    App().mainloop()
