# Design: final project polish, GUI and viva material

Date: 2026-11-07
Project: Parallel Image Convolution / Filtering (OpenMP), PDC course topic #21,
FAST-NUCES Karachi, Fall 2026. Shayan Ahmed, Rizwan Vadsariya, Aaqib.

## 1. Intent and why

This repository is a graded student project. Marks: proposal 1 (done), code 4,
viva 3, report 2. The course rules (docs/PDC_Project_List.docx) require OpenMP on a
problem large enough that the sequential version is "noticeably slow", a written
report with time / speedup / efficiency graphs over thread counts, and that **every
group member can explain and defend the full project in the viva**. Code "copied
without proper understanding" is treated as plagiarism.

The code already works, but:

- At the proposal's 12000x12000 size the sequential run takes 0.3 to 0.9 s, while
  the proposal promised "several seconds".
- The final benchmark has not been run at the proposal sizes (12000x12000 image;
  3000 images of 1024x1024). Earlier 32-bit builds could not; the 64-bit build now can.
- The report (REPORT.md, ~60 lines) is too thin for its 2 marks.
- The code uses constructs that are hard for students to explain (token-building
  macros, function-pointer tables), and the repository mixes docs, binaries and
  generated output with no clear layout.
- Everything is CLI-only. The students want a desktop GUI to demo with and to stop
  using the terminal entirely.

Success: the students can run, demo and explain the whole project from a GUI, the
numbers in the report come from proposal-size runs, and each of them can walk an
examiner through any file.

### Priority (added after review, overrides anything below that conflicts)

The project must stay **small and explainable**, in how it works and in the code
itself. The focus is the OpenMP C code the instructor requires. Every addition must
help explain or measure the OpenMP work; anything else is cut. The C program gets no
code whose only purpose is serving the GUI (no JSON output, no extra commands), and
the GUI stays a thin front end over the existing commands.

## 2. Goals

1. **Variable blur size.** Gaussian blur gets a size setting (5x5 up to at least
   15x15) so the sequential 12000x12000 run takes several seconds honestly, and so
   the report can show "small kernels are memory-bound and scale poorly, large
   kernels are compute-bound and scale well".
2. **Thread map.** Any parallel run can output an image coloured by which thread
   computed each pixel (rows show as stripes, tiles as blocks, dynamic scheduling as
   interleaved strips). This is the main visual viva aid.
3. **Scheduling comparison.** The row-parallel method can run with `static`,
   `dynamic` or `guided` scheduling, so the report can show why static is right for
   an evenly loaded problem.
4. **Simplified, explainable code and layout** (section 5).
5. **Desktop GUI** in the agreed Windows 10 native style (section 6).
6. **Clean final results** at proposal sizes, plots regenerated.
7. **Full report** (section 8) and a **viva guide** (section 9).

## 3. Non-goals

- No MPI or hybrid version. Topic #21 is OpenMP only.
- The proposal document is not changed (already submitted). Deviations are
  explained in the report instead.
- No new filter types beyond blur / sharpen / sobel. No kernel editor.
- The GUI does not do any image filtering or timing itself.

## 4. Anti-goals (would be a failure even if it works)

- Any parallel variant producing output that differs from the sequential one by
  even one byte. Verification stays exact, no tolerance.
- Code the students cannot explain line by line: clever macros, generated code,
  heavy abstraction, anything "AI-looking".
- A GUI that looks like a modern web/AI app. It must look like a plain Windows 10
  program (see mockup description in 6.1).
- Timings that are measured in Python or include process start-up, file I/O or
  image conversion.
- A GUI so large it becomes the project. It is a front end to the C program.

## 5. Code and repository

### 5.1 Constraints on the C code

- Plain C11, built with the 64-bit WinLibs GCC 16.2 at `C:\mingw64` via
  `scripts\build.ps1` (current logic of `build_windows.ps1`: static link, refuses
  32-bit unless `-Allow32Bit`). Must also still compile with gcc on Linux.
- **Portable CPU target instead of `-march=native`** (Claude's call): build with
  `-march=x86-64-v2` (SSE4.2, supported by this i7-3770 and any laptop from the last
  ~12 years). The exe that produces the report numbers is then the same exe that
  runs on the demo laptop. `-march=native` could emit instructions the laptop's CPU
  lacks (the exe would crash), or tune differently from what was measured.
- Integer arithmetic only, so every variant stays **bit-identical** to the
  sequential reference, including the new blur sizes (choose intermediate/accumulator
  widths so nothing overflows; 64-bit accumulators are fine).
- Timing stays in C, with the existing `wtime()` (QueryPerformanceCounter on
  Windows, `omp_get_wtime()` elsewhere). Report must explain why.
- Keep the existing measurement hygiene: parallel first-touch of buffers, thread
  team warm-up outside the timer, output poisoned before each run, best-of-N reps.
- Fix: the batch "one image per thread" strategy currently allocates and clears a
  buffer inside the timed region; move that outside the timer.
- Performance of the restructured code should stay close to today's for sharpen
  and sobel (rough guide: within ~15% at 8000x8000). If a simpler structure costs
  more than that, prefer the simpler one only if the students could not otherwise
  explain it, and say so in the report.
- **Blur uses one general-size implementation for every size, with no special 5x5
  fast path** (Claude's call). Every method and every size then runs the same code,
  so the size sweep compares like with like, and there is less code to explain.
  5x5 may get slower than today; that is acceptable (speedups are measured against
  the sequential run of the same code, and a slower T1 suits the "noticeably slow"
  requirement). Weights are the binomial row for the size (1-4-6-4-1 for 5x5,
  1-6-15-20-15-6-1 for 7x7, ...), so 5x5 output stays byte-identical to today's.

### 5.2 Explainability rules

- Each source file starts with a short plain-English comment: what it does and
  which course concept it shows.
- Prefer functions, loops, `switch`. No token-pasting or code-generating macros.
  A function-pointer table is acceptable only if a `switch` would be clearly worse.
- Every `#pragma omp` line has a one-line comment saying what it does in this spot.
- Names a student can say out loud: `filter_rows_parallel`, not `v_rows`.
- Method names shown to users: friendly name plus code name, e.g.
  "Tiled (omp_tiled)".

### 5.3 Methods to keep

| Code name | Friendly name | Notes |
|---|---|---|
| `seq` | Sequential | reference and T1 |
| `omp_rows` | Rows | static by default; `--schedule static\|dynamic\|guided` |
| `omp_tiled` | Tiled | `--tile HxW` |
| `seq_sep` | Separable (sequential) | blur only |
| `omp_sep` | Separable | blur only, two passes with a barrier |
| `omp_sep_fused` | Separable, fused | blur only, `--fuse-bh R` |

### 5.4 CLI (kept; the GUI drives it)

Existing sub-commands `gen`, `filter`, `bench`, `batch` stay (no new commands), plus
only these options:

- `--blur-size N` (odd, 3 to 25) for every command that runs blur.
- `--schedule static|dynamic|guided` for `omp_rows` (one value per run).
- `bench --methods a,b` to run only some methods (sequential always runs).
- `filter --verify`: also runs the sequential version, reports both times, the
  speedup and whether the output is identical.
- `filter --csv f.csv`: writes that one result as a CSV row, the same way `bench`
  and `batch` already write CSVs. The GUI reads results only from these CSV files
  and output images; it shows the normal text output as a log.
- Sweeps over several blur sizes or schedules are done by running `bench` several
  times (from `scripts\run_all.py`), not by more loops in C.
- The correctness sweep over awkward image sizes lives in the tests (Python), not
  in the C program.
- `filter --threadmap out.ppm`: writes the thread-ownership image. It comes from
  a **separate, untimed run** with the same settings, so recording ownership never
  affects a timing (Claude's call). Each pixel is coloured by the thread that wrote
  its final output value: for `omp_sep` that is the vertical (second) pass, for
  `omp_sep_fused` the thread that owned the row block. With `dynamic`/`guided` the
  map shows one example assignment and can differ from the timed run; the GUI says
  so in one line, and the viva guide explains why.
- `bench` and `batch` keep writing CSVs; their CSVs gain `blur_size` and
  `schedule` columns.

### 5.5 Repository layout (target)

```
README.md          one page: what it is, how to build, run the GUI, use the CLI
REPORT.md          the course report (section 8)
VIVA_GUIDE.md      study guide (section 9)
src/               pdc.h (one shared header) + image.c, filters.c, parallel.c,
                   bench.c, main.c. All the OpenMP methods live in parallel.c.
gui/               app.pyw (window and tabs), engine.py (runs pdc_conv.exe),
                   charts.py (matplotlib charts, also used by plot_results.py)
tests/             automated checks that every method matches the sequential one
scripts/           build.ps1, run_all.py, plot_results.py, package.ps1
results/           final CSVs, logs and plots (committed)
demo/              demo PNGs used by README/REPORT (committed)
docs/              proposal, project list, this design
.gitignore         pdc_conv.exe, *.dll, *.ppm, __pycache__/, .superpowers/
```

- `HowToRun.md` is merged into `README.md` and deleted.
- `pdc_conv.exe` and the old DLLs are removed from git (rebuilt by the script).
- `run_windows.ps1` becomes `run_all.py`: build, bench runs, batch, plots, demo
  images.

## 6. GUI

### 6.1 Look (decided with the students)

Windows 10 native, as in the approved mockup: Tkinter `ttk` with the default
`vista` theme on Windows 10 (verified available: Tk 8.6, Python 3.14). White title
bar, menu bar (File, Help), three tabs (Filter, Benchmark, Batch),
settings in group boxes on the left, results on the right, status bar at the
bottom with status text, CPU (cores/threads), free RAM and a progress bar
(indeterminate "busy" animation while the engine runs). Flat
grey buttons, the default button outlined in blue. Charts drawn with matplotlib
embedded in the window, styled plainly (white background, thin grey grid, default
colour cycle) to match.

### 6.2 Tech

Python 3 + Tkinter/ttk + Pillow + matplotlib (already in requirements.txt). The GUI
calls `pdc_conv.exe` as a subprocess; long runs happen in the background, the
window never freezes, and a Stop button kills the run.

The GUI never builds anything: on the development PC `scripts\build.ps1` is run
first, and in the packaged app (section 6.7) the exe is bundled. If the exe is
missing, the GUI says so in a message box and names the script.

Everything machine-dependent is detected at start-up, never hard-coded: logical
CPU count (thread choices and defaults, e.g. 1, 2, 4, ... up to that count), CPU
name, total and free RAM. The status bar and the RAM warning use the laptop's
values when run there.

### 6.3 Filter tab

- Input: open a JPG/PNG/BMP/PPM (converted to PPM by Pillow), or "Generated test
  image" with a size field.
- Filter: Blur / Sharpen / Sobel radio buttons; blur size control shown only for blur.
- Method dropdown (only valid methods for the chosen filter), schedule dropdown
  when Rows is chosen, threads (1 to number of logical CPUs).
- Short "how this method works" text under the method (2 to 3 sentences).
- Apply: runs `filter --verify`. Shows original and result side by side, plus a
  Result box: parallel time, sequential time, speedup, "Identical to sequential".
- "Show thread map" checkbox: shows the thread-ownership image (in place of or
  next to the result; builder's call) with a colour legend per thread.
- Save As (PNG/JPG/PPM).

### 6.4 Benchmark tab

Settings: image size (presets 2048, 4096, 8000, 12000), filter (one at a time),
blur size, thread counts (checkboxes, defaults 1, 2, 4, 8 limited to the CPU),
reps, schedule for Rows. A RAM check warns before a run that needs more memory
than is free. Results when the run finishes: one chart (speedup or efficiency,
switchable) and a table read from the run's CSV, with a verification column; a
log box shows the engine's normal text output as it runs. Tile size and fused
block rows stay at their defaults (CLI only).

### 6.5 Batch tab

Number of images, image size, filter (+ blur size), thread counts, reps. One chart
and a table comparing "split rows of each image" vs "one image per thread"
(speedup and images per second), plus the log box.

### 6.6 Menus

- File: Open image, Save result, Exit.
- Help: About (course, members), Open source folder.

No settings persistence, no "Run everything", no "open previous results", no
correctness-check dialog: the final results are produced by `scripts\run_all.py`
and the correctness sweep is in `tests\`.

### 6.7 Packaging for the demo laptop

The viva runs on a **different laptop** (unknown CPU, RAM, and whether Python or a
compiler is installed). So:

- `scripts\package.ps1` builds a self-contained folder and zip, e.g.
  `dist\PDC-Convolution\` + `dist\PDC-Convolution.zip`, containing:
  the GUI as a Windows exe (PyInstaller, one-folder mode; no Python needed on the
  laptop), the static `pdc_conv.exe`, the demo images, the final `results\`
  (so charts and tables can be shown even without re-running), the README, the
  REPORT and the VIVA_GUIDE.
- Unzip anywhere, double-click the GUI exe, it works. No install, no terminal.
- The source code stays viewable on the laptop: the zip also contains `src\`, and the
  GUI's Help menu has "Open source folder", for when the examiner asks to see code.
- README has a short optional section "Rebuilding on another PC" (install WinLibs to
  `C:\mingw64`, run `scripts\build.ps1`) in case the examiner asks for a compile.
- `dist\` is git-ignored; the zip is what gets copied to the laptop.
- The students must test the zip on the actual laptop well before the viva
  (see 12). Windows SmartScreen may warn about an unsigned exe the first time
  ("More info" then "Run anyway"); README mentions this.

## 7. Final results

After the code and GUI are done, run the full set at proposal sizes on the
students' PC (i7-3770, 4 cores / 8 threads, 8 GB RAM) with other programs closed:

- `bench` 12000x12000, all filters, threads 1/2/4/8, reps 3, blur at 5x5 and one
  large size (e.g. 15x15).
- Blur size sweep at 8 threads (for the memory-bound vs compute-bound graph).
- Scheduling comparison (static / dynamic / guided) for Rows.
- `batch` 3000 x 1024x1024, threads 1/2/4/8, reps 2.

Plots regenerated into `results\plots\`, demo images (including a thread map) into
`demo\`.

## 8. Report (REPORT.md)

Written from the final numbers. Must cover: objective and problem size; machine
and toolchain (incl. why 64-bit, why QPC timer); each method and the OpenMP
constructs it uses; verification method; results tables and graphs (time, speedup,
efficiency vs threads; speedup vs blur size; scheduling comparison; batch); analysis
(Amdahl's law with an estimated serial fraction, memory bandwidth limits for
sharpen and plain separable blur, why fusion helps, hyper-threading beyond 4
threads, per-image vs per-row reasons); deviations from the proposal; conclusion.

## 9. Viva guide (VIVA_GUIDE.md)

Plain-language study guide for all three members: walkthrough of every source file
and function; each OpenMP construct used and why; the measurement tricks and why
they matter; how to read each graph; likely examiner questions with model answers;
a 5-minute demo script using the GUI.

## 9a. Timeline and cut order

Submission and viva are within 20 days of 2026-11-07 (by ~2026-11-27). Order of
work, so the graded parts are safe early:

1. C changes and restructure (code: 4 marks).
2. Final results runs on this PC (needs other programs closed).
3. Report and viva guide drafts from those numbers (report 2 + viva 3 marks).
4. GUI: Filter tab first (the demo), then Benchmark, then Batch.
5. Packaging, then the students test the zip on the demo laptop. Target: the laptop
   test happens **at least a week before the viva**, leaving time to fix problems.
6. Final pass on report and viva guide (screenshots from the GUI, demo script
   timed on the laptop).

If time runs short, cut in this order (never the report or viva guide): the GUI's
Batch tab (batch results still come from the CLI and stay in the report), then the
GUI's Benchmark tab (the Filter tab plus the final plots are enough to demo), then
the scheduling comparison.

## 10. Left to the builder

- Exact split of `src/` into files, function names, internal data structures.
- How the GUI and CLI exchange data (format of the machine-readable mode).
- Thread map colours and whether it replaces or sits beside the result preview.
- Blur size limits beyond the 5..15 minimum, and how sizes are offered in the GUI.
- GUI internal structure (one file or a few).
- Order of work, as long as the final results and report come last.

## 11. Calls made by Claude (not explicitly confirmed by the students)

- Tkinter + subprocess-to-exe architecture; no GUI-side filtering.
- Keep all six methods; add scheduling only to Rows.
- Repository layout in 5.5, including deleting HowToRun.md and untracking binaries.
- This design lives in `docs/DESIGN.md` (can be deleted before submission).
- One general-size blur, no 5x5 fast path; binomial weights (5.1).
- Thread map from a separate untimed run, showing the final writer (5.4).
- GUI result files go to a scratch folder (`%LOCALAPPDATA%\PDC-Convolution\runs`)
  so they never overwrite the committed final CSVs in `results\`.
- Scope trimmed for explainability (after review): no JSON output or `check`
  command in C, one shared header, three GUI files, two menus, no GUI build step,
  settings persistence, "Run everything" or result loading.
- `-march=x86-64-v2` instead of `-march=native` (5.1).
- PyInstaller one-folder package with the exe, results and docs (6.7).
- Work order and cut order in 9a.
