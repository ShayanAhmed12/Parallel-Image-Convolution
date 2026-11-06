# PDC Convolution: Final Polish, GUI and Viva Material — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OpenMP convolution project small, explainable and submission-ready: a cleaner C engine with variable blur size, scheduling choice and thread maps; proposal-size results; a thin Windows 10 style GUI packaged for the demo laptop; full report and viva guide.

**Architecture:** The C program (`pdc_conv.exe`) does all filtering and timing; all OpenMP methods live in `src/parallel.c`. The Python GUI runs the exe with ordinary command-line options and reads the CSV files and images it writes. Nothing in C exists only to serve the GUI.

**Tech Stack:** C11 + OpenMP, WinLibs GCC 16.2 x86_64 at `C:\mingw64`, Python 3.14 + Tkinter/ttk (`vista` theme) + Pillow + matplotlib, `unittest` (stdlib), PyInstaller 6.22.

**Spec:** `docs/DESIGN.md`. Its "Priority" paragraph overrides everything: small, explainable, OpenMP C first.

## Global Constraints

- **Explainability first.** Each source file opens with a plain-English comment (what it does, which course concept); every `#pragma omp` line has a one-line comment; plain functions, loops and `switch`; no code-generating macros, no function-pointer tables; names you can say aloud. When a choice trades a little speed for much clearer code, take the clearer code.
- Every method's output is **byte-identical** to the sequential reference; integer arithmetic only.
- Build: `scripts\build.ps1`, `C:\mingw64\bin\gcc.exe`, `-O3 -march=x86-64-v2 -fopenmp -Wall -Wextra -std=gnu11 -static`; zero warnings; refuses 32-bit unless `-Allow32Bit`.
- Timing only in C via `wtime()`; keep first-touch, warm-up outside the timer, output poisoning, best-of-N.
- Methods (code / friendly): `seq` Sequential, `omp_rows` Rows, `omp_tiled` Tiled, `seq_sep` Separable (sequential), `omp_sep` Separable, `omp_sep_fused` Separable, fused.
- Blur sizes odd 3..25, binomial weights, one implementation for all sizes; 5x5 output identical to today's.
- GUI: ttk `vista` theme, Windows 10 look; no filtering/timing in Python; reads only the engine's CSVs/images; runs write to `%LOCALAPPDATA%\PDC-Convolution\runs\`.
- Tests: `python -m unittest discover -s tests -v` from the repo root after `scripts\build.ps1`.
- Branch `final-polish`; commit at the end of every task.

## Review Focus

1. **Paths with spaces, `+`, parentheses** (`D:\Projects\C++\...`, `photo (1).jpg`): engine calls use argument lists, never a shell string. Test in Task 9.
2. **Odd user images** (RGBA PNG, greyscale, 16-bit PNG, palette BMP, 1x1): converted to 8-bit RGB PPM first. Test in Task 9.
3. **Out of memory** (12000x12000 on an 8 GB laptop): engine exits 1 with `error: out of memory...`; GUI shows it in a message box and returns to idle. Test in Task 9.
4. **Stop during a long run:** process ends within 2 s, buttons re-enabled, no half-written CSV shown as a result. Test in Task 9.
5. **Invalid blur size** (4, 1, 27, `abc`): exit 1 with `error: blur size must be an odd number from 3 to 25`. Test in Task 3.

---

## File map

```
src/pdc.h          every type and function declaration, with a comment per item
src/image.c        die, aligned alloc, first_touch, wtime; Image, PPM I/O, test image, compare
src/filters.c      Filter (weights), blur/sharpen/sobel on a box of pixels, separable row helpers
src/parallel.c     the six methods, OpenMP schedules, thread-map recording and writing
src/bench.c        bench and batch commands (timing, verification, CSV)
src/main.c         argument parsing, gen and filter commands, usage text
gui/app.pyw        window, menus, the three tabs, status bar
gui/engine.py      run pdc_conv.exe in the background, stop it, system info, image conversion
gui/charts.py      matplotlib chart functions (GUI and plot script)
scripts/build.ps1, run_all.py, plot_results.py, package.ps1
tests/test_engine.py, tests/test_gui.py, tests/golden_sha256.txt
README.md, REPORT.md, VIVA_GUIDE.md
```

Removed: `src/pdc_conv.c`, `HowToRun.md`, `scripts/build_windows.ps1`, `scripts/run_windows.ps1`; untracked: `pdc_conv.exe`, `*.dll`.

## Engine CSV formats (exact headers)

- `filter --csv`: `filter,blur_size,method,schedule,threads,width,height,time_s,seq_time_s,speedup,identical` (last three empty without `--verify`; `identical` is `yes`/`no`)
- `bench`: `filter,blur_size,method,schedule,threads,time_s,speedup_vs_seq,efficiency,speedup_vs_self_1thread,verified` (`blur_size` empty for sharpen/sobel; `schedule` empty except `omp_rows`; `verified` = `ref|ok|FAIL`)
- `batch`: `strategy,filter,blur_size,threads,time_s,speedup_vs_seq,efficiency,images_per_s,verified`

Exit codes: 0 ok, 1 error (`error: ...` on stderr), 2 output mismatch.

Thread colours (`THREAD_COLORS`, matplotlib tab10, id modulo 10):
`(31,119,180) (255,127,14) (44,160,44) (214,39,40) (148,103,189) (140,86,75) (227,119,194) (127,127,127) (188,189,34) (23,190,207)`

---

### Task 1: Branch, hygiene, portable build, regression baseline

**Files:** Create `.gitignore`, `tests/__init__.py`, `tests/test_engine.py`, `tests/golden_sha256.txt`; rename `scripts/build_windows.ps1` → `scripts/build.ps1`; untrack exe/DLLs.

**Produces:** in `tests/test_engine.py`: `ROOT`, `EXE`, `run(*args, expect=0) -> CompletedProcess` (list args, text mode, asserts return code), `sha256(path) -> str`, `read_ppm(path) -> (w, h, bytes)`.

- [ ] **Step 1:** `git checkout -b final-polish`; commit the pending 64-bit build/timer/doc changes as `build: 64-bit static build and QPC timer`.
- [ ] **Step 2:** With the **current** exe: `gen --size 257x131`, then `filter --variant seq` for blur, sharpen, sobel; store `input/blur/sharpen/sobel` hashes in `tests/golden_sha256.txt`. Copy the exe to `pdc_conv_old.exe` (for Task 2's speed check).
- [ ] **Step 3: Tests:** `test_golden_outputs` (re-run Step 2 into a temp dir, hashes equal); `test_all_methods_match_sequential` (sizes `1x1 2x3 3x3 4x7 5x5 17x9 37x53 128x1 1x128 257x131`, `bench --size S --threads 1,3,4 --reps 1 --csv <tmp>` → exit 0 and `ALL VARIANTS MATCH`).
- [ ] **Step 4:** `.gitignore`: `pdc_conv.exe`, `pdc_conv_old.exe`, `*.dll`, `*.ppm`, `__pycache__/`, `.superpowers/`, `dist/`, `build/`; `git rm --cached pdc_conv.exe libgomp-1.dll pthreadGC2.dll`.
- [ ] **Step 5:** Rename the build script, `-march=native` → `-march=x86-64-v2`, rebuild, run tests. Expected: 2 PASS.
- [ ] **Step 6:** Commit `chore: gitignore, portable build, regression baseline`.

### Task 2: Split the engine into explainable files (no behaviour change)

**Files:** Create `src/pdc.h`, `src/image.c`, `src/filters.c`, `src/parallel.c`, `src/bench.c`, `src/main.c`; delete `src/pdc_conv.c`; `scripts/build.ps1` compiles `src\*.c`.

**Produces (`src/pdc.h`):**

```c
void   die(const char *fmt, ...);                /* "error: <msg>\n" on stderr, exit(1) */
void  *xalloc(size_t bytes);                      /* 64-byte aligned; dies "out of memory (try a smaller size)" */
void   xfree(void *p);
void   first_touch(void *buf, size_t bytes, int value);
double wtime(void);
double wtick(void);

typedef struct { int width, height; uint8_t *pixels; } Image;   /* RGB, 3 bytes/pixel, row-major */
Image  image_new(int width, int height);
void   image_free(Image *img);
Image  image_read_ppm(const char *path);
void   image_write_ppm(const char *path, const Image *img);
void   image_make_test(Image *img, uint32_t seed);
size_t image_count_differences(const Image *a, const Image *b);

typedef enum { FILTER_BLUR, FILTER_SHARPEN, FILTER_SOBEL } FilterKind;
typedef struct { FilterKind kind; int size; int radius; uint32_t weights[25]; int shift; } Filter;
Filter      filter_make(FilterKind kind, int blur_size);      /* Task 2: blur size 5 only */
const char *filter_name(FilterKind kind);
int         filter_from_name(const char *name, FilterKind *out);   /* 1 ok, 0 unknown */
void        filter_box(const Filter *f, const Image *in, Image *out, int y0, int y1, int x0, int x1);
void        blur_row_horizontal(const Filter *f, const uint8_t *src_row, uint32_t *dst_row, int width);
void        blur_row_vertical(const Filter *f, const uint32_t *const *rows, uint8_t *dst_row, int width);

typedef enum { METHOD_SEQ, METHOD_ROWS, METHOD_TILED, METHOD_SEP_SEQ, METHOD_SEP, METHOD_SEP_FUSED, METHOD_COUNT } Method;
typedef enum { SCHEDULE_STATIC, SCHEDULE_DYNAMIC, SCHEDULE_GUIDED } Schedule;
typedef struct {
    int threads; Schedule schedule; int tile_h, tile_w; int fuse_rows;
    uint8_t  *owner;     /* NULL, or width*height thread ids (Task 4) */
    uint32_t *scratch;   /* width*height*3, for METHOD_SEP_SEQ and METHOD_SEP */
} RunOptions;
RunOptions  run_options_default(void);    /* 1 thread, static, 16x2048, fuse 16, NULL, NULL */
void        run_method(Method m, const Filter *f, const Image *in, Image *out, const RunOptions *opt);
const char *method_name(Method m);
int         method_from_name(const char *name, Method *out);
int         method_blur_only(Method m);
int         method_is_parallel(Method m);
```

`filter_box` replaces `DEFINE_REGION`: one function with `switch (f->kind)` outside the pixel loops (three plain loop nests); border pixels use a gathered window. `run_method` sets the thread count and dispatches with a `switch`. The separable intermediate becomes `uint32_t`. Keep bench/batch text and CSV output unchanged in this task.

- [ ] **Step 1:** Create the files, moving today's logic.
- [ ] **Step 2:** Build (zero warnings); run tests. Expected: PASS.
- [ ] **Step 3: Speed check:** `bench --size 8000 --threads 1,8 --reps 3` with `pdc_conv_old.exe` and the new exe. Expected: sharpen and sobel within 15%. If slower, first check the `switch` is outside the pixel loop; accept a larger gap only if the fix would make the code clearly harder to explain, and note it for the report.
- [ ] **Step 4:** Commit `refactor: split engine into pdc.h, image, filters, parallel, bench, main`.

### Task 3: Variable blur size

**Files:** Modify `src/filters.c`, `src/pdc.h`, `src/main.c`, `src/bench.c`; Test `tests/test_engine.py`

**Produces:** `filter_make(FILTER_BLUR, n)` for odd 3..25: `radius = n/2`, `weights[i] = C(n-1, i)`, `shift = 2*(n-1)`; `BLUR_SIZE_MIN 3`, `BLUR_SIZE_MAX 25`; `--blur-size N` on `filter`, `bench`, `batch` (default 5). Arithmetic: row sums `uint32_t`, 2-D/vertical totals `uint64_t`, output `(total + (1ULL << (shift-1))) >> shift` (= today's `(sum+128)>>8` at 5x5).

- [ ] **Step 1: Failing tests:** `test_blur_5_matches_golden`; `test_blur_sizes_all_methods_match` (sizes 3, 7, 15, 25 × images `1x1 4x7 37x53 257x131`, `bench --kernel blur --blur-size N --threads 1,3` → `ALL VARIANTS MATCH`); `test_blur_extremes` (64x64 all-255 at size 25 with `seq` and `omp_sep` → all 255; all-0 → all 0); `test_invalid_blur_size` (`4 1 27 abc` → exit 1, message as Review Focus 5).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; time `filter --size 12000 --kernel blur --blur-size 15 --variant seq` (expect seconds; note it for the report).
- [ ] **Step 5:** Commit `feat: blur size 3..25 with exact binomial weights`.

### Task 4: Scheduling, thread map, `filter --verify` and `--csv`

**Files:** Modify `src/parallel.c`, `src/pdc.h`, `src/main.c`; Test `tests/test_engine.py`

**Produces:** `const char *schedule_name(Schedule); int schedule_from_name(const char *, Schedule *); extern const uint8_t THREAD_COLORS[10][3]; void threadmap_write_ppm(const char *path, const uint8_t *owner, int width, int height);`
- `omp_rows` uses `schedule(runtime)`; `run_method` calls `omp_set_schedule` with `(omp_sched_static,0)`, `(omp_sched_dynamic,1)`, `(omp_sched_guided,0)`. CLI `--schedule` on `filter` and `bench`.
- With `owner != NULL`, methods record `omp_get_thread_num()` once per row / tile / row-block they write (the vertical pass for `omp_sep`, the block for `omp_sep_fused`, all 0 for sequential methods).
- `filter --threadmap f.ppm`: a second, untimed run with `owner` set, written with `THREAD_COLORS[id % 10]`.
- `filter --verify`: runs `seq` too; prints `sequential: X s, speedup: Y x, identical: yes|NO`; exit 2 if not identical. `filter --csv f.csv`: one header + one row (format above).

- [ ] **Step 1: Failing tests:** `test_threadmap_static_rows` (64x40 sharpen, `omp_rows`, 2 threads → rows 0–19 colour 0, rows 20–39 colour 1); `test_threadmap_dynamic_interleaves` (64x400, dynamic, 4 threads → ≥3 colours and some colour in two separated row runs); `test_threadmap_tiled` (`--tile 8x16`, 4 threads, 64x32 → each tile one colour); `test_threadmap_sequential` (all colour 0); `test_schedules_match` (each schedule with `--verify` → `identical: yes`); `test_filter_csv` (header exact, `identical` = `yes`, `time_s` > 0).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; rerun Task 2's speed check for `omp_rows` static (no measurable cost from the `owner` check).
- [ ] **Step 5:** Commit `feat: OpenMP schedule choice, thread map, filter --verify/--csv`.

### Task 5: Bench and batch tidy-up

**Files:** Modify `src/bench.c`, `src/main.c`; Test `tests/test_engine.py`

- `bench`: `--methods a,b` (seq always runs), `--blur-size`, `--schedule`; CSV header as above. Scratch allocated only if a separable method runs.
- `batch`: `--blur-size`; CSV header as above. Fix: the "one image per thread" output buffers are allocated and first-touched **before** the timer starts.
- [ ] **Step 1: Failing tests:** `test_bench_csv` (`bench --size 64 --kernel blur --blur-size 3 --methods omp_rows --schedule dynamic --threads 1,2 --reps 1` → header exact, 3 rows: seq + rows×2, `schedule` = `dynamic` on rows lines); `test_batch_csv` (`batch --n 16 --bsize 64 --threads 1,2 --reps 1` → header exact, 5 rows, all `ok`).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: bench method selection, CSV columns, batch timing fix`.

### Task 6: Charts, plot script, `run_all.py`

**Files:** Create `gui/__init__.py`, `gui/charts.py`, `scripts/run_all.py`; rewrite `scripts/plot_results.py`; delete `scripts/run_windows.ps1`; Test `tests/test_gui.py` (+ small fixture CSVs in `tests/fixtures/`)

**Produces (`gui/charts.py`, each returns a matplotlib `Figure`, plain white style, tab10 colours, friendly method names, dashed "ideal" line where it applies):** `read_csv(path) -> list[dict]`; `speedup_chart(rows, metric="speedup"|"efficiency") -> Figure` (x = threads, one subplot per filter/blur size in `rows`); `blur_size_chart(rows, threads) -> Figure`; `schedule_chart(rows) -> Figure`; `batch_chart(rows, metric="speedup"|"images_per_s") -> Figure`.

`scripts/plot_results.py <results_dir>` writes `plots/speedup.png`, `plots/efficiency.png`, `plots/blur_sizes.png`, `plots/schedules.png`, `plots/batch.png` for whichever CSVs exist.

`scripts/run_all.py [--quick]` (builds first if `C:\mingw64` exists; `--quick` = tiny sizes, for the test). Steps, each `bench`/`batch` output also saved as `results/<name>.log`:
1. `bench --size 12000 --kernel all --threads 1,2,4,8 --reps 3` → `results/bench.csv` (blur 5)
2. same with `--kernel blur --blur-size 15` → appended into `results/bench.csv` by the script
3. blur sweep: for N in 3, 5, 9, 15, 21, 25: `bench --size 8000 --kernel blur --blur-size N --methods omp_rows,omp_tiled,omp_sep_fused --threads 1,4,8 --reps 2` → merged into `results/blur_sizes.csv`
4. schedules: for S in static, dynamic, guided: `bench --size 12000 --kernel blur --methods omp_rows --schedule S --threads 1,2,4,8 --reps 3` → merged into `results/schedules.csv`
5. `batch --n 3000 --bsize 1024 --kernel blur --threads 1,2,4,8 --reps 2` → `results/batch.csv`
6. plots
7. demo PNGs: `input`, `blur` (9x9), `sharpen`, `sobel` (1024², `omp_tiled`, 4 threads) and thread maps (sharpen, 4 threads): `threadmap_rows_static`, `threadmap_rows_dynamic`, `threadmap_tiled` (`--tile 128x256`); PPMs converted with Pillow, then deleted.

- [ ] **Step 1: Failing tests:** `test_plot_script` (fixtures → 5 PNGs > 10 KB); `test_run_all_quick` (temp copy of the layout → exit 0, 4 CSVs, 5 plots, 7 demo PNGs).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: charts, plot script and run_all.py`.

### Task 7: Final results run (needs the user)

- [ ] **Step 1:** Ask the user to close browsers, Discord, Steam etc. and confirm. Check free RAM ≥ 3.5 GB (blur at 12000² with the separable methods needs ~3 GB); if less, tell them the number and wait.
- [ ] **Step 2:** Run `python scripts/run_all.py` in the background (~15–25 min). On failure, report the step and output.
- [ ] **Step 3:** Check every `verified` is `ref`/`ok`, and look for outliers > 3x their neighbours (paging); rerun that step if found. Look at the 5 plots.
- [ ] **Step 4:** Commit `results: final proposal-size runs`.

### Task 8: README, REPORT, VIVA_GUIDE

**Files:** Rewrite `README.md`, `REPORT.md`; create `VIVA_GUIDE.md`; delete `HowToRun.md`

- [ ] **Step 1: README.md** (one page): what it is; run the packaged app; build from source (WinLibs to `C:\mingw64`, `scripts\build.ps1`); CLI reference; tests; folder map; SmartScreen note.
- [ ] **Step 2: REPORT.md** per spec 8, numbers only from `results/*.csv`, figures from `results/plots/` and `demo/`. Amdahl: serial fraction `f = (1/S - 1/p) / (1 - 1/p)` from the 8-thread speedup, calculation shown. Deviations: 32-bit → 64-bit toolchain; `omp_get_wtime` 1 ms → QPC; 5x5 sequential < 1 s → blur size; `-march=x86-64-v2`.
- [ ] **Step 3: VIVA_GUIDE.md** per spec 9: every file and function in 1–3 sentences (the `parallel.c` section is the longest); each `#pragma omp` and why; the measurement tricks; how to read each graph; ≥ 25 likely questions with model answers (separate output buffer, static vs dynamic, false sharing, why 8 threads ≠ 8x, why separable doesn't scale, fusion, first-touch, bit-exactness, QPC, Amdahl, thread map); a 5-minute demo script (timings filled in Task 12).
- [ ] **Step 4:** Check every number in REPORT.md against the CSVs. Commit `docs: README, report and viva guide`.

### Task 9: GUI engine wrapper and Filter tab

**Files:** Create `gui/engine.py`, `gui/app.pyw`; Test `tests/test_gui.py`

**Produces (`gui/engine.py`):**

```python
METHODS: list[tuple[str, str, bool]]       # (code, friendly, blur_only), spec order
METHOD_HELP: dict[str, str]                # code -> 2-3 sentence explanation
BLUR_SIZES = [3, 5, 7, 9, 11, 15, 21, 25]
def methods_for(filter_name: str) -> list[tuple[str, str]]
def thread_choices(logical_cpus: int) -> list[int]     # 8 -> [1,2,4,8]; 6 -> [1,2,4,6]; 12 -> [1,2,4,8,12]
def bench_memory_bytes(width: int, height: int, blur_separable: bool) -> int   # 3*w*h*3 (+ w*h*3*4 if blur_separable)
def batch_memory_bytes(size: int, threads: int) -> int # (8*3 + threads) * size*size*3
def to_ppm(src: Path, dst: Path) -> tuple[int, int]    # any Pillow mode -> 8-bit RGB P6
@dataclass
class SystemInfo: cpu_name: str; logical_cpus: int; free_ram: int
def system_info() -> SystemInfo                        # winreg ProcessorNameString, os.cpu_count(), GlobalMemoryStatusEx
def engine_path() -> Path                              # frozen: next to the GUI exe; else repo root
def runs_dir() -> Path                                 # %LOCALAPPDATA%\PDC-Convolution\runs (created)
class EngineRun:                                       # background subprocess, list args, CREATE_NO_WINDOW
    def __init__(self, args: list[str], on_line: Callable[[str], None],
                 on_done: Callable[[int, str], None], tk_root): ...   # on_done(code, stderr); -1 if stopped
    def stop(self) -> None                             # terminate; kill after 2 s
    running: bool
```

`app.pyw`: `App(tk.Tk)` with menu bar (File: Open image, Save result, Exit; Help: About, Open source folder), `ttk.Notebook` (Filter now; Benchmark/Batch in Task 10), status bar (status text | CPU name, n threads | free RAM | indeterminate `ttk.Progressbar` while busy). Segoe UI 9, theme colours only. If the engine is missing: message box naming `scripts\build.ps1`.

Filter tab (spec 6.3): input group (file + Browse… / generated test image + size); filter radios; blur size combobox (blur only); method combobox ("Friendly (code)"); schedule combobox (Rows only); threads combobox; method help text; Apply (default) and Save As…; "Show thread map" checkbox (map beside the result, colour legend "Thread 0..n-1", note for dynamic/guided: "Example assignment: dynamic scheduling can differ between runs"). Apply runs `filter --in … --verify --csv <runs>/filter.csv --out <runs>/result.ppm [--threadmap <runs>/map.ppm]`, then shows original | result (scaled to fit) and a Result group: parallel time, sequential time, speedup, "✔ Identical to sequential" (green) / "✘ Differs" (red).

- [ ] **Step 1: Failing tests** (Review Focus 1–4): `test_thread_choices`; `test_methods_for`; `test_bench_memory` (12000², separable → 3·432 000 000 + 1 728 000 000); `test_to_ppm_modes` (RGBA, L, I;16, palette BMP, 1x1); `test_engine_run_path_with_spaces` (input `my photo (1)+x.ppm`, real hidden Tk root, code 0, CSV `identical` yes); `test_engine_run_out_of_memory` (`bench --size 100000 --kernel blur --threads 1` → non-zero, stderr has `out of memory`); `test_engine_run_stop` (`bench --size 6000 --kernel blur --blur-size 25 --threads 1`, stop after 1 s → `on_done(-1, …)` within 2 s); `test_app_filter_apply` (App, generated 256², sobel, `omp_tiled`, 2 threads, Apply, pump events ≤ 10 s → "Identical" shown).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5: Manual check:** `pythonw gui\app.pyw`; each filter on a real JPG; thread map for Rows static, Rows dynamic, Tiled; screenshot compared with the approved Windows 10 mockup.
- [ ] **Step 6:** Commit `feat(gui): engine wrapper, window and Filter tab`.

### Task 10: GUI Benchmark and Batch tabs

**Files:** Modify `gui/app.pyw`; Test `tests/test_gui.py`

- Benchmark tab (spec 6.4): size presets 2048/4096/8000/12000; filter radios + blur size; thread checkboxes (`thread_choices`); reps spinbox (default 3); schedule combobox (Rows); RAM warning line when `bench_memory_bytes > 0.9 * free_ram` (confirm before running); Run / Stop; log box (engine text, live); after a successful run: chart (`speedup_chart`, Speedup/Efficiency switch) and `ttk.Treeview` table from the run's CSV with friendly names. A stopped or failed run shows the log only.
- Batch tab (spec 6.5): number of images (default 3000), size 512/1024/2048, filter + blur size, thread checkboxes, reps (default 2), RAM check (`batch_memory_bytes`), the explanation paragraph from the mockup, Run / Stop, log, then `batch_chart` (Speedup/Images per second) and table.
- [ ] **Step 1: Failing tests:** `test_bench_tab_run` (256², blur, threads 1,2, reps 1 → table has the CSV's row count, chart drawn); `test_batch_tab_run` (n 16, size 64, threads 1,2 → 5 rows); `test_bench_tab_stop` (start a long run, Stop → log shown, no table rows).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Manual check incl. Stop.
- [ ] **Step 6:** Commit `feat(gui): Benchmark and Batch tabs`.

### Task 11: Packaging

**Files:** Create `scripts/package.ps1`; `requirements.txt` adds `pyinstaller>=6.22`

- `package.ps1`: build engine; `python -m PyInstaller --noconfirm --windowed --name PDC-Convolution --distpath dist --workpath build\pyinstaller --paths gui gui\app.pyw`; copy `pdc_conv.exe`, `src\`, `results\`, `demo\`, `README.md`, `REPORT.md`, `VIVA_GUIDE.md` into `dist\PDC-Convolution\`; zip to `dist\PDC-Convolution.zip`.
- [ ] **Step 1:** Write and run it. Expected: zip created; note its size.
- [ ] **Step 2: Clean-machine check:** unzip to a temp path with a space; start from PowerShell with `$env:PATH='C:\Windows\System32;C:\Windows'`; Apply a filter, run a 2048² benchmark, Help → Open source folder. Expected: all work, no console window.
- [ ] **Step 3:** Commit `build: package for the demo laptop`.

### Task 12: Laptop test and final polish (needs the user)

- [ ] **Step 1:** Ask the user to copy the zip to the demo laptop (at least a week before the viva), run it, and report: CPU/RAM in the status bar, any SmartScreen prompt, whether Filter/Benchmark/thread map work, and the Filter timings.
- [ ] **Step 2:** Fix what they report (with a test where possible); re-package.
- [ ] **Step 3:** Add GUI screenshots to `docs/screenshots/`, reference them in README and REPORT; fill the demo script timings in VIVA_GUIDE.md.
- [ ] **Step 4:** All tests PASS; REPORT numbers match `results/`; `git status` clean. Commit `docs: screenshots and final demo script`.

---

## Cut order if time runs short (spec 9a)

Never cut Tasks 7, 8, 12. Drop in order: the Batch tab (Task 10), the Benchmark tab (Task 10), the schedule comparison (Task 4 schedule part, Task 6 step 4).
