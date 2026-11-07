# Parallel Image Convolution / Filtering (OpenMP)

PDC project, FAST-NUCES Karachi, Fall 2026, BCS-7B
Shayan Ahmed (23K-0551), Rizwan Vadsariya (23K-0005), Aaqib (23K-0625)

Blur (any odd size 3 to 25), sharpen and Sobel edge detection on 24-bit RGB images,
written sequentially first and then parallelised with OpenMP in six ways. Every
parallel method produces exactly the same bytes as the sequential one, and the
program checks that on every run. The written report is [REPORT.md](REPORT.md);
the study guide for the viva is [VIVA_GUIDE.md](VIVA_GUIDE.md).

## Run the app (demo laptop, nothing to install)

1. Copy `PDC-Convolution.zip` to the laptop and unzip it anywhere.
2. Double-click `PDC-Convolution.exe`.
   The first time, Windows SmartScreen may say "Windows protected your PC" because
   the exe is not signed: click **More info**, then **Run anyway**.

Tabs: **Filter** (one image, one method, compared with sequential, optional thread
map), **Benchmark** (every method at several thread counts, chart and table),
**Batch** (many small images). Help > Open source folder shows the C code.

## Build from source (development PC)

- Compiler: WinLibs GCC 16.2, x86_64, UCRT, extracted to `C:\mingw64`
  (or pass `-Compiler <path\to\gcc.exe>`). It must be 64-bit: the 12000 x 12000
  runs need about 3 GB in one process.
- Python 3 with `python -m pip install -r requirements.txt`.

```powershell
.\scripts\build.ps1                 # -> pdc_conv.exe (static, no DLLs needed)
pythonw gui\app.pyw                 # the app
python scripts\run_all.py           # every experiment in the report (~20 min, close other programs)
python scripts\run_all.py --quick   # the same steps on tiny images (seconds)
python -m unittest discover -s tests -v
.\scripts\package.ps1               # -> dist\PDC-Convolution.zip for the demo laptop
```

## Command line

```text
pdc_conv gen    --size N|WxH --out img.ppm
pdc_conv filter [--in img.ppm | --size N|WxH] --kernel blur|sharpen|sobel
                [--variant METHOD] [--threads P] [--out out.ppm]
                [--verify] [--csv filter.csv] [--threadmap map.ppm]
pdc_conv bench  [--size N|WxH] [--kernel all|blur|sharpen|sobel] [--methods m1,m2]
                [--threads 1,2,4,8] [--reps 3] [--csv bench.csv]
pdc_conv batch  [--n 2000] [--bsize 1024] [--kernel blur]
                [--threads 1,2,4,8] [--reps 2] [--csv batch.csv]

--blur-size N   odd blur window 3..25 (default 5): filter, bench, batch
--schedule S    static|dynamic|guided, loop schedule of omp_rows (default static)
--verify        filter: also run seq; print speedup and whether the output is identical
--threadmap F   filter: an image showing which thread computed each pixel
--tile HxW      omp_tiled tile size (default 16x2048);  --fuse-bh R: omp_sep_fused rows per block (16)
```

Methods: `seq`, `omp_rows`, `omp_tiled`, `seq_sep`, `omp_sep`, `omp_sep_fused`
(the last three are blur only). Exit codes: 0 ok, 1 error (`error: ...` on stderr),
2 an output differs from the sequential one.

## Folders

```text
src/        the C program: pdc.h, image.c, filters.c, parallel.c (all OpenMP methods), bench.c, main.c
gui/        the app: app.pyw (window), engine.py (runs pdc_conv.exe), charts.py (graphs)
scripts/    build.ps1, run_all.py, plot_results.py, package.ps1
tests/      unittest tests for the program and the app (+ small CSV fixtures)
results/    the final CSVs, logs and plots used in the report
demo/       example outputs and thread maps
docs/       proposal, project list, design and plan
```
