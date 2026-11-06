# Parallel Image Convolution / Filtering (OpenMP)

PDC project — FAST-NUCES Karachi, Fall 2026, BCS-7B
Shayan Ahmed (23K-0551), Rizwan Vadsariya (23K-0005), Aaqib (23K-0625)

Gaussian blur, sharpening and Sobel edge detection on 24-bit RGB images, written
sequentially first and then parallelised with OpenMP, following the proposal's plan.

## Build and run

Requires a **64-bit** MinGW-w64 GCC with OpenMP. The build script uses
`C:\mingw64\bin\gcc.exe` if it exists (WinLibs GCC 16.2, x86_64, UCRT, extracted
there), otherwise `gcc` from PATH; pass `-Compiler <path\to\gcc.exe>` to choose
another. The exe is linked statically, so no runtime DLLs need to sit next to it.
A 32-bit compiler is refused because the 12000x12000 blur needs ~2.1 GB in one
process; `-Allow32Bit` overrides that for small runs only.

Run the complete project natively in Windows PowerShell:

```powershell
python -m pip install -r requirements.txt  # one-time setup
.\scripts\run_windows.ps1
```

The proposal and project-list entry are kept in `docs/`. The written report is
in `REPORT.md`. Generated executable, demo images, benchmark CSVs, logs, and
plots are local outputs and can be regenerated with the command above.

For a shorter local run, use smaller values:

```powershell
.\scripts\run_windows.ps1 -Size 4096 -BatchN 300 -BatchSize 512 -Reps 1
```

Direct use:

```powershell
.\scripts\build_windows.ps1
.\pdc_conv.exe bench  --size 12000 --threads 1,2,4,8 --reps 3
.\pdc_conv.exe batch  --n 3000 --bsize 1024 --threads 1,2,4,8
.\pdc_conv.exe filter --in photo.ppm --kernel sobel --variant omp_tiled --threads 8 --out edges.ppm
.\pdc_conv.exe gen    --size 4096 --out test.ppm
```
Input/output images are binary PPM (P6). Convert with e.g. `convert photo.jpg photo.ppm`.

## What is implemented (mapped to the proposal)

| Proposal step | Where |
|---|---|
| Sequential blur / sharpen / sobel with border handling, baseline T1 | `variant seq` — clamp-to-edge borders via `gather()` |
| `#pragma omp parallel for`, static scheduling, separate output buffer | `omp_rows` (reads `in`, writes `out`, never in place) |
| Pixel-by-pixel verification vs sequential | automatic in `bench`/`batch`; every variant is compared byte-for-byte (`verify ok/MISMATCH`) |
| Cache-friendly tiling | `omp_tiled` — `collapse(2)` over tiles, tune with `--tile HxW` |
| Separable Gaussian (two 1-D passes) | `seq_sep`, `omp_sep` |
| Extra: cache-fused separable blur | `omp_sep_fused` — each thread blurs a block of rows into a small private buffer, so the intermediate never goes to main memory |
| Benchmark with `omp_get_wtime()`, speedup T1/Tp, efficiency S/p | `bench` → `results_bench.csv`, plotted by `scripts/plot_results.py`. Timing goes through `wtime()`: `omp_get_wtime()` on Linux; on Windows `QueryPerformanceCounter` (100 ns), because MinGW's `omp_get_wtime()` only ticks every 1 ms. The resolution in use is printed as `timer res=` in each log header |
| Secondary test: thousands of 1024² images, per-image vs per-row parallelism | `batch` → `results_batch.csv` |

Filters: 5x5 Gaussian (binomial 1-4-6-4-1), 3x3 sharpen, Sobel on luma with |gx|+|gy|.
All arithmetic is integer, so the separable and 2-D blurs are **bit-identical**, which makes
verification an exact comparison with no tolerance.

## Notes for the report

- **T1** is the plain sequential code (`seq`), not the OpenMP code run with 1 thread. The CSV also has
  `speedup_vs_self_1thread` so you can show OpenMP overhead separately.
- `seq_sep` / `omp_sep*` speedups are measured against the sequential **2-D** blur, so they combine the
  algorithmic gain (10 multiplies/pixel/channel instead of 25) with the parallel gain. For a pure scaling
  curve of the separable code, use the `speedup_vs_self_1thread` column.
- Buffers are first-touched in parallel so page faults stay out of the timed regions.
- With `-O3 -march=native` the sequential blur on 12000x12000 takes about 0.5 s on a modest CPU (the compiler
  vectorises the inner loops). That is still measurable, but the proposal says "several seconds"; if you want
  longer runs, use `--size 20000` (needs ~6 GB RAM for blur) or compare against `-O1` as a second baseline.
- Memory for blur at 12000x12000: input + reference + output + 16-bit intermediate ≈ 2.2 GB.
- Tile size matters a lot: on one test machine 256-wide tiles ran ~2x slower than 2048-wide tiles. Sweep
  `--tile` (and `--fuse-bh` for `omp_sep_fused`) on your own machine and report the best.
- Expect speedup to flatten past the number of physical cores (hyper-threads) and when memory bandwidth saturates;
  Sharpen has very little arithmetic per byte, so it is the most bandwidth-bound and scales worst.
  Blur and Sobel scale better. These are good points for the analysis section.

## Status of results

### Completed

- OpenMP blur, sharpening, Sobel, row, tile, separable, fused, and batch variants.
- Native Windows build, correctness checks, benchmark CSVs, logs, and plots.
- Written performance report in `REPORT.md`.
- Fallback performance run at 8000x8000 and 1500 images of 512x512 after the
  exact 12000x12000 run exceeded the 32-bit MinGW address-space limit.

- 64-bit build (WinLibs GCC 16.2, static). A 12000x12000 `bench` now allocates,
  runs and verifies every variant on this machine.

### Still required for exact proposal compliance

- Re-run the full workflow (`.\scripts\run_windows.ps1`) with the 64-bit build to
  replace the 8000x8000 / 512x512 results with the 12000x12000 and 3000 x 1024x1024
  runs, then update `REPORT.md`.
- The blur benchmark needs ~2.1 GB of free RAM. On an 8 GB machine, close browsers
  and other large programs first, or Windows pages to disk and blur timings become
  meaningless (seen as multi-second outliers).
