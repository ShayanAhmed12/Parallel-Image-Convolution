# Parallel Image Convolution / Filtering

**Course:** Parallel and Distributed Computing, FAST-NUCES Karachi, Fall 2026  
**Section:** BCS-7B  
**Members:** Shayan Ahmed (23K-0551), Rizwan Vadsariya (23K-0005), Aaqib (23K-0625)

## Objective

This project applies OpenMP to Gaussian blur, sharpening, and Sobel edge detection on large RGB images. The sequential implementation is the correctness reference. Parallel implementations distribute rows, tiles, or batches of images across threads.

## Implementation

- `seq`: sequential 2-D convolution and timing baseline.
- `omp_rows`: static OpenMP row parallelism.
- `omp_tiled`: cache-friendly tiled OpenMP parallelism.
- `seq_sep` and `omp_sep`: exact integer separable Gaussian blur.
- `omp_sep_fused`: fused separable blur with per-thread scratch space.
- `per_row` and `per_image`: batch-level parallel strategies.
- Every parallel result is compared byte-for-byte with the sequential result.

The implementation uses clamp-to-edge border handling, 24-bit binary PPM input/output, `omp_get_wtime()` timing, speedup, and efficiency calculations.

## Validation

The native Windows runner passed all awkward-size correctness cases, including 1x1, nonsquare images, and 257x131. The full benchmark output contained 36 rows with no verification failures. The batch output contained 9 rows with no verification failures.

## Measured Windows Run

The documented 12000x12000 run was attempted first, but the installed compiler targets 32-bit `mingw32`. Its address-space limit could not allocate the approximately 2.2 GB working set required by the full blur benchmark. The documented smaller-run fallback was used instead.

### Primary image: 8000x8000 RGB, 3 repetitions

| Kernel | Sequential (s) | Best 8-thread variant | Time (s) | Speedup |
|---|---:|---|---:|---:|
| Blur | 0.993 | `omp_rows` | 0.282 | 3.52x |
| Sharpen | 0.539 | `omp_tiled` | 0.145 | 3.72x |
| Sobel | 0.446 | `omp_tiled` | 0.123 | 3.63x |

All rows reported `ok` verification.

### Batch: 1500 images, 512x512 RGB, 2 repetitions

| Strategy | Threads | Time (s) | Speedup | Images/s |
|---|---:|---:|---:|---:|
| `per_row` | 8 | 2.342 | 2.80x | 640.5 |
| `per_image` | 8 | 1.971 | 3.32x | 761.0 |

Batch verification reported `ok` for both strategies and every thread count.

## Analysis

The tiled and row-parallel variants scale well through four threads, then efficiency declines as memory bandwidth and scheduling overhead become more significant. The batch test shows that per-image parallelism is faster than per-row parallelism for independent images because each thread receives a larger contiguous unit of work.

The separable blur is faster than the 2-D blur because it replaces a 25-tap neighborhood with two 5-tap passes while preserving exact integer output. Its speedup combines algorithmic improvement and thread parallelism, so `speedup_vs_self_1thread` should be used when analyzing pure scaling.

## Remaining Requirement

The exact proposal sizes still require a 64-bit OpenMP toolchain and sufficient memory: 12000x12000 for the primary test and approximately 3000 1024x1024 images for the secondary test. The implementation and runner support those values; this Windows environment cannot complete the primary allocation with its current 32-bit MinGW compiler.
