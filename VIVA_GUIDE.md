# Viva guide

A study guide for all three of us. Read it top to bottom once, then use the
questions at the end to test each other. Numbers come from [REPORT.md](REPORT.md).

## 1. The project in one minute

Convolution: every output pixel is a weighted sum of the input pixels in a small
window around it (3x3 for sharpen and Sobel, N x N for blur). Output pixels never
depend on each other, so the image can be split between threads in any way and the
result is the same. That makes it a clean **data-parallel** problem.

We wrote it sequentially first (`seq`), which is both the **correctness reference**
and the time **T1**. Then five parallel or faster versions, each a different way to
split the work or use the cache. For every run the program compares the output of
each method with the sequential output **byte by byte**, and reports
speedup = T1 / Tp and efficiency = speedup / p.

Everything uses whole-number arithmetic, so the outputs are exactly equal, not
"close". If a single byte differs, the program says MISMATCH and exits with code 2.

## 2. The files

The C program is `pdc_conv.exe`. The app (Python) only runs it and shows what it
writes; it never filters or times anything.

### src/pdc.h
The one header: every type and function the `.c` files share, grouped by file.
`Image` (width, height, pixels as R,G,B bytes row by row), `Filter` (kind, window
size, radius, blur weights, shift), `Method`, `Schedule`, `RunOptions`,
`BenchOptions`, `BatchOptions`.

### src/image.c: memory, timer, images
- `die`: prints `error: ...` and exits with code 1.
- `xalloc` / `xfree`: memory aligned to 64 bytes (one cache line), so no buffer
  starts in the middle of a cache line. Dies with "out of memory" if it fails.
- `first_touch`: writes a buffer once, in parallel (`omp parallel for`), before any
  timing. Windows only really gives a page of memory the first time it is written;
  doing that early keeps page faults out of the timed region.
- `wtime` / `wtick`: the timer. On Windows it reads `QueryPerformanceCounter`
  (100 ns resolution). MinGW's `omp_get_wtime` only ticks every 1 ms, which is too
  coarse for runs of 0.01 to 0.1 s.
- `image_new`, `image_free`: allocate (and first-touch) / free an image.
- `image_make_test`: the generated test image (colour gradients, a circle, a white
  square, stripes, a little noise), filled in parallel. Same seed = same image.
- `image_count_differences`: compares two images; `reduction(+ : differences)` adds
  up each thread's count at the end.
- `image_read_ppm`, `image_write_ppm`: binary PPM (P6) files, the simplest RGB format.

### src/filters.c: the maths, no parallelism
- `filter_make`: builds a filter. Blur weights are row N-1 of Pascal's triangle
  (5x5: 1 4 6 4 1), which approximates a Gaussian. One row sums to 2^(N-1), the
  whole window to 2^(2(N-1)), so dividing by the total is a right shift by
  `shift = 2(N-1)`. Rounding: add half (2^(shift-1)) before shifting.
  Sizes must be odd, 3 to 25.
- `blur_pixel`, `sharpen_pixel`, `sobel_pixel`: one output pixel from a window.
  Sharpen is `5*centre - up - down - left - right`, clamped to 0..255. Sobel turns
  pixels into brightness (`luma`), computes horizontal and vertical gradients gx,
  gy and outputs |gx| + |gy| in grey.
- `gather_window`: near the border part of the window is outside the image. It
  copies the window into a small buffer with outside pixels replaced by the nearest
  edge pixel ("clamp to edge"), so the same pixel functions work unchanged.
- `interior_columns`: in one row, which columns have their whole window inside the
  image (fast path) and which are border pixels (gathered window).
- `blur_box`, `sharpen_box`, `sobel_box`: filter a rectangle of output pixels
  (rows y0..y1, columns x0..x1). Each row: left border, interior, right border.
  Settings are copied into local variables first, because writing through a byte
  pointer would otherwise force the compiler to re-read them after every byte.
- `weighted_row_sums`, `blur_interior`: the fast blur for interior pixels. It works
  on 1024 bytes of a row at a time, loop order "each weight, then each byte", so the
  inner loop runs over consecutive bytes and the compiler can use SIMD.
- `filter_box`: the one entry point: "filter this rectangle", with a `switch` on
  the filter kind. Every 2-D method is just `filter_box` on different rectangles.
- `blur_row_horizontal`, `blur_row_vertical`: the two 1-D passes of the separable
  blur. Horizontal: weighted sums along one row into 32-bit numbers. Vertical: for
  one output row, weighted sum of N rows of those sums, then round and shift.

### src/parallel.c: the six methods (the OpenMP part, know this best)
- `run_options_default`: 1 thread, static schedule, 16 x 2048 tiles, 16 rows per
  fused block.
- `mark_owner`: only for the thread map. Writes `omp_get_thread_num()` into one byte
  per pixel of the rectangle a thread just wrote. Does nothing in timed runs
  (`owner` is NULL).
- `run_sequential` (**seq**): `filter_box` on the whole image. One thread.
- `run_rows` (**omp_rows**): `#pragma omp parallel for schedule(runtime)` over the
  rows; each iteration filters one row. `runtime` means the schedule is chosen in
  `run_method` by `omp_set_schedule`: static (default), dynamic (chunk 1) or guided.
- `run_tiled` (**omp_tiled**): two loops over tiles (rows of tiles, columns of
  tiles) with `#pragma omp parallel for collapse(2) schedule(static)`. collapse(2)
  merges the two loops into one list of tiles that is split between threads.
- `vertical_pass_row`: builds the list of the N rows of horizontal sums needed for
  output row y (clamped at the top and bottom) and calls `blur_row_vertical`.
- `run_separable_sequential` (**seq_sep**): horizontal pass over all rows into a
  full-size buffer of sums, then the vertical pass. One thread. Shows the algorithm
  gain alone: 2N multiplications per pixel instead of N x N.
- `run_separable` (**omp_sep**): one `#pragma omp parallel` region with two
  `#pragma omp for` loops: pass 1 then pass 2. The end of the first loop is an
  **implicit barrier**: pass 2 needs rows of sums that other threads produced, so
  nobody may start pass 2 until everyone has finished pass 1.
- `run_separable_fused` (**omp_sep_fused**): each thread allocates a small private
  buffer, then takes blocks of 16 output rows (`omp for schedule(static)`). For a
  block it does the horizontal pass on the 16 + 2r rows it needs, straight into its
  buffer, then the vertical pass from that buffer. No barrier is needed, and the
  buffer can stay in the cache if the image is not too wide. Cost: the 2r rows at
  block edges are computed twice. (At 12000 width the buffer is 2.9 MB, too big for
  the cache, so fusion did not beat `omp_sep` there; see question 11.)
- `run_method`: sets the thread count (`omp_set_num_threads`), sets the schedule for
  Rows (`omp_set_schedule`), and calls the method with a `switch`.
- `method_name`, `method_from_name`, `method_blur_only`, `method_is_parallel`,
  `schedule_name`, `schedule_from_name`: names for the command line and CSVs.
- `THREAD_COLORS`, `threadmap_write_ppm`: turn the owner bytes into a colour image
  (thread 0 blue, 1 orange, 2 green, 3 red, ...).

### src/bench.c: measuring
- `warm_up_threads`: an empty `#pragma omp parallel` region so the thread team is
  created before timing.
- `best_time`: run a method several times, keep the fastest.
- `run_bench`: sequential reference first (output + T1), then every method at every
  thread count. Before each method the output is filled with junk (0xA5), so a
  method that wrote nothing cannot pass. Prints a table and writes the CSV.
- `run_batch`: many small images. `per_row`: images one after another, each with
  the Rows method (threads share each image). `per_image`: `#pragma omp parallel`
  with `#pragma omp for schedule(dynamic, 1)` over the images, each thread filtering
  whole images into its own output image (allocated before the timer).

### src/main.c: the command line
Reads the options (`parse_thread_list`, `parse_size`, `parse_blur_size`,
`parse_filter_mask`, `parse_method_mask`) and runs `gen`, `filter`, `bench` or
`batch`. `command_filter` runs one method once; with `--verify` it also runs `seq`
and compares; with `--threadmap` it does a second, untimed run that records owners.
stdout is unbuffered so the app sees each line as it is printed.

### gui/ and scripts/
- `gui/app.pyw`: the window (Filter, Benchmark, Batch tabs). Builds a `pdc_conv`
  command line, runs it, shows the images, table and chart.
- `gui/engine.py`: runs the exe in the background (`EngineRun`), method names and
  help texts, CPU/RAM detection, image conversion to PPM.
- `gui/charts.py`: the graphs, shared by the app and the report.
- `scripts/build.ps1` (compile), `run_all.py` (all experiments), `plot_results.py`
  (graphs), `package.ps1` (the zip for the laptop).
- `tests/`: automatic checks that every method matches `seq` on awkward sizes
  (1x1, 2x3, 128x1, 257x131 ...), every blur size, every schedule, plus the app.

## 3. Every OpenMP construct we use, and why

| Where | Construct | Why |
|---|---|---|
| `run_rows` | `parallel for schedule(runtime)` | split rows; schedule chosen at run time to compare static / dynamic / guided |
| `run_method` | `omp_set_num_threads`, `omp_set_schedule` | thread count and Rows schedule from the command line |
| `run_tiled` | `parallel for collapse(2) schedule(static)` | two tile loops become one list of tiles, enough pieces for all threads |
| `run_separable` | `parallel` + two `for` | one team for both passes; implicit barrier between them is required |
| `run_separable_fused` | `parallel` + `for`, private buffer per thread | each thread keeps its intermediate in its own cache; no barrier |
| `run_batch` | `parallel for schedule(dynamic, 1)`; `parallel` + `for schedule(dynamic, 1)` | hand whole images to whichever thread is free |
| `image_count_differences` | `parallel for reduction(+ : differences)` | each thread counts privately; counts added at the end, no race |
| `first_touch`, `image_make_test` | `parallel for schedule(static)` | prepare buffers in parallel before timing |
| `warm_up_threads` | empty `parallel num_threads(p)` | create the threads before the timer starts |
| `mark_owner` | `omp_get_thread_num()` | thread map: which thread wrote this pixel |

Why there are no locks, atomics or critical sections: threads only **read** the
input image, and each output pixel is **written by exactly one thread**. Nothing is
shared for writing, so there is nothing to protect.

## 4. Measurement tricks (and why each matters)

1. **Best of N repetitions**: the fastest run is the one least disturbed by other
   programs.
2. **First touch before timing**: page faults (the OS giving memory for the first
   time) would otherwise be counted as filter time.
3. **Warm-up thread team**: creating threads takes time; it is done once, outside.
4. **Junk output before each method (0xA5)**: a broken method that writes nothing
   cannot pass verification by finding the previous method's output.
5. **Byte-by-byte verification** after every method, at every thread count.
6. **Precise timer (QPC)**: 1 ms resolution would be 10 % error on a 10 ms run.
7. **Large images** (12000 x 12000 = 432 MB per image) so each run is long enough
   to measure and the data does not fit in the cache, as in the proposal.
8. **Other programs closed** for the final run: timings need the whole CPU.

## 5. Reading the graphs (results/plots/)

- **speedup.png**: x = threads, y = T1 / Tp. Dashed line = ideal (p threads, p
  times faster). The closer a line is to the dashed one, the better it scales.
  Separable methods start above 1 at one thread because they do less work, not
  because of parallelism. The dotted flat line is `seq_sep` (one thread).
- **efficiency.png**: speedup / p. 1.0 = perfect use of every thread. It always
  falls as p grows (overheads, memory limits).
- **blur_sizes.png**: speedup with the most threads as the blur window grows. A
  bigger window = more arithmetic per byte loaded, so the work becomes
  compute-bound and scales better.
- **schedules.png**: Rows with static, dynamic and guided. Rows all cost the same,
  so static is already balanced; dynamic pays a small cost per row handed out.
- **batch.png**: many small images. Compare "threads share each image" with "one
  image per thread".
- **Thread maps** (demo/threadmap_*.png): each colour is a thread. Static rows =
  4 solid bands. Dynamic = thin interleaved stripes (each row goes to whichever
  thread is free, different every run). Tiled = tiles coloured by thread.

## 6. Likely questions, with model answers

1. **Why is convolution easy to parallelise?** Each output pixel depends only on
   the input, never on other output pixels. So any split of the output is correct
   and needs no synchronisation.
2. **Why do you write to a separate output image?** If we filtered in place, a
   thread could read a neighbour pixel that another thread has already overwritten,
   and the result would depend on timing (a race). Separate input and output make
   every read safe.
3. **Do you have any race conditions? Any locks?** No. The input is only read; every
   output pixel has exactly one writer. The only combined value, the difference
   count, uses `reduction`.
4. **What does `schedule(static)` do?** Divides the iterations into p equal
   consecutive blocks before the loop starts; thread 0 gets the first block, and so
   on. No cost at run time; best when every iteration costs the same.
5. **And dynamic? guided?** Dynamic: threads take the next chunk (here 1 row) when
   they finish one, which balances uneven work but costs a little per chunk.
   Guided: like dynamic, but chunks start large and shrink, so fewer hand-outs.
6. **Which schedule was best here, and why?** They were within 2 % (4.31x, 4.38x,
   4.31x at 8 threads). All rows cost about the same, so static is already balanced;
   dynamic's cost per hand-out (12000 rows) is tiny next to 0.2 ms of work per row.
   Dynamic would win if rows had very different costs.
7. **What is `collapse(2)`?** It merges two nested loops into one iteration space,
   so the tiles of the whole image are split between threads instead of only the
   rows of tiles. Gives more, smaller pieces to share.
8. **Why tiles at all?** A tile keeps the rows a thread is reading close together,
   so the window rows stay in the cache while it moves across the tile.
9. **What is the implicit barrier in `omp_sep`?** Every `omp for` ends with a
   barrier: all threads wait until all are done. Pass 2 (vertical) needs rows that
   other threads produced in pass 1, so this wait is required for correctness.
10. **Why does the plain separable blur scale badly?** It does little arithmetic per
    byte and writes then reads a full-size intermediate (4 bytes per channel, 1.7 GB
    at 12000²) through main memory. It is limited by memory bandwidth, which all
    cores share, plus the barrier.
11. **Did fusion help?** Only when the buffer fits in the cache. The idea: each
    thread uses a small buffer (16 + 2r rows), so the intermediate need not go to
    main memory and there is no barrier. At 8000 x 8000 it scaled well (8.7x to
    12.7x over `seq` for blur 9 and up). At 12000 x 12000 a row of sums is 144 KB,
    so the buffer is 2.9 MB (5x5) to 4.3 MB (15x15): bigger than L2 (256 KB), and 4
    of them exceed the 8 MB L3. Plus the 2r recomputed rows (+25 % work for 5x5,
    +88 % for 15x15). So there it was not faster than `omp_sep`. The fix would be
    column strips (narrow blocks); we kept the simple version and report it.
12. **Why not 8x speedup with 8 threads?** The test CPU (i7-3770) has 4 physical
    cores; threads 5-8 are hyper-threads sharing those cores. Also memory bandwidth
    is shared, there is a serial part (starting threads, borders, the barrier), and
    the OS and other programs use some CPU.
13. **What is hyper-threading and what does it give you here?** Two hardware threads
    per core share its execution units. It helps when one thread waits for memory;
    it cannot double compute power, so 4 -> 8 threads gives much less than 2x.
14. **What is Amdahl's law? What serial fraction did you get?** S(p) = 1 / (f +
    (1 - f) / p). Solving for f from the measured 8-thread speedup:
    f = (1/S - 1/p) / (1 - 1/p). The report shows the calculation. It includes real
    serial work and every other loss, so it is an upper estimate.
15. **What is false sharing? Can it happen here?** Two threads writing different
    variables in the same 64-byte cache line make the line bounce between cores.
    Here each thread writes whole rows or tiles (thousands of bytes), so only the
    first and last line of a thread's block can be shared with a neighbour, which is
    negligible. Buffers are 64-byte aligned.
16. **What is first touch and why do you do it?** The OS assigns physical memory
    when a page is first written. Doing it in advance, in parallel, keeps page faults
    out of the timing (and on multi-socket machines puts pages near their threads).
17. **How do you know the parallel outputs are correct?** Every method's output is
    compared byte by byte with the sequential one in every run (Verified column), and
    the tests check awkward sizes (1x1, 2x3, 128x1, 257x131 ...), all blur sizes and
    all schedules.
18. **Why integers instead of floating point?** Floating-point addition in a
    different order can give slightly different results. With integers, every method
    gives exactly the same bytes, so we can demand equality, not "close enough".
19. **How is the blur a Gaussian?** Binomial weights (rows of Pascal's triangle)
    approach a Gaussian curve. They are whole numbers whose total is a power of two,
    so the division is an exact shift.
20. **Why can blur be separated into two passes?** Its 2-D weights are the product
    of a row weight and a column weight (w[i] x w[j]). So you can sum along rows
    first, then along columns: 2N operations per pixel instead of N².
21. **Why is the separable version faster even with one thread?** Fewer operations
    (2N vs N²). That is an algorithm gain, not parallelism, so for scaling we also
    report speedup relative to the method's own 1-thread time.
22. **How do you handle borders?** Clamp to edge: a pixel outside the image takes
    the value of the nearest edge pixel. Border pixels gather their window into a
    small buffer; interior pixels read the image directly.
23. **Why the QPC timer?** MinGW's `omp_get_wtime` has 1 ms resolution, too coarse
    for 10 to 100 ms runs. `QueryPerformanceCounter` has 100 ns.
24. **Why a 64-bit compiler?** A 12000 x 12000 blur benchmark needs about 3 GB in one
    process; a 32-bit program can only address about 2 GB.
25. **Why does a bigger blur window scale better?** More multiplications per byte
    read: the threads spend their time computing (which scales with cores) instead
    of waiting for memory (which does not).
26. **Per-image vs per-row for many small images?** Per-row starts and ends a
    parallel loop (with a barrier) for every image and splits a small image into very
    small pieces. Per-image gives each thread whole images with no barrier between
    them, so it wastes less time waiting.
27. **What does the thread map show?** Which thread computed each pixel. Static
    rows: p solid bands. Dynamic: rows interleaved, different every run. It is made
    in a separate untimed run so recording owners does not affect the timing.
28. **Why is sharpen's speedup lower than blur's?** Sharpen does very little work per
    pixel (5 values), so it is limited by memory bandwidth, not by the cores.
29. **Could you use `#pragma omp simd` or GPU?** The compiler already vectorises the
    blur inner loops at -O3. GPU (CUDA/OpenCL) would be another project; the task
    was OpenMP on the CPU.
30. **What would you do with more time?** Measure on a CPU with more physical cores;
    try `proc_bind`/thread affinity; add a cache-blocked 2-D blur.

## 7. Numbers to remember (final run, i7-3770, 4 cores / 8 threads)

- 12000 x 12000 blur 5x5: `seq` 2.62 s; Rows / Tiled 8 threads 0.62 s (**4.2x**);
  4 threads **3.0x**; best overall `omp_sep` 4 threads 0.57 s (**4.6x**).
- Blur 15x15: `seq` 16.6 s; Tiled 8 threads 3.85 s (4.3x); `omp_sep` 4 threads
  1.45 s (**11.4x**, mostly from doing 2N instead of N² work).
- Sharpen: only **1.6x** (memory-bound, about 10 GB/s). Sobel: **2.9x**.
- Amdahl serial fraction (blur 5x5 Tiled): **0.107** at p = 4, **0.125** at p = 8.
- Schedules: static 4.31x, dynamic 4.38x, guided 4.31x: within 2 %.
- Batch (3000 x 1024²): one image per thread 4.30x vs threads share each image
  4.01x at 8 threads; 3.28x vs 2.71x at 4.
- Every result row verified identical to sequential.

## 8. Five-minute demo (with the app)

1. (30 s) Open the app. Point at the status bar: CPU name and thread count are read
   from the laptop. "The C program does all the work; this window only runs it."
2. (1 min) **Filter** tab, generated 2048 image, Blur 9, Sequential, Apply. Then
   Rows, all threads, Apply: show parallel time, sequential time, speedup and
   "Identical to sequential".
3. (1 min) Tick **Show thread map**. Rows + static: 4 or 8 solid bands. Switch to
   dynamic: interleaved stripes, "each row goes to whichever thread is free".
   Tiled: tiles by thread.
4. (1 min 30) **Benchmark** tab, 4096, Blur 5, threads 1 2 4 8, reps 2, Run. While
   it runs, explain the log (every line says ok). Show the speedup chart: separable
   lines start above 1 (less work), Rows and Tiled follow the ideal line up to 4
   threads and bend after (Hyper-Threading), ideal dashed line.
5. (1 min) Show `results/plots` from the final run (12000 x 12000) and the batch
   graph; Help > Open source folder, open `parallel.c` if asked.

Timings on the demo laptop (fill in after the laptop test): ...
