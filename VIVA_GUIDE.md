# Viva guide

Everything a team member needs to explain this project and answer questions about
it. Read it once from top to bottom, then use the questions in section 10 to test
each other. All numbers come from [REPORT.md](REPORT.md) (the final run on the
i7-3770 PC). How to run things is in [HOW_TO_RUN.md](HOW_TO_RUN.md).

**Contents**

1. The project in one minute
2. Background, in plain words
3. The three filters
4. The five methods (the OpenMP part: know this best)
5. Every OpenMP construct we use
6. Measuring fairly
7. Checking that the output is correct
8. The code, file by file
9. The results and what they mean
10. Questions you may be asked, with answers
11. Numbers to remember
12. The demo
13. Before the viva

---

## 1. The project in one minute

**What it does.** It applies three image filters (blur, sharpen, Sobel edge
detection) to large colour images, first with one thread and then with OpenMP on
all the cores of the CPU, and measures how much faster the parallel versions are.

**Say it like this (30 seconds):**

> "Convolution means every output pixel is a weighted sum of the input pixels
> around it. No output pixel depends on another output pixel, so we can give
> different parts of the image to different threads and the answer does not
> change. We wrote a sequential version first. It is our reference for correct
> output and our baseline time. Then we parallelised it with OpenMP in several
> ways (by rows, by tiles, a faster two-pass blur, and whole images per thread)
> and compared them. Every parallel output is checked byte by
> byte against the sequential one. On a 4-core, 8-thread CPU we get about 3x with
> 4 threads and 4.3x with 8 for the blur."

**The key words:**

- **T1**: time of the sequential program (one thread).
- **Tp**: time with p threads.
- **Speedup** = T1 / Tp. 4 threads and 4x faster = perfect.
- **Efficiency** = speedup / p. 1.0 = every thread fully used.

**Sizes from the proposal:** one 12000 x 12000 image (432 MB) and a batch of 3000
images of 1024 x 1024.

## 2. Background, in plain words

**Core, thread, Hyper-Threading.** A core is one real processor inside the CPU. A
thread is one stream of instructions. Our test CPU (Intel i7-3770) has **4 cores**,
and with **Hyper-Threading** each core can run 2 threads, so Windows sees **8
threads**. Two threads on one core share that core. The second thread only uses
the moments when the first is waiting, mostly for memory. So 8 threads on 4 cores
is never 8x.

**OpenMP.** A standard way to make C code parallel by adding `#pragma omp ...`
lines. The compiler (GCC with `-fopenmp`) turns them into code that starts threads
and splits loops between them. Example:

```c
#pragma omp parallel for
for (int y = 0; y < height; y++)
    filter_row(y);
```

Without the pragma, one thread does every row. With it, the rows are divided
between the threads. The loop body is unchanged.

**Data parallelism.** The same work on many independent pieces of data (here,
pixels). This is the easiest kind of problem to parallelise.

**Race condition.** Two threads use the same memory, at least one writes, and the
result depends on who goes first. We have none: threads only **read** the input,
and every output pixel is **written by exactly one thread**.

**Barrier.** A point where every thread waits until all threads have arrived.
Needed when the next step uses results from other threads. A barrier costs time,
because the fast threads wait for the slowest.

**Amdahl's law.** If a fraction f of the work cannot be parallelised, the best
speedup with p threads is S(p) = 1 / (f + (1 - f) / p). Even 10 % serial work
limits 8 threads to about 4.7x.

**Memory-bound vs compute-bound.** A *compute-bound* job spends its time
calculating, so more cores help. A *memory-bound* job spends its time waiting for
data from RAM. All cores share the same RAM, so more threads do not help much.
What decides it is **arithmetic per byte**: how much work is done for each byte
read.

**Cache.** Small, fast memory inside the CPU (here 256 KB L2 per core and 8 MB L3
shared). Data that is used again soon should stay in the cache. A 12000 x 12000
image is 432 MB, far too big for the cache, so it lives in RAM.

## 3. The three filters

Every output pixel is computed from a small **window** of input pixels around it,
separately for red, green and blue.

**Blur (N x N, N odd from 3 to 25; 5x5 is the main case).** A weighted average.
The weights come from Pascal's triangle and look like a bell (Gaussian) curve. For
5x5 each direction uses `1 4 6 4 1`. The 2-D weight is row weight x column weight.

Worked example, 3x3 blur (weights `1 2 1`), one colour:

```text
window          weights         products
10 10 10        1 2 1           10 20 10
10 50 10   x    2 4 2     =     20 200 20      sum = 320
10 10 10        1 2 1           10 20 10       weights add up to 16
```

320 / 16 = 20 (with rounding). The bright centre pixel is softened by its
neighbours. Because the weights always add up to a power of two (16 = 2^4), the
division is just a bit shift (`>> 4`), and everything stays a whole number.

**Sharpen (3x3).** `5 x centre - up - down - left - right`, then limited to 0..255.
Only 5 values per pixel, so very little arithmetic.

**Sobel edge detection (3x3).** First turn each pixel into brightness:
`(77 R + 150 G + 29 B) >> 8`. Then measure how fast brightness changes left to
right (gx) and top to bottom (gy) with weights 1 2 1. Output `|gx| + |gy|` (max
255) in grey: edges come out bright.

**Borders.** Near the edge, part of the window is outside the image. We use
**clamp to edge**: an outside pixel takes the value of the nearest edge pixel.

**Why whole numbers only?** With floating point, adding in a different order can
change the last digit. With integers, every method gives **exactly the same
bytes**, so we can demand equality instead of "close enough".

## 4. The five methods (the OpenMP part: know this best)

All in `src/parallel.c`. Each method writes into a separate output image.

### seq: the sequential reference

One thread filters the whole image. Its output is the **reference** that every
other method is compared with, and its time is **T1**.

### omp_rows: split the rows

```c
#pragma omp parallel for schedule(runtime)
for (int y = 0; y < in->height; y++)
    filter_box(f, in, out, y, y + 1, 0, in->width);   /* one row */
```

Each loop iteration is one row. `schedule(runtime)` means the way rows are handed
out is chosen when the program runs, with `omp_set_schedule`:

- **static**: before the loop, the rows are cut into p equal blocks; thread 0 gets
  the first block, thread 1 the next, and so on. No cost while running. Best when
  every row costs the same (true here).
- **dynamic, 1**: a thread that finishes takes the next single row. Balances uneven
  work, but each hand-out costs a little.
- **guided**: like dynamic, but starts with big pieces and ends with small ones.

The thread map shows the difference: static gives p solid bands, dynamic gives thin
interleaved stripes that change every run.

### omp_tiled: split into tiles

```c
#pragma omp parallel for collapse(2) schedule(static)
for (int ty = 0; ty < tiles_down; ty++)
    for (int tx = 0; tx < tiles_across; tx++)
        filter_box(f, in, out, y0, y1, x0, x1);        /* one tile */
```

The image is cut into tiles of 16 x 2048 pixels. **`collapse(2)`** joins the two
loops into one list of tiles, which is then divided between the threads. Without
it, only the outer loop (rows of tiles) would be divided, giving fewer pieces.
Tiles keep the rows a thread reads close together, so they stay in the cache.

### seq_sep: separable blur, one thread (blur only)

The blur weights are row weight x column weight, so the blur can be done as **two
1-D passes**: first sum along each row, then sum those down each column. That is
**2N** multiplications per pixel instead of **N x N** (10 instead of 25 for 5x5,
30 instead of 225 for 15x15). The row sums are kept in a full-size buffer of 32-bit
numbers. This one is single-threaded, to show the gain from the better algorithm
alone.

### omp_sep: separable blur, parallel (blur only)

```c
#pragma omp parallel
{
    #pragma omp for schedule(static)
    for (int y = 0; y < height; y++)  horizontal_pass(y);   /* pass 1 */
    /* implicit barrier here */
    #pragma omp for schedule(static)
    for (int y = 0; y < height; y++)  vertical_pass(y);     /* pass 2 */
}
```

One team of threads does both passes. Every `omp for` ends with an **implicit
barrier**. Here it is **required**: output row y needs the row sums of rows
y - r ... y + r, and some of those were made by other threads. So no thread may
start pass 2 until everyone has finished pass 1.

**We also tried a version without the barrier** ("fused": each thread did both
passes on blocks of 16 rows in a small private buffer). It was never faster: at
12000 width the buffer was 2.9 to 4.3 MB, too big for the 256 KB L2 cache, and
each block had to recompute extra rows. We removed it to keep the code small. The
report mentions it.

### Batch: many small images (`run_batch` in `src/bench.c`)

Two ways to use p threads on 3000 images of 1024 x 1024:

- **per_row ("threads share each image")**: images one after another, each with
  omp_rows. Every image starts and ends a parallel loop, with a barrier at the end.
- **per_image ("one image per thread")**: whole images handed to free threads:

```c
#pragma omp parallel
{
    Image *mine = &outputs[omp_get_thread_num()];   /* this thread's own output */
    #pragma omp for schedule(dynamic, 1)
    for (int n = 0; n < N; n++) filter_box(&f, &pool[n % pool_size], mine, ...);
}
```

No barrier between images. `dynamic, 1` keeps every thread busy until the end.
(The program makes 8 different test images and uses them in turn, to save memory.)

## 5. Every OpenMP construct we use

| Where | Construct | Why |
|---|---|---|
| `run_rows` | `parallel for schedule(runtime)` | split rows; schedule chosen at run time |
| `run_method` | `omp_set_num_threads`, `omp_set_schedule` | thread count and schedule from the command line |
| `run_tiled` | `parallel for collapse(2) schedule(static)` | two tile loops become one list of tiles |
| `run_separable` | `parallel` + two `omp for` | one team for both passes; the barrier between them is required |
| `run_batch` | `parallel` + `for schedule(dynamic, 1)`, `omp_get_thread_num` | hand whole images to free threads; each thread has its own output |
| `image_count_differences` | `parallel for reduction(+ : differences)` | each thread counts privately, counts added at the end, no race |
| `first_touch`, `image_make_test` | `parallel for schedule(static)` | prepare memory in parallel before timing |
| `warm_up_threads` | empty `parallel num_threads(p)` | create the threads before the timer starts |
| `mark_owner` | `omp_get_thread_num()` | thread map: which thread wrote this pixel |

**No locks, atomics or critical sections**, because nothing is written by two
threads. The only value all threads add to (the difference count) uses
`reduction`, which gives each thread a private counter and adds them up at the end.

## 6. Measuring fairly

All in `src/bench.c` and `src/image.c`.

1. **Best of N repetitions** (3 for the main runs). The fastest run is the one
   least disturbed by other programs.
2. **First touch before timing.** Windows only really gives a page of memory the
   first time it is written. We write every buffer once before the timer starts,
   so this cost is not counted as filter time.
3. **Warm-up thread team.** Creating threads takes time. An empty parallel region
   creates them once, before timing.
4. **Junk in the output (0xA5) before each method.** A broken method that wrote
   nothing would otherwise "pass" by finding the previous method's correct output.
5. **A precise timer.** MinGW's `omp_get_wtime` only ticks every 1 ms, which would
   be 10 % error on a 10 ms run. On Windows we use `QueryPerformanceCounter`
   (100 ns).
6. **Large images** (12000 x 12000), so each run is long enough to measure and the
   data does not fit in the cache.
7. **Other programs closed** during the final run.
8. **64-bit build.** The 12000 x 12000 blur needs about 3 GB; a 32-bit program can
   only use about 2 GB.

## 7. Checking that the output is correct

- `bench` compares every method's output with `seq`, **byte by byte**, at every
  thread count. The `verified` column says `ok` for every row of every final
  result.
- `batch` checks both strategies against the sequential output at every thread
  count, before timing.
- `filter --verify` (and the app's Apply) also runs `seq` and reports
  "identical" or not.
- If any byte differs, the program prints `MISMATCH` and exits with code 2.
- The automatic tests (`tests/`) run every method on awkward sizes (1x1, 2x3, 3x3,
  4x7, 17x9, 128x1, 1x128, 257x131 ...), every blur size from 3 to 25, all three
  schedules, and all-black / all-white images.

## 8. The code, file by file

The C program is `pdc_conv.exe`; it does **all** filtering and timing. The Python
app only runs it and shows what it writes.

**`src/pdc.h`**: the one header. The `Image` type (width, height, pixels as R,G,B
bytes row by row), `Filter` (kind, window size, weights, shift), the method and
schedule names, and every shared function. Also two small helpers, `clamp` and
`clamp_to_byte`.

**`src/image.c`**: memory, timer and image files.
- `xalloc`: memory aligned to 64 bytes (one cache line).
- `first_touch`: write a buffer once, in parallel, before timing.
- `wtime`: the precise timer.
- `image_make_test`: the generated test image (gradients, a circle, a square,
  stripes, a little noise). Same seed = same image.
- `image_count_differences`: compares two images, with `reduction`.
- `image_read_ppm`, `image_write_ppm`: the PPM format (a tiny header, then raw RGB
  bytes; the simplest image format there is).

**`src/filters.c`**: the maths, no OpenMP at all.
- `filter_make`: builds the blur weights from Pascal's triangle.
- `blur_pixel`, `sharpen_pixel`, `sobel_pixel`: one output pixel.
- `gather_window`: for border pixels, copies the window with clamp to edge.
- `blur_box`, `sharpen_box`, `sobel_box`: filter a rectangle (rows y0..y1, columns
  x0..x1). **`filter_box`** picks one of them. Every 2-D method is just
  `filter_box` called on different rectangles: the whole image, one row, or one
  tile.
- `blur_row_horizontal`, `blur_row_vertical`: the two passes of the separable blur.

**`src/parallel.c`**: the five methods (section 4), `run_method` (sets the thread
count and schedule, then calls the method), and the thread-map image.

**`src/bench.c`**: `run_bench` (every method at every thread count, table + CSV)
and `run_batch` (many small images). Each CSV row is written and flushed before the
table line is printed, so the app can show it straight away.

**`src/main.c`**: reads the command line and runs `gen`, `filter`, `bench` or
`batch`.

**`gui/`**: the app. `app.pyw` is the window, `engine.py` runs `pdc_conv.exe` in
the background, `charts.py` draws the graphs (also used for the report's graphs).

**`scripts/`**: `build.ps1` (compile), `run_all.py` (all experiments),
`plot_results.py` (graphs), `package.ps1` (the zip).

**`tests/`**: automatic tests for the program and the app.

## 9. The results and what they mean

### Main result: one 12000 x 12000 image

| | `seq` (T1) | 4 threads | 8 threads |
|---|---:|---:|---:|
| Blur 5x5, Rows / Tiled | 2.62 s | 3.0x | **4.2 to 4.3x** |
| Blur 15x15, Tiled | 16.6 s | 3.2x | 4.3x |
| Sharpen | 0.215 s | 1.6x | 1.6x |
| Sobel | 0.485 s | 2.5 to 2.6x | 2.8 to 2.9x |
| Blur 5x5, `omp_sep` | 2.62 s | **4.6x** | 4.3x |
| Blur 15x15, `omp_sep` | 16.6 s | **11.4x** | 6.9x |

### Why each result is what it is

**Up to 4 threads it scales well, after that much less.** There are 4 real cores.
Threads 5 to 8 are Hyper-Threads sharing those cores, so 4 to 8 threads adds only
about 1.4x.

**Sharpen stops at 1.6x: memory-bound.** Only 5 operations per value, so the
threads mostly wait for RAM (about 10 GB/s moved). More threads cannot make RAM
faster. Blur does far more arithmetic per byte, so it keeps scaling.

**Sobel is in between (2.8x)**: more work per pixel than sharpen, less than blur.

**A bigger blur window scales a little better** (blur sweep, 8000 x 8000): more
arithmetic per byte. The 2-D blur gets about 4.0 to 4.35x at 8 threads for every
size from 3 to 25.

**Separable blur is the fastest, but scales badly.** It does much less work (2N
instead of N x N), so even one thread is 2.1x (5x5) to 5.5x (15x15) faster than
`seq`. That is an **algorithm** gain, not parallelism. But it reads and writes a
big buffer for little arithmetic, so it is memory-bound: it scales only about 2x on
its own and gets **slower from 4 to 8 threads**. Speedups for these methods are
always measured against the same `seq` T1. The CSV also has
`speedup_vs_self_1thread` for scaling on its own.

**Amdahl's law.** Solving the formula for f using the measured speedup:
f = (1/S - 1/p) / (1 - 1/p). Blur 5x5 Tiled gives **f = 0.107 at p = 4** and
**0.125 at p = 8**. Our code has almost no truly serial part, so this f mostly
measures other losses: shared memory bandwidth, the barrier, thread start-up, and
Hyper-Threads not being real cores. It is an upper estimate.

**Schedules (static, dynamic, guided): no real winner.** 4.31x, 4.38x and 4.31x at
8 threads, within 2 %, which is less than the run-to-run variation. Every row costs
the same, so static is already balanced, and dynamic's cost per row (12000
hand-outs) is tiny next to the 0.2 ms each row takes.

**Batch: one image per thread wins** with 4 and 8 threads (3.28x vs 2.71x at 4,
4.30x vs 4.01x at 8). "Threads share each image" has a barrier after every one of
the 3000 images. With 2 threads the two are about equal.

**Every row of every result was verified identical to the sequential output.**

### Reading the graphs (`results/plots/`)

- **speedup.png**: x = threads, y = speedup. The dashed line is ideal. The closer
  to it, the better. Separable lines start above 1 because they do less work.
  A flat dotted line is a one-thread method (`seq_sep`).
- **efficiency.png**: speedup / threads. 1.0 is perfect; it always falls as threads
  grow.
- **time.png**: the plain seconds behind the other graphs.
- **blur_sizes.png**: speedup at 8 threads as the blur window grows.
- **schedules.png**: the three schedules, almost on top of each other.
- **batch.png**: the two batch strategies.
- **demo/threadmap_*.png**: each colour is a thread. Static rows = solid bands,
  dynamic = thin stripes, tiled = blocks.

## 10. Questions you may be asked, with answers

**About parallelism**

1. **Why is convolution easy to parallelise?** Each output pixel depends only on
   the input, never on other output pixels. Any split of the output is correct and
   needs no synchronisation.
2. **Why a separate output image?** If we filtered in place, a thread could read a
   neighbour that another thread had already changed: a race. Separate input and
   output make every read safe.
3. **Do you have race conditions, locks or atomics?** No. The input is only read,
   each output pixel has exactly one writer, and the one shared count uses
   `reduction`.
4. **What does `schedule(static)` do?** Splits the iterations into p equal blocks
   before the loop starts. No run-time cost. Best when every iteration costs the
   same.
5. **Dynamic and guided?** Dynamic: a thread takes the next chunk (here 1 row) when
   it is free; balances uneven work, small cost per chunk. Guided: chunks start big
   and shrink, so fewer hand-outs.
6. **Which schedule was best?** None measurably: within 2 %, less than the noise.
   Rows all cost the same, so static is already balanced. Dynamic would win if rows
   had very different costs.
7. **What is `collapse(2)`?** Joins two nested loops into one list of iterations,
   so all tiles are shared between threads, not just the rows of tiles.
8. **Why tiles?** The rows a thread reads stay close together, so they stay in the
   cache.
9. **What is the implicit barrier in `omp_sep`?** Every `omp for` ends with all
   threads waiting for each other. Pass 2 needs rows other threads made in pass 1,
   so the wait is required for a correct result.
10. **Why does the separable blur scale badly?** Little arithmetic per byte, and it
    writes then reads a full-size buffer (4 bytes per value, 1.7 GB at 12000²)
    through RAM. RAM bandwidth is shared by all cores. Plus the barrier.
11. **Could you avoid that barrier?** We tried a fused version: each thread did
    both passes on blocks of 16 rows in its own small buffer. It was never faster
    than `omp_sep` (0.651 s vs 0.567 s, 5x5, 4 threads): the buffer (2.9 to 4.3 MB)
    did not fit in the 256 KB L2, and blocks recomputed extra rows. So we removed
    it.
12. **What does `reduction(+ : differences)` do?** Each thread gets a private
    counter starting at 0; at the end OpenMP adds them into the shared one. No race,
    no lock.
13. **Why `omp_set_num_threads` and not an environment variable?** The program sets
    it for each run from the `--threads` list, so one run can test 1, 2, 4 and 8.

**About performance**

14. **Why not 8x with 8 threads?** Only 4 real cores; threads 5 to 8 are
    Hyper-Threads. Also shared memory bandwidth, the barrier and thread start-up,
    and other programs using some CPU.
15. **What is Hyper-Threading?** Two hardware threads share one core's execution
    units. It helps when one waits for memory, but it cannot double the computing
    power.
16. **What is Amdahl's law and what serial fraction did you get?**
    S(p) = 1 / (f + (1 - f) / p). From the measured speedup, f = 0.107 at 4 threads
    and 0.125 at 8 (blur 5x5 Tiled). It includes every loss, not only serial code,
    so it is an upper estimate.
17. **Why is sharpen's speedup so low?** Only 5 values per pixel: memory-bound. The
    threads wait for RAM, and RAM does not get faster with more threads.
18. **Why does a bigger blur scale better?** More arithmetic per byte: threads
    spend their time computing, which scales with cores.
19. **Why is the separable version faster even with one thread?** It does less
    work (2N instead of N²). That is a better algorithm, not parallelism, which is
    why we also report its speedup relative to its own one-thread time.
20. **Per-image vs per-row for many small images?** Per-row starts and ends a
    parallel loop, with a barrier, for every image. Per-image gives each thread
    whole images with no barrier between them, so less waiting.
21. **What is false sharing? Can it happen here?** Two threads writing different
    variables in the same 64-byte cache line make the line bounce between cores.
    Here each thread writes whole rows or tiles (thousands of bytes), so at most the
    edges of a block share a line: negligible. Buffers are 64-byte aligned.
22. **What is first touch?** The OS gives a page of physical memory only when it is
    first written. We write every buffer once before timing so this cost is not
    timed.

**About correctness and the code**

23. **How do you know the parallel outputs are correct?** Byte-by-byte comparison
    with `seq` for every method at every thread count, in every run, plus the
    automatic tests on awkward sizes.
24. **Why integers, not floating point?** Different addition orders can change
    floating-point results slightly. With integers every method gives exactly the
    same bytes, so we can demand equality.
25. **How is the blur a Gaussian?** Binomial weights (Pascal's triangle) approach
    the Gaussian bell curve. They are whole numbers that add up to a power of two,
    so the division is an exact shift.
26. **Why can the blur be split into two passes?** Its 2-D weight is row weight x
    column weight, so summing along rows and then down columns gives the same
    result with 2N instead of N² operations.
27. **How do you handle borders?** Clamp to edge: outside pixels take the value of
    the nearest edge pixel. Border pixels copy their window into a small buffer
    first; interior pixels read the image directly (faster).
28. **Why QueryPerformanceCounter?** MinGW's `omp_get_wtime` has 1 ms resolution,
    too coarse for short runs. QPC has 100 ns.
29. **Why a 64-bit compiler?** The 12000 x 12000 blur needs about 3 GB; a 32-bit
    program can use only about 2 GB.
30. **What does the thread map show?** Which thread computed each pixel. It is made
    in a separate, untimed run, so recording it does not slow the timing.
31. **What does the app do?** It only builds a `pdc_conv` command, runs it, and
    shows the images, table and charts from the files it writes. It shows the exact
    command, so every result can be repeated on the command line.
32. **Could you use SIMD or a GPU?** The compiler already uses SIMD (vector
    instructions) for the blur's inner loops at `-O3`. A GPU (CUDA) would be a
    different project; the task was OpenMP on the CPU.
33. **What would you do with more time?** Test on a CPU with more real cores, try
    thread affinity (`proc_bind`), and a cache-blocked 2-D blur.
34. **What changed from the proposal?** 64-bit compiler (memory), QPC timer, a
    variable blur size (3 to 25) to study arithmetic per byte, and extra
    experiments (schedules, thread maps) plus the app for the demo.

## 11. Numbers to remember

- CPU: i7-3770, **4 cores / 8 threads**, 8 MB L3.
- 12000 x 12000 blur 5x5: `seq` **2.62 s**; 8 threads **0.62 s = 4.2x**; 4 threads
  **3.0x**. Best overall: `omp_sep`, 4 threads, **0.57 s = 4.6x**.
- Blur 15x15: `seq` **16.6 s**; `omp_sep` 4 threads **1.45 s = 11.4x** (mostly from
  doing 2N instead of N² work).
- Sharpen **1.6x** (memory-bound). Sobel **2.8 to 2.9x**.
- Amdahl serial fraction: **0.107** (p = 4), **0.125** (p = 8).
- Schedules: 4.31x / 4.38x / 4.31x, so no winner.
- Batch, 3000 x 1024²: one image per thread **4.30x** vs threads share each image
  **4.01x** at 8 threads.
- **Every result verified identical to sequential.**

## 12. The demo (about 5 minutes, with the app)

1. **(30 s)** Open the app. Point at the status bar: CPU name and thread count are
   read from this laptop. "The C program does all the work; this window only runs
   it. The line at the bottom is the exact command."
2. **(1 min) Filter tab.** Generated image 2048, Blur 9, method Sequential, Apply.
   Then Rows with all threads, Apply: show parallel time, sequential time, speedup
   and "Identical to sequential".
3. **(1 min) Thread map** (on by default). Rows + static: solid bands, "each thread
   gets one block of rows". Switch to dynamic, Apply: stripes, "each row goes to
   whichever thread is free". Tiled: blocks.
4. **(1 min 30) Benchmark tab.** Size 4096, Blur 5, threads 1 2 4 8, Run. Rows fill
   the table one by one, and the Verified column says ok. On the chart: Rows and
   Tiled follow the dashed ideal line up to 4 threads, then bend (Hyper-Threading);
   separable lines start above 1 (less work).
5. **(1 min)** Show `results/plots` from the final 12000 x 12000 run and the batch
   graph. If asked about code: Help > Open source folder, open `parallel.c`.

The laptop's numbers will differ from the report (different CPU). That is normal:
explain the shape, not the exact numbers.

## 13. Before the viva

- [ ] Unzip `PDC-Convolution.zip` on the demo laptop and run the demo once (section
      12). Note that laptop's core count and speedups.
- [ ] Laptop on its charger, power mode "Best performance", other programs closed.
- [ ] Each of us can explain all five methods and point to their pragma in
      `parallel.c` (section 4).
- [ ] Each of us can explain why sharpen is 1.6x and blur is 4.3x (memory-bound vs
      compute-bound).
- [ ] Each of us can explain why 8 threads give about 4.3x, not 8x (4 cores,
      Hyper-Threading, Amdahl).
- [ ] Test each other with the questions in section 10, without reading the
      answers.
- [ ] Know where things are: report = `REPORT.md`, graphs = `results/plots/`,
      thread maps = `demo/`, code = `src/`.
