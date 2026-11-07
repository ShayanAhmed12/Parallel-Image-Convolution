/*
 * bench.c -- the "bench" and "batch" commands: timing and verification.
 *
 * bench : one large image. Runs the sequential method to get the reference
 *         output and the time T1, then every parallel method at every thread
 *         count p. Prints and saves time Tp, speedup T1/Tp, efficiency
 *         speedup/p, and whether the output is byte-identical to the reference.
 *
 * batch : many small images. Compares two ways of using the threads:
 *         per_row   -- all threads share the rows of one image, image after image
 *         per_image -- each thread takes whole images and filters them alone
 *
 * Measurement rules used everywhere:
 *   - best of several repetitions (the least disturbed run)
 *   - buffers are first touched before timing (no page faults in the timer)
 *   - the thread team is started once before timing
 *   - the output is overwritten with junk before each method, so a method
 *     that did nothing could never pass verification by accident
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pdc.h"

/* Start the thread team once so its start-up cost is not timed. */
static void warm_up_threads(int threads) {
    /* an empty parallel region: creates the threads, does nothing */
#pragma omp parallel num_threads(threads)
    { }
}

/* Run a method `reps` times and return the fastest time. */
static double best_time(Method m, const Filter *f, const Image *in, Image *out,
                        const RunOptions *opt, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = wtime();
        run_method(m, f, in, out, opt);
        double t = wtime() - t0;
        if (t < best) best = t;
    }
    return best;
}

static int filter_uses_scratch(unsigned filters) {
    return (filters & (1u << FILTER_BLUR)) != 0;
}

/* ------------------------------------------------------------------ */
/* bench                                                               */
/* ------------------------------------------------------------------ */

int run_bench(const BenchOptions *o) {
    const int W = o->width, H = o->height;
    const size_t bytes = (size_t)W * H * 3;
    printf("Image %dx%d (%.1f MB RGB), reps=%d (best-of), machine procs=%d, tile=%dx%d, "
           "fuse-bh=%d, timer res=%.0e s\n",
           W, H, bytes / 1048576.0, o->reps, omp_get_num_procs(), o->tile_h, o->tile_w,
           o->fuse_rows, wtick());

    Image in = image_new(W, H), reference = image_new(W, H), out = image_new(W, H);
    RunOptions opt = run_options_default();
    opt.tile_h = o->tile_h;
    opt.tile_w = o->tile_w;
    opt.fuse_rows = o->fuse_rows;
    if (filter_uses_scratch(o->filters)) {
        opt.scratch = xalloc(bytes * sizeof(uint32_t));
        first_touch(opt.scratch, bytes * sizeof(uint32_t), 0);
    }
    printf("Generating input...\n");
    image_make_test(&in, 12345);

    FILE *csv = fopen(o->csv, "w");
    if (!csv) die("cannot open %s for writing", o->csv);
    fprintf(csv, "kernel,variant,threads,time_s,speedup_vs_seq,efficiency,speedup_vs_self_1thread,verified\n");

    int all_match = 1;
    for (int k = FILTER_BLUR; k <= FILTER_SOBEL; k++) {
        if (!(o->filters & (1u << k))) continue;
        Filter f = filter_make((FilterKind)k, 5);
        printf("\n=== %s ===\n%-14s %4s %10s %9s %7s  %s\n", filter_name(f.kind), "variant", "thr",
               "time(s)", "speedup", "eff", "verify");

        /* the reference: sequential output and time T1 */
        double t1 = best_time(METHOD_SEQ, &f, &in, &reference, &opt, o->reps);
        printf("%-14s %4d %10.4f %9.2f %7.2f  reference (T1)\n", "seq", 1, t1, 1.0, 1.0);
        fprintf(csv, "%s,seq,1,%.6f,1.000,1.000,1.000,ref\n", filter_name(f.kind), t1);

        for (int m = METHOD_SEQ + 1; m < METHOD_COUNT; m++) {
            if (method_blur_only((Method)m) && f.kind != FILTER_BLUR) continue;
            int counts = method_is_parallel((Method)m) ? o->n_threads : 1;
            double own_one_thread_time = 0;
            for (int i = 0; i < counts; i++) {
                opt.threads = method_is_parallel((Method)m) ? o->threads[i] : 1;
                warm_up_threads(opt.threads);
                first_touch(out.pixels, bytes, 0xA5);   /* junk, so stale data cannot pass */
                double t = best_time((Method)m, &f, &in, &out, &opt, o->reps);
                size_t differences = image_count_differences(&out, &reference);
                if (differences) all_match = 0;
                if (opt.threads == 1) own_one_thread_time = t;
                double speedup = t1 / t, efficiency = speedup / opt.threads;
                char self_speedup[32] = "";
                if (own_one_thread_time > 0)
                    snprintf(self_speedup, sizeof self_speedup, "%.3f", own_one_thread_time / t);
                printf("%-14s %4d %10.4f %9.2f %7.2f  %s\n", method_name((Method)m), opt.threads, t,
                       speedup, efficiency, differences ? "MISMATCH" : "ok");
                fprintf(csv, "%s,%s,%d,%.6f,%.3f,%.3f,%s,%s\n", filter_name(f.kind),
                        method_name((Method)m), opt.threads, t, speedup, efficiency, self_speedup,
                        differences ? "FAIL" : "ok");
                fflush(stdout);
            }
        }
    }
    fclose(csv);
    printf("\n%s -- results written to %s\n",
           all_match ? "ALL VARIANTS MATCH THE SEQUENTIAL REFERENCE" : "*** MISMATCH DETECTED ***", o->csv);
    image_free(&in);
    image_free(&reference);
    image_free(&out);
    if (opt.scratch) xfree(opt.scratch);
    return all_match ? 0 : 2;
}

/* ------------------------------------------------------------------ */
/* batch                                                               */
/* ------------------------------------------------------------------ */

#define POOL_MAX 8   /* distinct test images, reused in turn */

int run_batch(const BatchOptions *o) {
    const int N = o->count, S = o->size;
    const int pool_size = N < POOL_MAX ? N : POOL_MAX;
    const size_t bytes = (size_t)S * S * 3;
    const Filter f = filter_make(o->filter, 5);
    printf("Batch: %d images of %dx%d, kernel=%s, pool of %d distinct images cycled, reps=%d, "
           "timer res=%.0e s\n", N, S, S, filter_name(f.kind), pool_size, o->reps, wtick());

    Image pool[POOL_MAX], reference[POOL_MAX], check[POOL_MAX];
    for (int i = 0; i < pool_size; i++) {
        pool[i] = image_new(S, S);
        reference[i] = image_new(S, S);
        check[i] = image_new(S, S);
        image_make_test(&pool[i], 1000u + (uint32_t)i);
    }
    Image shared_out = image_new(S, S);
    RunOptions opt = run_options_default();

    /* verification: both strategies must reproduce the sequential output */
    for (int i = 0; i < pool_size; i++) run_method(METHOD_SEQ, &f, &pool[i], &reference[i], &opt);
    int ok = 1;
    omp_set_num_threads(o->threads[o->n_threads - 1]);
    /* per-image style: one image per thread */
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < pool_size; i++) filter_box(&f, &pool[i], &check[i], 0, S, 0, S);
    for (int i = 0; i < pool_size; i++)
        if (image_count_differences(&check[i], &reference[i])) ok = 0;
    opt.threads = o->threads[o->n_threads - 1];
    for (int i = 0; i < pool_size; i++) {
        first_touch(check[i].pixels, bytes, 0xA5);
        run_method(METHOD_ROWS, &f, &pool[i], &check[i], &opt);
        if (image_count_differences(&check[i], &reference[i])) ok = 0;
    }
    printf("Verification (per-image & per-row vs sequential): %s\n", ok ? "ok" : "MISMATCH");

    FILE *csv = fopen(o->csv, "w");
    if (!csv) die("cannot open %s for writing", o->csv);
    fprintf(csv, "strategy,threads,time_s,speedup_vs_seq,efficiency,images_per_s,verified\n");

    /* T1: every image filtered sequentially */
    double t1 = 1e30;
    opt.threads = 1;
    for (int r = 0; r < o->reps; r++) {
        double t0 = wtime();
        for (int i = 0; i < N; i++) run_method(METHOD_SEQ, &f, &pool[i % pool_size], &shared_out, &opt);
        double t = wtime() - t0;
        if (t < t1) t1 = t;
    }
    printf("\n%-10s %4s %10s %9s %7s %10s\n", "strategy", "thr", "time(s)", "speedup", "eff", "img/s");
    printf("%-10s %4d %10.3f %9.2f %7.2f %10.1f\n", "seq", 1, t1, 1.0, 1.0, N / t1);
    fprintf(csv, "seq,1,%.6f,1.000,1.000,%.2f,%s\n", t1, N / t1, ok ? "ok" : "FAIL");

    for (int i = 0; i < o->n_threads; i++) {
        const int p = o->threads[i];
        opt.threads = p;
        warm_up_threads(p);

        /* strategy A, per_row: images one after another, rows split between threads */
        double t_rows = 1e30;
        for (int r = 0; r < o->reps; r++) {
            double t0 = wtime();
            for (int n = 0; n < N; n++) run_method(METHOD_ROWS, &f, &pool[n % pool_size], &shared_out, &opt);
            double t = wtime() - t0;
            if (t < t_rows) t_rows = t;
        }

        /* strategy B, per_image: each thread filters whole images on its own */
        double t_images = 1e30;
        omp_set_num_threads(p);
        for (int r = 0; r < o->reps; r++) {
            double t0 = wtime();
            /* each thread gets its own output image */
#pragma omp parallel
            {
                Image mine = image_new(S, S);
                /* hand out images one at a time to whichever thread is free */
#pragma omp for schedule(dynamic, 1)
                for (int n = 0; n < N; n++) filter_box(&f, &pool[n % pool_size], &mine, 0, S, 0, S);
                image_free(&mine);
            }
            double t = wtime() - t0;
            if (t < t_images) t_images = t;
        }

        printf("%-10s %4d %10.3f %9.2f %7.2f %10.1f\n", "per_row", p, t_rows, t1 / t_rows,
               t1 / t_rows / p, N / t_rows);
        printf("%-10s %4d %10.3f %9.2f %7.2f %10.1f\n", "per_image", p, t_images, t1 / t_images,
               t1 / t_images / p, N / t_images);
        fprintf(csv, "per_row,%d,%.6f,%.3f,%.3f,%.2f,%s\n", p, t_rows, t1 / t_rows, t1 / t_rows / p,
                N / t_rows, ok ? "ok" : "FAIL");
        fprintf(csv, "per_image,%d,%.6f,%.3f,%.3f,%.2f,%s\n", p, t_images, t1 / t_images,
                t1 / t_images / p, N / t_images, ok ? "ok" : "FAIL");
        fflush(stdout);
    }
    fclose(csv);
    printf("\nresults written to %s\n", o->csv);
    for (int i = 0; i < pool_size; i++) {
        image_free(&pool[i]);
        image_free(&reference[i]);
        image_free(&check[i]);
    }
    image_free(&shared_out);
    return ok ? 0 : 2;
}
