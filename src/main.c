/*
 * main.c -- reads the command line and runs one command.
 *
 *   pdc_conv gen    --size N|WxH --out img.ppm
 *   pdc_conv filter [--in img.ppm | --size N|WxH] --kernel blur|sharpen|sobel
 *                   [--variant METHOD] [--threads P] [--out out.ppm]
 *   pdc_conv bench  [--size N|WxH] [--kernel all|blur|sharpen|sobel]
 *                   [--threads 1,2,4,8] [--reps 3] [--csv results_bench.csv]
 *   pdc_conv batch  [--n 2000] [--bsize 1024] [--kernel blur]
 *                   [--threads 1,2,4,8] [--reps 2] [--csv results_batch.csv]
 *   --blur-size N (odd, 3..25, default 5) works with filter, bench and batch
 *   tuning:         --tile HxW (default 16x2048)   --fuse-bh R (default 16)
 *
 * No parallel work happens here; see parallel.c and bench.c.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pdc.h"

static void usage(void) {
    puts("pdc_conv -- parallel image convolution (OpenMP)\n\n"
         "  pdc_conv gen    --size N|WxH --out img.ppm\n"
         "  pdc_conv filter [--in img.ppm | --size N|WxH] --kernel blur|sharpen|sobel\n"
         "                  [--variant seq|omp_rows|omp_tiled|seq_sep|omp_sep|omp_sep_fused]\n"
         "                  [--threads P] [--out out.ppm]\n"
         "  pdc_conv bench  [--size N|WxH] [--kernel all|blur|sharpen|sobel]\n"
         "                  [--threads 1,2,4,8] [--reps 3] [--csv results_bench.csv]\n"
         "  pdc_conv batch  [--n 2000] [--bsize 1024] [--kernel blur]\n"
         "                  [--threads 1,2,4,8] [--reps 2] [--csv results_batch.csv]\n\n"
         "  --blur-size N   odd blur window 3..25 (default 5) for filter, bench, batch\n"
         "  tuning:         --tile HxW (omp_tiled, default 16x2048)  --fuse-bh R (default 16)");
}

/* "1,2,4,8" -> {1,2,4,8}; returns how many numbers were read */
static int parse_thread_list(const char *text, int *out, int max) {
    char buf[256];
    strncpy(buf, text, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    int n = 0;
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ",")) {
        int v = atoi(t);
        if (v < 1) die("thread counts must be 1 or more");
        out[n++] = v;
    }
    if (n == 0) die("empty thread list");
    return n;
}

/* "15" -> 15; anything that is not an odd number from 3 to 25 is an error */
static int parse_blur_size(const char *text) {
    char *end;
    long n = strtol(text, &end, 10);
    if (end == text || *end || n < BLUR_SIZE_MIN || n > BLUR_SIZE_MAX || n % 2 == 0)
        die("blur size must be an odd number from %d to %d", BLUR_SIZE_MIN, BLUR_SIZE_MAX);
    return (int)n;
}

/* "4096" -> 4096x4096, "800x600" -> 800x600 */
static void parse_size(const char *text, int *width, int *height) {
    int a = 0, b = 0;
    if (sscanf(text, "%dx%d", &a, &b) == 2) { *width = a; *height = b; }
    else { a = atoi(text); *width = *height = a; }
    if (*width < 1 || *height < 1) die("bad --size %s", text);
}

/* "all" -> every filter; otherwise one filter name. Returns a bit mask. */
static unsigned parse_filter_mask(const char *text) {
    if (!strcmp(text, "all")) return (1u << FILTER_BLUR) | (1u << FILTER_SHARPEN) | (1u << FILTER_SOBEL);
    FilterKind k;
    if (!filter_from_name(text, &k)) die("unknown kernel %s (blur|sharpen|sobel|all)", text);
    return 1u << k;
}

/* Default thread counts: 1, 2, 4, ... up to the number of logical CPUs. */
static int default_thread_list(int *list) {
    int procs = omp_get_num_procs(), n = 0;
    for (int t = 1; t <= procs && n < MAX_THREAD_COUNTS - 1; t *= 2) list[n++] = t;
    if (list[n - 1] != procs) list[n++] = procs;
    return n;
}

/* filter: apply one method once and report the time */
static int command_filter(const char *in_path, int width, int height, FilterKind kind, int blur_size,
                          Method method, const RunOptions *base, const char *out_path) {
    Filter f = filter_make(kind, blur_size);
    if (method_blur_only(method) && kind != FILTER_BLUR)
        die("method %s only supports --kernel blur", method_name(method));

    Image in;
    if (in_path) in = image_read_ppm(in_path);
    else { in = image_new(width, height); image_make_test(&in, 12345); }
    Image out = image_new(in.width, in.height);

    RunOptions opt = *base;
    size_t bytes = (size_t)in.width * in.height * 3;
    if (method == METHOD_SEP_SEQ || method == METHOD_SEP) {
        opt.scratch = xalloc(bytes * sizeof(uint32_t));
        first_touch(opt.scratch, bytes * sizeof(uint32_t), 0);
    }

    double t0 = wtime();
    run_method(method, &f, &in, &out, &opt);
    double t = wtime() - t0;
    int threads = method_is_parallel(method) ? opt.threads : 1;
    printf("%s / %s on %dx%d with %d thread(s): %.4f s\n", filter_name(kind), method_name(method),
           in.width, in.height, threads, t);
    if (out_path) { image_write_ppm(out_path, &out); printf("wrote %s\n", out_path); }

    image_free(&in);
    image_free(&out);
    if (opt.scratch) xfree(opt.scratch);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *command = argv[1];

    int width = 4096, height = 4096, reps = -1, count = 2000, batch_size = 1024, blur_size = 5;
    int threads[MAX_THREAD_COUNTS], n_threads = 0;
    const char *kernel = NULL, *in_path = NULL, *out_path = NULL, *csv = NULL;
    const char *method_text = "omp_rows";
    RunOptions opt = run_options_default();

    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!value) { fprintf(stderr, "missing value for %s\n", arg); usage(); return 1; }
        i++;
        if      (!strcmp(arg, "--size"))    parse_size(value, &width, &height);
        else if (!strcmp(arg, "--kernel"))  kernel = value;
        else if (!strcmp(arg, "--blur-size")) blur_size = parse_blur_size(value);
        else if (!strcmp(arg, "--threads")) n_threads = parse_thread_list(value, threads, MAX_THREAD_COUNTS);
        else if (!strcmp(arg, "--reps"))    reps = atoi(value);
        else if (!strcmp(arg, "--csv"))     csv = value;
        else if (!strcmp(arg, "--in"))      in_path = value;
        else if (!strcmp(arg, "--out"))     out_path = value;
        else if (!strcmp(arg, "--variant")) method_text = value;
        else if (!strcmp(arg, "--n"))       count = atoi(value);
        else if (!strcmp(arg, "--bsize"))   batch_size = atoi(value);
        else if (!strcmp(arg, "--tile")) {
            if (sscanf(value, "%dx%d", &opt.tile_h, &opt.tile_w) != 2 || opt.tile_h < 1 || opt.tile_w < 1)
                die("bad --tile %s (use HxW)", value);
        }
        else if (!strcmp(arg, "--fuse-bh")) {
            opt.fuse_rows = atoi(value);
            if (opt.fuse_rows < 1) die("bad --fuse-bh %s", value);
        }
        else { fprintf(stderr, "unknown option: %s\n", arg); usage(); return 1; }
    }
    if (n_threads == 0) n_threads = default_thread_list(threads);

    if (!strcmp(command, "gen")) {
        if (!out_path) die("gen needs --out");
        Image img = image_new(width, height);
        image_make_test(&img, 12345);
        image_write_ppm(out_path, &img);
        printf("wrote %s (%dx%d)\n", out_path, width, height);
        image_free(&img);
        return 0;
    }

    if (!strcmp(command, "filter")) {
        FilterKind kind;
        Method method;
        if (!filter_from_name(kernel ? kernel : "blur", &kind)) die("filter needs one kernel: blur|sharpen|sobel");
        if (!method_from_name(method_text, &method)) die("unknown --variant %s", method_text);
        opt.threads = threads[0];
        return command_filter(in_path, width, height, kind, blur_size, method, &opt, out_path);
    }

    if (!strcmp(command, "bench")) {
        BenchOptions b;
        memset(&b, 0, sizeof b);
        b.width = width;
        b.height = height;
        b.filters = parse_filter_mask(kernel ? kernel : "all");
        b.blur_size = blur_size;
        memcpy(b.threads, threads, sizeof threads);
        b.n_threads = n_threads;
        b.reps = reps > 0 ? reps : 3;
        b.tile_h = opt.tile_h;
        b.tile_w = opt.tile_w;
        b.fuse_rows = opt.fuse_rows;
        b.csv = csv ? csv : "results_bench.csv";
        return run_bench(&b);
    }

    if (!strcmp(command, "batch")) {
        BatchOptions b;
        memset(&b, 0, sizeof b);
        if (!filter_from_name(kernel ? kernel : "blur", &b.filter)) die("batch needs one kernel: blur|sharpen|sobel");
        if (count < 1 || batch_size < 1) die("bad --n or --bsize");
        b.count = count;
        b.blur_size = blur_size;
        b.size = batch_size;
        memcpy(b.threads, threads, sizeof threads);
        b.n_threads = n_threads;
        b.reps = reps > 0 ? reps : 2;
        b.csv = csv ? csv : "results_batch.csv";
        return run_batch(&b);
    }

    usage();
    return 1;
}
