/*
 * main.c -- reads the command line and runs one command.
 *
 *   pdc_conv gen    --size N|WxH --out img.ppm
 *   pdc_conv filter [--in img.ppm | --size N|WxH] --kernel blur|sharpen|sobel
 *                   [--variant METHOD] [--threads P] [--out out.ppm]
 *                   [--verify] [--csv f.csv] [--threadmap map.ppm]
 *   pdc_conv bench  [--size N|WxH] [--kernel all|blur|sharpen|sobel] [--methods m1,m2]
 *                   [--threads 1,2,4,8] [--reps 3] [--csv results_bench.csv]
 *   pdc_conv batch  [--n 2000] [--bsize 1024] [--kernel blur]
 *                   [--threads 1,2,4,8] [--reps 2] [--csv results_batch.csv]
 *   --blur-size N (odd, 3..25, default 5) works with filter, bench and batch
 *   --schedule static|dynamic|guided (omp_rows only, default static): filter, bench
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
         "                  [--verify] [--csv filter.csv] [--threadmap map.ppm]\n"
         "  pdc_conv bench  [--size N|WxH] [--kernel all|blur|sharpen|sobel] [--methods m1,m2]\n"
         "                  [--threads 1,2,4,8] [--reps 3] [--csv results_bench.csv]\n"
         "  pdc_conv batch  [--n 2000] [--bsize 1024] [--kernel blur]\n"
         "                  [--threads 1,2,4,8] [--reps 2] [--csv results_batch.csv]\n\n"
         "  --blur-size N   odd blur window 3..25 (default 5) for filter, bench, batch\n"
         "  --schedule S    static|dynamic|guided, loop schedule of omp_rows (default static)\n"
         "  --verify        filter: also run seq, report speedup and whether output is identical\n"
         "  --threadmap F   filter: write an image showing which thread computed each pixel\n"
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

/* "omp_rows,omp_tiled" -> a bit mask of methods; seq is always included */
static unsigned parse_method_mask(const char *text) {
    char buf[256];
    strncpy(buf, text, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    unsigned mask = 1u << METHOD_SEQ;
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        Method m;
        if (!method_from_name(t, &m)) die("unknown method %s", t);
        mask |= 1u << m;
    }
    return mask;
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

/* What the filter command was asked to do. */
typedef struct {
    const char *in_path;     /* NULL: generate a test image of width x height */
    int width, height;
    FilterKind kind;
    int blur_size;
    Method method;
    const char *out_path;    /* optional result image */
    int verify;              /* also run seq: report speedup and compare outputs */
    const char *csv;         /* optional one-row CSV of the result */
    const char *threadmap;   /* optional thread map image */
} FilterJob;

static uint32_t *new_scratch(const Image *img) {
    size_t bytes = (size_t)img->width * img->height * 3 * sizeof(uint32_t);
    uint32_t *scratch = xalloc(bytes);
    first_touch(scratch, bytes, 0);
    return scratch;
}

/* filter: apply one method once and report the time */
static int command_filter(const FilterJob *job, const RunOptions *base) {
    Filter f = filter_make(job->kind, job->blur_size);
    if (method_blur_only(job->method) && job->kind != FILTER_BLUR)
        die("method %s only supports --kernel blur", method_name(job->method));

    Image in;
    if (job->in_path) in = image_read_ppm(job->in_path);
    else { in = image_new(job->width, job->height); image_make_test(&in, 12345); }
    Image out = image_new(in.width, in.height);

    RunOptions opt = *base;
    if (job->method == METHOD_SEP_SEQ || job->method == METHOD_SEP) opt.scratch = new_scratch(&in);
    int threads = method_is_parallel(job->method) ? opt.threads : 1;

    double t0 = wtime();
    run_method(job->method, &f, &in, &out, &opt);
    double t = wtime() - t0;
    printf("%s / %s on %dx%d with %d thread(s): %.4f s\n", filter_name(job->kind),
           method_name(job->method), in.width, in.height, threads, t);

    /* --verify: the sequential reference, its time, and a byte-by-byte comparison */
    double seq_time = 0;
    int identical = 1;
    if (job->verify) {
        Image reference = image_new(in.width, in.height);
        RunOptions seq_opt = run_options_default();
        t0 = wtime();
        run_method(METHOD_SEQ, &f, &in, &reference, &seq_opt);
        seq_time = wtime() - t0;
        identical = image_count_differences(&out, &reference) == 0;
        printf("sequential: %.4f s, speedup: %.2f x, identical: %s\n", seq_time, seq_time / t,
               identical ? "yes" : "NO");
        image_free(&reference);
    }

    if (job->out_path) { image_write_ppm(job->out_path, &out); printf("wrote %s\n", job->out_path); }

    /* --threadmap: a second, untimed run that records which thread wrote each pixel.
     * With dynamic or guided scheduling this split can differ from the timed run. */
    if (job->threadmap) {
        opt.owner = xalloc((size_t)in.width * in.height);
        run_method(job->method, &f, &in, &out, &opt);
        threadmap_write_ppm(job->threadmap, opt.owner, in.width, in.height);
        printf("wrote %s\n", job->threadmap);
        xfree(opt.owner);
    }

    if (job->csv) {
        FILE *csv = fopen(job->csv, "w");
        if (!csv) die("cannot open %s for writing", job->csv);
        fprintf(csv, "filter,blur_size,method,schedule,threads,width,height,time_s,seq_time_s,speedup,identical\n");
        char size_text[8] = "";   /* blur size column: empty for sharpen and sobel */
        if (job->kind == FILTER_BLUR) snprintf(size_text, sizeof size_text, "%d", f.size);
        fprintf(csv, "%s,%s,%s,%s,%d,%d,%d,%.6f,", filter_name(job->kind), size_text, method_name(job->method),
                job->method == METHOD_ROWS ? schedule_name(opt.schedule) : "", threads, in.width, in.height, t);
        if (job->verify) fprintf(csv, "%.6f,%.3f,%s\n", seq_time, seq_time / t, identical ? "yes" : "no");
        else fprintf(csv, ",,\n");
        fclose(csv);
    }

    image_free(&in);
    image_free(&out);
    if (opt.scratch) xfree(opt.scratch);
    return identical ? 0 : 2;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *command = argv[1];

    int width = 4096, height = 4096, reps = -1, count = 2000, batch_size = 1024, blur_size = 5;
    int threads[MAX_THREAD_COUNTS], n_threads = 0;
    const char *kernel = NULL, *in_path = NULL, *out_path = NULL, *csv = NULL, *threadmap = NULL;
    int verify = 0;
    const char *method_text = "omp_rows";
    unsigned method_mask = (1u << METHOD_COUNT) - 1;   /* bench: every method */
    RunOptions opt = run_options_default();

    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--verify")) { verify = 1; continue; }   /* the only option without a value */
        const char *value = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!value) { fprintf(stderr, "missing value for %s\n", arg); usage(); return 1; }
        i++;
        if      (!strcmp(arg, "--size"))    parse_size(value, &width, &height);
        else if (!strcmp(arg, "--kernel"))  kernel = value;
        else if (!strcmp(arg, "--blur-size")) blur_size = parse_blur_size(value);
        else if (!strcmp(arg, "--schedule")) {
            if (!schedule_from_name(value, &opt.schedule)) die("unknown --schedule %s (static|dynamic|guided)", value);
        }
        else if (!strcmp(arg, "--threadmap")) threadmap = value;
        else if (!strcmp(arg, "--threads")) n_threads = parse_thread_list(value, threads, MAX_THREAD_COUNTS);
        else if (!strcmp(arg, "--reps"))    reps = atoi(value);
        else if (!strcmp(arg, "--csv"))     csv = value;
        else if (!strcmp(arg, "--in"))      in_path = value;
        else if (!strcmp(arg, "--out"))     out_path = value;
        else if (!strcmp(arg, "--variant")) method_text = value;
        else if (!strcmp(arg, "--methods")) method_mask = parse_method_mask(value);
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
        FilterJob job = { in_path, width, height, kind, blur_size, method, out_path, verify, csv, threadmap };
        return command_filter(&job, &opt);
    }

    if (!strcmp(command, "bench")) {
        BenchOptions b;
        memset(&b, 0, sizeof b);
        b.width = width;
        b.height = height;
        b.filters = parse_filter_mask(kernel ? kernel : "all");
        b.blur_size = blur_size;
        b.methods = method_mask;
        memcpy(b.threads, threads, sizeof threads);
        b.n_threads = n_threads;
        b.reps = reps > 0 ? reps : 3;
        b.tile_h = opt.tile_h;
        b.tile_w = opt.tile_w;
        b.fuse_rows = opt.fuse_rows;
        b.schedule = opt.schedule;
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
