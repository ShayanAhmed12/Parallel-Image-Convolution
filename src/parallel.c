/*
 * parallel.c -- the six ways of running a filter over an image.
 *
 * This is the OpenMP part of the project. Every method computes exactly the
 * same output image; they differ only in how the work is split between
 * threads and how well they use the cache.
 *
 *   seq            plain loop, one thread: the reference result and time T1
 *   omp_rows       "#pragma omp parallel for" over the image rows
 *   omp_tiled      the image is cut into tiles; threads take whole tiles
 *   seq_sep        separable blur: horizontal pass, then vertical pass
 *   omp_sep        separable blur with both passes parallel over rows
 *   omp_sep_fused  separable blur where each thread does both passes on a
 *                  small block of rows, so the intermediate stays in cache
 *
 * Course concepts: data parallelism, loop scheduling (static, dynamic,
 * guided), collapse(2), the implicit barrier between two "omp for" loops,
 * per-thread private buffers.
 *
 * Thread map: when opt->owner is not NULL, each method also records which
 * thread wrote each pixel (one byte per pixel). It is only used for a
 * separate, untimed run that draws the picture of the work split.
 *
 * Why it is safe: threads only read the input image and each output pixel is
 * written by exactly one thread, so there are no races and no locks.
 */
#include <omp.h>
#include <string.h>

#include "pdc.h"

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

RunOptions run_options_default(void) {
    RunOptions opt;
    memset(&opt, 0, sizeof opt);
    opt.threads = 1;
    opt.schedule = SCHEDULE_STATIC;
    opt.tile_h = 16;
    opt.tile_w = 2048;
    opt.fuse_rows = 16;
    return opt;
}

/* Thread map: write this thread's number into the owner bytes of the
 * rectangle rows y0..y1-1, columns x0..x1-1. Does nothing for timed runs. */
static void mark_owner(uint8_t *owner, int width, int y0, int y1, int x0, int x1) {
    if (!owner) return;
    int id = omp_get_thread_num();          /* 0 outside a parallel region */
    for (int y = y0; y < y1; y++) memset(owner + (size_t)y * width + x0, id, (size_t)(x1 - x0));
}

/* ------------------------------------------------------------------ */
/* 2-D methods: filter_box on different pieces of the image             */
/* ------------------------------------------------------------------ */

static void run_sequential(const Filter *f, const Image *in, Image *out, uint8_t *owner) {
    filter_box(f, in, out, 0, in->height, 0, in->width);
    mark_owner(owner, in->width, 0, in->height, 0, in->width);
}

static void run_rows(const Filter *f, const Image *in, Image *out, uint8_t *owner) {
    /* split the rows between the threads. schedule(runtime) means the split
     * is chosen by omp_set_schedule() in run_method:
     *   static  - thread 0 gets the first block of rows, thread 1 the next, ...
     *   dynamic - a free thread takes the next single row (load balancing)
     *   guided  - like dynamic, but with big pieces first and small ones last */
#pragma omp parallel for schedule(runtime)
    for (int y = 0; y < in->height; y++) {
        filter_box(f, in, out, y, y + 1, 0, in->width);
        mark_owner(owner, in->width, y, y + 1, 0, in->width);
    }
}

static void run_tiled(const Filter *f, const Image *in, Image *out, const RunOptions *opt) {
    const int th = opt->tile_h, tw = opt->tile_w;
    const int tiles_down = (in->height + th - 1) / th;
    const int tiles_across = (in->width + tw - 1) / tw;
    /* collapse(2) turns the two tile loops into one list of tiles, which
     * is then split between the threads */
#pragma omp parallel for collapse(2) schedule(static)
    for (int ty = 0; ty < tiles_down; ty++)
        for (int tx = 0; tx < tiles_across; tx++) {
            int y0 = ty * th, y1 = (y0 + th < in->height) ? y0 + th : in->height;
            int x0 = tx * tw, x1 = (x0 + tw < in->width) ? x0 + tw : in->width;
            filter_box(f, in, out, y0, y1, x0, x1);
            mark_owner(opt->owner, in->width, y0, y1, x0, x1);
        }
}

/* ------------------------------------------------------------------ */
/* separable blur                                                      */
/* ------------------------------------------------------------------ */

/* Vertical pass for output row y, reading the horizontal sums of rows
 * y - radius ... y + radius from the full-size scratch buffer. */
static void vertical_pass_row(const Filter *f, const uint32_t *sums, Image *out, int y) {
    const size_t row_len = (size_t)out->width * 3;
    const uint32_t *rows[BLUR_SIZE_MAX];
    for (int j = 0; j < f->size; j++)
        rows[j] = sums + (size_t)clamp_int(y - f->radius + j, 0, out->height - 1) * row_len;
    blur_row_vertical(f, rows, out->pixels + (size_t)y * row_len, out->width);
}

static void run_separable_sequential(const Filter *f, const Image *in, Image *out, uint32_t *sums,
                                     uint8_t *owner) {
    const size_t row_len = (size_t)in->width * 3;
    for (int y = 0; y < in->height; y++)
        blur_row_horizontal(f, in->pixels + y * row_len, sums + y * row_len, in->width);
    for (int y = 0; y < in->height; y++)
        vertical_pass_row(f, sums, out, y);
    mark_owner(owner, in->width, 0, in->height, 0, in->width);
}

static void run_separable(const Filter *f, const Image *in, Image *out, uint32_t *sums, uint8_t *owner) {
    const size_t row_len = (size_t)in->width * 3;
    /* one team of threads for both passes */
#pragma omp parallel
    {
        /* pass 1: rows split between threads */
#pragma omp for schedule(static)
        for (int y = 0; y < in->height; y++)
            blur_row_horizontal(f, in->pixels + y * row_len, sums + y * row_len, in->width);
        /* the end of the loop above is an implicit barrier: pass 2 starts only
         * when every thread has finished pass 1, because a row's vertical sum
         * needs rows that other threads produced */
#pragma omp for schedule(static)
        for (int y = 0; y < in->height; y++) {
            vertical_pass_row(f, sums, out, y);
            mark_owner(owner, in->width, y, y + 1, 0, in->width);   /* who wrote the final row */
        }
    }
}

/* Fused: each thread takes a block of `fuse_rows` output rows, blurs the
 * rows it needs horizontally into its own small buffer (block + radius rows
 * above and below), then immediately does the vertical pass from that
 * buffer. No barrier is needed, and the intermediate can stay in the cache
 * if the buffer is small enough (it is about 20 rows x width x 12 bytes:
 * 2.9 MB at width 12000, too big for this CPU's caches -- see REPORT.md).
 * The price: rows on the edge of a block are blurred horizontally twice. */
static void run_separable_fused(const Filter *f, const Image *in, Image *out, int fuse_rows,
                                uint8_t *owner) {
    const int W = in->width, H = in->height, r = f->radius;
    const size_t row_len = (size_t)W * 3;
    const int blocks = (H + fuse_rows - 1) / fuse_rows;
    /* each thread runs this block with its own buffer */
#pragma omp parallel
    {
        uint32_t *buffer = xalloc((size_t)(fuse_rows + 2 * r) * row_len * sizeof(uint32_t));
        const uint32_t *rows[BLUR_SIZE_MAX];
        /* blocks of rows split between threads */
#pragma omp for schedule(static)
        for (int b = 0; b < blocks; b++) {
            int y0 = b * fuse_rows, y1 = (y0 + fuse_rows < H) ? y0 + fuse_rows : H;
            int needed = (y1 - y0) + 2 * r;   /* rows y0-r ... y1-1+r */
            for (int k = 0; k < needed; k++) {
                int src_y = clamp_int(y0 - r + k, 0, H - 1);
                blur_row_horizontal(f, in->pixels + (size_t)src_y * row_len, buffer + k * row_len, W);
            }
            for (int y = y0; y < y1; y++) {
                for (int j = 0; j < f->size; j++) rows[j] = buffer + (size_t)(y - y0 + j) * row_len;
                blur_row_vertical(f, rows, out->pixels + (size_t)y * row_len, W);
            }
            mark_owner(owner, W, y0, y1, 0, W);
        }
        xfree(buffer);
    }
}

/* ------------------------------------------------------------------ */
/* choosing a method                                                   */
/* ------------------------------------------------------------------ */

void run_method(Method m, const Filter *f, const Image *in, Image *out, const RunOptions *opt) {
    omp_set_num_threads(method_is_parallel(m) ? opt->threads : 1);
    /* the loop schedule used by schedule(runtime) in run_rows;
     * chunk 0 means "the default": equal blocks for static, adaptive for guided */
    switch (opt->schedule) {
    case SCHEDULE_STATIC:  omp_set_schedule(omp_sched_static, 0); break;
    case SCHEDULE_DYNAMIC: omp_set_schedule(omp_sched_dynamic, 1); break;
    case SCHEDULE_GUIDED:  omp_set_schedule(omp_sched_guided, 0); break;
    }
    switch (m) {
    case METHOD_SEQ:       run_sequential(f, in, out, opt->owner); break;
    case METHOD_ROWS:      run_rows(f, in, out, opt->owner); break;
    case METHOD_TILED:     run_tiled(f, in, out, opt); break;
    case METHOD_SEP_SEQ:   run_separable_sequential(f, in, out, opt->scratch, opt->owner); break;
    case METHOD_SEP:       run_separable(f, in, out, opt->scratch, opt->owner); break;
    case METHOD_SEP_FUSED: run_separable_fused(f, in, out, opt->fuse_rows, opt->owner); break;
    case METHOD_COUNT:     break;
    }
}

static const char *const METHOD_NAMES[METHOD_COUNT] = {
    "seq", "omp_rows", "omp_tiled", "seq_sep", "omp_sep", "omp_sep_fused",
};

const char *method_name(Method m) { return METHOD_NAMES[m]; }

int method_from_name(const char *name, Method *out) {
    for (int m = 0; m < METHOD_COUNT; m++)
        if (!strcmp(name, METHOD_NAMES[m])) { *out = (Method)m; return 1; }
    return 0;
}

int method_blur_only(Method m) {
    return m == METHOD_SEP_SEQ || m == METHOD_SEP || m == METHOD_SEP_FUSED;
}

static const char *const SCHEDULE_NAMES[] = { "static", "dynamic", "guided" };

const char *schedule_name(Schedule s) { return SCHEDULE_NAMES[s]; }

int schedule_from_name(const char *name, Schedule *out) {
    for (int s = 0; s < 3; s++)
        if (!strcmp(name, SCHEDULE_NAMES[s])) { *out = (Schedule)s; return 1; }
    return 0;
}

/* thread map colours (matplotlib "tab10"): thread t is drawn in THREAD_COLORS[t % 10] */
const uint8_t THREAD_COLORS[10][3] = {
    {31, 119, 180}, {255, 127, 14}, {44, 160, 44}, {214, 39, 40}, {148, 103, 189},
    {140, 86, 75}, {227, 119, 194}, {127, 127, 127}, {188, 189, 34}, {23, 190, 207},
};

void threadmap_write_ppm(const char *path, const uint8_t *owner, int width, int height) {
    Image map = image_new(width, height);
    for (size_t i = 0; i < (size_t)width * height; i++)
        memcpy(map.pixels + i * 3, THREAD_COLORS[owner[i] % 10], 3);
    image_write_ppm(path, &map);
    image_free(&map);
}

int method_is_parallel(Method m) {
    return m == METHOD_ROWS || m == METHOD_TILED || m == METHOD_SEP || m == METHOD_SEP_FUSED;
}
