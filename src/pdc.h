/*
 * pdc.h -- everything the program's files share: types and function declarations.
 *
 * Parallel Image Convolution / Filtering (OpenMP)
 * PDC course project, FAST-NUCES Karachi, Fall 2026
 * Shayan Ahmed (23K-0551), Rizwan Vadsariya (23K-0005), Aaqib (23K-0625)
 *
 * How the files fit together:
 *   image.c     memory, timing, reading/writing images, the generated test image
 *   filters.c   the maths of blur / sharpen / sobel for a rectangle of pixels
 *   parallel.c  the six ways of running a filter: sequential and OpenMP versions
 *   bench.c     the "bench" and "batch" commands: timing and verification
 *   main.c      reads the command line and runs the chosen command
 */
#ifndef PDC_H
#define PDC_H

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* image.c -- helpers                                                  */
/* ------------------------------------------------------------------ */

/* Print "error: <message>" and stop the program with exit code 1. */
void die(const char *fmt, ...);

/* Allocate memory aligned to 64 bytes (one cache line). Stops the program if
 * there is not enough memory. Free it with xfree(). */
void *xalloc(size_t bytes);
void xfree(void *p);

/* Fill a buffer in parallel. Done once before timing so that every thread
 * "first touches" its own part of the memory and page faults never happen
 * inside a timed region. */
void first_touch(void *buf, size_t bytes, int value);

/* Wall-clock time in seconds, and the timer's resolution. */
double wtime(void);
double wtick(void);

/* ------------------------------------------------------------------ */
/* image.c -- images                                                   */
/* ------------------------------------------------------------------ */

/* A 24-bit RGB image: 3 bytes per pixel (red, green, blue), rows stored one
 * after another with no gaps. Pixel (x, y) starts at pixels[(y*width + x)*3]. */
typedef struct {
    int width, height;
    uint8_t *pixels;
} Image;

Image  image_new(int width, int height);                 /* filled with 0 */
void   image_free(Image *img);
Image  image_read_ppm(const char *path);                 /* binary PPM (P6) */
void   image_write_ppm(const char *path, const Image *img);
void   image_make_test(Image *img, uint32_t seed);       /* same picture for the same seed */
size_t image_count_differences(const Image *a, const Image *b);  /* number of differing bytes */

/* ------------------------------------------------------------------ */
/* filters.c -- the filters                                            */
/* ------------------------------------------------------------------ */

typedef enum { FILTER_BLUR, FILTER_SHARPEN, FILTER_SOBEL } FilterKind;

#define BLUR_SIZE_MIN 3    /* blur sizes are odd: 3, 5, 7, ... 25 */
#define BLUR_SIZE_MAX 25

/* A filter and its settings. For blur, size is the window width (5 means 5x5),
 * weights[] holds one row of binomial weights (1 4 6 4 1 for size 5) and
 * 2^shift is the sum of all size*size weights, so dividing by it is a shift. */
typedef struct {
    FilterKind kind;
    int size;      /* window width: blur size, or 3 for sharpen and sobel */
    int radius;    /* size / 2: how far the window reaches from the centre */
    uint32_t weights[BLUR_SIZE_MAX];
    int shift;
} Filter;

Filter      filter_make(FilterKind kind, int blur_size);   /* blur_size ignored unless blur */
const char *filter_name(FilterKind kind);                    /* "blur", "sharpen", "sobel" */
int         filter_from_name(const char *name, FilterKind *out);  /* 1 = ok, 0 = unknown */

/* Apply the filter to the output pixels in rows [y0, y1) and columns [x0, x1).
 * Reads only from `in`, writes only to `out`, so any number of threads can
 * call it at once on different boxes. Every method is built from this. */
void filter_box(const Filter *f, const Image *in, Image *out, int y0, int y1, int x0, int x1);

/* The two halves of a separable blur (blur only):
 * horizontal: blur one image row sideways into a row of 32-bit sums;
 * vertical:   combine `size` such rows (above, at and below the output row)
 *             into one finished output row. */
void blur_row_horizontal(const Filter *f, const uint8_t *src_row, uint32_t *dst_row, int width);
void blur_row_vertical(const Filter *f, const uint32_t *const *rows, uint8_t *dst_row, int width);

/* ------------------------------------------------------------------ */
/* parallel.c -- the methods                                           */
/* ------------------------------------------------------------------ */

typedef enum {
    METHOD_SEQ,        /* "seq"           sequential, the reference (T1)          */
    METHOD_ROWS,       /* "omp_rows"      OpenMP parallel for over rows           */
    METHOD_TILED,      /* "omp_tiled"     OpenMP over rectangular tiles           */
    METHOD_SEP_SEQ,    /* "seq_sep"       separable blur, sequential   (blur only) */
    METHOD_SEP,        /* "omp_sep"       separable blur, two parallel passes      */
    METHOD_SEP_FUSED,  /* "omp_sep_fused" separable blur, passes fused per block   */
    METHOD_COUNT
} Method;

typedef enum { SCHEDULE_STATIC, SCHEDULE_DYNAMIC, SCHEDULE_GUIDED } Schedule;

/* Settings for one run of a method. */
typedef struct {
    int threads;              /* number of OpenMP threads (parallel methods only) */
    Schedule schedule;        /* loop schedule for METHOD_ROWS */
    int tile_h, tile_w;       /* tile size for METHOD_TILED */
    int fuse_rows;            /* rows per block for METHOD_SEP_FUSED */
    uint8_t *owner;           /* NULL, or width*height bytes: which thread wrote each pixel */
    uint32_t *scratch;        /* width*height*3 sums, needed by METHOD_SEP_SEQ and METHOD_SEP */
} RunOptions;

RunOptions  run_options_default(void);   /* 1 thread, static, 16x2048 tiles, 16 fused rows */
void        run_method(Method m, const Filter *f, const Image *in, Image *out, const RunOptions *opt);
const char *method_name(Method m);
int         method_from_name(const char *name, Method *out);   /* 1 = ok, 0 = unknown */
int         method_blur_only(Method m);
int         method_is_parallel(Method m);

/* ------------------------------------------------------------------ */
/* bench.c -- the bench and batch commands                             */
/* ------------------------------------------------------------------ */

#define MAX_THREAD_COUNTS 32

typedef struct {
    int width, height;
    unsigned filters;                    /* bit (1 << FilterKind) for each filter to test */
    int blur_size;
    int threads[MAX_THREAD_COUNTS], n_threads;
    int reps;                            /* best of this many runs */
    int tile_h, tile_w, fuse_rows;
    const char *csv;
} BenchOptions;

typedef struct {
    int count, size;                     /* number of images, each size x size */
    FilterKind filter;
    int blur_size;
    int threads[MAX_THREAD_COUNTS], n_threads;
    int reps;
    const char *csv;
} BatchOptions;

int run_bench(const BenchOptions *o);    /* returns 0 if every output matched, 2 if not */
int run_batch(const BatchOptions *o);

#endif
