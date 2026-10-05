/*
 * pdc_conv.c -- Parallel Image Convolution / Filtering (OpenMP)
 * Parallel and Distributed Computing (PDC), FAST-NUCES Karachi, Fall 2026
 * Shayan Ahmed (23K-0551), Rizwan Vadsariya (23K-0005), Aaqib (23K-0625)
 *
 * Filters (24-bit RGB, interleaved, clamp-to-edge borders, integer math so
 * every variant is bit-exact against the sequential reference):
 *   blur    : 5x5 Gaussian (binomial 1-4-6-4-1, separable, sum 256)
 *   sharpen : 3x3 [0 -1 0; -1 5 -1; 0 -1 0]
 *   sobel   : 3x3 Sobel on luma, |gx|+|gy| clamped to 255 (gray output)
 *
 * Implementations (variants):
 *   seq            sequential 2-D convolution     (correctness reference, T1)
 *   omp_rows       #pragma omp parallel for schedule(static) over rows
 *   omp_tiled      cache tiling, collapse(2) over (tile-row, tile-col)
 *   seq_sep        sequential separable blur (two 1-D passes)          [blur]
 *   omp_sep        separable blur, both passes parallel                [blur]
 *   omp_sep_fused  separable blur, row-blocks fused so the intermediate
 *                  stays in cache (per-thread scratch)                 [blur]
 *
 * Sub-commands:  gen | filter | bench | batch     (see usage())
 */
#define _POSIX_C_SOURCE 200809L
#ifdef _WIN32
#include <malloc.h>
#endif
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* utilities                                                           */
/* ------------------------------------------------------------------ */
typedef enum { K_BLUR = 0, K_SHARPEN, K_SOBEL, K_COUNT } kernel_t;
static const char *KNAME[K_COUNT] = {"blur", "sharpen", "sobel"};

static volatile uint64_t g_sink; /* defeats dead-store elimination */
static int g_tile_h = 16, g_tile_w = 2048; /* omp_tiled tile size   */
static int g_fuse_bh = 16;                /* omp_sep_fused block   */

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void *xalloc(size_t bytes) {
    if (bytes == 0) bytes = 64;
#ifdef _WIN32
    void *p = _aligned_malloc(bytes, 64);
    if (!p)
        die("out of memory (try a smaller --size)");
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, (bytes + 63) & ~(size_t)63) != 0 || !p)
        die("out of memory (try a smaller --size)");
#endif
    return p;
}

static void xfree(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Parallel first-touch so pages are spread across threads/NUMA nodes and
 * page-fault cost never lands inside a timed region. */
static void par_fill(void *buf, size_t bytes, int val) {
    const size_t chunk = (size_t)1 << 20;
    long nch = (long)((bytes + chunk - 1) / chunk);
#pragma omp parallel for schedule(static)
    for (long c = 0; c < nch; c++) {
        size_t o = (size_t)c * chunk;
        size_t l = (bytes - o < chunk) ? bytes - o : chunk;
        memset((char *)buf + o, val, l);
    }
}

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline uint8_t clamp255(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static size_t count_diff(const uint8_t *a, const uint8_t *b, size_t n) {
    size_t d = 0;
#pragma omp parallel for reduction(+ : d) schedule(static)
    for (long i = 0; i < (long)n; i++) d += (a[i] != b[i]);
    return d;
}

/* ------------------------------------------------------------------ */
/* synthetic image + PPM I/O                                           */
/* ------------------------------------------------------------------ */
static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* Deterministic: gradient + circle + rectangle + stripes + noise. */
static void synth(uint8_t *img, int W, int H, uint32_t seed) {
    long cx = W / 2, cy = H / 2;
    long rad = (W < H ? W : H) / 4, r2 = rad * rad;
    int stripe = W / 48 + 1;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int r = (int)(((long)x * 255) / (W > 1 ? W - 1 : 1));
            int g = (int)(((long)y * 255) / (H > 1 ? H - 1 : 1));
            int b = 255 - (r + g) / 2;
            long dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy < r2) { r = 255 - r; g = 255 - g; }
            if (x > W / 8 && x < W * 3 / 8 && y > H / 8 && y < H / 3)
                r = g = b = 240;
            if (x > W * 5 / 8 && y > H * 5 / 8 && (((x + y) / stripe) & 1))
                r = g = b = 30;
            uint32_t h = hash32((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ seed);
            r += (int)(h & 31) - 16;
            g += (int)((h >> 5) & 31) - 16;
            b += (int)((h >> 10) & 31) - 16;
            uint8_t *p = img + ((size_t)y * W + x) * 3;
            p[0] = clamp255(r); p[1] = clamp255(g); p[2] = clamp255(b);
        }
    }
}

static int ppm_int(FILE *f) {
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); }
        else if (c == EOF) return -1;
        else if (c >= '0' && c <= '9') break;
    }
    int v = 0;
    while (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); c = fgetc(f); }
    return v;
}

static uint8_t *read_ppm(const char *path, int *W, int *H) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open input image");
    if (fgetc(f) != 'P' || fgetc(f) != '6') die("input must be a binary PPM (P6)");
    *W = ppm_int(f); *H = ppm_int(f);
    int maxv = ppm_int(f);
    if (*W <= 0 || *H <= 0 || maxv != 255) die("unsupported PPM header (need maxval 255)");
    uint8_t *img = xalloc((size_t)*W * *H * 3);
    if (fread(img, 1, (size_t)*W * *H * 3, f) != (size_t)*W * *H * 3) die("truncated PPM");
    fclose(f);
    return img;
}

static void write_ppm(const char *path, const uint8_t *img, int W, int H) {
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot open output file");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(img, 1, (size_t)W * H * 3, f);
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* per-pixel window kernels                                            */
/* p -> top-left of the (2R+1)^2 window, s = row stride in bytes       */
/* ------------------------------------------------------------------ */
static const int G[5] = {1, 4, 6, 4, 1};

static inline void blur_win(const uint8_t *p, ptrdiff_t s, uint8_t *o) {
    for (int c = 0; c < 3; c++) {
        int sum = 0;
        for (int j = 0; j < 5; j++) {
            const uint8_t *r = p + j * s + c;
            int h = G[0] * r[0] + G[1] * r[3] + G[2] * r[6] + G[3] * r[9] + G[4] * r[12];
            sum += G[j] * h;
        }
        o[c] = (uint8_t)((sum + 128) >> 8);
    }
}

static inline void sharpen_win(const uint8_t *p, ptrdiff_t s, uint8_t *o) {
    for (int c = 0; c < 3; c++) {
        int v = 5 * p[s + 3 + c] - p[3 + c] - p[s + c] - p[s + 6 + c] - p[2 * s + 3 + c];
        o[c] = clamp255(v);
    }
}

static inline int luma(const uint8_t *q) { return (77 * q[0] + 150 * q[1] + 29 * q[2]) >> 8; }

static inline void sobel_win(const uint8_t *p, ptrdiff_t s, uint8_t *o) {
    int a = luma(p), b = luma(p + 3), c = luma(p + 6);
    int d = luma(p + s), f = luma(p + s + 6);
    int g = luma(p + 2 * s), h = luma(p + 2 * s + 3), i = luma(p + 2 * s + 6);
    int gx = (c + 2 * f + i) - (a + 2 * d + g);
    int gy = (g + 2 * h + i) - (a + 2 * b + c);
    int m = abs(gx) + abs(gy);
    o[0] = o[1] = o[2] = (uint8_t)(m > 255 ? 255 : m);
}

/* Copy a clamped (2R+1)^2 neighbourhood: this is the border-handling path. */
static void gather(const uint8_t *in, int W, int H, int x, int y, int R, uint8_t *win) {
    int n = 2 * R + 1;
    for (int j = 0; j < n; j++) {
        int yy = clampi(y - R + j, 0, H - 1);
        for (int i = 0; i < n; i++) {
            int xx = clampi(x - R + i, 0, W - 1);
            const uint8_t *q = in + ((size_t)yy * W + xx) * 3;
            uint8_t *d = win + (j * n + i) * 3;
            d[0] = q[0]; d[1] = q[1]; d[2] = q[2];
        }
    }
}

/* region_<k>(in,out,W,H, y0,y1, x0,x1): filter the pixel box [y0,y1)x[x0,x1).
 * Interior pixels read the image directly; border pixels go through gather().
 * Every sequential / parallel variant is built from this one function. */
#define DEFINE_REGION(NAME, R, WINFN)                                                   \
    static void NAME(const uint8_t *in, uint8_t *out, int W, int H, int y0, int y1,     \
                     int x0, int x1) {                                                  \
        const ptrdiff_t s = (ptrdiff_t)W * 3;                                           \
        for (int y = y0; y < y1; y++) {                                                 \
            int yin = (y >= (R) && y < H - (R));                                        \
            int xa = yin ? (x0 > (R) ? x0 : (R)) : x1;                                  \
            if (xa > x1) xa = x1;                                                       \
            int xb = yin ? (x1 < W - (R) ? x1 : W - (R)) : x1;                          \
            if (xb < xa) xb = xa;                                                       \
            for (int x = x0; x < xa; x++) {                                             \
                uint8_t win[25 * 3];                                                    \
                gather(in, W, H, x, y, (R), win);                                       \
                WINFN(win, (ptrdiff_t)(2 * (R) + 1) * 3, out + ((size_t)y * W + x) * 3); \
            }                                                                           \
            for (int x = xa; x < xb; x++)                                               \
                WINFN(in + ((size_t)(y - (R)) * W + (x - (R))) * 3, s,                  \
                      out + ((size_t)y * W + x) * 3);                                   \
            for (int x = xb; x < x1; x++) {                                             \
                uint8_t win[25 * 3];                                                    \
                gather(in, W, H, x, y, (R), win);                                       \
                WINFN(win, (ptrdiff_t)(2 * (R) + 1) * 3, out + ((size_t)y * W + x) * 3); \
            }                                                                           \
        }                                                                               \
    }

DEFINE_REGION(region_blur, 2, blur_win)
DEFINE_REGION(region_sharpen, 1, sharpen_win)
DEFINE_REGION(region_sobel, 1, sobel_win)

typedef void (*region_fn)(const uint8_t *, uint8_t *, int, int, int, int, int, int);
static const region_fn REGION[K_COUNT] = {region_blur, region_sharpen, region_sobel};

/* ------------------------------------------------------------------ */
/* 2-D variants: sequential / row-parallel / tiled                      */
/* ------------------------------------------------------------------ */
static void run_seq(kernel_t k, const uint8_t *in, uint8_t *out, int W, int H) {
    REGION[k](in, out, W, H, 0, H, 0, W);
}

static void run_omp_rows(kernel_t k, const uint8_t *in, uint8_t *out, int W, int H) {
    region_fn f = REGION[k];
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) f(in, out, W, H, y, y + 1, 0, W);
}

static void run_omp_tiled(kernel_t k, const uint8_t *in, uint8_t *out, int W, int H) {
    region_fn f = REGION[k];
    const int th = g_tile_h, tw = g_tile_w;
    const int nty = (H + th - 1) / th, ntx = (W + tw - 1) / tw;
#pragma omp parallel for collapse(2) schedule(static)
    for (int ty = 0; ty < nty; ty++)
        for (int tx = 0; tx < ntx; tx++) {
            int y0 = ty * th, y1 = y0 + th < H ? y0 + th : H;
            int x0 = tx * tw, x1 = x0 + tw < W ? x0 + tw : W;
            f(in, out, W, H, y0, y1, x0, x1);
        }
}

/* ------------------------------------------------------------------ */
/* separable Gaussian blur (two 1-D passes, exact integer arithmetic)   */
/* ------------------------------------------------------------------ */
static inline void hpix(const uint8_t *s, uint16_t *t, int x, int W) {
    for (int c = 0; c < 3; c++) {
        int sum = 0;
        for (int i = 0; i < 5; i++) sum += G[i] * s[clampi(x - 2 + i, 0, W - 1) * 3 + c];
        t[x * 3 + c] = (uint16_t)sum;
    }
}

/* horizontal pass of one row: uint8 row -> uint16 row (weights sum 16) */
static void hpass_row(const uint8_t *s, uint16_t *t, int W) {
    int n = W * 3;
    int lead = W < 5 ? W : 2;
    for (int x = 0; x < lead; x++) hpix(s, t, x, W);
    if (W >= 5) {
        for (int x = W - 2; x < W; x++) hpix(s, t, x, W);
        for (int k = 6; k < n - 6; k++)
            t[k] = (uint16_t)(G[0] * s[k - 6] + G[1] * s[k - 3] + G[2] * s[k] +
                              G[3] * s[k + 3] + G[4] * s[k + 6]);
    }
}

/* vertical pass of one row from five intermediate rows -> final uint8 row */
static void vpass_row(const uint16_t *r0, const uint16_t *r1, const uint16_t *r2,
                      const uint16_t *r3, const uint16_t *r4, uint8_t *out, int n) {
    for (int k = 0; k < n; k++) {
        int sum = r0[k] + 4 * r1[k] + 6 * r2[k] + 4 * r3[k] + r4[k];
        out[k] = (uint8_t)((sum + 128) >> 8);
    }
}

static void vpass_global(const uint16_t *tmp, uint8_t *out, int W, int H, int y) {
    size_t n = (size_t)W * 3;
    const uint16_t *r[5];
    for (int j = 0; j < 5; j++) r[j] = tmp + (size_t)clampi(y - 2 + j, 0, H - 1) * n;
    vpass_row(r[0], r[1], r[2], r[3], r[4], out + (size_t)y * n, (int)n);
}

static void blur_sep_seq(const uint8_t *in, uint8_t *out, uint16_t *tmp, int W, int H) {
    size_t n = (size_t)W * 3;
    for (int y = 0; y < H; y++) hpass_row(in + y * n, tmp + y * n, W);
    for (int y = 0; y < H; y++) vpass_global(tmp, out, W, H, y);
}

static void blur_sep_omp(const uint8_t *in, uint8_t *out, uint16_t *tmp, int W, int H) {
    size_t n = (size_t)W * 3;
#pragma omp parallel
    {
#pragma omp for schedule(static) /* implicit barrier: pass 2 needs neighbours' pass 1 */
        for (int y = 0; y < H; y++) hpass_row(in + y * n, tmp + y * n, W);
#pragma omp for schedule(static)
        for (int y = 0; y < H; y++) vpass_global(tmp, out, W, H, y);
    }
}

/* Row-block fusion: each thread blurs a block of rows horizontally into a
 * small private buffer (block + 4 halo rows) and immediately runs the vertical
 * pass on it, so the intermediate never goes out to main memory. */
static void blur_sep_fused_omp(const uint8_t *in, uint8_t *out, int W, int H) {
    size_t n = (size_t)W * 3;
    int bh = g_fuse_bh, nb = (H + bh - 1) / bh;
#pragma omp parallel
    {
        uint16_t *buf = xalloc((size_t)(bh + 4) * n * sizeof(uint16_t));
#pragma omp for schedule(static)
        for (int b = 0; b < nb; b++) {
            int y0 = b * bh, y1 = y0 + bh < H ? y0 + bh : H;
            for (int idx = 0; idx < (y1 - y0) + 4; idx++) {
                int src = clampi(y0 - 2 + idx, 0, H - 1);
                hpass_row(in + (size_t)src * n, buf + (size_t)idx * n, W);
            }
            for (int y = y0; y < y1; y++) {
                const uint16_t *b0 = buf + (size_t)(y - y0) * n;
                vpass_row(b0, b0 + n, b0 + 2 * n, b0 + 3 * n, b0 + 4 * n, out + (size_t)y * n, (int)n);
            }
        }
        xfree(buf);
    }
}

/* ------------------------------------------------------------------ */
/* variant table                                                       */
/* ------------------------------------------------------------------ */
typedef void (*variant_fn)(kernel_t, const uint8_t *, uint8_t *, uint16_t *, int, int);
typedef struct {
    const char *name;
    variant_fn fn;
    int blur_only; /* only valid for the blur kernel */
    int parallel;  /* honours the thread count       */
} variant_t;

static void v_seq(kernel_t k, const uint8_t *in, uint8_t *out, uint16_t *t, int W, int H) { (void)t; run_seq(k, in, out, W, H); }
static void v_rows(kernel_t k, const uint8_t *in, uint8_t *out, uint16_t *t, int W, int H) { (void)t; run_omp_rows(k, in, out, W, H); }
static void v_tiled(kernel_t k, const uint8_t *in, uint8_t *out, uint16_t *t, int W, int H) { (void)t; run_omp_tiled(k, in, out, W, H); }
static void v_sep_seq(kernel_t k, const uint8_t *in, uint8_t *out, uint16_t *t, int W, int H) { (void)k; blur_sep_seq(in, out, t, W, H); }
static void v_sep_omp(kernel_t k, const uint8_t *in, uint8_t *out, uint16_t *t, int W, int H) { (void)k; blur_sep_omp(in, out, t, W, H); }
static void v_sep_fused(kernel_t k, const uint8_t *in, uint8_t *out, uint16_t *t, int W, int H) { (void)k; (void)t; blur_sep_fused_omp(in, out, W, H); }

static const variant_t VARIANTS[] = {
    {"seq", v_seq, 0, 0},           /* index 0 = baseline T1 */
    {"omp_rows", v_rows, 0, 1},
    {"omp_tiled", v_tiled, 0, 1},
    {"seq_sep", v_sep_seq, 1, 0},
    {"omp_sep", v_sep_omp, 1, 1},
    {"omp_sep_fused", v_sep_fused, 1, 1},
};
#define NV ((int)(sizeof(VARIANTS) / sizeof(VARIANTS[0])))

/* ------------------------------------------------------------------ */
/* CLI helpers                                                         */
/* ------------------------------------------------------------------ */
static int parse_list(const char *s, int *out, int max) {
    char buf[256];
    strncpy(buf, s, sizeof buf - 1); buf[sizeof buf - 1] = 0;
    int n = 0;
    for (char *t = strtok(buf, ","); t && n < max; t = strtok(NULL, ",")) {
        int v = atoi(t);
        if (v < 1) die("thread counts must be >= 1");
        out[n++] = v;
    }
    return n;
}

static void parse_size(const char *s, int *W, int *H) {
    int a = 0, b = 0;
    if (sscanf(s, "%dx%d", &a, &b) == 2) { *W = a; *H = b; }
    else { a = atoi(s); *W = *H = a; }
    if (*W < 1 || *H < 1) die("bad --size");
}

static int parse_kernel_mask(const char *s) {
    if (!strcmp(s, "all")) return (1 << K_COUNT) - 1;
    for (int k = 0; k < K_COUNT; k++) if (!strcmp(s, KNAME[k])) return 1 << k;
    die("unknown kernel (blur|sharpen|sobel|all)");
    return 0;
}

static void default_threads(int *list, int *n) {
    int p = omp_get_num_procs(), c = 0;
    for (int t = 1; t <= p && c < 30; t *= 2) list[c++] = t;
    if (list[c - 1] != p) list[c++] = p;
    *n = c;
}

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
         "  common tuning:  --tile HxW (omp_tiled, default 16x2048)  --fuse-bh R (default 16)");
}

/* ------------------------------------------------------------------ */
/* bench: single big image, all variants x thread counts                */
/* ------------------------------------------------------------------ */
static double best_time(const variant_t *v, kernel_t k, const uint8_t *in, uint8_t *out,
                        uint16_t *tmp, int W, int H, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        v->fn(k, in, out, tmp, W, H);
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
    }
    g_sink += out[(size_t)W * 3 / 2];
    return best;
}

static int bench(int W, int H, int kmask, const int *pl, int np, int reps, const char *csv) {
    size_t bytes = (size_t)W * H * 3;
    int need_tmp = kmask & (1 << K_BLUR);
    printf("Image %dx%d (%.1f MB RGB), reps=%d (best-of), machine procs=%d, tile=%dx%d, fuse-bh=%d\n",
           W, H, bytes / 1048576.0, reps, omp_get_num_procs(), g_tile_h, g_tile_w, g_fuse_bh);
    uint8_t *in = xalloc(bytes), *ref = xalloc(bytes), *out = xalloc(bytes);
    uint16_t *tmp = need_tmp ? xalloc(bytes * sizeof(uint16_t)) : NULL;
    printf("Generating input...\n");
    synth(in, W, H, 12345);
    par_fill(ref, bytes, 0);
    par_fill(out, bytes, 0);
    if (tmp) par_fill(tmp, bytes * sizeof(uint16_t), 0);

    FILE *f = fopen(csv, "w");
    if (!f) die("cannot open csv for writing");
    fprintf(f, "kernel,variant,threads,time_s,speedup_vs_seq,efficiency,speedup_vs_self_1thread,verified\n");

    int all_ok = 1;
    for (int k = 0; k < K_COUNT; k++) {
        if (!(kmask & (1 << k))) continue;
        printf("\n=== %s ===\n%-14s %4s %10s %9s %7s  %s\n", KNAME[k], "variant", "thr",
               "time(s)", "speedup", "eff", "verify");
        omp_set_num_threads(1);
        double tseq = best_time(&VARIANTS[0], (kernel_t)k, in, ref, tmp, W, H, reps);
        printf("%-14s %4d %10.4f %9.2f %7.2f  reference (T1)\n", "seq", 1, tseq, 1.0, 1.0);
        fprintf(f, "%s,seq,1,%.6f,1.000,1.000,1.000,ref\n", KNAME[k], tseq);

        for (int vi = 1; vi < NV; vi++) {
            const variant_t *v = &VARIANTS[vi];
            if (v->blur_only && k != K_BLUR) continue;
            int cnt = v->parallel ? np : 1;
            double t_self1 = 0;
            for (int pi = 0; pi < cnt; pi++) {
                int p = v->parallel ? pl[pi] : 1;
                omp_set_num_threads(p);
#pragma omp parallel
                { g_sink += 0; } /* spin up the thread team outside the timer */
                par_fill(out, bytes, 0xA5); /* stale data can't fake a pass */
                double t = best_time(v, (kernel_t)k, in, out, tmp, W, H, reps);
                size_t bad = count_diff(out, ref, bytes);
                if (bad) all_ok = 0;
                if (p == 1) t_self1 = t;
                double sp = tseq / t, ef = sp / p;
                char selfbuf[32] = "";
                if (t_self1 > 0) snprintf(selfbuf, sizeof selfbuf, "%.3f", t_self1 / t);
                printf("%-14s %4d %10.4f %9.2f %7.2f  %s\n", v->name, p, t, sp, ef,
                       bad ? "MISMATCH" : "ok");
                fprintf(f, "%s,%s,%d,%.6f,%.3f,%.3f,%s,%s\n", KNAME[k], v->name, p, t, sp, ef,
                        selfbuf, bad ? "FAIL" : "ok");
                fflush(stdout);
            }
        }
    }
    fclose(f);
    printf("\n%s -- results written to %s\n", all_ok ? "ALL VARIANTS MATCH THE SEQUENTIAL REFERENCE" : "*** MISMATCH DETECTED ***", csv);
    xfree(in); xfree(ref); xfree(out); xfree(tmp);
    return all_ok ? 0 : 2;
}

/* ------------------------------------------------------------------ */
/* batch: many small images; per-image vs per-row parallelism           */
/* ------------------------------------------------------------------ */
static int batch(int N, int S, kernel_t k, const int *pl, int np, int reps, const char *csv) {
    const int POOL = N < 8 ? N : 8;
    size_t bytes = (size_t)S * S * 3;
    printf("Batch: %d images of %dx%d, kernel=%s, pool of %d distinct images cycled, reps=%d\n",
           N, S, S, KNAME[k], POOL, reps);
    uint8_t *pool[8], *ref[8], *chk[8];
    for (int i = 0; i < POOL; i++) {
        pool[i] = xalloc(bytes); ref[i] = xalloc(bytes); chk[i] = xalloc(bytes);
        synth(pool[i], S, S, 1000u + (uint32_t)i);
        par_fill(chk[i], bytes, 0);
    }
    uint8_t *shared_out = xalloc(bytes);
    par_fill(shared_out, bytes, 0);

    /* correctness: sequential reference vs per-image-parallel vs per-row-parallel */
    for (int i = 0; i < POOL; i++) run_seq(k, pool[i], ref[i], S, S);
    int ok = 1;
    omp_set_num_threads(pl[np - 1]);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < POOL; i++) run_seq(k, pool[i], chk[i], S, S);
    for (int i = 0; i < POOL; i++) if (count_diff(chk[i], ref[i], bytes)) ok = 0;
    for (int i = 0; i < POOL; i++) {
        par_fill(chk[i], bytes, 0xA5);
        run_omp_rows(k, pool[i], chk[i], S, S);
        if (count_diff(chk[i], ref[i], bytes)) ok = 0;
    }
    printf("Verification (per-image & per-row vs sequential): %s\n", ok ? "ok" : "MISMATCH");

    FILE *f = fopen(csv, "w");
    if (!f) die("cannot open csv for writing");
    fprintf(f, "strategy,threads,time_s,speedup_vs_seq,efficiency,images_per_s,verified\n");

    double tseq = 1e30;
    omp_set_num_threads(1);
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        for (int i = 0; i < N; i++) run_seq(k, pool[i % POOL], shared_out, S, S);
        double t = omp_get_wtime() - t0;
        if (t < tseq) tseq = t;
    }
    g_sink += shared_out[S];
    printf("\n%-10s %4s %10s %9s %7s %10s\n", "strategy", "thr", "time(s)", "speedup", "eff", "img/s");
    printf("%-10s %4d %10.3f %9.2f %7.2f %10.1f\n", "seq", 1, tseq, 1.0, 1.0, N / tseq);
    fprintf(f, "seq,1,%.6f,1.000,1.000,%.2f,%s\n", tseq, N / tseq, ok ? "ok" : "FAIL");

    for (int pi = 0; pi < np; pi++) {
        int p = pl[pi];
        omp_set_num_threads(p);
#pragma omp parallel
        { g_sink += 0; }
        /* strategy A: parallelise inside each image (rows), images one by one */
        double ta = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = omp_get_wtime();
            for (int i = 0; i < N; i++) run_omp_rows(k, pool[i % POOL], shared_out, S, S);
            double t = omp_get_wtime() - t0;
            if (t < ta) ta = t;
        }
        g_sink += shared_out[S];
        /* strategy B: one image per thread, each image processed sequentially */
        double tb = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = omp_get_wtime();
            uint64_t local = 0;
#pragma omp parallel reduction(+ : local)
            {
                uint8_t *o = xalloc(bytes);
                memset(o, 0, bytes);
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < N; i++) {
                    run_seq(k, pool[i % POOL], o, S, S);
                    local += o[(size_t)(i % S) * 3];
                }
                xfree(o);
            }
            double t = omp_get_wtime() - t0;
            g_sink += local;
            if (t < tb) tb = t;
        }
        printf("%-10s %4d %10.3f %9.2f %7.2f %10.1f\n", "per_row", p, ta, tseq / ta, tseq / ta / p, N / ta);
        printf("%-10s %4d %10.3f %9.2f %7.2f %10.1f\n", "per_image", p, tb, tseq / tb, tseq / tb / p, N / tb);
        fprintf(f, "per_row,%d,%.6f,%.3f,%.3f,%.2f,%s\n", p, ta, tseq / ta, tseq / ta / p, N / ta, ok ? "ok" : "FAIL");
        fprintf(f, "per_image,%d,%.6f,%.3f,%.3f,%.2f,%s\n", p, tb, tseq / tb, tseq / tb / p, N / tb, ok ? "ok" : "FAIL");
        fflush(stdout);
    }
    fclose(f);
    printf("\nresults written to %s\n", csv);
    for (int i = 0; i < POOL; i++) { xfree(pool[i]); xfree(ref[i]); xfree(chk[i]); }
    xfree(shared_out);
    return ok ? 0 : 2;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];
    int W = 4096, H = 4096, reps = -1, N = 2000, bsize = 1024, nthreads = 1;
    int pl[32], np = 0;
    const char *kname = NULL, *in_path = NULL, *out_path = NULL, *csv = NULL, *variant = "omp_rows";

    for (int i = 2; i < argc; i++) {
#define IS(s) (!strcmp(argv[i], s) && i + 1 < argc)
        if (IS("--size")) parse_size(argv[++i], &W, &H);
        else if (IS("--kernel")) kname = argv[++i];
        else if (IS("--threads")) { np = parse_list(argv[++i], pl, 32); nthreads = pl[0]; }
        else if (IS("--reps")) reps = atoi(argv[++i]);
        else if (IS("--csv")) csv = argv[++i];
        else if (IS("--in")) in_path = argv[++i];
        else if (IS("--out")) out_path = argv[++i];
        else if (IS("--variant")) variant = argv[++i];
        else if (IS("--n")) N = atoi(argv[++i]);
        else if (IS("--bsize")) bsize = atoi(argv[++i]);
        else if (IS("--tile")) { if (sscanf(argv[++i], "%dx%d", &g_tile_h, &g_tile_w) != 2 || g_tile_h < 1 || g_tile_w < 1) die("bad --tile (HxW)"); }
        else if (IS("--fuse-bh")) { g_fuse_bh = atoi(argv[++i]); if (g_fuse_bh < 1) die("bad --fuse-bh"); }
        else { fprintf(stderr, "unknown/incomplete option: %s\n", argv[i]); usage(); return 1; }
    }
    if (np == 0) default_threads(pl, &np);

    if (!strcmp(cmd, "gen")) {
        if (!out_path) die("gen needs --out");
        uint8_t *img = xalloc((size_t)W * H * 3);
        synth(img, W, H, 12345);
        write_ppm(out_path, img, W, H);
        printf("wrote %s (%dx%d)\n", out_path, W, H);
        return 0;
    }
    if (!strcmp(cmd, "filter")) {
        int km = parse_kernel_mask(kname ? kname : "blur");
        kernel_t k = K_BLUR;
        for (int j = 0; j < K_COUNT; j++) if (km == (1 << j)) k = (kernel_t)j;
        if (km == (1 << K_COUNT) - 1) die("filter needs one kernel, not 'all'");
        const variant_t *v = NULL;
        for (int j = 0; j < NV; j++) if (!strcmp(VARIANTS[j].name, variant)) v = &VARIANTS[j];
        if (!v) die("unknown --variant");
        if (v->blur_only && k != K_BLUR) die("that variant only supports --kernel blur");
        uint8_t *in;
        if (in_path) in = read_ppm(in_path, &W, &H);
        else { in = xalloc((size_t)W * H * 3); synth(in, W, H, 12345); }
        size_t bytes = (size_t)W * H * 3;
        uint8_t *out = xalloc(bytes);
        uint16_t *tmp = xalloc(bytes * sizeof(uint16_t));
        par_fill(out, bytes, 0); par_fill(tmp, bytes * sizeof(uint16_t), 0);
        omp_set_num_threads(v->parallel ? nthreads : 1);
        double t0 = omp_get_wtime();
        v->fn(k, in, out, tmp, W, H);
        double t = omp_get_wtime() - t0;
        printf("%s / %s on %dx%d with %d thread(s): %.4f s\n", KNAME[k], v->name, W, H,
               v->parallel ? nthreads : 1, t);
        if (out_path) { write_ppm(out_path, out, W, H); printf("wrote %s\n", out_path); }
        return 0;
    }
    if (!strcmp(cmd, "bench")) {
        return bench(W, H, parse_kernel_mask(kname ? kname : "all"), pl, np, reps > 0 ? reps : 3,
                     csv ? csv : "results_bench.csv");
    }
    if (!strcmp(cmd, "batch")) {
        int km = parse_kernel_mask(kname ? kname : "blur");
        kernel_t k = K_BLUR;
        for (int j = 0; j < K_COUNT; j++) if (km == (1 << j)) k = (kernel_t)j;
        if (km == (1 << K_COUNT) - 1) die("batch needs one kernel, not 'all'");
        if (N < 1 || bsize < 1) die("bad --n / --bsize");
        return batch(N, bsize, k, pl, np, reps > 0 ? reps : 2, csv ? csv : "results_batch.csv");
    }
    usage();
    return 1;
}
