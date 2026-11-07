/*
 * filters.c -- the maths of the three filters, with no parallelism.
 *
 *   blur    : Gaussian blur, a size x size window of binomial weights
 *             (5x5 uses 1 4 6 4 1 in each direction)
 *   sharpen : 3x3 window  [ 0 -1  0 ; -1  5 -1 ;  0 -1  0 ]
 *   sobel   : edge detection on brightness, |gx| + |gy|, grey output
 *
 * Course concept: convolution. Every output pixel is a weighted sum of the
 * input pixels in a small window around it. Output pixels never depend on
 * each other, so any split of the output between threads gives the same
 * result -- this is what makes the problem easy to parallelise.
 *
 * Only whole numbers are used (no floating point), so every method gives
 * exactly the same bytes and results can be compared with no tolerance.
 *
 * Borders: pixels outside the image are treated as copies of the nearest
 * edge pixel ("clamp to edge").
 */
#include <stdlib.h>
#include <string.h>

#include "pdc.h"

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline uint8_t clamp_to_byte(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* ------------------------------------------------------------------ */
/* filter settings                                                     */
/* ------------------------------------------------------------------ */

Filter filter_make(FilterKind kind, int blur_size) {
    Filter f;
    memset(&f, 0, sizeof f);
    f.kind = kind;
    if (kind != FILTER_BLUR) {
        f.size = 3;
        f.radius = 1;
        return f;
    }
    if (blur_size < BLUR_SIZE_MIN || blur_size > BLUR_SIZE_MAX || blur_size % 2 == 0)
        die("blur size must be an odd number from %d to %d", BLUR_SIZE_MIN, BLUR_SIZE_MAX);
    f.size = blur_size;
    f.radius = blur_size / 2;
    /* binomial weights: row (size-1) of Pascal's triangle, e.g. 1 4 6 4 1.
     * C(n,i) = C(n,i-1) * (n-i+1) / i is always a whole number; the largest
     * (size 25: C(24,12) = 2704156) fits easily in 32 bits. */
    f.weights[0] = 1;
    for (int i = 1; i < f.size; i++)
        f.weights[i] = f.weights[i - 1] * (uint32_t)(f.size - i) / (uint32_t)i;
    /* one row sums to 2^(size-1), the whole size x size window to 2^(2(size-1)) */
    f.shift = 2 * (f.size - 1);
    return f;
}

const char *filter_name(FilterKind kind) {
    switch (kind) {
    case FILTER_BLUR:    return "blur";
    case FILTER_SHARPEN: return "sharpen";
    case FILTER_SOBEL:   return "sobel";
    }
    return "?";
}

int filter_from_name(const char *name, FilterKind *out) {
    if (!strcmp(name, "blur"))    { *out = FILTER_BLUR;    return 1; }
    if (!strcmp(name, "sharpen")) { *out = FILTER_SHARPEN; return 1; }
    if (!strcmp(name, "sobel"))   { *out = FILTER_SOBEL;   return 1; }
    return 0;
}


/* ------------------------------------------------------------------ */
/* one output pixel                                                    */
/* p points at the top-left pixel of the window, stride is the number  */
/* of bytes from one window row to the next, o receives 3 bytes.       */
/* ------------------------------------------------------------------ */

static inline void blur_pixel(const uint8_t *p, ptrdiff_t stride, uint8_t *o,
                              int size, const uint32_t *w, int shift) {
    const uint64_t half = (uint64_t)1 << (shift - 1);   /* for rounding to nearest */
    for (int c = 0; c < 3; c++) {                        /* red, green, blue */
        uint64_t total = 0;
        for (int j = 0; j < size; j++) {
            const uint8_t *row = p + j * stride + c;
            uint32_t row_sum = 0;
            for (int i = 0; i < size; i++) row_sum += w[i] * row[i * 3];
            total += (uint64_t)w[j] * row_sum;
        }
        o[c] = (uint8_t)((total + half) >> shift);
    }
}

static inline void sharpen_pixel(const uint8_t *p, ptrdiff_t stride, uint8_t *o) {
    for (int c = 0; c < 3; c++) {
        int centre = p[stride + 3 + c];
        int up = p[3 + c], down = p[2 * stride + 3 + c];
        int left = p[stride + c], right = p[stride + 6 + c];
        o[c] = clamp_to_byte(5 * centre - up - down - left - right);
    }
}

/* brightness of one RGB pixel, 0..255 (weights 0.30, 0.59, 0.11 times 256) */
static inline int luma(const uint8_t *q) {
    return (77 * q[0] + 150 * q[1] + 29 * q[2]) >> 8;
}

static inline void sobel_pixel(const uint8_t *p, ptrdiff_t stride, uint8_t *o) {
    /* a b c        gx = right column - left column  (weights 1 2 1)
     * d . f        gy = bottom row   - top row      (weights 1 2 1)
     * g h i */
    int a = luma(p), b = luma(p + 3), c = luma(p + 6);
    int d = luma(p + stride), f = luma(p + stride + 6);
    int g = luma(p + 2 * stride), h = luma(p + 2 * stride + 3), i = luma(p + 2 * stride + 6);
    int gx = (c + 2 * f + i) - (a + 2 * d + g);
    int gy = (g + 2 * h + i) - (a + 2 * b + c);
    int magnitude = abs(gx) + abs(gy);
    o[0] = o[1] = o[2] = (uint8_t)(magnitude > 255 ? 255 : magnitude);
}

/* ------------------------------------------------------------------ */
/* borders                                                             */
/* ------------------------------------------------------------------ */

/* Near the border part of the window lies outside the image. Copy the window
 * into a small buffer, replacing outside pixels by the nearest edge pixel,
 * so the same pixel functions work unchanged. */
static void gather_window(const Image *in, int x, int y, int radius, uint8_t *window) {
    int n = 2 * radius + 1;
    for (int j = 0; j < n; j++) {
        int yy = clamp_int(y - radius + j, 0, in->height - 1);
        for (int i = 0; i < n; i++) {
            int xx = clamp_int(x - radius + i, 0, in->width - 1);
            const uint8_t *src = in->pixels + ((size_t)yy * in->width + xx) * 3;
            uint8_t *dst = window + (j * n + i) * 3;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
        }
    }
}

/* In row y, find the columns [*xa, *xb) inside [x0, x1) whose whole window
 * lies inside the image. Columns before xa and from xb on are border pixels. */
static void interior_columns(int y, int r, int W, int H, int x0, int x1, int *xa, int *xb) {
    if (y < r || y >= H - r) {      /* the whole row is near the top or bottom edge */
        *xa = *xb = x1;
        return;
    }
    *xa = clamp_int(r, x0, x1);
    *xb = clamp_int(W - r, *xa, x1);
}

/* ------------------------------------------------------------------ */
/* a box of output pixels                                              */
/*                                                                     */
/* Each filter has its own loop with the same three parts per row:     */
/* left border pixels, interior pixels (read straight from the image), */
/* right border pixels (read from a gathered window).                  */
/*                                                                     */
/* Settings are copied into local variables first. The output is       */
/* written through a byte pointer, and C lets a byte pointer point at  */
/* anything, so otherwise the compiler would have to re-read the       */
/* settings after every byte written -- several times slower.          */
/* ------------------------------------------------------------------ */

/* Blur works on CHUNK bytes of a row at a time: the temporary sums then fit
 * in the fastest cache (12 KB), and the inner loops run over consecutive
 * bytes, which lets the compiler use SIMD instructions. */
#define CHUNK 1024

/* sums[k] = w[0]*p[k] + w[1]*p[k+3] + ... + w[size-1]*p[k+3*(size-1)], k < n.
 * p[k] and p[k+3] are the same colour in neighbouring pixels, so this blurs
 * n bytes of one row sideways. Loop order "each weight, then each byte"
 * keeps the inner loop over consecutive bytes. */
static inline void weighted_row_sums(const uint8_t *p, int n, const uint32_t *w, int size, uint32_t *sums) {
    for (int k = 0; k < n; k++) sums[k] = 0;
    for (int i = 0; i < size; i++) {
        const uint8_t *q = p + 3 * i;
        for (int k = 0; k < n; k++) sums[k] += w[i] * q[k];
    }
}

/* Interior pixels [xa, xb) of output row y:
 * out = (sum over window rows j of w[j] * (horizontal sum of row j)) / 2^shift */
static void blur_interior(const uint8_t *src, ptrdiff_t stride, uint8_t *out_row, int y, int xa, int xb,
                          int r, int size, const uint32_t *w, int shift) {
    const uint64_t half = (uint64_t)1 << (shift - 1);
    uint32_t sums[CHUNK];
    uint64_t total[CHUNK];
    for (int start = xa * 3; start < xb * 3; start += CHUNK) {
        int n = (xb * 3 - start < CHUNK) ? xb * 3 - start : CHUNK;
        for (int k = 0; k < n; k++) total[k] = 0;
        for (int j = 0; j < size; j++) {
            const uint8_t *window_row = src + (size_t)(y - r + j) * stride + start - r * 3;
            weighted_row_sums(window_row, n, w, size, sums);
            for (int k = 0; k < n; k++) total[k] += (uint64_t)w[j] * sums[k];
        }
        for (int k = 0; k < n; k++) out_row[start + k] = (uint8_t)((total[k] + half) >> shift);
    }
}

static void blur_box(const Filter *f, const Image *in, Image *out, int y0, int y1, int x0, int x1) {
    const int W = in->width, H = in->height, r = f->radius, size = f->size, shift = f->shift;
    const ptrdiff_t stride = (ptrdiff_t)W * 3;
    const uint8_t *src = in->pixels;
    uint8_t *dst = out->pixels;
    uint32_t w[BLUR_SIZE_MAX];
    memcpy(w, f->weights, sizeof w);
    uint8_t window[BLUR_SIZE_MAX * BLUR_SIZE_MAX * 3];

    for (int y = y0; y < y1; y++) {
        uint8_t *out_row = dst + (size_t)y * stride;
        int xa, xb;
        interior_columns(y, r, W, H, x0, x1, &xa, &xb);
        for (int x = x0; x < xa; x++) {
            gather_window(in, x, y, r, window);
            blur_pixel(window, size * 3, out_row + x * 3, size, w, shift);
        }
        blur_interior(src, stride, out_row, y, xa, xb, r, size, w, shift);
        for (int x = xb; x < x1; x++) {
            gather_window(in, x, y, r, window);
            blur_pixel(window, size * 3, out_row + x * 3, size, w, shift);
        }
    }
}

static void sharpen_box(const Image *in, Image *out, int y0, int y1, int x0, int x1) {
    const int W = in->width, H = in->height;
    const ptrdiff_t stride = (ptrdiff_t)W * 3;
    const uint8_t *src = in->pixels;
    uint8_t *dst = out->pixels;
    uint8_t window[3 * 3 * 3];

    for (int y = y0; y < y1; y++) {
        uint8_t *out_row = dst + (size_t)y * stride;
        int xa, xb;
        interior_columns(y, 1, W, H, x0, x1, &xa, &xb);
        for (int x = x0; x < xa; x++) {
            gather_window(in, x, y, 1, window);
            sharpen_pixel(window, 9, out_row + x * 3);
        }
        for (int x = xa; x < xb; x++)
            sharpen_pixel(src + (size_t)(y - 1) * stride + (size_t)(x - 1) * 3, stride, out_row + x * 3);
        for (int x = xb; x < x1; x++) {
            gather_window(in, x, y, 1, window);
            sharpen_pixel(window, 9, out_row + x * 3);
        }
    }
}

static void sobel_box(const Image *in, Image *out, int y0, int y1, int x0, int x1) {
    const int W = in->width, H = in->height;
    const ptrdiff_t stride = (ptrdiff_t)W * 3;
    const uint8_t *src = in->pixels;
    uint8_t *dst = out->pixels;
    uint8_t window[3 * 3 * 3];

    for (int y = y0; y < y1; y++) {
        uint8_t *out_row = dst + (size_t)y * stride;
        int xa, xb;
        interior_columns(y, 1, W, H, x0, x1, &xa, &xb);
        for (int x = x0; x < xa; x++) {
            gather_window(in, x, y, 1, window);
            sobel_pixel(window, 9, out_row + x * 3);
        }
        for (int x = xa; x < xb; x++)
            sobel_pixel(src + (size_t)(y - 1) * stride + (size_t)(x - 1) * 3, stride, out_row + x * 3);
        for (int x = xb; x < x1; x++) {
            gather_window(in, x, y, 1, window);
            sobel_pixel(window, 9, out_row + x * 3);
        }
    }
}

void filter_box(const Filter *f, const Image *in, Image *out, int y0, int y1, int x0, int x1) {
    switch (f->kind) {
    case FILTER_BLUR:    blur_box(f, in, out, y0, y1, x0, x1); break;
    case FILTER_SHARPEN: sharpen_box(in, out, y0, y1, x0, x1); break;
    case FILTER_SOBEL:   sobel_box(in, out, y0, y1, x0, x1); break;
    }
}

/* ------------------------------------------------------------------ */
/* separable blur: a 2-D binomial blur equals a horizontal 1-D blur     */
/* followed by a vertical one (size + size multiplies per pixel instead */
/* of size * size). The sums are kept exact, so the result is identical */
/* to filter_box.                                                       */
/* ------------------------------------------------------------------ */

/* one border pixel of the horizontal pass: neighbours outside the row are
 * replaced by the nearest edge pixel */
static void horizontal_border_pixel(const uint8_t *src_row, uint32_t *dst_row, int x, int width,
                                    int r, int size, const uint32_t *w) {
    for (int c = 0; c < 3; c++) {
        uint32_t sum = 0;
        for (int i = 0; i < size; i++) sum += w[i] * src_row[clamp_int(x - r + i, 0, width - 1) * 3 + c];
        dst_row[x * 3 + c] = sum;
    }
}

void blur_row_horizontal(const Filter *f, const uint8_t *src_row, uint32_t *dst_row, int width) {
    const int r = f->radius, size = f->size;
    uint32_t w[BLUR_SIZE_MAX];
    memcpy(w, f->weights, sizeof w);
    int xa = r < width ? r : width;                 /* interior pixels are [xa, xb) */
    int xb = width - r > xa ? width - r : xa;
    for (int x = 0; x < xa; x++) horizontal_border_pixel(src_row, dst_row, x, width, r, size, w);
    for (int start = xa * 3; start < xb * 3; start += CHUNK) {
        int n = (xb * 3 - start < CHUNK) ? xb * 3 - start : CHUNK;
        weighted_row_sums(src_row + start - r * 3, n, w, size, dst_row + start);
    }
    for (int x = xb; x < width; x++) horizontal_border_pixel(src_row, dst_row, x, width, r, size, w);
}

void blur_row_vertical(const Filter *f, const uint32_t *const *rows, uint8_t *dst_row, int width) {
    const int size = f->size, shift = f->shift;
    const uint64_t half = (uint64_t)1 << (shift - 1);
    uint32_t w[BLUR_SIZE_MAX];
    memcpy(w, f->weights, sizeof w);
    uint64_t total[CHUNK];
    for (int start = 0; start < width * 3; start += CHUNK) {
        int n = (width * 3 - start < CHUNK) ? width * 3 - start : CHUNK;
        for (int k = 0; k < n; k++) total[k] = 0;
        for (int j = 0; j < size; j++) {
            const uint32_t *row = rows[j] + start;
            for (int k = 0; k < n; k++) total[k] += (uint64_t)w[j] * row[k];
        }
        for (int k = 0; k < n; k++) dst_row[start + k] = (uint8_t)((total[k] + half) >> shift);
    }
}
