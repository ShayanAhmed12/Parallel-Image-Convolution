/*
 * image.c -- memory, timing, and images.
 *
 * Nothing here is a filter. This file provides the supporting pieces:
 * aligned memory, a precise timer, reading and writing PPM image files,
 * the generated test image, and comparing two images byte by byte.
 *
 * Course concept: "first touch". On a multi-core machine the operating system
 * really allocates a memory page the first time it is written. first_touch()
 * writes every buffer once, in parallel and before any timing, so page faults
 * never land inside a measured region.
 */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <malloc.h>
#include <windows.h>
#endif
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pdc.h"

/* ------------------------------------------------------------------ */
/* errors and memory                                                   */
/* ------------------------------------------------------------------ */

void die(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}

void *xalloc(size_t bytes) {
    if (bytes == 0) bytes = 64;
    bytes = (bytes + 63) & ~(size_t)63;   /* round up to whole cache lines */
#ifdef _WIN32
    void *p = _aligned_malloc(bytes, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) p = NULL;
#endif
    if (!p) die("out of memory (try a smaller size)");
    return p;
}

void xfree(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

void first_touch(void *buf, size_t bytes, int value) {
    const size_t chunk = (size_t)1 << 20;   /* 1 MB pieces */
    ptrdiff_t chunks = (ptrdiff_t)((bytes + chunk - 1) / chunk);
    /* each thread writes its own share of the 1 MB pieces */
#pragma omp parallel for schedule(static)
    for (ptrdiff_t c = 0; c < chunks; c++) {
        size_t start = (size_t)c * chunk;
        size_t length = (bytes - start < chunk) ? bytes - start : chunk;
        memset((char *)buf + start, value, length);
    }
}

/* ------------------------------------------------------------------ */
/* timer                                                               */
/* ------------------------------------------------------------------ */
/* MinGW's omp_get_wtime() only ticks every 1 ms, too coarse for runs of
 * 0.1 s. On Windows we read QueryPerformanceCounter (100 ns) instead; on
 * other systems omp_get_wtime() is already precise. */
#ifdef _WIN32
static double qpc_period = 0;

double wtime(void) {
    LARGE_INTEGER now;
    if (qpc_period == 0) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        qpc_period = 1.0 / (double)frequency.QuadPart;
    }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * qpc_period;
}

double wtick(void) {
    if (qpc_period == 0) wtime();
    return qpc_period;
}
#else
double wtime(void) { return omp_get_wtime(); }
double wtick(void) { return omp_get_wtick(); }
#endif

/* ------------------------------------------------------------------ */
/* images                                                              */
/* ------------------------------------------------------------------ */

Image image_new(int width, int height) {
    Image img;
    size_t bytes = (size_t)width * height * 3;
    img.width = width;
    img.height = height;
    img.pixels = xalloc(bytes);
    first_touch(img.pixels, bytes, 0);
    return img;
}

void image_free(Image *img) {
    xfree(img->pixels);
    img->pixels = NULL;
}

static inline uint8_t clamp_to_byte(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* Small integer hash, used to add repeatable "noise" to the test image. */
static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* The generated test picture: colour gradient, inverted circle, light
 * rectangle, dark stripes and noise -- edges and textures for every filter. */
void image_make_test(Image *img, uint32_t seed) {
    int W = img->width, H = img->height;
    long cx = W / 2, cy = H / 2;
    long rad = (W < H ? W : H) / 4, r2 = rad * rad;
    int stripe = W / 48 + 1;
    /* each thread draws its own rows */
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
            uint8_t *p = img->pixels + ((size_t)y * W + x) * 3;
            p[0] = clamp_to_byte(r);
            p[1] = clamp_to_byte(g);
            p[2] = clamp_to_byte(b);
        }
    }
}

size_t image_count_differences(const Image *a, const Image *b) {
    size_t n = (size_t)a->width * a->height * 3, differences = 0;
    /* threads count their own share, OpenMP adds the counts together */
#pragma omp parallel for reduction(+ : differences) schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
        differences += (a->pixels[i] != b->pixels[i]);
    return differences;
}

/* ------------------------------------------------------------------ */
/* PPM files: "P6", width, height, 255, then the raw RGB bytes          */
/* ------------------------------------------------------------------ */

/* Read the next number in a PPM header, skipping spaces and # comments. */
static int read_header_number(FILE *f) {
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); }
        else if (c == EOF) return -1;
        else if (c >= '0' && c <= '9') break;
    }
    int value = 0;
    while (c >= '0' && c <= '9') { value = value * 10 + (c - '0'); c = fgetc(f); }
    return value;
}

Image image_read_ppm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open input image %s", path);
    if (fgetc(f) != 'P' || fgetc(f) != '6') die("input must be a binary PPM (P6)");
    int width = read_header_number(f), height = read_header_number(f);
    int maxval = read_header_number(f);
    if (width <= 0 || height <= 0 || maxval != 255) die("unsupported PPM header (need maxval 255)");
    Image img = image_new(width, height);
    size_t bytes = (size_t)width * height * 3;
    if (fread(img.pixels, 1, bytes, f) != bytes) die("truncated PPM file");
    fclose(f);
    return img;
}

void image_write_ppm(const char *path, const Image *img) {
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot open output file %s", path);
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->pixels, 1, (size_t)img->width * img->height * 3, f);
    fclose(f);
}
