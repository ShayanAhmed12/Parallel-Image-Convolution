"""Charts drawn from the CSV files the engine writes.

Used by the app (gui/app.pyw) and by scripts/plot_results.py, so the report
and the app show the same pictures. Every function returns a matplotlib
Figure; nothing here runs the engine or does any timing.
"""
import csv
from collections import defaultdict

from matplotlib.figure import Figure

from .engine import METHOD_NAMES

# one fixed colour per method, so a method looks the same on every chart
METHOD_COLORS = {"seq": "tab:gray", "omp_rows": "tab:blue", "omp_tiled": "tab:orange",
                 "seq_sep": "tab:purple", "omp_sep": "tab:green"}
STRATEGY_NAMES = {"seq": "Sequential", "per_row": "Threads share each image", "per_image": "One image per thread"}

# metric -> (CSV column, y-axis label)
METRICS = {"time": ("time_s", "time (s)"),
           "speedup": ("speedup_vs_seq", "speedup (T1 / Tp)"),
           "efficiency": ("efficiency", "efficiency (speedup / p)"),
           "images_per_s": ("images_per_s", "images per second")}


def read_csv(path):
    """All complete rows of an engine CSV, as dicts (column name -> text).
    A row still being written (the engine is running) has missing values and is left out."""
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if None not in row.values()]


def _lines(rows, key, x, y, keep=lambda row: True):
    """Group CSV rows into chart lines: {row[key]: [(x, y), ...] sorted by x}."""
    lines = defaultdict(list)
    for row in rows:
        if keep(row):
            lines[row[key]].append((int(row[x]), float(row[y])))
    return {name: sorted(points) for name, points in lines.items()}


def _draw(ax, lines, names, colors=None):
    """One line per entry; a line with a single point (a one-thread method) is drawn flat and dotted.
    Returns the x values used, for the axis ticks."""
    for name, points in lines.items():
        style = dict(label=names.get(name, name), color=(colors or {}).get(name))
        if len(points) == 1:
            ax.axhline(points[0][1], ls=":", lw=1.4, **style)
        else:
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", **style)
    return sorted({p[0] for points in lines.values() for p in points})


def _ideal(ax, metric, threads, t1=None):
    """Perfect scaling: time = T1 / p, speedup = p, efficiency = 1."""
    if metric == "speedup":
        ax.plot(threads, threads, "k--", lw=1, label="ideal")
    elif metric == "efficiency":
        ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")
    elif metric == "time" and t1:
        ax.plot(threads, [t1 / p for p in threads], "k--", lw=1, label="ideal (T1 / p)")


def _finish(ax, title, xlabel, ylabel, xticks):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _panel_title(row):
    if row["filter"] == "blur":
        return f"Blur {row['blur_size']}x{row['blur_size']}"
    return row["filter"].capitalize()


def speedup_chart(rows, metric="speedup"):
    """bench.csv: time, speedup or efficiency against threads, one panel per filter and blur size."""
    column, ylabel = METRICS[metric]
    panels = list(dict.fromkeys(_panel_title(r) for r in rows))      # in CSV order
    fig = Figure(figsize=(4.6 * max(1, len(panels)), 4.2))
    for i, title in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1)
        # seq is left out of speedup and efficiency (always 1), and is the reference line on the time chart
        lines = _lines(rows, "method", "threads", column,
                       lambda r: _panel_title(r) == title and (r["method"] != "seq" or metric == "time"))
        threads = _draw(ax, lines, METHOD_NAMES, METHOD_COLORS)
        _ideal(ax, metric, threads, lines.get("seq", [(1, None)])[0][1])
        _finish(ax, title, "threads", ylabel, threads)
    fig.tight_layout()
    return fig


def blur_size_chart(rows, threads):
    """blur_sizes.csv: speedup at `threads` threads against blur size, one line per method."""
    lines = _lines(rows, "method", "blur_size", "speedup_vs_seq",
                   lambda r: r["method"] != "seq" and int(r["threads"]) == threads)
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    sizes = _draw(ax, lines, METHOD_NAMES, METHOD_COLORS)
    ax.axhline(threads, color="k", ls="--", lw=1, label=f"ideal ({threads} threads)")
    _finish(ax, f"Speedup with {threads} threads as the blur window grows",
            "blur size (N for an N x N window)", "speedup (T1 / Tp)", sizes)
    fig.tight_layout()
    return fig


def schedule_chart(rows):
    """schedules.csv: Rows method, speedup against threads, one line per loop schedule."""
    lines = _lines(rows, "schedule", "threads", "speedup_vs_seq", lambda r: r["method"] == "omp_rows")
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    threads = _draw(ax, lines, {s: f"schedule({s})" for s in lines})
    _ideal(ax, "speedup", threads)
    _finish(ax, "Rows method: static vs dynamic vs guided scheduling", "threads", "speedup (T1 / Tp)", threads)
    fig.tight_layout()
    return fig


def batch_chart(rows, metric="speedup"):
    """batch.csv: the two ways of using threads on many small images."""
    column, ylabel = METRICS[metric]
    # on the images-per-second chart, the sequential rate is a flat reference line
    lines = _lines(rows, "strategy", "threads", column, lambda r: r["strategy"] != "seq" or metric != "speedup")
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    threads = _draw(ax, lines, STRATEGY_NAMES, {"seq": "k"})
    _ideal(ax, metric, threads)
    _finish(ax, "Many small images", "threads", ylabel, threads)
    fig.tight_layout()
    return fig
