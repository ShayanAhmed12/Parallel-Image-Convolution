"""Charts drawn from the CSV files the engine writes.

Used by the app (gui/app.pyw) and by scripts/plot_results.py, so the report
and the app show the same pictures. Every function returns a matplotlib
Figure; nothing here runs the engine or does any timing.
"""
import csv
from collections import defaultdict

from matplotlib.figure import Figure

from .engine import METHODS

# what the methods are called on screen, e.g. "omp_tiled" -> "Tiled"
METHOD_LABELS = {code: name for code, name, _ in METHODS}
# one fixed colour per method, so a method looks the same on every chart
METHOD_COLORS = {"seq": "tab:gray", "omp_rows": "tab:blue", "omp_tiled": "tab:orange",
                 "seq_sep": "tab:purple", "omp_sep": "tab:green", "omp_sep_fused": "tab:red"}
STRATEGY_LABELS = {"per_row": "Threads share each image", "per_image": "One image per thread"}


def read_csv(path):
    """All rows of an engine CSV as a list of dicts (column name -> text)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _panel_title(row):
    if row["filter"] == "blur":
        return f"Blur {row['blur_size']}x{row['blur_size']}"
    return row["filter"].capitalize()


def _style(ax, xlabel, ylabel, xticks):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(xticks))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# metric -> (bench.csv column, y-axis label)
METRICS = {"time": ("time_s", "time (s)"),
           "speedup": ("speedup_vs_seq", "speedup (T1 / Tp)"),
           "efficiency": ("efficiency", "efficiency (speedup / p)")}


def _ideal_line(ax, metric, threads, methods=None):
    """Perfect scaling: time = T1 / p, speedup = p, efficiency = 1."""
    threads = sorted(threads)
    if metric == "time":
        if methods and "seq" in methods:
            t1 = methods["seq"][0][1]
            ax.plot(threads, [t1 / p for p in threads], "k--", lw=1, label="ideal (T1 / p)")
    elif metric == "speedup":
        ax.plot(threads, threads, "k--", lw=1, label="ideal")
    else:
        ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")


def speedup_chart(rows, metric="speedup"):
    """bench.csv: time, speedup or efficiency against threads, one panel per filter and blur size."""
    column, ylabel = METRICS[metric]
    panels = defaultdict(lambda: defaultdict(list))      # title -> method -> [(threads, value)]
    for r in rows:
        if r["method"] != "seq" or metric == "time":     # seq is the reference line on the time chart
            panels[_panel_title(r)][r["method"]].append((int(r["threads"]), float(r[column])))

    fig = Figure(figsize=(4.6 * max(1, len(panels)), 4.2))
    for i, (title, methods) in enumerate(panels.items()):
        ax = fig.add_subplot(1, len(panels), i + 1)
        threads = {t for points in methods.values() for t, _ in points}
        for method, points in methods.items():
            points.sort()
            label = METHOD_LABELS.get(method, method)
            if len(points) == 1:     # a one-thread method: a flat reference line
                ax.axhline(points[0][1], ls=":", lw=1.4, color=METHOD_COLORS.get(method), label=label)
            else:
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                        color=METHOD_COLORS.get(method), label=label)
        _ideal_line(ax, metric, threads, methods)
        ax.set_title(title)
        _style(ax, "threads", ylabel, threads)
    fig.tight_layout()
    return fig


def blur_size_chart(rows, threads):
    """blur_sizes.csv: speedup at `threads` threads against blur size, one line per method."""
    lines = defaultdict(list)                            # method -> [(blur size, speedup)]
    for r in rows:
        if r["method"] != "seq" and int(r["threads"]) == threads:
            lines[r["method"]].append((int(r["blur_size"]), float(r["speedup_vs_seq"])))
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    sizes = set()
    for method, points in lines.items():
        points.sort()
        sizes.update(s for s, _ in points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                label=METHOD_LABELS.get(method, method))
    ax.axhline(threads, color="k", ls="--", lw=1, label=f"ideal ({threads} threads)")
    ax.set_title(f"Speedup with {threads} threads as the blur window grows")
    _style(ax, "blur size (N for an N x N window)", "speedup (T1 / Tp)", sizes)
    fig.tight_layout()
    return fig


def schedule_chart(rows):
    """schedules.csv: Rows method, speedup against threads, one line per loop schedule."""
    lines = defaultdict(list)                            # schedule -> [(threads, speedup)]
    for r in rows:
        if r["method"] == "omp_rows":
            lines[r["schedule"]].append((int(r["threads"]), float(r["speedup_vs_seq"])))
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    threads = set()
    for schedule, points in lines.items():
        points.sort()
        threads.update(t for t, _ in points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=f"schedule({schedule})")
    _ideal_line(ax, "speedup", threads)
    ax.set_title("Rows method: static vs dynamic vs guided scheduling")
    _style(ax, "threads", "speedup (T1 / Tp)", threads)
    fig.tight_layout()
    return fig


def batch_chart(rows, metric="speedup"):
    """batch.csv: the two ways of using threads on many small images."""
    column = "speedup_vs_seq" if metric == "speedup" else "images_per_s"
    lines = defaultdict(list)                            # strategy -> [(threads, value)]
    sequential = None
    for r in rows:
        if r["strategy"] == "seq":
            sequential = float(r[column])
        else:
            lines[r["strategy"]].append((int(r["threads"]), float(r[column])))
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    threads = set()
    for strategy, points in lines.items():
        points.sort()
        threads.update(t for t, _ in points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                label=STRATEGY_LABELS.get(strategy, strategy))
    if metric == "speedup":
        _ideal_line(ax, "speedup", threads)
        ylabel = "speedup (T1 / Tp)"
    else:
        if sequential is not None:
            ax.axhline(sequential, color="k", ls=":", lw=1, label="sequential")
        ylabel = "images per second"
    ax.set_title("Many small images")
    _style(ax, "threads", ylabel, threads)
    fig.tight_layout()
    return fig
