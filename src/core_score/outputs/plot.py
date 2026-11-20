"""Two plots per threshold: every complex as a dot ordered by its CORE Score, and the
distribution of CORE Scores."""

from pathlib import Path

FONT = "Optima"  # the paper's typeface; matplotlib's default is used where it is missing
INK, MUTED, LINE, DOT = "#16232b", "#5c6b73", "#c8ccce", "#2f6f8f"


def plot_core_scores(scores_by_threshold, output):
    """Write core_scores.png, every complex lowest first, and core_score_distribution.png, a
    histogram of the scores. Names end in the threshold when there is more than one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for threshold, values in scores_by_threshold.items():
        scores = sorted(value for value in values if value is not None)
        suffix = "" if len(scores_by_threshold) == 1 else f"_{threshold:g}A"
        for name, draw_plot in (("core_scores", draw), ("core_score_distribution", draw_distribution)):
            with plt.rc_context(style(matplotlib)):
                figure, axes = plt.subplots(figsize=(8, 5))
                draw_plot(axes, scores)
                figure.tight_layout()
                path = Path(output) / f"{name}{suffix}.png"
                figure.savefig(path, dpi=220)
                plt.close(figure)
            written.append(path)
    return written


def style(matplotlib):
    """Plain, light and airy: hairline axes, horizontal rules only, set in Optima if this
    machine has it."""
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    font = {"font.family": "sans-serif", "font.sans-serif": [FONT]} if FONT in available else {}
    return {
        **font, "font.size": 12,
        "text.color": INK, "axes.labelcolor": MUTED, "axes.labelsize": 12.5,
        "axes.edgecolor": LINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    }


def draw(axes, scores):
    from matplotlib.ticker import MaxNLocator

    count = len(scores)
    axes.scatter(range(1, count + 1), scores, s=dot_size(count), color=DOT,
                 alpha=0.75 if count > 500 else 1.0, linewidths=0, zorder=4)
    axes.set_ylim(0, 1.04)
    axes.set_xlim(0, count + 1)
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.grid(axis="y", color=LINE, linewidth=0.7, alpha=0.6, zorder=0)
    axes.set_axisbelow(True)
    axes.set_xlabel("Complexes, ordered by CORE Score", labelpad=10)
    axes.set_ylabel("CORE Score", labelpad=10)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.xaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    title(axes, complexes(count))


def draw_distribution(axes, scores):
    import numpy as np

    axes.hist(scores, bins=np.linspace(0, 1, 41), color=DOT, edgecolor="white", linewidth=0.6, zorder=4)
    axes.set_xlim(0, 1)
    axes.yaxis.get_major_locator().set_params(integer=True)
    axes.grid(axis="y", color=LINE, linewidth=0.7, alpha=0.6, zorder=0)
    axes.set_axisbelow(True)
    axes.set_xlabel("CORE Score", labelpad=10)
    axes.set_ylabel("Complexes", labelpad=10)
    title(axes, complexes(len(scores)))


def title(axes, subtitle):
    axes.set_title("CORE Score", loc="left", fontsize=17, pad=26, color=INK)
    axes.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                  fontsize=11.5, color=MUTED)


def complexes(count):
    return f"{count:,} complex{'' if count == 1 else 'es'}"


def dot_size(count):
    """Smaller dots the more complexes there are, so a big benchmark still reads as a curve."""
    return 26 if count <= 200 else (12 if count <= 1000 else 5)
