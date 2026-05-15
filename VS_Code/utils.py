import time
import csv
import json
import logging
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
logging.getLogger("rasterio").setLevel(logging.ERROR)

#  palette
PASTEL = {
    "vegetation":     "#A8D5B5",
    "non_vegetation": "#F5C89A",
    "water_road":     "#A3C4E8",
    "accent":         "#C4A8D5",
    "neutral":        "#D4D0C8",
    "bg":             "#FAFAF8",
    "grid":           "#EDEBE5",
    "text":           "#3A3A38",
    "text_muted":     "#7A7A76",
    "blue":           "#A3C4E8",
    "green":          "#A8D5B5",
    "orange":         "#F5C89A",
    "purple":         "#C4A8D5",
}

# Shared matplotlib style
def _apply_base_style(ax, grid_axis="y"):
    ax.set_facecolor(PASTEL["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PASTEL["grid"])
    ax.spines["bottom"].set_color(PASTEL["grid"])
    ax.tick_params(colors=PASTEL["text_muted"], labelsize=9)
    ax.xaxis.label.set_color(PASTEL["text_muted"])
    ax.yaxis.label.set_color(PASTEL["text_muted"])
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=PASTEL["grid"], linewidth=0.8, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=PASTEL["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def pretty(self) -> str:
        return fmt_time(self.elapsed)

def log(msg: str):
    print(f"[INFO] {msg}")

def log_table(rows: list, headers: list, col_widths: list):
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))

def is_band_valid(band: np.ndarray, threshold: float = 1.0) -> bool:
    return float(np.std(band)) > threshold

# Elbow / silhouette plot
def save_elbow_plot(ks, wcss, sil, best_k, stem, plot_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                    facecolor="white", constrained_layout=True)

    for ax, y, ylabel, color in [
        (ax1, wcss, "WCSS (inertia)", PASTEL["blue"]),
        (ax2, sil,  "Silhouette score", PASTEL["green"]),
    ]:
        _apply_base_style(ax, grid_axis="y")
        ax.fill_between(ks, y, alpha=0.18, color=color, zorder=1)
        ax.plot(ks, y, "-o", color=color, linewidth=2, markersize=6,
                markerfacecolor="white", markeredgewidth=1.8,
                markeredgecolor=color, zorder=3)
        ki = ks.index(best_k)
        ax.plot(best_k, y[ki], "o", color=PASTEL["accent"],
                markersize=10, zorder=4, markeredgewidth=0)
        ax.axvline(best_k, color=PASTEL["accent"], linewidth=1.4,
                   linestyle="--", alpha=0.7, zorder=2,
                   label=f"selected k = {best_k}")
        ax.set_xlabel("number of clusters (k)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9, frameon=False,
                  labelcolor=PASTEL["text_muted"])

    fig.suptitle(f"optimal k selection — {stem}",
                 fontsize=12, color=PASTEL["text"], y=1.02)
    plt.savefig(Path(plot_dir) / f"{stem}_k_selection.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

# k-scores CSV
def save_k_scores_csv(ks, wcss, sil, best_k, stem, plot_dir):
    out = Path(plot_dir) / f"{stem}_k_scores.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "wcss", "silhouette", "selected"])
        for k, w, s in zip(ks, wcss, sil):
            writer.writerow([k, round(w, 4), round(s, 6),
                             "YES" if k == best_k else ""])

def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
