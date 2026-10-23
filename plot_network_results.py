#!/usr/bin/env python3
"""Plot the fixed n=1024 benchmark comparison from sibling JSON results.

Run: python3 plot_network_results.py
Writes network_thresholds.pdf (vector) and network_thresholds.png (300 dpi)
in the figures/ subdirectory. Requires NumPy and Matplotlib.
"""
from pathlib import Path
import json
import os
import tempfile

# Keep the font cache in a writable temporary directory on restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "positivity-paper-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "network_results.json").read_text())["results"]
families = ("cycle", "square_torus", "heterogeneous_ring")
labels = ("Cycle", "Grid", "Weighted ring")
quantities = ("H0_over_h_star", "H1_over_h_star", "H2_over_h_star")
colors = ("#8A8A8A", "#0072B2", "#E69F00")

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.1), sharey=True)
fig.subplots_adjust(left=0.103, right=0.993, bottom=0.20, top=0.78,
                    wspace=0.16)
positions = np.arange(len(families))
width = 0.22
for ax, theta in zip(axes, (0.5, 0.75)):
    selected = {row["family"]: row for row in data
                if row["n"] == 1024 and row["theta"] == theta}
    assert set(selected) == set(families)
    for offset, quantity, color in zip((-1, 0, 1), quantities, colors):
        values = [selected[family][quantity] for family in families]
        ax.bar(positions + offset * width, values, width=width,
               color=color, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axhline(1, color="black", linewidth=0.7, linestyle=(0, (4, 2)), zorder=4)
    ax.set_title(rf"$\theta={theta:g}$", pad=5)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.07)
    ax.set_xlim(-0.55, 2.55)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1), ("0", ".25", ".50", ".75", "1"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.45)
    ax.tick_params(axis="x", pad=3)
axes[0].set_ylabel("Fraction of reference threshold", labelpad=5, fontsize=8.5)

handles = [Patch(facecolor=color, edgecolor="none", label=label)
           for color, label in zip(colors, (r"$H_0/h_*$", r"$H_1/h_*$", r"$H_2/h_*$"))]
handles.append(Line2D([0], [0], color="black", linewidth=0.7,
                      linestyle=(0, (4, 2)), label="Reference"))
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 1.012),
           frameon=False, ncol=4, handlelength=1.8, columnspacing=1.5,
           handletextpad=0.5, borderaxespad=0)

destination = ROOT / "figures"
destination.mkdir(exist_ok=True)
fig.savefig(destination / "network_thresholds.pdf")
fig.savefig(destination / "network_thresholds.png", dpi=300)
plt.close(fig)
