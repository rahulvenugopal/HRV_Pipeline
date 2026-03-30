# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:51:05 2026
- Visualises group-level cardiac autonomic activity as Poincaré
ellipses using pre-computed MeanNN, SD1, SD2
- Individual subject ellipses drawn semi-transparently to reveal spread
- Outliers with SD1 or SD2 > Q3 + 1.5×IQR (per group-condition) are removed
  before plotting
  
Read the three papers below
- Brennan M et al. (2001) Do existing measures of Poincare plot geometry reflect
  nonlinear features of heart rate variability? IEEE Trans. Biomed. Eng.
- Guzik P et al. (2007) Correlations between Poincaré plot and conventional HRV
  measures for the same 1200-beat RR intervals. Folia Cardiologica.
- Tulppo MP et al. (1996) Quantitative beat-to-beat analysis of heart rate
  dynamics during exercise. Am J Physiol Heart Circ Physiol.
  
@author: Rahul Venugopal and Claude
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

import tkinter as tk
from tkinter import filedialog

import warnings
warnings.filterwarnings("ignore")

# Set folders
root = tk.Tk(); root.withdraw()
FILEPATH  = filedialog.askopenfilename(title="Select HRV CSV file",
                                       filetypes=[("CSV files", "*.csv")])
SAVE_PATH = filedialog.asksaveasfilename(title="Save output PNG as",
                                         defaultextension=".png",
                                         filetypes=[("PNG files", "*.png")])

# Groups and conditions — order controls subplot row / column positions
GROUPS     = ["NC", "TAU", "YTAU"]
CONDITIONS = ["PRE", "POST"]

# One colour per group (colour-blind distinguishable palette)
GROUP_COLORS = {
    "NC":   "#1976D2",   # blue
    "TAU":  "#D32F2F",   # red
    "YTAU": "#388E3C",   # green
}

# Tukey fence multiplier for IQR-based outlier exclusion
# 1.5 = standard "mild" outliers   |   3.0 = "extreme" outliers only
OUTLIER_MULTIPLIER = 1.5
OUTLIER_COLUMNS    = ("SD1", "SD2")

# Load the data
df = pd.read_csv(FILEPATH)

# Normalise column names and string labels to avoid whitespace / case issues
df.columns      = df.columns.str.strip()
df["Group"]     = df["Group"].str.strip().str.upper()
df["Condition"] = df["Condition"].str.strip().str.upper()

# Show the number of subjects in each sub groups-conditions
for grp in GROUPS:
    for cond in CONDITIONS:
        n = len(df[(df["Group"] == grp) & (df["Condition"] == cond)])
        if n > 0:
            print(f"  {grp:>5s} × {cond:<5s}: n = {n}")

# Any subject whose SD1 OR SD2 falls outside that fence is excluded.
mask_keep = pd.Series(True, index=df.index)   # start: keep everyone

for (grp, cond), sub in df.groupby(["Group", "Condition"]):
    for col in OUTLIER_COLUMNS:
        q1    = sub[col].quantile(0.25)
        q3    = sub[col].quantile(0.75)
        iqr   = q3 - q1
        lower = q1 - OUTLIER_MULTIPLIER * iqr
        upper = q3 + OUTLIER_MULTIPLIER * iqr

        # Boolean mask: True where value is outside the fence
        out_mask = (sub[col] < lower) | (sub[col] > upper)

        if out_mask.any():
            subjects = sub.loc[out_mask, "Subject_no"].tolist()
            vals     = sub.loc[out_mask, col].round(1).tolist()
            print(
                f"  {grp} × {cond} | {col}: "
                f"fence [{lower:.1f}, {upper:.1f}]  →  "
                f"removed subject(s) {subjects}  (values: {vals})"
            )
            # Flag the outlier rows for removal
            mask_keep.loc[sub[out_mask].index] = False

df_clean    = df[mask_keep].copy().reset_index(drop=True)
df_excluded = df[~mask_keep].copy().reset_index(drop=True)

# print selected subject count
print(f"  Retained : {len(df_clean)} / {len(df)} subjects")

# Setup a color function for neat plots
def _hex_to_rgba(hex_color, alpha):
    """Convert a hex colour string to an (R, G, B, A) tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b   = (int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)

# Compute shared axis limits across ALL group-condition combos ONCE
# so every subplot uses the same scale — enables direct visual comparison.
# Bounding-box half-extent for a 45°-rotated ellipse: Δ = √((SD1²+SD2²)/2)
half_extent_all = np.sqrt((df_clean["SD1"]**2 + df_clean["SD2"]**2) / 2)
GLOBAL_LO = max((df_clean["MeanNN"] - half_extent_all).min() - 40, 0)
GLOBAL_HI =     (df_clean["MeanNN"] + half_extent_all).max() + 40

#%% Set up the plot
fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(12, 12),
)
# hspace / wspace: vertical and horizontal gap between subplots
# as a fraction of the average subplot size — smaller = tighter
plt.subplots_adjust(hspace=0.25, wspace=0, top=1, bottom=0.06)

# Loop over every Group × Condition
for row_idx, group in enumerate(GROUPS):
    for col_idx, condition in enumerate(CONDITIONS):

        ax    = axes[row_idx, col_idx]
        color = GROUP_COLORS[group]
        
        if group == "NC" and condition == "POST":
            ax.set_visible(False)   # completely blank — no box, no text
            continue

        # Filter to this Group × Condition
        sub = df_clean[
            (df_clean["Group"] == group) & (df_clean["Condition"] == condition)
        ].copy()

        # Empty cell fallback (e.g. future groups with missing conditions)
        if sub.empty:
            ax.set_facecolor("#EEEEEE")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#999999", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{group} — {condition}", fontsize=11, color="#AAAAAA")
            continue

        # Shared axis limits — same across all subplots for direct comparison
        lo, hi = GLOBAL_LO, GLOBAL_HI

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        #Line of identity
        ax.plot([lo, hi], [lo, hi],
                color="#AAAAAA", linewidth=0.9, linestyle="--", zorder=1)

        # Individual subject ellipses (semi-transparent)
        for _, row in sub.iterrows():
            ind_ellipse = Ellipse(
                xy=(row["MeanNN"], row["MeanNN"]),
                width=2 * row["SD1"],    # full minor axis
                height=2 * row["SD2"],   # full major axis
                angle=45,
                linewidth=0.7,
                zorder=2,
            )
            ind_ellipse.set_facecolor(_hex_to_rgba(color, 0.12))
            ind_ellipse.set_edgecolor(_hex_to_rgba(color, 0.45))
            ax.add_patch(ind_ellipse)

        # Group-mean ellipse (bold, outline only)
        mn_grp  = sub["MeanNN"].mean()
        sd1_grp = sub["SD1"].mean()
        sd2_grp = sub["SD2"].mean()

        mean_ellipse = Ellipse(
            xy=(mn_grp, mn_grp),
            width=2 * sd1_grp,
            height=2 * sd2_grp,
            angle=45,
            linewidth=2.8,
            zorder=5,
        )
        mean_ellipse.set_facecolor(_hex_to_rgba(color, 0.0))   # transparent fill
        mean_ellipse.set_edgecolor(_hex_to_rgba(color, 1.0))   # solid bold edge
        ax.add_patch(mean_ellipse)

        # Dot at the group-mean centre (sits on the LoI)
        ax.plot(mn_grp, mn_grp, marker="o", color=color,
                markersize=5, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)

        # Annotation box (bottom-right corner)
        ax.text(
            0.97, 0.04,
            (
                f"$\\overline{{\\mathrm{{MeanNN}}}}$ = {mn_grp:.0f} ms\n"
                f"$\\overline{{\\mathrm{{SD1}}}}$   = {sd1_grp:.1f} ms\n"
                f"$\\overline{{\\mathrm{{SD2}}}}$   = {sd2_grp:.1f} ms"
            ),
            transform=ax.transAxes,
            fontsize=8,
            ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="#CCCCCC", alpha=0.88),
        )

        # Titles, labels, grid
        ax.set_title(
            f"{group} — {condition}  (n = {len(sub)})",
            fontsize=11, fontweight="bold", color=color, pad=6,
        )
        ax.set_xlabel("RR$_n$ (ms)", fontsize=9)
        ax.set_ylabel("RR$_{n+1}$ (ms)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)

legend_elements = []
for grp, col in GROUP_COLORS.items():
    legend_elements.append(
        mpatches.Patch(facecolor=col, edgecolor=col, alpha=0.3,
                       label=f"{grp} — individual subject")
    )
    legend_elements.append(
        Line2D([0], [0], color=col, linewidth=2.5, label=f"{grp} — group mean")
    )
legend_elements.append(
    Line2D([0], [0], color="#AAAAAA", linewidth=0.9,
           linestyle="--", label="Line of identity")
)

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor="#DDDDDD",
    bbox_to_anchor=(0.5, -0.05),
)

# Save the plot
fig.savefig(SAVE_PATH, dpi=600, bbox_inches="tight")
plt.close()