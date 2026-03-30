# -*- coding: utf-8 -*-
"""
Poincaré Plot Bootstrap Variability — Group-Level Cardiac Autonomic Activity
=============================================================================

PURPOSE
-------
Visualises the UNCERTAINTY of the group-mean Poincaré ellipse using
non-parametric case bootstrap resampling (subjects as the unit of resampling).

For each Group × Condition panel, three visual layers are drawn on top of
each other:
  Layer 1 — Ghost ellipses : 300 of the 1000 bootstrapped mean ellipses,
             each drawn at very low opacity, forming a density cloud that
             shows where the mean ellipse could plausibly fall.
  Layer 2 — 95 % CI band  : a shaded ring between the 2.5th and 97.5th
             percentile of radial distance (from the observed mean centre)
             computed across ALL 1000 bootstrap ellipses at 360 angles.
             This is the primary inferential visual.
  Layer 3 — Observed mean  : the same bold mean ellipse as in the original
             averaging script, so both plots can be compared directly.

WHY BOOTSTRAP (not parametric CI)?
-----------------------------------
SD1 and SD2 are not guaranteed to be normally distributed, especially with
the small-to-moderate sample sizes typical in psychophysiology studies
(n = 14-24 here). Bootstrap percentile intervals make no distributional
assumption — they let the data speak for themselves.

For small n, a parametric bootstrap could alternatively be preferred
(Wikipedia — Bootstrapping, statistics, 2026).

KEY REFERENCES
--------------
Poincare ellipse geometry:
  Tulppo MP et al. (1996) Quantitative beat-to-beat analysis of heart rate
    dynamics during exercise. Am J Physiol Heart Circ Physiol.
  Brennan M et al. (2001) Do existing measures of Poincare plot geometry
    reflect nonlinear features of HRV? IEEE Trans. Biomed. Eng.
  Guzik P et al. (2007) Correlations between Poincare plot and conventional
    HRV measures. Folia Cardiologica.

Bootstrap confidence intervals (methodology):
  Efron B & Tibshirani RJ (1993) An Introduction to the Bootstrap.
    Chapman & Hall/CRC.
    The foundational textbook. The percentile interval used here is
    described in Ch. 13.
  Hesterberg TC (2015) What teachers should know about the bootstrap:
    resampling in the undergraduate statistics curriculum.
    Am. Statistician, 69(4), 371-386. PMC4784504.
    Discusses when percentile bootstrap works and its limitations.
  Efron B (1987) Better bootstrap confidence intervals.
    J. Am. Stat. Assoc., 82(397), 171-185.
    Introduces BCa — a more accurate alternative for skewed distributions.

NOTE: No published paper appears to have applied bootstrap CIs specifically
to Poincare plot ellipse variability across subjects at group level. This
script operationalises the general bootstrap percentile interval framework
(Efron 1993) in that specific context.

@author: Rahul Venugopal and Claude
"""

# =============================================================================
# 0.  IMPORTS
# =============================================================================
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

# =============================================================================
# 1.  FILE SELECTION  (GUI dialogs — same as original script)
# =============================================================================
root = tk.Tk()
root.withdraw()   # hide the empty Tk window

FILEPATH = filedialog.askopenfilename(
    title="Select HRV CSV file",
    filetypes=[("CSV files", "*.csv")]
)

SAVE_PATH = filedialog.asksaveasfilename(
    title="Save output PNG as",
    defaultextension=".png",
    filetypes=[("PNG files", "*.png")]
)

# =============================================================================
# 2.  STUDY-DESIGN CONSTANTS
# =============================================================================
# Order controls subplot row / column positions  (row = group, col = condition)
GROUPS     = ["NC", "TAU", "YTAU"]
CONDITIONS = ["PRE", "POST"]

# One colour per group — colour-blind distinguishable palette
GROUP_COLORS = {
    "NC":   "#1976D2",   # blue
    "TAU":  "#D32F2F",   # red
    "YTAU": "#388E3C",   # green
}

# =============================================================================
# 3.  BOOTSTRAP SETTINGS
# =============================================================================
N_BOOT   = 1000   # number of bootstrap resamples per group-condition

# Percentile bounds for the CI band:  2.5 / 97.5  ->  95 % interval
CI_LO    = 2.5
CI_HI    = 97.5

# Angular resolution of the CI band ring
# 360 gives one point per degree around the full ellipse
N_ANGLES = 360

# Number of ghost ellipses to draw (a subset of N_BOOT avoids overplotting)
N_GHOSTS = 300

# Fixed random seed — guarantees the same figure on every run
RNG = np.random.default_rng(seed=42)

# =============================================================================
# 4.  OUTLIER REMOVAL SETTINGS  (identical to original script)
# =============================================================================
# Tukey IQR fence: remove subjects whose SD1 OR SD2 exceeds Q3 + 1.5*IQR
# The fence is computed separately inside each group-condition cell so that
# a group with naturally higher HRV does not inflate the fence for another.
OUTLIER_MULTIPLIER = 1.5
OUTLIER_COLUMNS    = ("SD1", "SD2")

# =============================================================================
# 5.  LOAD & NORMALISE DATA
# =============================================================================
df = pd.read_csv(FILEPATH)

# Strip whitespace from column names and string columns to prevent silent
# key-mismatch errors (e.g. "Group " != "Group")
df.columns      = df.columns.str.strip()
df["Group"]     = df["Group"].str.strip().str.upper()
df["Condition"] = df["Condition"].str.strip().str.upper()

print("\nRaw subject counts per group x condition:")
for grp in GROUPS:
    for cond in CONDITIONS:
        n = len(df[(df["Group"] == grp) & (df["Condition"] == cond)])
        if n > 0:
            print(f"  {grp:>5s} x {cond:<5s}: n = {n}")

# =============================================================================
# 6.  OUTLIER REMOVAL  (Tukey IQR fence, per group-condition cell)
# =============================================================================
# Start with a boolean Series that keeps every row; we will flip outliers False
mask_keep = pd.Series(True, index=df.index)

print("\nOutlier removal:")
for (grp, cond), sub in df.groupby(["Group", "Condition"]):
    for col in OUTLIER_COLUMNS:
        q1  = sub[col].quantile(0.25)
        q3  = sub[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_MULTIPLIER * iqr
        upper = q3 + OUTLIER_MULTIPLIER * iqr

        # True where the value VIOLATES the fence
        out_mask = (sub[col] < lower) | (sub[col] > upper)

        if out_mask.any():
            subjects = sub.loc[out_mask, "Subject_no"].tolist()
            vals     = sub.loc[out_mask, col].round(1).tolist()
            print(
                f"  {grp} x {cond} | {col}: fence [{lower:.1f}, {upper:.1f}]"
                f"  ->  removed {subjects}  (values: {vals})"
            )
            mask_keep.loc[sub[out_mask].index] = False

df_clean = df[mask_keep].copy().reset_index(drop=True)
print(f"\n  Retained: {len(df_clean)} / {len(df)} subjects\n")

# =============================================================================
# 7.  GLOBAL AXIS LIMITS
#     Same scale across ALL subplots so panels can be compared directly.
# =============================================================================
# For a 45-degree-rotated ellipse the axis-aligned bounding-box half-extent
# is sqrt((SD1^2 + SD2^2) / 2).  Add 40 ms padding on each side.
half_ext  = np.sqrt((df_clean["SD1"] ** 2 + df_clean["SD2"] ** 2) / 2)
GLOBAL_LO = max((df_clean["MeanNN"] - half_ext).min() - 40, 0)
GLOBAL_HI =     (df_clean["MeanNN"] + half_ext).max() + 40

# =============================================================================
# 8.  COLOUR HELPER  (hex string -> matplotlib RGBA tuple)
# =============================================================================
# matplotlib accepts (R, G, B, A) tuples where each component is 0-1.
# This small inline conversion is used repeatedly below.
def _hex_to_rgba(hex_color, alpha):
    h       = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)

# =============================================================================
# 9.  FIGURE SETUP
# =============================================================================
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
plt.subplots_adjust(hspace=0.25, wspace=0, top=1, bottom=0.08)

# =============================================================================
# 10.  MAIN LOOP — one subplot per Group x Condition
# =============================================================================
for row_idx, group in enumerate(GROUPS):
    for col_idx, condition in enumerate(CONDITIONS):

        ax    = axes[row_idx, col_idx]
        color = GROUP_COLORS[group]

        # NC has no POST condition in this study design — hide that cell
        if group == "NC" and condition == "POST":
            ax.set_visible(False)
            continue

        # Filter to this Group x Condition
        sub = df_clean[
            (df_clean["Group"] == group) &
            (df_clean["Condition"] == condition)
        ].copy()

        # If no data remain after outlier removal, show a placeholder
        if sub.empty:
            ax.set_facecolor("#EEEEEE")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=11, color="#999999", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{group} - {condition}", fontsize=11, color="#AAAAAA")
            continue

        n_sub = len(sub)   # number of subjects in this cell

        # ------------------------------------------------------------------
        # AXIS COSMETICS
        # ------------------------------------------------------------------
        lo, hi = GLOBAL_LO, GLOBAL_HI
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)

        # Line of identity (diagonal where RR_n == RR_n+1)
        ax.plot([lo, hi], [lo, hi],
                color="#AAAAAA", linewidth=0.9, linestyle="--", zorder=1)

        # ------------------------------------------------------------------
        # EXTRACT ARRAYS FOR THIS CELL
        # ------------------------------------------------------------------
        mn_arr  = sub["MeanNN"].values   # (n_sub,)
        sd1_arr = sub["SD1"].values
        sd2_arr = sub["SD2"].values

        # Observed (non-bootstrapped) group means — used as the reference
        mn_obs  = mn_arr.mean()
        sd1_obs = sd1_arr.mean()
        sd2_obs = sd2_arr.mean()

        # ==================================================================
        # STEP 1 — PARAMETRIC CASE BOOTSTRAP
        # ==================================================================
        # "Case bootstrap" resamples whole subjects (rows) with replacement.
        # Each of the N_BOOT resamples has the same size as the original
        # group (n_sub) — some subjects may appear multiple times, others
        # not at all, which is what creates the variability.
        #
        # For each resample b we compute the group mean of MeanNN, SD1, SD2.
        # This gives three arrays of length N_BOOT, each representing a
        # distribution of plausible group-mean values.
        #
        # All N_BOOT resamples are generated in one vectorised call:
        #   boot_idx shape = (N_BOOT, n_sub)
        #   each row is one resample of n_sub subject indices drawn 0..n_sub-1
        # ------------------------------------------------------------------

        boot_idx = RNG.integers(0, n_sub, size=(N_BOOT, n_sub))
        #   Shape (N_BOOT, n_sub) — integer indices into the subject arrays

        # Fancy-index each array then average along the subject axis (axis=1)
        # Result shape for each: (N_BOOT,)
        boot_mn  = mn_arr[boot_idx].mean(axis=1)
        boot_sd1 = sd1_arr[boot_idx].mean(axis=1)
        boot_sd2 = sd2_arr[boot_idx].mean(axis=1)

        # 95 % percentile CI for each scalar parameter (used in annotation)
        sd1_ci_lo = np.percentile(boot_sd1, CI_LO)
        sd1_ci_hi = np.percentile(boot_sd1, CI_HI)
        sd2_ci_lo = np.percentile(boot_sd2, CI_LO)
        sd2_ci_hi = np.percentile(boot_sd2, CI_HI)
        mn_ci_lo  = np.percentile(boot_mn,  CI_LO)
        mn_ci_hi  = np.percentile(boot_mn,  CI_HI)

        print(
            f"  {group} x {condition} (n={n_sub})"
            f" | MeanNN [{mn_ci_lo:.0f}-{mn_ci_hi:.0f}] ms"
            f"  SD1 [{sd1_ci_lo:.1f}-{sd1_ci_hi:.1f}]"
            f"  SD2 [{sd2_ci_lo:.1f}-{sd2_ci_hi:.1f}]"
        )

        # ==================================================================
        # STEP 2 — LAYER 1: GHOST ELLIPSES  (bootstrap density cloud)
        # ==================================================================
        # Draw N_GHOSTS bootstrap ellipses at very low alpha (fill only).
        # Where many ghost ellipses overlap the colour darkens, giving an
        # intuitive density impression — the darker the region, the more
        # likely the true group mean ellipse falls there.
        #
        # N_GHOSTS < N_BOOT avoids full saturation from too many patches.
        # ------------------------------------------------------------------

        ghost_idx = RNG.choice(N_BOOT, size=min(N_GHOSTS, N_BOOT), replace=False)

        for b in ghost_idx:
            ghost_e = Ellipse(
                xy=(boot_mn[b], boot_mn[b]),   # centre on the Line of Identity
                width=2 * boot_sd1[b],          # full minor axis (Brennan 2001)
                height=2 * boot_sd2[b],         # full major axis
                angle=45,                        # Poincare convention (Tulppo 1996)
                linewidth=0,                     # no visible edge — fill only
                zorder=2,
            )
            ghost_e.set_facecolor(_hex_to_rgba(color, 0.04))
            ax.add_patch(ghost_e)

        # ==================================================================
        # STEP 3 — LAYER 2: 95 % CI BAND  (radial percentile ring)
        # ==================================================================
        # APPROACH: for every angle theta around the ellipse (0 to 360 deg)
        #   a) compute the (x, y) boundary point on EACH of the N_BOOT
        #      bootstrap ellipses at that angle
        #   b) compute each point's radial distance from the OBSERVED mean
        #      centre (mn_obs, mn_obs)
        #   c) take the 2.5th and 97.5th percentile of those radial distances
        #   d) project those two percentile radii back along the UNIT VECTOR
        #      from the observed centre to the OBSERVED mean ellipse boundary
        #
        # This produces an inner ring (x_lo, y_lo) and outer ring (x_hi, y_hi).
        # Filling the polygon between them gives the shaded CI band.
        #
        # Why radial distance?
        #   The ellipse boundary is a 2-D curve; we need one scalar per angle
        #   to compute percentiles.  Radial distance from a fixed reference
        #   point is the simplest choice that faithfully represents the
        #   geometric spread of the bootstrap ellipses.
        # ------------------------------------------------------------------

        thetas = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

        # Rotation constants for the 45-degree Poincare convention
        cos45 = np.cos(np.radians(45))
        sin45 = np.sin(np.radians(45))

        # -- 3a: Boundary points for ALL bootstrap ellipses at ALL angles --
        # Parametric rotated ellipse (Brennan 2001):
        #   x(t) = cx + SD1*cos(t)*cos45 - SD2*sin(t)*sin45
        #   y(t) = cx + SD1*cos(t)*sin45 + SD2*sin(t)*cos45
        # where cx is the centre (= MeanNN, sitting on the Line of Identity).
        #
        # boot_mn, boot_sd1, boot_sd2 each have shape (N_BOOT,).
        # thetas has shape (N_ANGLES,).
        # Using [:, None] broadcasts (N_BOOT,) -> (N_BOOT, 1) so that
        # multiplying by a (N_ANGLES,) array gives (N_BOOT, N_ANGLES).

        bx = (boot_mn[:, None]
              + boot_sd1[:, None] * np.cos(thetas)[None, :] * cos45
              - boot_sd2[:, None] * np.sin(thetas)[None, :] * sin45)

        by = (boot_mn[:, None]
              + boot_sd1[:, None] * np.cos(thetas)[None, :] * sin45
              + boot_sd2[:, None] * np.sin(thetas)[None, :] * cos45)

        # -- 3b: Radial distance from the observed mean centre --------------
        # shape: (N_BOOT, N_ANGLES)
        dx = bx - mn_obs
        dy = by - mn_obs
        r  = np.sqrt(dx ** 2 + dy ** 2)

        # -- 3c: Percentile radii at each angle ----------------------------
        # np.percentile over axis=0 collapses the N_BOOT dimension.
        # r_lo_vals, r_hi_vals both have shape (N_ANGLES,).
        r_lo_vals = np.percentile(r, CI_LO, axis=0)
        r_hi_vals = np.percentile(r, CI_HI, axis=0)

        # -- 3d: Unit vectors from centre to observed mean ellipse ---------
        # We project the percentile radii along the direction defined by
        # the OBSERVED mean ellipse at each angle so the CI band is aligned
        # with the mean ellipse rather than being a symmetric ring.
        ref_x = (mn_obs
                 + sd1_obs * np.cos(thetas) * cos45
                 - sd2_obs * np.sin(thetas) * sin45)
        ref_y = (mn_obs
                 + sd1_obs * np.cos(thetas) * sin45
                 + sd2_obs * np.sin(thetas) * cos45)

        ux    = ref_x - mn_obs          # displacement vector components
        uy    = ref_y - mn_obs
        r_ref = np.sqrt(ux ** 2 + uy ** 2)

        ux /= r_ref                     # normalise to unit vectors
        uy /= r_ref

        # Inner and outer CI ring coordinates
        x_lo = mn_obs + r_lo_vals * ux
        y_lo = mn_obs + r_lo_vals * uy
        x_hi = mn_obs + r_hi_vals * ux
        y_hi = mn_obs + r_hi_vals * uy

        # -- 3e: Closed polygon = outer ring forward + inner ring reversed --
        # Concatenating like this traces: outer edge -> jump back -> inner
        # edge (reversed direction) -> close.  ax.fill() fills the enclosed
        # area automatically.
        band_x = np.concatenate([x_hi, x_lo[::-1], [x_hi[0]]])
        band_y = np.concatenate([y_hi, y_lo[::-1], [y_hi[0]]])

        ax.fill(band_x, band_y,
                color=color, alpha=0.28, zorder=3, linewidth=0)

        # Thin crisp outlines on both edges of the band
        ax.plot(np.append(x_hi, x_hi[0]), np.append(y_hi, y_hi[0]),
                color=color, alpha=0.55, linewidth=0.6, zorder=4)
        ax.plot(np.append(x_lo, x_lo[0]), np.append(y_lo, y_lo[0]),
                color=color, alpha=0.55, linewidth=0.6, zorder=4)

        # ==================================================================
        # STEP 4 — LAYER 3: OBSERVED MEAN ELLIPSE  (bold, on top)
        # ==================================================================
        # Identical to the original script — the group average of MeanNN,
        # SD1, SD2. Drawn last (highest zorder) so the CI band frames it.
        # ------------------------------------------------------------------

        mean_e = Ellipse(
            xy=(mn_obs, mn_obs),
            width=2 * sd1_obs,
            height=2 * sd2_obs,
            angle=45,
            linewidth=2.8,
            zorder=5,
        )
        mean_e.set_facecolor(_hex_to_rgba(color, 0.0))   # transparent fill
        mean_e.set_edgecolor(_hex_to_rgba(color, 1.0))   # solid bold edge
        ax.add_patch(mean_e)

        # Dot at the mean centre, sitting on the Line of Identity
        ax.plot(mn_obs, mn_obs,
                marker="o", color=color, markersize=5, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)

        # ==================================================================
        # STEP 5 — ANNOTATION BOX
        # ==================================================================
        # Shows observed mean and its 95 % bootstrap CI for each parameter.
        # CI width naturally reflects sample size: wider for smaller n.
        # ------------------------------------------------------------------

        ax.text(
            0.97, 0.04,
            (
                f"$\\overline{{\\mathrm{{MeanNN}}}}$ = {mn_obs:.0f} ms"
                f"  [{mn_ci_lo:.0f}-{mn_ci_hi:.0f}]\n"
                f"$\\overline{{\\mathrm{{SD1}}}}$   = {sd1_obs:.1f} ms"
                f"  [{sd1_ci_lo:.1f}-{sd1_ci_hi:.1f}]\n"
                f"$\\overline{{\\mathrm{{SD2}}}}$   = {sd2_obs:.1f} ms"
                f"  [{sd2_ci_lo:.1f}-{sd2_ci_hi:.1f}]"
            ),
            transform=ax.transAxes,
            fontsize=7.5,
            ha="right", va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="#CCCCCC", alpha=0.90),
        )

        # Labels and title
        ax.set_title(
            f"{group} - {condition}  (n = {n_sub})",
            fontsize=11, fontweight="bold", color=color, pad=6,
        )
        ax.set_xlabel("RR$_n$ (ms)", fontsize=9)
        ax.set_ylabel("RR$_{n+1}$ (ms)", fontsize=9)
        ax.tick_params(labelsize=8)

# =============================================================================
# 11.  FIGURE LEGEND
# =============================================================================
legend_elements = []

for grp, col in GROUP_COLORS.items():
    legend_elements.append(
        mpatches.Patch(facecolor=col, edgecolor="none", alpha=0.18,
                       label=f"{grp} - bootstrap cloud (n={N_BOOT})")
    )
    legend_elements.append(
        mpatches.Patch(facecolor=col, edgecolor=col, alpha=0.35,
                       label=f"{grp} - 95 % CI band")
    )
    legend_elements.append(
        Line2D([0], [0], color=col, linewidth=2.5,
               label=f"{grp} - observed mean ellipse")
    )

legend_elements.append(
    Line2D([0], [0], color="#AAAAAA", linewidth=0.9,
           linestyle="--", label="Line of identity")
)

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=9,
    frameon=True,
    framealpha=0.9,
    edgecolor="#DDDDDD",
    bbox_to_anchor=(0.5, -0.07),
)

# =============================================================================
# 12.  SAVE
# =============================================================================
fig.savefig(SAVE_PATH, dpi=600, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {SAVE_PATH}")
