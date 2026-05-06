from pathlib import Path
# %% [markdown]
# # Phase 2 — Learn to Read TESS Lightcurves
#
# **Goal:** Build visual intuition for transit signals using real TESS data.
# We'll work through four target types:
#   1. Hot Jupiter — big, obvious dips (WASP-17 b, TIC 103633434)
#   2. Sub-Neptune — shallower signal (TOI-1431 b, TIC 402026209)
#   3. Eclipsing Binary — distinct V-shape / secondary eclipse
#   4. Known False Positive — centroid shift, contamination
#
# Run cells in order. Each section builds on the previous one.

# %% [markdown]
# ## Setup

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
import lightkurve as lk

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

print(f"lightkurve version: {lk.__version__}")
print("Setup complete. Ready to hunt planets.")

# %% [markdown]
# ---
# ## Target 1 — Hot Jupiter: WASP-17 b (TIC 103633434)
#
# Period ~3.74 d, depth ~1.5% (15,000 ppm). Should be unmissable.

# %%
TARGET_1 = "TIC 103633434"
print(f"Searching MAST for {TARGET_1}...")

results = lk.search_lightcurve(TARGET_1, mission="TESS", author="SPOC")
print(results)

# %%
# Download all available sectors (SPOC 2-min cadence)
lcs_1 = results.download_all(quality_bitmask='default')
print(f"Downloaded {len(lcs_1)} light curves")

# %%
# Stitch sectors together
lc_1 = lcs_1.stitch()
print(f"Baseline: {lc_1.time.value.min():.1f} – {lc_1.time.value.max():.1f} BTJD")
print(f"N points: {len(lc_1)}")

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Raw
lc_1.plot(ax=axes[0], label='Raw flux')
axes[0].set_title(f"{TARGET_1} — WASP-17 b (Hot Jupiter)")

# Flattened (remove stellar variability with a running median / Savitzky-Golay)
lc_flat_1 = lc_1.flatten(window_length=401)
lc_flat_1.plot(ax=axes[1], label='Flattened', color='steelblue')
axes[1].set_title("Flattened lightcurve")
axes[1].set_ylim(0.975, 1.015)

# Sigma-clip outliers
lc_clean_1 = lc_flat_1.remove_outliers(sigma=4)
lc_clean_1.plot(ax=axes[2], label='Cleaned', color='darkorange')
axes[2].set_title("After outlier removal")
axes[2].set_ylim(0.975, 1.015)

plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent.parent / 'data/wasp17_raw.png'), bbox_inches='tight')
plt.show()
print("Raw → flattened → cleaned. Notice the ~1.5% dips every ~3.74 days.")

# %%
# BLS period search
print("Running Box Least Squares period search...")

blsm_1 = lc_clean_1.to_periodogram(method='bls',
                                     period=np.arange(0.5, 20, 0.005),
                                     duration=np.arange(0.05, 0.25, 0.01))
best_period_1 = blsm_1.period_at_max_power
best_t0_1 = blsm_1.transit_time_at_max_power
print(f"Best BLS period: {best_period_1:.4f}")
print(f"Transit epoch  : {best_t0_1:.4f} BTJD")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

blsm_1.plot(ax=axes[0])
axes[0].set_title("BLS Periodogram — WASP-17 b")
axes[0].axvline(best_period_1.value, color='red', lw=1.5, ls='--', label=f'P={best_period_1.value:.3f}d')
axes[0].legend()

# Phase-fold on best period
lc_fold_1 = lc_clean_1.fold(period=best_period_1, epoch_time=best_t0_1)
lc_fold_1.scatter(ax=axes[1], alpha=0.3, s=5, label='All data')
lc_fold_1.bin(time_bin_size=0.01).plot(ax=axes[1], color='red', lw=2, label='10-min bins')
axes[1].set_title(f"Phase-folded — P={best_period_1.value:.3f} d")
axes[1].set_xlim(-0.3, 0.3)
axes[1].set_ylim(0.975, 1.010)
axes[1].legend()

plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent.parent / 'data/wasp17_bls_fold.png'), bbox_inches='tight')
plt.show()

# %%
# Skip BLS for now — phase-fold at the KNOWN period to see the planet first
# WASP-17 b has a well-known period of 3.7354 days
PERIOD_1 = 3.7354
T0_1 = 1684.39  # approximate transit epoch in BTJD

lc_fold_1 = lc_clean_1.fold(period=PERIOD_1, epoch_time=T0_1)
lc_binned_1 = lc_fold_1.bin(time_bin_size=0.008)

fig, ax = plt.subplots(figsize=(12, 6))
lc_fold_1.scatter(ax=ax, alpha=0.15, s=2, color='gray', label='All data')
lc_binned_1.plot(ax=ax, color='red', linewidth=2.5, label='Binned (12 min)')
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(0.980, 1.005)
ax.set_title('WASP-17 b — THIS IS A PLANET', fontsize=16)
ax.set_xlabel('Phase (days from transit center)')
ax.set_ylabel('Normalized Flux')
ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.3, label='Normal brightness')
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()

depth = (1.0 - lc_binned_1.flux.value.min()) * 100
print(f"\nTransit depth: {depth:.2f}%")
print(f"That means WASP-17 b blocks about {depth:.2f}% of its star's light.")
print(f"The planet orbits every {PERIOD_1} days ({PERIOD_1 * 24:.1f} hours).")
print(f"You are looking at the shadow of a planet 1,300 light years away.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

blsm_1.plot(ax=axes[0])
axes[0].set_title("BLS Periodogram — WASP-17 b")
axes[0].axvline(best_period_1.value, color='red', lw=1.5, ls='--', label=f'P={best_period_1.value:.3f}d')
axes[0].legend()

# Phase-fold on best period
lc_fold_1 = lc_clean_1.fold(period=best_period_1, epoch_time=best_t0_1)
lc_fold_1.scatter(ax=axes[1], alpha=0.3, s=5, label='All data')
lc_fold_1.bin(time_bin_size=0.01).plot(ax=axes[1], color='red', lw=2, label='10-min bins')
axes[1].set_title(f"Phase-folded — P={best_period_1.value:.3f} d")
axes[1].set_xlim(-0.3, 0.3)
axes[1].set_ylim(0.975, 1.010)
axes[1].legend()

plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent.parent / 'data/wasp17_bls_fold.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# **What to notice:**
# - BLS picks up the period cleanly — strong, isolated peak
# - Phase-folded transit: flat baseline, symmetric U-shaped dip
# - Depth ≈ 15,000 ppm. A Jupiter-radius planet around a Sun-like star
# - Duration ≈ 3.5 hours out of 89.6-hour orbit

# %% [markdown]
# ---
# ## Target 2 — Sub-Neptune: TOI-1431 b (TIC 402026209)
#
# Period ~2.65 d, depth ~430 ppm. Ten times shallower than WASP-17 b.

# %%
TARGET_2 = "TIC 402026209"
print(f"Searching MAST for {TARGET_2} (TOI-1431 b)...")

results_2 = lk.search_lightcurve(TARGET_2, mission="TESS", author="SPOC")
print(results_2)

# %%
lcs_2 = results_2.download_all(quality_bitmask='default')
lc_2 = lcs_2.stitch().flatten(window_length=401).remove_outliers(sigma=4)

print(f"N points after cleaning: {len(lc_2)}")

# %%
blsm_2 = lc_2.to_periodogram(method='bls',
                               period=np.arange(0.5, 15, 0.001),
                               duration=np.arange(0.02, 0.15, 0.002))
best_period_2 = blsm_2.period_at_max_power
best_t0_2 = blsm_2.transit_time_at_max_power

print(f"Best BLS period: {best_period_2:.4f}")
print(f"Known period   : 2.6503 d")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

blsm_2.plot(ax=axes[0])
axes[0].set_title("BLS Periodogram — TOI-1431 b (Sub-Neptune)")
axes[0].axvline(best_period_2.value, color='red', lw=1.5, ls='--')

lc_fold_2 = lc_2.fold(period=best_period_2, epoch_time=best_t0_2)
lc_fold_2.scatter(ax=axes[1], alpha=0.3, s=5)
lc_fold_2.bin(time_bin_size=0.01).plot(ax=axes[1], color='red', lw=2, label='10-min bins')
axes[1].set_title(f"Phase-folded — P={best_period_2.value:.3f} d")
axes[1].set_xlim(-0.2, 0.2)
axes[1].legend()

plt.tight_layout()
plt.savefig(str(Path(__file__).resolve().parent.parent / 'data/toi1431_bls_fold.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# **What to notice:**
# - Same analysis but the transit depth is ~430 ppm — subtle in single transits
# - Phase-folding reveals it clearly — that's the power of TESS multi-sector coverage
# - Scatter in the phase-fold is dominated by photon noise + systematics

# %% [markdown]
# ---
# ## Target 3 — Eclipsing Binary: TIC 454141135 (KIC-like EB)
#
# EBs are the #1 false positive. They look like planets but aren't.
# Diagnostic: secondary eclipse, ellipsoidal variation, odd/even depth differences.

# %%
TARGET_3 = "TIC 454141135"
print(f"Searching for eclipsing binary {TARGET_3}...")

results_3 = lk.search_lightcurve(TARGET_3, mission="TESS", author="SPOC")
if len(results_3) == 0:
    # Fallback: use a known EB from the catalog
    print("Primary TIC not found, trying TIC 271893367 (known EB)...")
    results_3 = lk.search_lightcurve("TIC 271893367", mission="TESS", author="SPOC")

print(results_3)

# %%
if len(results_3) > 0:
    lcs_3 = results_3.download_all(quality_bitmask='default')
    lc_3 = lcs_3.stitch().flatten(window_length=201).remove_outliers(sigma=5)

    blsm_3 = lc_3.to_periodogram(method='bls',
                                   period=np.arange(0.3, 20, 0.001),
                                   duration=np.arange(0.02, 0.3, 0.005))
    best_period_3 = blsm_3.period_at_max_power
    best_t0_3 = blsm_3.transit_time_at_max_power

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Raw
    lc_3.scatter(ax=axes[0, 0], alpha=0.3, s=2)
    axes[0, 0].set_title(f"Eclipsing Binary — {TARGET_3}")

    # BLS periodogram
    blsm_3.plot(ax=axes[0, 1])
    axes[0, 1].set_title("BLS Periodogram")
    axes[0, 1].axvline(best_period_3.value, color='red', lw=1.5, ls='--',
                        label=f'P={best_period_3.value:.3f}d')
    axes[0, 1].legend()

    # Phase-fold at full period (shows primary + secondary)
    lc_fold_3 = lc_3.fold(period=best_period_3, epoch_time=best_t0_3)
    lc_fold_3.scatter(ax=axes[1, 0], alpha=0.3, s=3)
    lc_fold_3.bin(time_bin_size=0.01).plot(ax=axes[1, 0], color='red', lw=2)
    axes[1, 0].set_title(f"Phase-fold at P={best_period_3.value:.3f} d\n(Look for secondary at phase ±0.5)")

    # Phase-fold at HALF period (secondary eclipse appears at same phase as primary)
    lc_fold_half = lc_3.fold(period=best_period_3 / 2, epoch_time=best_t0_3)
    lc_fold_half.scatter(ax=axes[1, 1], alpha=0.3, s=3, color='gray')
    lc_fold_half.bin(time_bin_size=0.01).plot(ax=axes[1, 1], color='darkorange', lw=2)
    axes[1, 1].set_title("Phase-fold at P/2 — EB signature:\nodd/even depth mismatch")

    plt.tight_layout()
    plt.savefig(str(Path(__file__).resolve().parent.parent / 'data/eb_diagnostics.png'), bbox_inches='tight')
    plt.show()

# %% [markdown]
# **EB diagnostics to look for:**
# - Secondary eclipse visible at phase ~0.5 (if circular orbit)
# - Ellipsoidal variations — sinusoidal modulation between eclipses
# - Odd/even depth mismatch: if P/2 fold shows two different depths, it's an EB
# - ExoMiner++ should score this low (< 0.3)

# %% [markdown]
# ---
# ## Target 4 — Fresh TESS Sector Search
#
# Pick the most recent available sector and look for new signals.

# %%
# What sectors are currently available?
print("Checking most recent TESS sectors available on MAST...")
recent = lk.search_lightcurve("TIC 402026209", mission="TESS")
print(recent)
print("\nLatest available sector:", recent[-1] if len(recent) > 0 else "none")

# %%
# Search for any star in the latest sector — let's pick a quiet solar analog
# TIC 261136679 = a quiet G dwarf observed in multiple sectors
TARGET_4 = "TIC 261136679"
print(f"Downloading latest sector for {TARGET_4}...")

results_4 = lk.search_lightcurve(TARGET_4, mission="TESS", author="SPOC")
print(results_4)

if len(results_4) > 0:
    lc_4_raw = results_4[-1].download(quality_bitmask='default')
    lc_4 = lc_4_raw.flatten(window_length=301).remove_outliers(sigma=4)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    lc_4_raw.plot(ax=axes[0], label='Raw')
    axes[0].set_title(f"{TARGET_4} — Raw flux (latest sector)")

    blsm_4 = lc_4.to_periodogram(method='bls',
                                   period=np.arange(0.5, 14, 0.001),
                                   duration=np.arange(0.02, 0.2, 0.003))
    blsm_4.plot(ax=axes[1])
    best_p4 = blsm_4.period_at_max_power
    axes[1].axvline(best_p4.value, color='red', lw=1.5, ls='--',
                    label=f'Peak P={best_p4.value:.3f}d  (SDE={blsm_4.max_power:.1f})')
    axes[1].legend()
    axes[1].set_title("BLS — Flag anything with SDE > 8")

    plt.tight_layout()
    plt.savefig(str(Path(__file__).resolve().parent.parent / 'data/fresh_search.png'), bbox_inches='tight')
    plt.show()

    print(f"\nTop BLS peak: P={best_p4:.3f}, SDE={blsm_4.max_power:.2f}")
    if blsm_4.max_power > 8:
        print("SDE > 8 — worth folding and inspecting!")
    else:
        print("SDE < 8 — likely noise or stellar variability, not a confident transit.")

# %% [markdown]
# ---
# ## Summary: Key Transit Diagnostics
#
# | Feature | Planet | Eclipsing Binary | Systematic |
# |---------|--------|-----------------|------------|
# | Transit shape | Flat-bottom (U) | V-shaped | irregular |
# | Secondary eclipse | No | Yes (at phase 0.5) | No |
# | Odd/even depth | Equal | Different | Varies |
# | Centroid shift | < 1 arcsec | Can be large | Correlated with BJD |
# | BLS SDE | > 8 | > 8 | Usually < 7 |
# | ExoMiner++ score | High | Low | Very low |
#
# **Rule of thumb:** If BLS SDE > 9 AND transit is U-shaped AND no secondary eclipse
# AND centroid is stable → submit for follow-up.

# %%
print("Phase 2 complete!")
print(f"Plots saved to {Path(__file__).resolve().parent.parent / 'data'}")
print("Next: open the Dashboard → Candidate Browser to load the Zenodo ExoMiner++ catalog.")

# %%
