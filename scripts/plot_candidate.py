"""
Shared 4-panel transit plot for a TESS lightcurve.

    Panel 1 — Raw flux (all cadences)
    Panel 2 — Flattened & cleaned
    Panel 3 — Phase-folded with 10-min bins
    Panel 4 — Annotated transit zoom (ingress / mid / egress arrows)

Used by scan_sector.py (saving PNGs) and dashboard.py (interactive display).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from astropy import units as u


BG      = "#0d1117"
PANEL   = "#161b22"
SPINE   = "#30363d"
TICK    = "#8b949e"
LABEL   = "#c9d1d9"
TITLE   = "#e6edf3"
BLUE    = "#58a6ff"
ORANGE  = "#f0883e"
GRAY    = "#484f58"
RED     = "#ff7b72"
GREEN   = "#3fb950"


def _strip(arr) -> np.ndarray:
    """Convert masked / Quantity arrays to plain float64 ndarray."""
    a = np.asarray(arr, dtype=float)
    if np.ma.is_masked(a):
        a = np.ma.filled(a, fill_value=np.nan)
    return a


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=8)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(TITLE)
    for spine in ax.spines.values():
        spine.set_color(SPINE)


def make_4panel(
    lc_raw,
    lc_flat,
    period_d: float,
    t0_btjd: float,
    tic_id: int,
    sector: int | str,
    depth_ppm: float,
    bls_power: float = 0.0,
    out_path=None,
):
    """
    Generate the 4-panel transit figure.

    Parameters
    ----------
    lc_raw    : lightkurve LightCurve  (un-normalised, for panel 1)
    lc_flat   : lightkurve LightCurve  (normalised + flattened, for panels 2-4)
    period_d  : orbital period in days
    t0_btjd   : transit epoch in BTJD
    tic_id    : TESS Input Catalog ID
    sector    : sector number or label string
    depth_ppm : transit depth in ppm
    bls_power : BLS power (shown in title)
    out_path  : if given, save PNG here and return None; else return Figure

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
                  left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in (ax1, ax2, ax3, ax4):
        _style_ax(ax)

    # ── Panel 1 : Raw lightcurve ──────────────────────────────────────────
    t_raw = _strip(lc_raw.time.value)
    f_raw = _strip(lc_raw.flux.value)
    ax1.scatter(t_raw, f_raw, s=1, alpha=0.35, color=BLUE,
                rasterized=True, linewidths=0)
    ax1.set_xlabel("BTJD", fontsize=9)
    ax1.set_ylabel("Flux (e⁻/s)", fontsize=9)
    ax1.set_title("Raw Lightcurve", fontsize=10, fontweight="bold")

    # ── Panel 2 : Flattened & cleaned ────────────────────────────────────
    t_flat = _strip(lc_flat.time.value)
    f_flat = _strip(lc_flat.flux.value)
    ax2.scatter(t_flat, f_flat, s=1, alpha=0.35, color=ORANGE,
                rasterized=True, linewidths=0)
    ax2.axhline(1.0, color=SPINE, lw=0.8, ls="--", zorder=0)
    ax2.set_xlabel("BTJD", fontsize=9)
    ax2.set_ylabel("Normalised Flux", fontsize=9)
    ax2.set_title("Flattened & Cleaned", fontsize=10, fontweight="bold")

    # ── Phase fold ────────────────────────────────────────────────────────
    lc_fold = lc_flat.fold(period=period_d * u.day,
                           epoch_time=t0_btjd * u.day)
    lc_bin  = lc_fold.bin(time_bin_size=0.01)

    ph_all  = _strip(lc_fold.phase.value)
    fl_all  = _strip(lc_fold.flux.value)
    ph_bin  = _strip(lc_bin.phase.value)
    fl_bin  = _strip(lc_bin.flux.value)

    # ── Panel 3 : Full phase-fold ─────────────────────────────────────────
    ax3.scatter(ph_all, fl_all, s=1, alpha=0.15, color=GRAY,
                rasterized=True, linewidths=0)
    ax3.plot(ph_bin, fl_bin, color=RED, lw=2, label="10-min bins", zorder=5)
    ax3.axhline(1.0, color=SPINE, lw=0.8, ls="--", zorder=0)
    ax3.set_xlabel("Phase", fontsize=9)
    ax3.set_ylabel("Normalised Flux", fontsize=9)
    ax3.set_title(f"Phase-folded   P = {period_d:.5f} d", fontsize=10, fontweight="bold")
    ax3.set_xlim(-0.5, 0.5)
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=LABEL,
               edgecolor=SPINE, framealpha=0.8)

    # ── Panel 4 : Annotated transit zoom ─────────────────────────────────
    ZOOM = 0.13   # ±13% of period shown
    zm      = np.abs(ph_all) < ZOOM
    zm_bin  = np.abs(ph_bin) < ZOOM

    ax4.scatter(ph_all[zm], fl_all[zm], s=4, alpha=0.35, color=GRAY,
                rasterized=True, linewidths=0)

    if zm_bin.any():
        bph = ph_bin[zm_bin]
        bfl = fl_bin[zm_bin]
        ax4.plot(bph, bfl, color=RED, lw=2.5, zorder=5)

        baseline  = float(np.nanmedian(fl_bin[np.abs(ph_bin) > 0.08]))
        mid_idx   = int(np.nanargmin(bfl))
        mid_ph    = float(bph[mid_idx])
        mid_fl    = float(bfl[mid_idx])
        half_lev  = baseline - (baseline - mid_fl) * 0.5

        below     = bph[bfl < half_lev]
        ing_ph    = float(below[0])  if len(below) else mid_ph - 0.02
        egr_ph    = float(below[-1]) if len(below) else mid_ph + 0.02

        y_ann = baseline + 0.55 * abs(baseline - mid_fl)
        arrkw = dict(arrowstyle="->", lw=1.2, color=RED)
        txkw  = dict(color=RED, fontsize=8, fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=2, foreground=PANEL)])

        ax4.annotate("Ingress",   xy=(ing_ph, half_lev),
                     xytext=(ing_ph - 0.045, y_ann),
                     arrowprops=arrkw, **txkw, ha="right")
        ax4.annotate("Mid-transit", xy=(mid_ph, mid_fl),
                     xytext=(mid_ph, y_ann + 0.3 * abs(y_ann - mid_fl)),
                     arrowprops=arrkw, **txkw, ha="center")
        ax4.annotate("Egress",    xy=(egr_ph, half_lev),
                     xytext=(egr_ph + 0.045, y_ann),
                     arrowprops=arrkw, **txkw, ha="left")

        # Depth double-arrow
        x_dep = max(ZOOM * 0.55, egr_ph + 0.025)
        ax4.annotate("", xy=(x_dep, mid_fl), xytext=(x_dep, baseline),
                     arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
        ax4.text(x_dep + 0.006, (mid_fl + baseline) / 2,
                 f"{depth_ppm:.0f}\nppm",
                 color=GREEN, fontsize=7, va="center", linespacing=1.2,
                 path_effects=[pe.withStroke(linewidth=2, foreground=PANEL)])

        ax4.axhline(baseline, color=SPINE, lw=0.8, ls="--", label="baseline")

    ax4.set_xlabel("Phase", fontsize=9)
    ax4.set_ylabel("Normalised Flux", fontsize=9)
    ax4.set_title("Transit Detail", fontsize=10, fontweight="bold")
    ax4.set_xlim(-ZOOM, ZOOM)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=LABEL,
               edgecolor=SPINE, framealpha=0.8)

    # ── Supertitle ───────────────────────────────────────────────────────
    power_str = f"   BLS power = {bls_power:.1f}" if bls_power else ""
    fig.suptitle(
        f"TIC {tic_id}  ·  Sector {sector}  ·  "
        f"P = {period_d:.5f} d  ·  depth = {depth_ppm:.0f} ppm{power_str}",
        fontsize=12, color=TITLE, fontweight="bold", y=0.97,
    )

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return None
    return fig
