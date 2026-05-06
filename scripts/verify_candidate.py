#!/usr/bin/env python3
"""
verify_candidate.py — Multi-sector consistency verification for TESS transit candidates.

Downloads all available TESS sectors for a star, phase-folds each sector
at the candidate period, measures transit depth and SNR per sector, and
scores the overall consistency. A real planet should show a consistent dip
across most sectors it was observed in.

Usage:
    python verify_candidate.py --tic 402026209 --period 2.6503
    python verify_candidate.py --tic 103633434 --period 3.7354 --t0 1326.5
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
VERIFY_DIR    = DATA_DIR / "verification"
SCRIPTS_DIR   = Path(__file__).parent

# Import colour constants and helpers from plot_candidate.py
sys.path.insert(0, str(SCRIPTS_DIR))
from plot_candidate import _strip, _style_ax, BG, PANEL, SPINE, TICK, LABEL, TITLE
from plot_candidate import BLUE, ORANGE, GRAY, RED, GREEN

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_SNR           = 2.0    # per-sector SNR to count as "has signal"
MAX_DEPTH_RATIO   = 3.0    # sector depth must be within 3x of median to count
MIN_IN_TRANSIT    = 2      # minimum binned points in transit window
REAL_THRESHOLD    = 0.70   # consistency_score >= this → "Likely real planet"
INCON_THRESHOLD   = 0.30   # consistency_score >= this → "Inconclusive"


def _log(*args, **kwargs) -> None:
    """Best-effort console logging that must never crash verification."""
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except (ValueError, OSError):
        pass


# ── Sector download ───────────────────────────────────────────────────────────

def download_all_sectors(tic_id: int, author: str = "SPOC") -> list:
    """
    Return list of (sector_int, LightCurve) for every available sector.
    Falls back to author=None if SPOC returns nothing.
    """
    import lightkurve as lk
    from astropy import units as u

    target = f"TIC {tic_id}"
    sr = lk.search_lightcurve(target, mission="TESS", author=author, exptime=120)
    if len(sr) == 0:
        # Retry without author filter
        sr = lk.search_lightcurve(target, mission="TESS")

    if len(sr) == 0:
        return []

    lcs = sr.download_all(quality_bitmask="default")
    if lcs is None or len(lcs) == 0:
        return []

    # Split by sector
    sector_map = {}
    for lc in lcs:
        sec = getattr(lc, "sector", None)
        if sec is None:
            sec = lc.meta.get("SECTOR") or lc.meta.get("sector")
        if sec is None:
            continue
        sec = int(sec)
        if sec not in sector_map:
            sector_map[sec] = lc
        # keep only one lightcurve per sector (prefer existing)

    return sorted(sector_map.items())   # [(sector_int, lc), ...]


# ── Per-sector measurement ────────────────────────────────────────────────────

def measure_sector(lc, period: float, t0: float, window_length: int = 401) -> dict:
    """
    Flatten a sector's lightcurve, phase-fold at period/t0, measure transit.

    Returns dict with:
        sector, depth_ppm, noise_ppm, snr, n_points, n_in_transit, has_signal
    """
    from astropy import units as u

    sec = getattr(lc, "sector", None)
    if sec is None:
        sec = lc.meta.get("SECTOR") or lc.meta.get("sector") or "?"

    if len(lc) < 100:
        return {"sector": sec, "depth_ppm": 0.0, "noise_ppm": 0.0, "snr": 0.0,
                "n_points": len(lc), "n_in_transit": 0, "has_signal": False,
                "skip_reason": "too few points"}

    try:
        lc_flat = lc.normalize().flatten(window_length=window_length).remove_outliers(sigma=4)
    except Exception:
        return {"sector": sec, "depth_ppm": 0.0, "noise_ppm": 0.0, "snr": 0.0,
                "n_points": len(lc), "n_in_transit": 0, "has_signal": False,
                "skip_reason": "flatten failed"}

    if len(lc_flat) < 50:
        return {"sector": sec, "depth_ppm": 0.0, "noise_ppm": 0.0, "snr": 0.0,
                "n_points": len(lc_flat), "n_in_transit": 0, "has_signal": False,
                "skip_reason": "too few points after flatten"}

    try:
        lc_fold = lc_flat.fold(period=period * u.day, epoch_time=t0 * u.day)
        lc_bin  = lc_fold.bin(time_bin_size=0.01)
    except Exception:
        return {"sector": sec, "depth_ppm": 0.0, "noise_ppm": 0.0, "snr": 0.0,
                "n_points": len(lc_flat), "n_in_transit": 0, "has_signal": False,
                "skip_reason": "fold/bin failed"}

    flux_bin  = _strip(lc_bin.flux.value)
    phase_bin = _strip(lc_bin.phase.value)   # in DAYS, range = [-period/2, period/2]

    # ── Dynamically locate transit minimum ────────────────────────────────
    # Phase is in DAYS.  Search the central ±15% of the period (up to ±0.20 d)
    # so we find the transit even if t0 has a small error.
    half_period  = period / 2.0
    search_half  = min(0.20, half_period * 0.30)   # ±20% of half-period, capped at 0.20d
    baseline_min = min(0.25, half_period * 0.40)   # out-of-transit starts beyond this

    search_mask = np.abs(phase_bin) < search_half
    valid_s     = search_mask & np.isfinite(flux_bin)

    if not valid_s.any():
        # Fallback: search full central half
        search_mask = np.abs(phase_bin) < (half_period * 0.80)
        valid_s     = search_mask & np.isfinite(flux_bin)

    if not valid_s.any():
        return {"sector": sec, "depth_ppm": 0.0, "noise_ppm": 0.0, "snr": 0.0,
                "n_points": len(lc_flat), "n_in_transit": 0, "has_signal": False,
                "skip_reason": "no data near transit"}

    # Minimum of binned data in search region = transit center
    mid_idx     = int(np.nanargmin(flux_bin[valid_s]))
    transit_min = float(flux_bin[valid_s][mid_idx])
    mid_phase   = float(phase_bin[valid_s][mid_idx])   # transit center in phase-days

    # Out-of-transit baseline: far from detected transit center
    out_mask = np.abs(phase_bin - mid_phase) > baseline_min
    out_flux = flux_bin[out_mask & np.isfinite(flux_bin)]

    # In-transit: narrow window around detected center
    in_mask = np.abs(phase_bin - mid_phase) < 0.06
    n_in    = int(np.sum(in_mask & np.isfinite(flux_bin)))

    if out_flux.size < 5 or not np.any(np.isfinite(out_flux)):
        return {"sector": sec, "depth_ppm": 0.0, "noise_ppm": 0.0, "snr": 0.0,
                "n_points": len(lc_flat), "n_in_transit": n_in, "has_signal": False,
                "skip_reason": "insufficient out-of-transit baseline"}

    baseline   = float(np.nanmedian(out_flux))
    noise_norm = float(np.nanstd(out_flux))

    depth_frac  = max(0.0, (baseline - transit_min) / baseline) if baseline > 0 else 0.0
    depth_ppm   = depth_frac * 1e6
    noise_ppm   = (noise_norm / baseline) * 1e6 if baseline > 0 else 999999.0
    snr         = depth_ppm / noise_ppm if noise_ppm > 0 else 0.0

    return {
        "sector":       int(sec),
        "depth_ppm":    round(depth_ppm, 1),
        "noise_ppm":    round(noise_ppm, 1),
        "snr":          round(snr, 2),
        "n_points":     len(lc_flat),
        "n_in_transit": n_in,
        "has_signal":   False,   # filled in by compute_consistency
        "skip_reason":  None,
    }


# ── Consistency scoring ───────────────────────────────────────────────────────

def compute_consistency(sector_results: list) -> dict:
    """
    Score multi-sector measurements for self-consistency.

    Returns dict with consistency_score, verdict, n_sectors_checked,
    n_sectors_consistent, median_depth_ppm, notes.
    """
    # Only consider sectors that didn't error out and have a real depth measurement
    valid = [r for r in sector_results if r.get("skip_reason") is None and r["depth_ppm"] > 0]

    if not valid:
        return {
            "consistency_score": 0.0,
            "verdict": "No valid sector measurements — cannot verify",
            "n_sectors_checked": len(sector_results),
            "n_sectors_consistent": 0,
            "median_depth_ppm": 0.0,
            "eb_warning": None,
        }

    if len(valid) == 1:
        r = valid[0]
        # Single sector — report SNR but note it's unverifiable
        snr_note = f"SNR={r['snr']:.1f}"
        verdict = (
            f"Only 1 sector available — cannot verify multi-sector consistency "
            f"({snr_note}, depth={r['depth_ppm']:.0f} ppm)"
        )
        # Mark has_signal based on single-sector SNR
        r["has_signal"] = r["snr"] >= MIN_SNR and r["n_in_transit"] >= MIN_IN_TRANSIT
        _eb_warning_1 = None
        if r["depth_ppm"] > 2000:
            _eb_warning_1 = (
                "⚠️ Deep signal — could be an eclipsing binary. "
                "Check for secondary eclipse at phase 0.5 and V-shaped transit profile "
                "before trusting this as a planet."
            )
        return {
            "consistency_score": 1.0 if r["has_signal"] else 0.0,
            "verdict": verdict,
            "n_sectors_checked": 1,
            "n_sectors_consistent": 1 if r["has_signal"] else 0,
            "median_depth_ppm": r["depth_ppm"],
            "eb_warning": _eb_warning_1,
        }

    depths = np.array([r["depth_ppm"] for r in valid])
    median_depth = float(np.median(depths))

    consistent = []
    for r in valid:
        depth_ok   = (r["depth_ppm"] >= median_depth / MAX_DEPTH_RATIO and
                      r["depth_ppm"] <= median_depth * MAX_DEPTH_RATIO)
        snr_ok     = r["snr"] >= MIN_SNR
        transit_ok = r["n_in_transit"] >= MIN_IN_TRANSIT
        r["has_signal"] = depth_ok and snr_ok and transit_ok
        consistent.append(r["has_signal"])

    n_consistent     = sum(consistent)
    n_total          = len(valid)
    consistency_score = n_consistent / n_total

    if consistency_score >= REAL_THRESHOLD:
        verdict = f"Likely real planet — signal in {n_consistent}/{n_total} sectors"
    elif consistency_score >= INCON_THRESHOLD:
        verdict = f"Inconclusive — signal in {n_consistent}/{n_total} sectors"
    else:
        verdict = f"Likely not a planet — signal in only {n_consistent}/{n_total} sectors"

    # EB warning for deep signals
    eb_warning = None
    if median_depth > 2000:
        eb_warning = (
            "⚠️ Deep signal — could be an eclipsing binary. "
            "Check for secondary eclipse at phase 0.5 and V-shaped transit profile "
            "before trusting this as a planet."
        )

    return {
        "consistency_score": round(consistency_score, 3),
        "verdict": verdict,
        "n_sectors_checked": n_total,
        "n_sectors_consistent": n_consistent,
        "median_depth_ppm": round(median_depth, 1),
        "eb_warning": eb_warning,
    }


# ── Verification plot ─────────────────────────────────────────────────────────

def make_verification_plot(sector_data: list, sector_results: list,
                           period: float, t0: float,
                           tic_id: int, verdict: str,
                           out_path=None):
    """
    Multi-panel verification figure.
    Top rows: one subplot per sector (phase-folded), green/red border.
    Bottom row: combined stitched phase-fold.
    """
    from astropy import units as u

    # Build result lookup by sector
    res_by_sec = {r["sector"]: r for r in sector_results if "sector" in r}

    n_sectors = len(sector_data)
    if n_sectors == 0:
        return None

    NCOLS = min(4, n_sectors)
    n_sector_rows = math.ceil(n_sectors / NCOLS)
    n_rows_total  = n_sector_rows + 1   # +1 for combined panel

    fig = plt.figure(figsize=(4 * NCOLS, 3.5 * n_rows_total + 0.8))
    fig.patch.set_facecolor(BG)

    gs = GridSpec(n_rows_total, NCOLS, figure=fig,
                  hspace=0.55, wspace=0.35,
                  top=0.92, bottom=0.06, left=0.06, right=0.98)

    # ── Per-sector subplots ───────────────────────────────────────────────
    for idx, (sec_num, lc) in enumerate(sector_data):
        row = idx // NCOLS
        col = idx % NCOLS
        ax  = fig.add_subplot(gs[row, col])
        _style_ax(ax)

        res = res_by_sec.get(int(sec_num), {})
        has_signal = res.get("has_signal", False)
        depth      = res.get("depth_ppm", 0.0)
        snr        = res.get("snr", 0.0)
        skip       = res.get("skip_reason")

        # Border colour
        border_color = GREEN if has_signal else (ORANGE if snr >= 1 else RED)
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2.0)

        if skip:
            ax.text(0.5, 0.5, f"Sector {sec_num}\n({skip})",
                    transform=ax.transAxes, ha="center", va="center",
                    color=TICK, fontsize=7)
        else:
            try:
                lc_flat = lc.normalize().flatten(window_length=401).remove_outliers(sigma=4)
                lc_fold = lc_flat.fold(period=period * u.day, epoch_time=t0 * u.day)
                lc_bin  = lc_fold.bin(time_bin_size=0.01)

                ph_all = _strip(lc_fold.phase.value)
                fl_all = _strip(lc_fold.flux.value)
                ph_bin = _strip(lc_bin.phase.value)
                fl_bin = _strip(lc_bin.flux.value)

                ax.scatter(ph_all, fl_all, s=1, alpha=0.15, color=GRAY,
                           rasterized=True, linewidths=0)
                ax.plot(ph_bin, fl_bin, color=RED if has_signal else ORANGE,
                        lw=1.5, zorder=5)
                ax.axhline(1.0, color=SPINE, lw=0.6, ls="--")
                ax.set_xlim(-0.5, 0.5)
            except Exception:
                ax.text(0.5, 0.5, "plot error", transform=ax.transAxes,
                        ha="center", va="center", color=TICK, fontsize=7)

        signal_tag = "✓" if has_signal else "✗"
        ax.set_title(
            f"Sector {sec_num}  {signal_tag}\n"
            f"depth={depth:.0f} ppm  SNR={snr:.1f}",
            fontsize=7, color=TITLE if has_signal else TICK,
        )
        ax.set_xlabel("Phase", fontsize=6)
        ax.tick_params(labelsize=6)

    # Fill any empty cells in the last row
    total_cells = n_sector_rows * NCOLS
    for idx in range(n_sectors, total_cells):
        row = idx // NCOLS
        col = idx % NCOLS
        ax  = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # ── Combined stitched phase-fold ──────────────────────────────────────
    ax_comb = fig.add_subplot(gs[n_sector_rows, :])
    _style_ax(ax_comb)

    try:
        import lightkurve as lk
        all_flat = []
        for sec_num, lc in sector_data:
            try:
                lc_flat = lc.normalize().flatten(window_length=401).remove_outliers(sigma=4)
                all_flat.append(lc_flat)
            except Exception:
                pass

        if all_flat:
            from astropy import units as u
            if len(all_flat) > 1:
                lc_stitch = lk.LightCurveCollection(all_flat).stitch()
            else:
                lc_stitch = all_flat[0]

            lc_fold_all = lc_stitch.fold(period=period * u.day, epoch_time=t0 * u.day)
            lc_bin_all  = lc_fold_all.bin(time_bin_size=0.005)

            ph_a = _strip(lc_fold_all.phase.value)
            fl_a = _strip(lc_fold_all.flux.value)
            ph_b = _strip(lc_bin_all.phase.value)
            fl_b = _strip(lc_bin_all.flux.value)

            ax_comb.scatter(ph_a, fl_a, s=0.5, alpha=0.10, color=GRAY,
                            rasterized=True, linewidths=0)
            ax_comb.plot(ph_b, fl_b, color=BLUE, lw=2, zorder=5,
                         label=f"Stitched ({len(all_flat)} sectors)")
            ax_comb.axhline(1.0, color=SPINE, lw=0.8, ls="--")
            ax_comb.set_xlim(-0.5, 0.5)
            ax_comb.legend(fontsize=8, facecolor=PANEL, labelcolor=LABEL,
                           edgecolor=SPINE, framealpha=0.8)
    except Exception as e:
        ax_comb.text(0.5, 0.5, f"Combined plot error: {e}",
                     transform=ax_comb.transAxes, ha="center", color=TICK)

    ax_comb.set_title("Combined — all sectors stitched", fontsize=9,
                      color=TITLE, fontweight="bold")
    ax_comb.set_xlabel("Phase", fontsize=8)
    ax_comb.set_ylabel("Normalised Flux", fontsize=8)

    # ── Supertitle ────────────────────────────────────────────────────────
    fig.suptitle(
        f"TIC {tic_id}  ·  P = {period:.5f} d  ·  {verdict}",
        fontsize=11, color=TITLE, fontweight="bold", y=0.975,
    )

    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return None
    return fig


# ── t0 auto-detection ─────────────────────────────────────────────────────────

def find_t0(lc, period: float) -> float:
    """
    Find transit epoch by folding at the given period and locating the flux minimum.

    Avoids BLS (which requires len(periods) > 1) by folding with an arbitrary
    epoch, then reading off the phase of the deepest bin.
    """
    from astropy import units as u

    lc_flat = lc.normalize().flatten(window_length=401).remove_outliers(sigma=4)

    # Fold with first time point as reference epoch
    arb_t0 = float(lc_flat.time[0].value)
    lc_fold = lc_flat.fold(period=period * u.day, epoch_time=arb_t0 * u.day)
    lc_bin  = lc_fold.bin(time_bin_size=0.01)

    ph = np.asarray(lc_bin.phase.value, dtype=float)   # in DAYS
    fl = np.asarray(lc_bin.flux.value,  dtype=float)

    valid = np.isfinite(fl)
    if not valid.any():
        return arb_t0

    # Phase of the deepest bin = offset of transit from arb_t0
    mid_idx = int(np.nanargmin(fl))
    return arb_t0 + float(ph[mid_idx])


# ── Main entry point ──────────────────────────────────────────────────────────

def verify(tic_id: int, period: float, t0: float | None = None,
           out_dir: Path | None = None) -> dict:
    """
    Full verification pipeline for one candidate.

    Returns dict with:
        tic_id, period, t0, sector_results, consistency_score, verdict,
        n_sectors_checked, n_sectors_consistent, median_depth_ppm, figure
    """
    # Output directory
    if out_dir is None:
        out_dir = VERIFY_DIR / f"TIC_{tic_id}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[verify] TIC {tic_id}  P={period:.5f}d")

    # Download all sectors
    sector_data = download_all_sectors(tic_id)
    if not sector_data:
        result = {
            "tic_id": tic_id, "period": period, "t0": t0,
            "sector_results": [],
            "consistency_score": 0.0,
            "verdict": "No TESS data found for this star",
            "n_sectors_checked": 0,
            "n_sectors_consistent": 0,
            "median_depth_ppm": 0.0,
            "eb_warning": None,
            "figure": None,
        }
        _save(result, out_dir)
        return result

    _log(f"[verify] Found {len(sector_data)} sector(s): "
         f"{[s for s, _ in sector_data]}")

    # Auto-detect t0 if not given
    if t0 is None:
        for sec_num, lc in sector_data:
            if len(lc) >= 200:
                try:
                    t0 = find_t0(lc, period)
                    _log(f"[verify] Auto-detected t0={t0:.5f} from sector {sec_num}")
                    break
                except Exception:
                    continue
    if t0 is None:
        t0 = float(sector_data[0][1].time[0].value)  # fallback

    # Per-sector measurements
    sector_results = []
    for sec_num, lc in sector_data:
        r = measure_sector(lc, period, t0)
        r["sector"] = int(sec_num)
        sector_results.append(r)
        sr = r.get("skip_reason") or f"depth={r['depth_ppm']:.0f}ppm SNR={r['snr']:.1f}"
        _log(f"[verify]   Sector {sec_num}: {sr}")

    # Consistency scoring
    consistency = compute_consistency(sector_results)
    _log(f"[verify] {consistency['verdict']}")

    # Plot
    fig = make_verification_plot(
        sector_data, sector_results, period, t0, tic_id, consistency["verdict"],
        out_path=str(out_dir / "verification_plot.png"),
    )

    result = {
        "tic_id":              tic_id,
        "period":              period,
        "t0":                  round(t0, 5),
        "sector_results":      sector_results,
        **consistency,
        "eb_warning":          consistency.get("eb_warning"),
        "figure":              None,   # not JSON-serialisable; returned for in-process callers
    }

    _save(result, out_dir)

    # Return figure separately (not in JSON)
    result["figure"] = fig
    return result


def _save(result: dict, out_dir: Path):
    """Save verification.json (excluding the figure object)."""
    safe = {k: v for k, v in result.items() if k != "figure"}
    (out_dir / "verification.json").write_text(json.dumps(safe, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-sector transit verification")
    ap.add_argument("--tic",     type=int,   required=True,  help="TESS Input Catalog ID")
    ap.add_argument("--period",  type=float, required=True,  help="Candidate period (days)")
    ap.add_argument("--t0",      type=float, default=None,   help="Transit epoch BTJD (auto if omitted)")
    ap.add_argument("--out-dir", type=str,   default=None,   help="Output directory (default: data/verification/TIC_XXXXX/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    result  = verify(args.tic, args.period, t0=args.t0, out_dir=out_dir)

    _log()
    _log(f"═══ Verification Result ══════════════════════")
    _log(f"  TIC:          {result['tic_id']}")
    _log(f"  Period:       {result['period']:.5f} d")
    _log(f"  Sectors:      {result['n_sectors_checked']} checked, "
         f"{result['n_sectors_consistent']} consistent")
    _log(f"  Consistency:  {result['consistency_score']:.0%}")
    _log(f"  Verdict:      {result['verdict']}")
    _log(f"  Plot:         {VERIFY_DIR}/TIC_{result['tic_id']}/verification_plot.png")


if __name__ == "__main__":
    main()
