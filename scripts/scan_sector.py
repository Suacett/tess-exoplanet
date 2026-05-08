#!/usr/bin/env python3
"""
Exoplanet sector scanner — BLS search across all SPOC 2-min targets.

Usage:
    python scan_sector.py --sector 99
    python scan_sector.py --sector 99 --limit 20
    python scan_sector.py --sector 99 --workers 16
    python scan_sector.py --sector 99 --limit 20 --watch
    python scan_sector.py --sector 99 --prefetch   # local FITS only, zero network — run prefetch_sector.py first
"""
import argparse
import base64
import csv
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import threading
from pathlib import Path

import numpy as np
import requests
import psutil

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    MofNCompleteColumn, TimeRemainingColumn, TextColumn,
)
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
TESS_DIR      = DATA_DIR / "tess"
RESULTS_DIR   = DATA_DIR / "results"
TRACKING_FILE = DATA_DIR / "scanned_sectors.json"
SCRIPTS_DIR   = Path(__file__).parent

# GPU BLS service — opt-in. Set GPU_BLS_URL=http://<host>:9876 to enable.
# Unset (default) = CPU-only mode; behaviour is identical for end users.
_GPU_URL   = os.environ.get("GPU_BLS_URL")   # None = CPU-only
_GPU_BATCH = 100                              # stars per HTTP request


CANDIDATE_THRESHOLD = 7.0    # ⚡ alert threshold
PLOT_THRESHOLD      = 9.0    # generate 4-panel PNG above this
LOG_EVERY           = 10     # print live stats every N stars


# ── Logging (file only — console output via rich) ─────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.NullHandler()],
)
# Suppress lightkurve's chatty per-file messages
# Note: do NOT suppress astropy logger here — astropy uses a custom Logger subclass
# that must initialise before setLevel is called, otherwise its _set_defaults breaks.
# lightkurve suppression is safe because lightkurve uses standard logging.
logging.getLogger("lightkurve").setLevel(logging.ERROR)
logging.getLogger("lightkurve.io").setLevel(logging.ERROR)


# ── Target discovery ──────────────────────────────────────────────────────────

def get_tic_ids_for_sector(sector: int) -> list:
    url = (
        f"https://archive.stsci.edu/missions/tess/download_scripts/sector/"
        f"tesscurl_sector_{sector:02d}_lc.sh"
    )
    console.log(f"Fetching sector {sector} target list…")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    sector_str = f"{sector:04d}"
    # dict.fromkeys preserves insertion order while deduplicating
    # (a TIC can appear multiple times if observed on >1 CCD/camera)
    tic_ids = list(dict.fromkeys(
        int(m)
        for m in re.findall(rf"s{sector_str}-(\d+)-\d{{4}}-s_lc\.fits", resp.text)
    ))
    return tic_ids


def find_local_fits(sector: int, tic_id: int) -> Path | None:
    for dirname in (f"sector{sector:02d}", f"sector{sector}"):
        d = TESS_DIR / dirname
        if d.exists():
            hits = list(d.glob(f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"))
            if hits:
                return hits[0]
    return None


def _load_and_flatten(args: tuple):
    """Load + flatten one lightcurve. Thread-safe (ThreadPoolExecutor).
    Returns (tic_id, time_list, flux_list) or None on failure."""
    tic_id, sector = args
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import lightkurve as lk
        local_path = find_local_fits(sector, tic_id)
        if local_path:
            lc = lk.read(str(local_path), quality_bitmask="default")
        else:
            sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS",
                                       author="SPOC", sector=sector)
            if len(sr) == 0:
                return None
            lc = sr[0].download(quality_bitmask="default")
        if lc is None or len(lc) < 200:
            return None
        lc_flat = lc.normalize().flatten(window_length=401).remove_outliers(sigma=4)
        if len(lc_flat) < 200:
            return None
        return (tic_id, lc_flat.time.value.tolist(), lc_flat.flux.value.tolist())
    except Exception as exc:
        logging.debug(f"TIC {tic_id} load: {exc}")
        return None


def _depth_from_arrays(time_list, flux_list, period, t0):
    """Compute depth/duration/SNR from pre-flattened arrays and BLS results."""
    import warnings
    warnings.filterwarnings("ignore")
    import lightkurve as lk
    from astropy import units as u
    time_arr  = np.array(time_list, dtype=float)
    flux_arr  = np.array(flux_list, dtype=float)
    time_span = time_arr[-1] - time_arr[0]
    n_transits = max(1, int(time_span / period))
    lc      = lk.LightCurve(time=time_arr, flux=flux_arr)
    lc_fold = lc.fold(period=period * u.day, epoch_time=t0 * u.day)
    lc_bin  = lc_fold.bin(time_bin_size=0.01)
    flux_b  = np.ma.filled(np.asarray(lc_bin.flux.value), fill_value=np.nan).astype(float)
    phase_b = np.asarray(lc_bin.phase.value, dtype=float)
    baseline  = float(np.nanmedian(flux_b[np.abs(phase_b) > 0.15]))
    in_tr     = flux_b[np.abs(phase_b) < 0.05]
    min_flux  = float(np.nanmin(in_tr)) if in_tr.size else baseline
    depth_ppm = (baseline - min_flux) / baseline * 1e6
    half_lev  = baseline - (baseline - min_flux) * 0.5
    dur_hours = float(np.sum(flux_b < half_lev) * 0.01 * period * 24)
    snr       = depth_ppm / (float(np.nanstd(flux_b)) * 1e6 + 1e-9)
    return {
        "depth_ppm":      round(depth_ppm, 1),
        "duration_hours": round(dur_hours, 3),
        "snr":            round(snr, 2),
        "n_transits":     n_transits,
    }


def _gpu_bls_batch(batch):
    """POST a batch of stars to the GPU BLS service.
    Returns list of result dicts, or None on any error (caller falls back to CPU)."""
    if not _GPU_URL:
        return None
    try:
        resp = requests.post(
            _GPU_URL + "/bls",
            json={"stars": [{"tic_id": t, "time": tm, "flux": fl}
                             for t, tm, fl in batch]},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logging.warning(f"GPU batch error: {exc}")
        return None


# ── Per-star worker ───────────────────────────────────────────────────────────

def process_tic(args: tuple) -> dict | None:
    tic_id, sector, period_step = args
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import lightkurve as lk
        from astropy import units as u

        local_path = find_local_fits(sector, tic_id)
        if local_path:
            lc = lk.read(str(local_path), quality_bitmask="default")
        else:
            sr = lk.search_lightcurve(
                f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector,
            )
            if len(sr) == 0:
                return None
            lc = sr[0].download(quality_bitmask="default")

        if lc is None or len(lc) < 200:
            return None

        lc_flat = lc.normalize().flatten(window_length=401).remove_outliers(sigma=4)
        if len(lc_flat) < 200:
            return None

        periods   = np.arange(0.5, 14.0, period_step)
        durations = np.arange(0.05, 0.20, 0.02)
        blsm = lc_flat.to_periodogram(method="bls", period=periods, duration=durations)

        best_p  = float(blsm.period_at_max_power.value)
        best_t0 = float(blsm.transit_time_at_max_power.value)

        # Compute SDE (Signal Detection Efficiency) — normalised power, ~7-50 for real signals
        power_arr = np.asarray(blsm.power.value, dtype=float)
        p_mean = float(np.mean(power_arr))
        p_std  = float(np.std(power_arr))
        sde    = (float(blsm.max_power) - p_mean) / p_std if p_std > 0 else 0.0

        # Number of transits observed
        time_span  = float(lc_flat.time[-1].value - lc_flat.time[0].value)
        n_transits = max(1, int(time_span / best_p))

        lc_fold = lc_flat.fold(period=best_p * u.day, epoch_time=best_t0 * u.day)
        lc_bin  = lc_fold.bin(time_bin_size=0.01)

        flux  = np.ma.filled(np.asarray(lc_bin.flux.value), fill_value=np.nan).astype(float)
        phase = np.asarray(lc_bin.phase.value, dtype=float)

        baseline  = float(np.nanmedian(flux[np.abs(phase) > 0.15]))
        in_tr     = flux[np.abs(phase) < 0.05]
        min_flux  = float(np.nanmin(in_tr)) if in_tr.size else baseline
        depth_ppm = (baseline - min_flux) / baseline * 1e6

        half_lev      = baseline - (baseline - min_flux) * 0.5
        below_half    = np.sum(flux < half_lev) * 0.01
        dur_hours     = below_half * best_p * 24

        snr = depth_ppm / (np.nanstd(flux) * 1e6) if np.nanstd(flux) > 0 else 0.0

        return {
            "tic_id":         tic_id,
            "period":         round(best_p, 5),
            "depth_ppm":      round(depth_ppm, 1),
            "duration_hours": round(dur_hours, 3),
            "bls_power":      round(sde, 4),   # SDE, not raw power
            "snr":            round(snr, 2),
            "t0":             round(best_t0, 5),
            "n_transits":     n_transits,
        }

    except Exception as exc:
        logging.debug(f"TIC {tic_id}: {exc}")
        return None


# ── Post-scan: plot generation ────────────────────────────────────────────────

def _plot_one(args: tuple) -> tuple:
    """Worker function for parallel plot generation — must be top-level for pickling."""
    r, sector, plot_dir_str, scripts_dir_str = args
    import sys, warnings
    sys.path.insert(0, scripts_dir_str)
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import lightkurve as lk
    from plot_candidate import make_4panel

    tic_id = r["tic_id"]
    plot_dir = Path(plot_dir_str)
    out_png = plot_dir / f"tic_{tic_id}_s{sector:02d}.png"
    try:
        local = find_local_fits(sector, tic_id)
        if local:
            lc_raw = lk.read(str(local), quality_bitmask="default")
        else:
            sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS",
                                       author="SPOC", sector=sector)
            if len(sr) == 0:
                return (tic_id, None, "no data")
            lc_raw = sr[0].download(quality_bitmask="default")
        lc_flat = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)
        make_4panel(
            lc_raw=lc_raw,
            lc_flat=lc_flat,
            period_d=r["period"],
            t0_btjd=r["t0"],
            tic_id=tic_id,
            sector=sector,
            depth_ppm=r["depth_ppm"],
            bls_power=r["bls_power"],
            out_path=str(out_png),
        )
        return (tic_id, out_png, None)
    except Exception as e:
        return (tic_id, None, str(e))


def generate_plots(candidates: list, sector: int, plot_dir: Path,
                   workers: int = 1) -> list:
    """Generate 4-panel PNGs in parallel. Returns list of (tic_id, path)."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    strong = [r for r in candidates if r["bls_power"] >= PLOT_THRESHOLD]
    if not strong:
        return []

    console.print(Rule(f"[bold]Generating {len(strong)} plots (BLS power ≥ {PLOT_THRESHOLD}) — {workers} workers[/bold]"))

    work = [(r, sector, str(plot_dir), str(SCRIPTS_DIR)) for r in strong]
    generated = []
    with mp.Pool(processes=min(workers, len(strong)),
                 maxtasksperchild=8) as pool:
        for tic_id, out_png, err in pool.imap_unordered(_plot_one, work, chunksize=4):
            if err:
                console.log(f"  [red]✗[/red] TIC {tic_id}: {err}")
            else:
                generated.append((tic_id, out_png))
                console.log(f"  [green]✓[/green] TIC {tic_id} → {out_png.name}")
    return generated


# ── Post-scan: HTML report ────────────────────────────────────────────────────

def generate_html_report(
    sector: int,
    candidates: list,
    n_scanned: int,
    elapsed: float,
    plot_dir: Path,
    out_html: Path,
    n_raw: int = 0,
    filter_counts: dict = None,
):
    def b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode()

    fc = filter_counts or {}
    n_filtered = sum(fc.values())
    filter_line = ""
    if n_raw:
        pct_filt = 100 * n_filtered / max(n_raw, 1)
        filter_line = (
            f"<p style='color:#8b949e;font-size:13px;margin-bottom:16px'>"
            f"🔍 {n_scanned:,} stars searched &nbsp;→&nbsp; "
            f"📊 {n_raw:,} raw BLS detections &nbsp;→&nbsp; "
            f"🗑️ {n_filtered:,} filtered ({pct_filt:.1f}%) &nbsp;→&nbsp; "
            f"✅ {len(candidates):,} candidates</p>"
        )

    rows = ""
    for i, r in enumerate(candidates[:50], 1):
        cls  = r.get("classification", "")
        sde  = r["bls_power"]
        power_class = "high" if sde >= 12 else "med" if sde >= PLOT_THRESHOLD else "low"
        rows += f"""
        <tr class="{power_class}">
          <td>{i}</td>
          <td><a href="#tic{r['tic_id']}">TIC {r['tic_id']}</a></td>
          <td>{r['period']:.5f}</td>
          <td>{r['depth_ppm']:.1f}</td>
          <td>{r['duration_hours']:.2f}</td>
          <td><b>{sde:.2f}</b></td>
          <td>{r['snr']:.2f}</td>
          <td>{cls}</td>
        </tr>"""

    cards = ""
    for r in candidates:
        if r["bls_power"] < PLOT_THRESHOLD:
            continue
        png = plot_dir / f"tic_{r['tic_id']}_s{sector:02d}.png"
        img_tag = (
            f'<img src="data:image/png;base64,{b64(png)}" width="100%">'
            if png.exists() else "<p><i>Plot not generated</i></p>"
        )
        priority = "🔴 HIGH PRIORITY" if r["bls_power"] >= 12 else "⚡ Candidate"
        cards += f"""
        <div class="card" id="tic{r['tic_id']}">
          <h2>{priority} &nbsp; TIC {r['tic_id']}</h2>
          <p>Period: <b>{r['period']:.5f} d</b> &nbsp;|&nbsp;
             Depth: <b>{r['depth_ppm']:.0f} ppm</b> &nbsp;|&nbsp;
             BLS power: <b>{r['bls_power']:.2f}</b> &nbsp;|&nbsp;
             Duration: <b>{r['duration_hours']:.2f} h</b></p>
          {img_tag}
        </div>"""

    n_cands  = len(candidates)
    n_strong = sum(1 for r in candidates if r["bls_power"] >= PLOT_THRESHOLD)
    n_planet = sum(1 for r in candidates if r.get("classification") == "Planet candidate")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sector {sector} Scan Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif;
          padding: 24px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  .meta {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 16px 24px; text-align: center; min-width: 140px; }}
  .stat .val {{ font-size: 32px; font-weight: bold; color: #58a6ff; }}
  .stat .lab {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  h2.sec {{ color: #c9d1d9; margin: 32px 0 12px; border-bottom: 1px solid #30363d;
             padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 600; padding: 10px 12px;
        text-align: left; border-bottom: 1px solid #30363d; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr.high td {{ background: rgba(255,123,114,0.08); }}
  tr.high td:nth-child(6) {{ color: #ff7b72; font-weight: bold; }}
  tr.med td {{ background: rgba(88,166,255,0.06); }}
  tr.med td:nth-child(6) {{ color: #58a6ff; }}
  tr:hover td {{ background: rgba(255,255,255,0.04); }}
  a {{ color: #58a6ff; text-decoration: none; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 24px; margin-bottom: 24px; }}
  .card h2 {{ color: #e6edf3; margin-bottom: 8px; font-size: 18px; }}
  .card p {{ color: #8b949e; font-size: 13px; margin-bottom: 16px; }}
  .card img {{ border-radius: 4px; }}
</style>
</head>
<body>
<h1>⭐ Sector {sector} Scan Report</h1>
<p class="meta">Generated on scan completion &nbsp;|&nbsp; {n_scanned:,} stars scanned in {elapsed:.1f}s</p>
{filter_line}
<div class="stats">
  <div class="stat"><div class="val">{n_scanned:,}</div><div class="lab">Stars Scanned</div></div>
  <div class="stat"><div class="val">{n_planet}</div><div class="lab">🟢 Planet Candidates</div></div>
  <div class="stat"><div class="val">{n_strong}</div><div class="lab">Strong SDE (≥{PLOT_THRESHOLD})</div></div>
  <div class="stat"><div class="val">{elapsed:.0f}s</div><div class="lab">Scan Time</div></div>
</div>

<h2 class="sec">All Candidates (ranked by SDE)</h2>
<table>
  <thead>
    <tr><th>#</th><th>TIC ID</th><th>Period (d)</th>
        <th>Depth (ppm)</th><th>Duration (h)</th><th>SDE</th><th>SNR</th><th>Classification</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<h2 class="sec">Transit Plots (BLS power ≥ {PLOT_THRESHOLD})</h2>
{cards if cards else '<p style="color:#8b949e">No strong signals detected.</p>'}

</body>
</html>
"""
    out_html.write_text(html)
    console.log(f"[green]HTML report:[/green] {out_html}")


# ── Live CPU watcher (--watch flag) ──────────────────────────────────────────

def _watch_cpu(stop_event: threading.Event, interval: float = 10.0):
    """Background thread: print per-core CPU % every `interval` seconds."""
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        try:
            per_core = psutil.cpu_percent(percpu=True)
            ram = psutil.virtual_memory()
            bar = lambda p: ("█" * int(p / 5)).ljust(20) + f" {p:4.0f}%"
            lines = [f"[dim]── CPU / RAM ──────────────────────────────[/dim]"]
            for i, p in enumerate(per_core):
                color = "green" if p < 50 else "yellow" if p < 80 else "red"
                lines.append(f"[dim]  Core {i:>2}[/dim] [{color}]{bar(p)}[/{color}]")
            lines.append(
                f"[dim]  RAM    {ram.percent:4.0f}%  "
                f"{ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB[/dim]"
            )
            console.print("\n".join(lines))
        except Exception:
            pass


# ── Banner & summary ─────────────────────────────────────────────────────────

def print_banner(sector: int, n_stars: int, workers: int):
    ram = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    freq_str = f"{cpu_freq.current/1000:.2f} GHz" if cpu_freq else "N/A"

    console.print(Panel(
        f"[bold cyan]TESS Sector {sector}[/bold cyan]\n\n"
        f"  Stars to scan : [bold white]{n_stars:,}[/bold white]\n"
        f"  CPU cores     : [bold white]{workers}[/bold white] of {os.cpu_count()} available  ({freq_str})\n"
        f"  RAM available : [bold white]{ram.available/1e9:.1f} GB[/bold white]"
        f"  /  {ram.total/1e9:.1f} GB total\n"
        f"  BLS grid      : [dim]P 0.5–14 d, dur 0.05–0.20 d[/dim]",
        title="[bold yellow]⭐ Exoplanet BLS Scanner[/bold yellow]",
        border_style="bright_blue",
        padding=(1, 4),
    ))


def print_summary(candidates: list, n_scanned: int, n_skip: int, elapsed: float,
                  n_raw: int = 0, filter_counts: dict = None):
    n_cands  = len(candidates)
    rate     = n_scanned / elapsed if elapsed > 0 else 0
    fc       = filter_counts or {}

    # Header rule
    console.print()
    console.print(Rule("[bold yellow]Scan Complete[/bold yellow]", style="bright_blue"))

    # Stats row
    stats = Table(box=None, show_header=False, padding=(0, 3))
    stats.add_column(style="dim")
    stats.add_column(style="bold white")
    stats.add_row("🔍 Stars searched",    f"{n_scanned:,}")
    stats.add_row("📊 Raw BLS detections",f"{n_raw:,}")
    if fc:
        filt_parts = []
        if fc.get("depth_hi"):  filt_parts.append(f"{fc['depth_hi']:,} eclipsing binaries (depth >50k ppm)")
        if fc.get("depth_lo"):  filt_parts.append(f"{fc['depth_lo']:,} too shallow (<50 ppm)")
        if fc.get("period_lo"): filt_parts.append(f"{fc['period_lo']:,} bad period (<0.2 d)")
        if fc.get("n_transits"):filt_parts.append(f"{fc['n_transits']:,} single-transit (<2 observed)")
        if fc.get("sde"):       filt_parts.append(f"{fc['sde']:,} noise (SDE <{CANDIDATE_THRESHOLD})")
        n_filtered = sum(fc.values())
        stats.add_row("🗑️  Filtered out", f"{n_filtered:,}  (" + ", ".join(filt_parts) + ")")
    n_planet   = sum(1 for r in candidates if r.get("classification") == "Planet candidate")
    n_inspect  = sum(1 for r in candidates if r.get("classification") == "Needs inspection")
    n_eb       = sum(1 for r in candidates if r.get("classification") == "Eclipsing binary")
    stats.add_row("🟢 Planet candidates", f"{n_planet}")
    stats.add_row("🟡 Needs inspection",  f"{n_inspect}")
    stats.add_row("🔴 Eclipsing binary",  f"{n_eb}")
    stats.add_row("Elapsed",             f"{elapsed:.1f}s  ({rate:.2f} stars/s)")
    console.print(stats)
    console.print()

    if not candidates:
        console.print("[dim]No BLS candidates found after filtering.[/dim]")
        return

    # Top-5 table
    top5 = candidates[:5]
    t = Table(
        title="[bold]Top Candidates[/bold]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        show_lines=False,
        padding=(0, 2),
    )
    t.add_column("TIC ID",         justify="right")
    t.add_column("Period (d)",     justify="right")
    t.add_column("Depth (ppm)",    justify="right")
    t.add_column("Dur (h)",        justify="right")
    t.add_column("SDE",            justify="right")
    t.add_column("Class",          justify="left")

    for r in top5:
        sde = r["bls_power"]
        style = "bold red" if sde >= 12 else "yellow" if sde >= PLOT_THRESHOLD else "white"
        t.add_row(
            str(r["tic_id"]),
            f"{r['period']:.5f}",
            f"{r['depth_ppm']:.1f}",
            f"{r['duration_hours']:.2f}",
            Text(f"{sde:.2f}", style=style),
            r.get("classification", ""),
        )
    console.print(t)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLS scan of a TESS sector")
    parser.add_argument("--sector",  type=int, required=True)
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only first N stars (for testing)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--output",  type=str, default=None,
                        help="Results directory (default: data/results/sectorNN/)")
    parser.add_argument("--no-plots",  action="store_true",
                        help="Skip 4-panel PNG generation")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--watch",  action="store_true",
                        help="Print periodic per-core CPU stats while scanning")
    parser.add_argument("--period-step", type=float, default=0.02,
                        help="BLS period grid step in days (default 0.02, was 0.01)")
    parser.add_argument("--prefetch", action="store_true",
                        help="Use only local FITS files — no downloading. "
                             "Run prefetch_sector.py first to populate the cache.")
    args = parser.parse_args()

    sector  = args.sector
    workers = min(args.workers, os.cpu_count() or 16)

    out_dir  = Path(args.output) if args.output else RESULTS_DIR / f"sector{sector:02d}"
    plot_dir = out_dir / "plots"
    out_csv  = out_dir / "bls_results.csv"
    out_html = out_dir / "scan_report.html"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also write a log file
    fh = logging.FileHandler(out_dir / "scan.log")
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)

    # ── Fetch targets ────────────────────────────────────────────────────
    if args.prefetch:
        # Local-only mode: discover TIC IDs from files already on disk
        dl_dir = TESS_DIR / f"sector{sector:02d}"
        if not dl_dir.exists():
            console.print(f"[red]No local FITS data at {dl_dir}[/red]")
            console.print(
                f"[yellow]Run first:[/yellow]  "
                f"python prefetch_sector.py --sector {sector}"
            )
            sys.exit(1)
        sector_str = f"{sector:04d}"
        tic_ids = list(dict.fromkeys(
            int(m.group(1))
            for f in sorted(dl_dir.glob("*_lc.fits"))
            for m in [re.search(rf"s{sector_str}-(\d+)-\d{{4}}-s_lc\.fits", f.name)]
            if m
        ))
        if not tic_ids:
            console.print(f"[red]No *_lc.fits files found in {dl_dir}[/red]")
            sys.exit(1)
        console.log(
            f"[green]Prefetch mode:[/green] found {len(tic_ids):,} local FITS "
            f"files in {dl_dir}"
        )
    else:
        try:
            tic_ids = get_tic_ids_for_sector(sector)
        except Exception as e:
            console.print(f"[red]Could not fetch target list: {e}[/red]")
            sys.exit(1)

    if args.limit:
        tic_ids = tic_ids[: args.limit]

    n_total = len(tic_ids)
    work    = [(tid, sector, args.period_step) for tid in tic_ids]

    print_banner(sector, n_total, workers)

    # ── Optional --watch thread ──────────────────────────────────────────
    stop_watch = threading.Event()
    if args.watch:
        watch_thread = threading.Thread(target=_watch_cpu, args=(stop_watch,), daemon=True)
        watch_thread.start()

    # Pre-filter constants
    MAX_DEPTH_PPM  = 50_000   # deeper = almost certainly EB
    MIN_DEPTH_PPM  = 50       # shallower = noise
    MIN_PERIOD_D   = 0.2      # shorter = unphysical for planets
    MIN_TRANSITS   = 2        # need at least 2 to confirm periodicity

    # ── Parallel BLS scan ────────────────────────────────────────────────
    all_results = []   # everything that BLS ran on
    candidates  = []   # after filtering
    t0          = time.time()   # shared start time for both CPU and GPU paths

    # ── GPU probe ────────────────────────────────────────────────────────
    _use_gpu = False
    if _GPU_URL:
        try:
            _use_gpu = requests.get(_GPU_URL + "/health", timeout=2).ok
        except Exception:
            pass
        console.print(
            f"[green]GPU service online → {_GPU_URL}[/green]" if _use_gpu
            else "[yellow]GPU service unreachable — using CPU[/yellow]"
        )

    _gpu_scan_done = False
    if _use_gpu:
        from concurrent.futures import ThreadPoolExecutor
        import math as _math
        skipped = 0
        _preprocessed = []
        console.print(
            f"[cyan]Loading {n_total:,} lightcurves ({min(workers, 16)} threads)…[/cyan]"
        )
        with ThreadPoolExecutor(max_workers=min(workers, 16)) as _ex:
            for _r in _ex.map(_load_and_flatten,
                               [(tid, sector) for tid in tic_ids]):
                if _r is None:
                    skipped += 1
                else:
                    _preprocessed.append(_r)
        _star_lut = {r[0]: (r[1], r[2]) for r in _preprocessed}
        _n_bat = max(1, _math.ceil(len(_preprocessed) / _GPU_BATCH))
        console.print(
            f"[cyan]GPU BLS: {len(_preprocessed):,} stars in {_n_bat} batches[/cyan]"
        )
        _done = 0
        _pf   = out_dir / "scan_progress.json"
        for _b0 in range(0, len(_preprocessed), _GPU_BATCH):
            _batch   = _preprocessed[_b0:_b0 + _GPU_BATCH]
            _gpu_res = _gpu_bls_batch(_batch)
            if _gpu_res is None:
                # Batch CPU fallback — GPU service unavailable for this batch
                _cw = [(r[0], sector, args.period_step) for r in _batch]
                with mp.Pool(processes=min(workers, len(_cw))) as _p:
                    for _cr in _p.imap_unordered(process_tic, _cw):
                        if _cr is not None:
                            all_results.append(_cr)
                        _done += 1
            else:
                for _gr in _gpu_res:
                    _tid = _gr["tic_id"]
                    if _tid in _star_lut:
                        _d = _depth_from_arrays(
                            *_star_lut[_tid], _gr["period"], _gr["t0"]
                        )
                        all_results.append({
                            "tic_id":    _tid,
                            "period":    _gr["period"],
                            "t0":        _gr["t0"],
                            "bls_power": _gr["sde"],
                            **_d,
                        })
                _done  += len(_batch)
                skipped += len(_batch) - len(_gpu_res)
            _el = time.time() - t0
            _rt = _done / _el if _el > 0 else 0
            console.print(
                f"  GPU batch {_b0 // _GPU_BATCH + 1}/{_n_bat}: "
                f"{_done + skipped}/{n_total} "
                f"({'GPU' if _gpu_res else 'CPU fallback'})"
            )
            try:
                _pf.write_text(json.dumps({
                    "sector": sector, "done": _done + skipped,
                    "total": n_total, "candidates": len(all_results),
                    "rate": round(_rt, 2),
                }))
            except Exception:
                pass
        _gpu_scan_done = True

    if not _gpu_scan_done:
        skipped = 0

        bls_label = (
            "[bold cyan]BLS (local) sector {task.fields[sector]}[/bold cyan]"
            if args.prefetch else
            "[bold cyan]Scanning sector {task.fields[sector]}[/bold cyan]"
        )
        progress_cols = [
            SpinnerColumn(),
            TextColumn(bls_label),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[rate]:.2f}/s[/dim]"),
        ]

        progress_file = out_dir / "scan_progress.json"

        with Progress(*progress_cols, console=console, transient=False) as prog:
            task = prog.add_task("Scanning…", total=n_total, sector=sector, rate=0.0)

            with mp.Pool(processes=workers) as pool:
                for i, result in enumerate(
                    pool.imap_unordered(process_tic, work, chunksize=16), 1
                ):
                    elapsed_now = time.time() - t0
                    rate = i / elapsed_now if elapsed_now > 0 else 0
                    prog.update(task, advance=1, rate=rate)

                    if result is None:
                        skipped += 1
                    else:
                        all_results.append(result)

                    # Write progress for dashboard to read
                    if i % LOG_EVERY == 0 or i == n_total:
                        try:
                            progress_file.write_text(json.dumps({
                                "sector": sector, "total": n_total, "done": i,
                                "candidates": len(all_results), "rate": round(rate, 2),
                            }))
                        except Exception:
                            pass

    stop_watch.set()
    elapsed = time.time() - t0

    # ── Apply pre-filters and classify ───────────────────────────────────
    filter_counts = {"sde": 0, "depth_hi": 0, "depth_lo": 0, "period_lo": 0, "n_transits": 0}
    for r in all_results:
        sde    = r["bls_power"]
        depth  = r["depth_ppm"]
        period = r["period"]
        n_tr   = r.get("n_transits", 2)

        if sde < CANDIDATE_THRESHOLD:
            filter_counts["sde"] += 1
            continue
        if depth > MAX_DEPTH_PPM:
            filter_counts["depth_hi"] += 1
            r["classification"] = "Eclipsing binary"
            candidates.append(r)   # keep in CSV but classified as EB
            continue
        if depth < MIN_DEPTH_PPM:
            filter_counts["depth_lo"] += 1
            continue
        if period < MIN_PERIOD_D:
            filter_counts["period_lo"] += 1
            continue
        if n_tr < MIN_TRANSITS:
            filter_counts["n_transits"] += 1
            continue

        # Classify
        if depth <= 20_000 and sde >= PLOT_THRESHOLD and 0.5 <= period <= 100:
            r["classification"] = "Planet candidate"
        elif sde >= PLOT_THRESHOLD:
            r["classification"] = "Needs inspection"
        else:
            r["classification"] = "Weak signal"
        candidates.append(r)

        # Live print for strong candidates
        sde_val = r["bls_power"]
        if sde_val >= CANDIDATE_THRESHOLD:
            color = "bold red" if sde_val >= 12 else "bold yellow"
            console.print(
                f"  [bold green]⚡ CANDIDATE[/bold green]  "
                f"TIC [cyan]{r['tic_id']}[/cyan]  "
                f"P=[green]{r['period']:.4f} d[/green]  "
                f"depth=[magenta]{r['depth_ppm']:.0f} ppm[/magenta]  "
                f"SDE=[{color}]{sde_val:.2f}[/{color}]  "
                f"[dim]{r['classification']}[/dim]"
            )

    # ── Sort and save CSV ────────────────────────────────────────────────
    candidates.sort(key=lambda r: r["bls_power"], reverse=True)

    fieldnames = ["tic_id", "period", "depth_ppm", "duration_hours",
                  "bls_power", "snr", "t0", "n_transits", "classification",
                  "n_sectors_checked", "n_sectors_consistent",
                  "consistency_score", "verified", "eb_warning"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    console.log(f"[green]CSV saved:[/green] {out_csv}  ({len(candidates)} candidates)")

    # Write filter stats for hunt.py to read
    stats_file = out_dir / "scan_filter_stats.json"
    stats_file.write_text(json.dumps({
        "n_raw": len(all_results),
        "n_candidates": len(candidates),
        "filter_counts": filter_counts,
    }))

    # ── Print summary ────────────────────────────────────────────────────
    print_summary(candidates, len(tic_ids) - skipped, skipped, elapsed,
                  n_raw=len(all_results), filter_counts=filter_counts)

    # ── Generate plots ───────────────────────────────────────────────────
    if not args.no_plots:
        generated = generate_plots(candidates, sector, plot_dir, workers=workers)
    else:
        generated = []

    # ── Generate HTML report ─────────────────────────────────────────────
    if not args.no_report:
        generate_html_report(
            sector=sector,
            candidates=candidates,
            n_scanned=len(tic_ids) - skipped,
            elapsed=elapsed,
            plot_dir=plot_dir,
            out_html=out_html,
            n_raw=len(all_results),
            filter_counts=filter_counts,
        )

    # ── Update tracking ──────────────────────────────────────────────────
    try:
        tracking = json.loads(TRACKING_FILE.read_text()) if TRACKING_FILE.exists() else {}
        scanned  = set(tracking.get("scanned", []))
        scanned.add(sector)
        tracking["scanned"] = sorted(scanned)
        TRACKING_FILE.write_text(json.dumps(tracking, indent=2))
    except Exception as e:
        console.log(f"[yellow]Warning: could not update tracking file: {e}[/yellow]")

    console.print()
    console.print(Rule(style="bright_blue"))
    console.print(f"  [dim]Results:[/dim]  {out_dir}")
    if out_html.exists():
        console.print(f"  [dim]Report: [/dim]  [link={out_html}]{out_html}[/link]")
    console.print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
