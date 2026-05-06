#!/usr/bin/env python3
"""
BLS transit search on Kepler lightcurves.

Kepler observed ~200,000 stars for 4 continuous years with higher photometric
precision than TESS. The 4-year baseline means BLS can stack many more transits,
making weaker signals detectable. This is where the 16-core Threadripper earns
its keep — Kepler lightcurves are ~70,000 cadences (vs ~18,000 for one TESS sector).

Target selection (in priority order):
  1. --tic-file: user-supplied list of KIC IDs
  2. --catalog koi: Kepler Objects of Interest (known transit signals)
  3. --catalog bright: Bright Kepler targets (Kpmag 8–12)
  4. --catalog random: Random sample from KIC

Usage:
    python scan_kepler.py --batch 1000
    python scan_kepler.py --batch 500 --catalog koi --workers 16
    python scan_kepler.py --kic-file my_kic_ids.txt
    python scan_kepler.py --batch 100 --period-max 300
"""
import argparse
import base64
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import requests
import psutil

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    MofNCompleteColumn, TimeRemainingColumn, TextColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "kepler_results"
SCRIPTS_DIR = Path(__file__).parent

CANDIDATE_THRESHOLD = 7.0
PLOT_THRESHOLD      = 9.0
LOG_EVERY           = 10

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.NullHandler()],
)


# ── Target list fetchers ──────────────────────────────────────────────────────

def fetch_koi_kic_ids(limit: int) -> list:
    """
    Fetch KIC IDs of Kepler Objects of Interest from NASA ExoplanetArchive.
    These are stars with known transit signals — ideal for validating BLS.
    """
    console.log("Fetching KOI target list from NASA ExoplanetArchive…")
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+kepid+from+cumulative&format=csv"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    kic_ids = []
    seen = set()
    for line in resp.text.splitlines():
        line = line.strip()
        if line and not line.startswith("kepid") and line.isdigit():
            kic = int(line)
            if kic not in seen:
                seen.add(kic)
                kic_ids.append(kic)
                if limit and len(kic_ids) >= limit:
                    break
    console.log(f"Fetched {len(kic_ids)} KOI KIC IDs")
    return kic_ids


def fetch_bright_kic_ids(limit: int) -> list:
    """
    Fetch bright Kepler target KIC IDs (Kpmag 8–12) via MAST.
    """
    console.log("Fetching bright Kepler targets from MAST Catalog…")
    try:
        from astroquery.mast import Catalogs
        results = Catalogs.query_criteria(
            catalog="Kepler",
            Kepmag=[8, 12],
        )
        kic_ids = [int(r["kepid"]) for r in results[:limit]]
        console.log(f"Fetched {len(kic_ids)} bright KIC IDs")
        return kic_ids
    except ImportError:
        console.print("[yellow]astroquery not installed — falling back to KOI list[/yellow]")
        return fetch_koi_kic_ids(limit)
    except Exception as e:
        console.print(f"[yellow]Bright target query failed ({e}) — falling back to KOI list[/yellow]")
        return fetch_koi_kic_ids(limit)


def fetch_random_kic_ids(limit: int) -> list:
    """Query MAST for a random sample of Kepler targets."""
    console.log("Fetching random Kepler targets from MAST…")
    try:
        from astroquery.mast import Catalogs
        results = Catalogs.query_criteria(
            catalog="Kepler",
            Kepmag=[10, 14],
        )
        import random
        rows = list(results)
        random.shuffle(rows)
        kic_ids = [int(r["kepid"]) for r in rows[:limit]]
        console.log(f"Fetched {len(kic_ids)} random KIC IDs")
        return kic_ids
    except Exception as e:
        console.print(f"[yellow]Random query failed ({e}) — falling back to KOI list[/yellow]")
        return fetch_koi_kic_ids(limit)


def load_kic_file(path: str) -> list:
    """Load KIC IDs from a text file (one per line)."""
    kic_ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    kic_ids.append(int(line))
                except ValueError:
                    pass
    return kic_ids


# ── Per-star BLS worker ───────────────────────────────────────────────────────

def process_kic(args: tuple) -> dict | None:
    """Download a Kepler lightcurve and run BLS. Returns result dict or None."""
    kic_id, period_max = args
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import lightkurve as lk
        from astropy import units as u

        # Download all available Kepler quarters and stitch
        sr = lk.search_lightcurve(
            f"KIC {kic_id}", mission="Kepler", author="Kepler", cadence="long",
        )
        if len(sr) == 0:
            return None

        lcs = sr.download_all(quality_bitmask="default")
        if lcs is None:
            return None

        lc_raw  = lcs.stitch() if len(lcs) > 1 else lcs[0]
        if len(lc_raw) < 500:
            return None

        lc_flat = lc_raw.normalize().flatten(
            window_length=1001  # wider window for 4-year baseline
        ).remove_outliers(sigma=4)
        if len(lc_flat) < 500:
            return None

        # BLS with wider period range — Kepler's 4-year baseline enables this
        periods   = np.arange(0.5, float(period_max), 0.01)
        durations = np.arange(0.01, 0.25, 0.01)
        blsm = lc_flat.to_periodogram(method="bls", period=periods, duration=durations)

        best_p  = float(blsm.period_at_max_power.value)
        best_t0 = float(blsm.transit_time_at_max_power.value)
        power   = float(blsm.max_power)

        lc_fold = lc_flat.fold(period=best_p * u.day, epoch_time=best_t0 * u.day)
        lc_bin  = lc_fold.bin(time_bin_size=0.005)  # finer bins for Kepler

        flux  = np.ma.filled(np.asarray(lc_bin.flux.value), fill_value=np.nan).astype(float)
        phase = np.asarray(lc_bin.phase.value, dtype=float)

        baseline  = float(np.nanmedian(flux[np.abs(phase) > 0.15]))
        in_tr     = flux[np.abs(phase) < 0.05]
        min_flux  = float(np.nanmin(in_tr)) if in_tr.size else baseline
        depth_ppm = (baseline - min_flux) / baseline * 1e6

        half_lev   = baseline - (baseline - min_flux) * 0.5
        below_half = np.sum(flux < half_lev) * 0.005
        dur_hours  = below_half * best_p * 24

        snr = depth_ppm / (np.nanstd(flux) * 1e6) if np.nanstd(flux) > 0 else 0.0

        # Count available quarters (data coverage)
        n_quarters = len(sr)

        return {
            "kic_id":         kic_id,
            "period":         round(best_p, 5),
            "depth_ppm":      round(depth_ppm, 1),
            "duration_hours": round(dur_hours, 3),
            "bls_power":      round(power, 4),
            "snr":            round(snr, 2),
            "t0":             round(best_t0, 5),
            "quarters":       n_quarters,
            "n_points":       len(lc_flat),
        }

    except Exception as exc:
        logging.debug(f"KIC {kic_id}: {exc}")
        return None


# ── Summary & output ──────────────────────────────────────────────────────────

def print_banner(batch_name: str, n_stars: int, workers: int, period_max: float):
    ram = psutil.virtual_memory()
    console.print(Panel(
        f"[bold cyan]Kepler BLS Scan — {batch_name}[/bold cyan]\n\n"
        f"  Stars to scan : [bold white]{n_stars:,}[/bold white]\n"
        f"  CPU workers   : [bold white]{workers}[/bold white] of {os.cpu_count()} available\n"
        f"  RAM available : [bold white]{ram.available/1e9:.1f} GB[/bold white] / {ram.total/1e9:.1f} GB\n"
        f"  BLS period    : [dim]0.5 – {period_max:.0f} days (4-year Kepler baseline)[/dim]\n"
        f"  Note          : [dim]Kepler LCs are ~70k pts vs ~18k for TESS; expect ~3× longer[/dim]",
        title="[bold yellow]🌟 Kepler BLS Scanner[/bold yellow]",
        border_style="bright_blue",
        padding=(1, 4),
    ))


def save_results(candidates: list, out_csv: Path, out_html: Path, n_scanned: int, elapsed: float):
    candidates.sort(key=lambda r: r["bls_power"], reverse=True)

    fieldnames = ["kic_id", "period", "depth_ppm", "duration_hours",
                  "bls_power", "snr", "t0", "quarters", "n_points"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)
    console.log(f"[green]CSV:[/green] {out_csv}")

    # Simple HTML report
    rate   = n_scanned / elapsed if elapsed > 0 else 0
    rows   = ""
    for i, r in enumerate(candidates[:50], 1):
        cls = "high" if r["bls_power"] >= 12 else "med" if r["bls_power"] >= PLOT_THRESHOLD else ""
        rows += (
            f'<tr class="{cls}"><td>{i}</td><td>KIC {r["kic_id"]}</td>'
            f'<td>{r["period"]:.5f}</td><td>{r["depth_ppm"]:.1f}</td>'
            f'<td>{r["duration_hours"]:.2f}</td><td><b>{r["bls_power"]:.2f}</b></td>'
            f'<td>{r["snr"]:.2f}</td><td>{r["quarters"]}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Kepler BLS Scan</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1100px;margin:0 auto}}
h1{{color:#58a6ff;margin-bottom:8px}}.meta{{color:#8b949e;font-size:14px;margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#161b22;color:#8b949e;padding:10px 12px;text-align:left;border-bottom:1px solid #30363d}}
td{{padding:8px 12px;border-bottom:1px solid #21262d}}
tr.high td{{background:rgba(255,123,114,0.08)}}tr.med td{{background:rgba(88,166,255,0.06)}}
tr:hover td{{background:rgba(255,255,255,0.04)}}</style></head><body>
<h1>🌟 Kepler BLS Scan</h1>
<p class="meta">{n_scanned} stars · {len(candidates)} candidates · {elapsed:.0f}s ({rate:.2f} stars/s)</p>
<table><thead><tr><th>#</th><th>KIC</th><th>Period(d)</th><th>Depth(ppm)</th>
<th>Dur(h)</th><th>BLS Power</th><th>SNR</th><th>Quarters</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
    out_html.write_text(html)
    console.log(f"[green]HTML:[/green] {out_html}")


def main():
    parser = argparse.ArgumentParser(
        description="BLS transit search on Kepler lightcurves"
    )
    parser.add_argument("--batch",      type=int, default=1000,
                        help="Number of stars to scan (default: 1000)")
    parser.add_argument("--catalog",    choices=["koi", "bright", "random"],
                        default="koi",
                        help="Target catalog (default: koi — known transit hosts)")
    parser.add_argument("--kic-file",   type=str, default=None,
                        help="Text file of KIC IDs (one per line)")
    parser.add_argument("--workers",    type=int, default=os.cpu_count(),
                        help="Parallel workers (default: all CPUs)")
    parser.add_argument("--period-max", type=float, default=100.0,
                        help="Max BLS period in days (default: 100)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output directory (default: data/kepler_results/batchN/)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Limit stars (alias for --batch, overrides it)")
    args = parser.parse_args()

    batch_size = args.limit or args.batch
    workers    = min(args.workers, os.cpu_count() or 16)

    # ── Get target list ─────────────────────────────────────────────────
    if args.kic_file:
        kic_ids   = load_kic_file(args.kic_file)
        batch_name = Path(args.kic_file).stem
    elif args.catalog == "koi":
        kic_ids   = fetch_koi_kic_ids(batch_size)
        batch_name = "KOI catalog"
    elif args.catalog == "bright":
        kic_ids   = fetch_bright_kic_ids(batch_size)
        batch_name = "Bright Kepler targets"
    else:
        kic_ids   = fetch_random_kic_ids(batch_size)
        batch_name = "Random Kepler sample"

    if not kic_ids:
        console.print("[red]No targets found. Exiting.[/red]")
        sys.exit(1)

    kic_ids = kic_ids[:batch_size]

    # ── Set up output ───────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_dir = Path(args.output)
    else:
        existing = sorted(RESULTS_DIR.glob("batch*/"))
        batch_n  = len(existing) + 1
        out_dir  = RESULTS_DIR / f"batch{batch_n:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "bls_results.csv"
    out_html = out_dir / "scan_report.html"

    fh = logging.FileHandler(out_dir / "scan.log")
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)

    print_banner(batch_name, len(kic_ids), workers, args.period_max)

    # ── BLS scan ────────────────────────────────────────────────────────
    work = [(kic_id, args.period_max) for kic_id in kic_ids]

    candidates: list = []
    skipped = 0
    t0      = time.time()

    progress_cols = [
        SpinnerColumn(),
        TextColumn("[bold cyan]Kepler BLS ({task.fields[batch]})[/bold cyan]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[rate]:.2f}/s  {task.fields[cands]} cands[/dim]"),
    ]

    with Progress(*progress_cols, console=console, transient=False) as prog:
        task = prog.add_task(
            "Scanning…", total=len(kic_ids), batch=batch_name[:20],
            rate=0.0, cands=0,
        )
        with mp.Pool(processes=workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(process_kic, work, chunksize=2), 1
            ):
                elapsed_now = time.time() - t0
                rate = i / elapsed_now if elapsed_now > 0 else 0
                prog.update(task, advance=1, rate=rate, cands=len(candidates))

                if result is None:
                    skipped += 1
                else:
                    candidates.append(result)
                    pwr = result["bls_power"]
                    if pwr >= CANDIDATE_THRESHOLD:
                        color = "bold red" if pwr >= 12 else "bold yellow"
                        prog.console.print(
                            f"  [bold green]⚡ CANDIDATE[/bold green]  "
                            f"KIC [cyan]{result['kic_id']}[/cyan]  "
                            f"P=[green]{result['period']:.4f} d[/green]  "
                            f"depth=[magenta]{result['depth_ppm']:.0f} ppm[/magenta]  "
                            f"power=[{color}]{pwr:.2f}[/{color}]  "
                            f"Q={result['quarters']}"
                        )

    elapsed  = time.time() - t0
    n_scanned = len(kic_ids) - skipped
    rate      = n_scanned / elapsed if elapsed > 0 else 0

    console.print()
    console.print(Rule("[bold yellow]Scan Complete[/bold yellow]", style="bright_blue"))

    stats = Table(box=None, show_header=False, padding=(0, 3))
    stats.add_column(style="dim")
    stats.add_column(style="bold white")
    stats.add_row("Stars scanned",   f"{n_scanned:,}")
    stats.add_row("Stars skipped",   f"{skipped}")
    stats.add_row("Elapsed",         f"{elapsed:.1f}s  ({rate:.2f} stars/s)")
    stats.add_row("Candidates found", f"{len(candidates)}")
    console.print(stats)

    if candidates:
        candidates.sort(key=lambda r: r["bls_power"], reverse=True)
        t = Table(
            title="[bold]Top Kepler Candidates[/bold]",
            box=box.ROUNDED, border_style="bright_blue",
            header_style="bold cyan", padding=(0, 2),
        )
        t.add_column("KIC ID",      justify="right")
        t.add_column("Period (d)",  justify="right")
        t.add_column("Depth (ppm)",  justify="right")
        t.add_column("Dur (h)",     justify="right")
        t.add_column("BLS Power",   justify="right")
        t.add_column("Quarters",    justify="right")
        for r in candidates[:8]:
            pwr   = r["bls_power"]
            style = "bold red" if pwr >= 12 else "yellow" if pwr >= PLOT_THRESHOLD else "white"
            t.add_row(
                str(r["kic_id"]),
                f"{r['period']:.5f}",
                f"{r['depth_ppm']:.1f}",
                f"{r['duration_hours']:.2f}",
                Text(f"{pwr:.4f}", style=style),
                str(r["quarters"]),
            )
        console.print(t)

    save_results(candidates, out_csv, out_html, n_scanned, elapsed)
    console.print(f"\n  [dim]Results:[/dim]  {out_dir}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
