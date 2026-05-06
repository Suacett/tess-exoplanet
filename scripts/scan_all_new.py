#!/usr/bin/env python3
"""
Scan all TESS sectors not yet in the ExoMiner++ catalog (sectors >= 68).
Prefetches each sector, then runs BLS with all CPU cores.
Designed to be left running overnight.

Usage:
    python scan_all_new.py                       # all unscanned sectors 68+
    python scan_all_new.py --sectors 68 69 70   # specific sectors
    python scan_all_new.py --limit 50           # limit stars per sector (for testing)
    python scan_all_new.py --no-prefetch        # skip download phase (use cached files)
    python scan_all_new.py --workers 16
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console()

SCRIPTS_DIR   = Path(__file__).parent
DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR   = DATA_DIR / "results"
TRACKING_FILE = DATA_DIR / "scanned_sectors.json"
EXOMINER_MAX_SECTOR = 67   # sectors 1-67 are in the published catalog


def load_tracking() -> dict:
    if TRACKING_FILE.exists():
        return json.loads(TRACKING_FILE.read_text())
    return {"scanned": [], "available": []}


def get_unscanned_new_sectors(tracking: dict, available: list) -> list:
    """Return sectors >= 68 that are available but not yet scanned."""
    scanned = set(tracking.get("scanned", []))
    return sorted(s for s in available if s > EXOMINER_MAX_SECTOR and s not in scanned)


def read_scan_progress(sector: int) -> dict:
    """Read the live progress JSON written by scan_sector.py."""
    p = RESULTS_DIR / f"sector{sector:02d}" / "scan_progress.json"
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return {}


def run_subprocess(cmd: list, log_path: Path) -> tuple[int, float]:
    """
    Run a command, tee stdout+stderr to log_path AND console.
    Returns (returncode, elapsed_seconds).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=SCRIPTS_DIR,
        )
        for line in proc.stdout:
            # Strip ANSI codes for log file, keep them for console
            import re
            clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
            log_fh.write(clean)
            log_fh.flush()
            console.print(line.rstrip(), markup=False, highlight=False)
        proc.wait()
    return proc.returncode, time.time() - t0


def print_master_header(sectors: list, current_idx: int, sector_results: list):
    """Print the master progress panel."""
    done    = current_idx
    total_s = len(sectors)
    total_c = sum(r.get("candidates", 0) for r in sector_results)
    total_stars = sum(r.get("n_scanned", 0) for r in sector_results)

    panel_lines = [
        f"[bold cyan]Multi-Sector Scan — Sectors {sectors[0]}–{sectors[-1]}[/bold cyan]\n",
        f"  Overall   : {done}/{total_s} sectors complete",
        f"  Stars     : {total_stars:,} processed",
        f"  Candidates: [bold yellow]{total_c}[/bold yellow] total",
    ]
    if done < total_s:
        remaining = sectors[done:]
        panel_lines.append(f"  Remaining : {remaining}")

    console.print(Panel(
        "\n".join(panel_lines),
        title="[bold yellow]⭐ Overnight Scan Progress[/bold yellow]",
        border_style="bright_blue",
        padding=(0, 2),
    ))


def main():
    parser = argparse.ArgumentParser(description="Scan all new TESS sectors (68+)")
    parser.add_argument("--sectors", type=int, nargs="+", default=None,
                        help="Specific sectors to scan (default: all unscanned >= 68)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Stars per sector limit (for testing)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="BLS worker processes (default: all CPUs)")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Skip download phase (assume FITS already cached)")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip 4-panel plot generation")
    parser.add_argument("--no-report",  action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--dl-threads", type=int, default=16,
                        help="Download threads per sector (default: 16)")
    args = parser.parse_args()

    python = sys.executable
    workers = min(args.workers or os.cpu_count(), os.cpu_count() or 16)

    # ── Determine which sectors to scan ──────────────────────────────────
    if args.sectors:
        sectors_to_scan = sorted(args.sectors)
        console.print(f"[dim]Scanning specified sectors: {sectors_to_scan}[/dim]")
    else:
        tracking = load_tracking()
        # Get available sectors from MAST
        console.print("[dim]Querying MAST for available sectors…[/dim]")
        try:
            import requests, re
            resp = requests.get(
                "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html",
                timeout=30,
            )
            available = sorted(
                {int(m) for m in re.findall(r"sector_(\d+)_lc", resp.text) if int(m) < 500}
            )
        except Exception as e:
            console.print(f"[red]Could not query MAST: {e}[/red]")
            sys.exit(1)

        sectors_to_scan = get_unscanned_new_sectors(tracking, available)
        if not sectors_to_scan:
            console.print("[green]No new unscanned sectors found (68+). All up to date.[/green]")
            sys.exit(0)
        console.print(
            f"Found [bold]{len(sectors_to_scan)}[/bold] unscanned sectors "
            f"(> {EXOMINER_MAX_SECTOR}): {sectors_to_scan}"
        )

    n_sectors     = len(sectors_to_scan)
    sector_results: list[dict] = []
    wall_start    = time.time()

    console.print()
    console.print(Rule(
        f"[bold yellow]Starting overnight scan — {n_sectors} sectors[/bold yellow]",
        style="bright_blue",
    ))
    console.print()

    for idx, sector in enumerate(sectors_to_scan):
        sector_start = time.time()
        print_master_header(sectors_to_scan, idx, sector_results)
        console.print()
        console.print(Rule(
            f"[bold cyan]Sector {sector}  ({idx + 1}/{n_sectors})[/bold cyan]",
            style="cyan",
        ))
        console.print()

        out_dir = RESULTS_DIR / f"sector{sector:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: Prefetch ──────────────────────────────────────────
        if not args.no_prefetch:
            console.print(f"[bold]▶ Phase 1: Downloading sector {sector}…[/bold]")
            dl_cmd = [python, str(SCRIPTS_DIR / "prefetch_sector.py"),
                      "--sector", str(sector),
                      "--threads", str(args.dl_threads)]
            if args.limit:
                dl_cmd += ["--limit", str(args.limit)]
            rc, dl_elapsed = run_subprocess(dl_cmd, out_dir / "prefetch.log")
            if rc != 0:
                console.print(f"[red]Prefetch failed for sector {sector} (exit {rc}). Skipping.[/red]")
                sector_results.append({"sector": sector, "ok": False, "phase": "prefetch"})
                continue
            console.print(f"[dim]  Prefetch done in {dl_elapsed:.0f}s[/dim]")
            console.print()

        # ── Phase 2: BLS scan ─────────────────────────────────────────
        console.print(f"[bold]▶ Phase 2: BLS scanning sector {sector} with {workers} workers…[/bold]")
        scan_cmd = [python, str(SCRIPTS_DIR / "scan_sector.py"),
                    "--sector", str(sector),
                    "--workers", str(workers)]
        if not args.no_prefetch:
            scan_cmd.append("--prefetch")
        if args.limit:
            scan_cmd += ["--limit", str(args.limit)]
        if args.no_plots:
            scan_cmd.append("--no-plots")
        if args.no_report:
            scan_cmd.append("--no-report")

        rc, bls_elapsed = run_subprocess(scan_cmd, out_dir / "scan_stdout.log")
        sector_elapsed = time.time() - sector_start

        # ── Read results ─────────────────────────────────────────────
        csv_path = out_dir / "bls_results.csv"
        n_scanned = 0
        n_cands   = 0
        if csv_path.exists():
            try:
                import csv as csv_mod
                rows = list(csv_mod.DictReader(open(csv_path)))
                n_scanned = len(rows)
                n_cands   = sum(1 for r in rows if float(r.get("bls_power", 0)) >= 7)
            except Exception:
                pass

        sector_results.append({
            "sector":     sector,
            "ok":         rc == 0,
            "n_scanned":  n_scanned,
            "candidates": n_cands,
            "elapsed":    sector_elapsed,
        })

        status = "[green]✓ OK[/green]" if rc == 0 else f"[red]✗ exit {rc}[/red]"
        console.print(
            f"\n  {status}  Sector {sector}: {n_scanned} stars, "
            f"{n_cands} candidates, {sector_elapsed:.0f}s total\n"
        )

    # ── Final summary ─────────────────────────────────────────────────────
    wall_elapsed = time.time() - wall_start
    console.print()
    console.print(Rule("[bold yellow]All Sectors Complete[/bold yellow]", style="bright_blue"))

    t = Table(
        title="[bold]Multi-Sector Scan Summary[/bold]",
        box=box.ROUNDED, border_style="bright_blue",
        header_style="bold cyan", show_lines=False, padding=(0, 2),
    )
    t.add_column("Sector",     justify="right")
    t.add_column("Stars",      justify="right")
    t.add_column("Candidates", justify="right")
    t.add_column("Time",       justify="right")
    t.add_column("Status",     justify="center")

    for r in sector_results:
        t.add_row(
            str(r["sector"]),
            f"{r.get('n_scanned', 0):,}",
            f"{r.get('candidates', 0)}",
            f"{r.get('elapsed', 0):.0f}s",
            "[green]OK[/green]" if r.get("ok") else "[red]FAIL[/red]",
        )
    console.print(t)

    total_stars = sum(r.get("n_scanned", 0) for r in sector_results)
    total_cands = sum(r.get("candidates", 0) for r in sector_results)
    console.print(
        f"\n  Total: [bold]{total_stars:,}[/bold] stars, "
        f"[bold yellow]{total_cands}[/bold yellow] candidates, "
        f"{wall_elapsed / 3600:.1f}h wall time\n"
    )


if __name__ == "__main__":
    main()
