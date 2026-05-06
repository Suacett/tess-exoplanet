#!/usr/bin/env python3
"""
Prefetch all TESS SPOC 2-min FITS lightcurves for a sector.
Downloads in parallel, saves to /opt/exoplanet/data/tess/sectorXX/.
No BLS processing — download only.

Usage:
    python prefetch_sector.py --sector 99
    python prefetch_sector.py --sector 99 --threads 32
    python prefetch_sector.py --sector 99 --limit 100   # test run

After prefetch, run BLS with zero network wait:
    python scan_sector.py --sector 99 --prefetch --workers 16
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    MofNCompleteColumn, TimeRemainingColumn, TextColumn,
)
from rich.rule import Rule

console = Console()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TESS_DIR = DATA_DIR / "tess"


def get_sector_download_map(sector: int) -> list:
    """
    Parse the MAST curl script and return [{tic_id, filename, url}, ...].
    Deduplicates by TIC ID (first occurrence wins for multi-CCD targets).
    """
    url = (
        f"https://archive.stsci.edu/missions/tess/download_scripts/sector/"
        f"tesscurl_sector_{sector:02d}_lc.sh"
    )
    console.log(f"Fetching sector {sector} file list from MAST…")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    sector_str = f"{sector:04d}"
    seen: dict = {}
    for filename, dl_url in re.findall(
        r'-o\s+"?([^\s"]+)"?\s+(https?://\S+)', resp.text
    ):
        m = re.search(rf"s{sector_str}-(\d+)-\d{{4}}-s_lc\.fits", filename)
        if m:
            tic_id = int(m.group(1))
            if tic_id not in seen:
                seen[tic_id] = {"tic_id": tic_id, "filename": filename, "url": dl_url}
    return list(seen.values())


def download_one(args: tuple) -> tuple:
    """
    Download a single FITS file.
    Returns (tic_id, nbytes_fetched, ok, was_cached).
    Uses .tmp rename to prevent corrupt files on interrupted downloads.
    """
    tic_id, url, dest_str, retries = args
    dest = Path(dest_str)

    if dest.exists():
        return (tic_id, dest.stat().st_size, True, True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            nbytes = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    nbytes += len(chunk)
            tmp.rename(dest)
            return (tic_id, nbytes, True, False)
        except Exception as exc:
            import logging
            logging.debug(f"TIC {tic_id} attempt {attempt + 1}: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    return (tic_id, 0, False, False)


def run_prefetch(
    sector: int,
    dl_dir: Path,
    threads: int,
    limit: int | None = None,
    retries: int = 3,
) -> dict:
    """
    Core prefetch logic. Returns stats dict.
    Separated from main() so scan_all_new.py can call it directly.
    """
    try:
        download_map = get_sector_download_map(sector)
    except Exception as e:
        console.print(f"[red]Failed to fetch file list: {e}[/red]")
        return {"ok": False, "error": str(e)}

    if not download_map:
        console.print(
            f"[red]No FITS files found for sector {sector}. "
            "Sector may not yet be available on MAST.[/red]"
        )
        return {"ok": False, "error": "no files"}

    if limit:
        download_map = download_map[:limit]

    n_total    = len(download_map)
    n_cached   = sum(1 for e in download_map if (dl_dir / e["filename"]).exists())
    n_to_fetch = n_total - n_cached

    console.print(Panel(
        f"[bold cyan]TESS Sector {sector} — Prefetch[/bold cyan]\n\n"
        f"  Files in sector  : [bold white]{n_total:,}[/bold white]\n"
        f"  Already cached   : [bold white]{n_cached:,}[/bold white]\n"
        f"  To download      : [bold white]{n_to_fetch:,}[/bold white]\n"
        f"  Download threads : [bold white]{threads}[/bold white]\n"
        f"  Save directory   : [dim]{dl_dir}[/dim]",
        title="[bold yellow]⬇  Sector Prefetch[/bold yellow]",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    if n_to_fetch == 0:
        console.print("[green]All files already cached. Nothing to download.[/green]")
        return {
            "ok": True, "n_ok": n_cached, "n_fail": 0, "n_cached": n_cached,
            "total_bytes": 0, "elapsed": 0.0,
        }

    dl_dir.mkdir(parents=True, exist_ok=True)
    work = [
        (e["tic_id"], e["url"], str(dl_dir / e["filename"]), retries)
        for e in download_map
        if not (dl_dir / e["filename"]).exists()
    ]

    progress_cols = [
        SpinnerColumn(),
        TextColumn("[bold cyan]Sector {task.fields[sector]}[/bold cyan]"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[speed]}[/dim]"),
        TextColumn("[red]{task.fields[fail]}[/red]"),
    ]

    n_ok = n_cached
    n_fail = 0
    total_bytes = 0
    t0 = time.time()

    with Progress(*progress_cols, console=console, transient=False) as prog:
        task = prog.add_task(
            "Downloading…", total=n_total, sector=sector, speed="— MB/s", fail=""
        )
        prog.update(task, advance=n_cached)

        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {pool.submit(download_one, item): item[0] for item in work}
            for fut in as_completed(futures):
                tic_id, nbytes, ok, was_cached = fut.result()
                total_bytes   += nbytes
                elapsed_now    = time.time() - t0
                speed_mb       = total_bytes / elapsed_now / 1e6 if elapsed_now > 0 else 0.0
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    prog.console.print(
                        f"  [red]✗[/red] TIC {tic_id} failed after {retries} retries"
                    )
                fail_str = f"{n_fail} failed" if n_fail else ""
                prog.update(task, advance=1, speed=f"{speed_mb:.1f} MB/s", fail=fail_str)

    elapsed   = time.time() - t0
    speed_avg = total_bytes / elapsed / 1e6 if elapsed > 0 else 0.0

    try:
        disk_gb = sum(f.stat().st_size for f in dl_dir.glob("*_lc.fits")) / 1e9
        disk_str = f"  Disk usage  : {disk_gb:.2f} GB total in {dl_dir}\n"
    except Exception:
        disk_str = ""

    console.print()
    console.print(Rule("[bold green]Download Complete[/bold green]", style="bright_blue"))
    console.print(
        f"  [green]✓[/green]  {n_ok:,} files OK"
        + (f"  [red]✗[/red]  {n_fail} failed" if n_fail else "") + "\n"
        f"  Downloaded  : {total_bytes / 1e6:.1f} MB in {elapsed:.1f}s "
        f"({speed_avg:.1f} MB/s avg)\n"
        + disk_str
    )
    console.print(
        f"\n  [dim]Now run:[/dim]  "
        f"python scan_sector.py --sector {sector} --prefetch --workers 16"
    )

    return {
        "ok": True, "n_ok": n_ok, "n_fail": n_fail, "n_cached": n_cached,
        "total_bytes": total_bytes, "elapsed": elapsed, "speed_avg": speed_avg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download all TESS SPOC 2-min FITS lightcurves for a sector"
    )
    parser.add_argument("--sector",  type=int, required=True, help="TESS sector number")
    parser.add_argument("--threads", type=int, default=16,
                        help="Parallel download threads (default: 16)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Download only first N files (for testing)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries per file on failure (default: 3)")
    args = parser.parse_args()

    dl_dir = TESS_DIR / f"sector{args.sector:02d}"
    result = run_prefetch(args.sector, dl_dir, args.threads, args.limit, args.retries)
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
