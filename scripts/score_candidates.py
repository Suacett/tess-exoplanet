#!/usr/bin/env python3
"""
Score BLS candidates using the ExoMiner++ neural network (via Podman).

Reads a BLS results CSV, filters to strong candidates, builds the ExoMiner++
input CSV, runs the Podman pipeline, then merges scores back to the results.

Usage:
    python score_candidates.py --sector 10
    python score_candidates.py --sector 10 --threshold 7.0
    python score_candidates.py --sector 10 --model exominer++_single
    python score_candidates.py --csv /path/to/bls_results.csv --sector 10
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console()

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
CANDS_DIR   = DATA_DIR / "candidates"

EXOMINER_IMAGE   = "ghcr.io/nasa/exominer:latest"
DEFAULT_MODEL    = "exominer++_single"
DEFAULT_THRESHOLD = 9.0


def check_podman() -> bool:
    """Return True if podman is available and the ExoMiner image is pulled."""
    try:
        r = subprocess.run(
            ["podman", "image", "exists", EXOMINER_IMAGE],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_bls_candidates(csv_path: Path, threshold: float) -> list[dict]:
    """Load BLS results and return rows above threshold, sorted by power."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if float(row.get("bls_power", 0)) >= threshold:
                    rows.append(row)
            except ValueError:
                pass
    return sorted(rows, key=lambda r: float(r.get("bls_power", 0)), reverse=True)


def build_tics_csv(candidates: list[dict], sector: int, out_path: Path) -> None:
    """Write the ExoMiner++ input CSV: tic_id, sector_run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tic_id", "sector_run"])
        seen = set()
        for row in candidates:
            tic_id = str(row["tic_id"]).strip()
            sr     = f"{sector}-{sector}"
            if tic_id not in seen:
                writer.writerow([tic_id, sr])
                seen.add(tic_id)
    console.log(f"[green]ExoMiner input:[/green] {out_path}  ({len(seen)} TICs)")


def run_exominer(
    tics_file: Path,
    run_dir: Path,
    model: str,
    num_processes: int,
) -> bool:
    """
    Run ExoMiner++ via Podman. Returns True on success.
    See /opt/exoplanet/EXOMINER_HOWTO.md for full documentation.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "podman", "run", "--rm",
        "-v", f"{tics_file}:/tics_tbl.csv:Z",
        "-v", f"{run_dir}:/outputs:Z",
        EXOMINER_IMAGE,
        "--tic_ids_fp=/tics_tbl.csv",
        "--output_dir=/outputs",
        "--data_collection_mode=2min",
        "--num_processes", str(num_processes),
        "--num_jobs", str(max(1, num_processes // 4)),
        "--download_spoc_data_products=true",
        "--stellar_parameters_source=ticv8",
        "--ruwe_source=gaiadr2",
        f"--exominer_model={model}",
    ]

    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    console.print()

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_path = run_dir / "exominer_run.log"
    with open(log_path, "w") as log_fh:
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            console.print(f"[dim]{line.rstrip()}[/dim]", markup=False)

    proc.wait()
    elapsed = time.time() - t0
    console.print(f"\n[dim]ExoMiner++ finished in {elapsed:.0f}s (exit {proc.returncode})[/dim]")
    return proc.returncode == 0


def merge_scores(
    bls_csv: Path,
    predictions_csv: Path,
    out_csv: Path,
) -> int:
    """
    Merge ExoMiner++ scores into the BLS results CSV.
    Returns number of rows matched.
    """
    # Load ExoMiner predictions indexed by tic_id
    scores: dict[str, float] = {}
    try:
        with open(predictions_csv) as f:
            for row in csv.DictReader(f):
                tic = str(row.get("tic_id", "")).strip()
                score_col = next(
                    (c for c in row if "score" in c.lower()), None
                )
                if tic and score_col:
                    try:
                        scores[tic] = float(row[score_col])
                    except ValueError:
                        pass
    except FileNotFoundError:
        console.print(f"[red]Predictions file not found: {predictions_csv}[/red]")
        return 0

    # Merge
    matched = 0
    rows = []
    with open(bls_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = (reader.fieldnames or []) + ["exominer_score"]
        for row in reader:
            tic = str(row.get("tic_id", "")).strip()
            row["exominer_score"] = scores.get(tic, "")
            if row["exominer_score"] != "":
                matched += 1
            rows.append(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.log(
        f"[green]Merged CSV saved:[/green] {out_csv}  "
        f"({matched}/{len(rows)} rows with ExoMiner++ score)"
    )
    return matched


def main():
    parser = argparse.ArgumentParser(
        description="Score BLS candidates with ExoMiner++"
    )
    parser.add_argument("--sector",    type=int, required=True,
                        help="TESS sector number (used for sector_run column)")
    parser.add_argument("--csv",       type=str, default=None,
                        help="BLS results CSV (default: data/results/sectorXX/bls_results.csv)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Min BLS power to include (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--model",     type=str, default=DEFAULT_MODEL,
                        help=f"ExoMiner++ model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers",   type=int, default=os.cpu_count(),
                        help="Podman --num_processes (default: all CPUs)")
    args = parser.parse_args()

    sector   = args.sector
    bls_csv  = Path(args.csv) if args.csv else \
               RESULTS_DIR / f"sector{sector:02d}" / "bls_results.csv"
    run_name = f"sector{sector:02d}_exominer"
    run_dir  = CANDS_DIR / run_name
    tics_csv = run_dir / "tics_tbl.csv"
    out_csv  = RESULTS_DIR / f"sector{sector:02d}" / "bls_exominer_results.csv"

    # ── Validate inputs ──────────────────────────────────────────────────
    if not bls_csv.exists():
        console.print(f"[red]BLS results not found: {bls_csv}[/red]")
        console.print(f"[dim]Run scan_sector.py --sector {sector} first.[/dim]")
        sys.exit(1)

    if not check_podman():
        console.print(
            f"[red]Podman not available or ExoMiner image not pulled.[/red]\n"
            f"[dim]Pull with:  podman pull {EXOMINER_IMAGE}[/dim]"
        )
        sys.exit(1)

    # ── Load and filter candidates ───────────────────────────────────────
    candidates = load_bls_candidates(bls_csv, args.threshold)
    if not candidates:
        console.print(
            f"[yellow]No candidates with BLS power ≥ {args.threshold} "
            f"found in {bls_csv}[/yellow]"
        )
        sys.exit(0)

    console.print(Panel(
        f"[bold cyan]ExoMiner++ Scoring — Sector {sector}[/bold cyan]\n\n"
        f"  BLS CSV        : [dim]{bls_csv}[/dim]\n"
        f"  Candidates (≥{args.threshold:.0f}) : [bold white]{len(candidates)}[/bold white]\n"
        f"  Model          : [bold white]{args.model}[/bold white]\n"
        f"  Run directory  : [dim]{run_dir}[/dim]\n"
        f"  Workers        : [bold white]{args.workers}[/bold white]",
        title="[bold yellow]🤖 ExoMiner++[/bold yellow]",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    # Show top candidates to be scored
    t = Table(box=box.SIMPLE, header_style="bold cyan", padding=(0, 1))
    t.add_column("TIC ID",     justify="right")
    t.add_column("Period (d)", justify="right")
    t.add_column("Depth(ppm)", justify="right")
    t.add_column("BLS Power",  justify="right")
    for r in candidates[:10]:
        t.add_row(
            str(r["tic_id"]),
            f"{float(r.get('period', 0)):.4f}",
            f"{float(r.get('depth_ppm', 0)):.0f}",
            f"{float(r.get('bls_power', 0)):.2f}",
        )
    if len(candidates) > 10:
        t.add_row("…", "…", "…", f"({len(candidates)} total)")
    console.print(t)

    # ── Build input CSV ──────────────────────────────────────────────────
    build_tics_csv(candidates, sector, tics_csv)

    # ── Run ExoMiner++ ───────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Running ExoMiner++ pipeline[/bold]", style="bright_blue"))
    ok = run_exominer(tics_csv, run_dir / "output", args.model, args.workers)

    if not ok:
        console.print("[red]ExoMiner++ pipeline failed. Check logs at:[/red]")
        console.print(f"  {run_dir / 'output' / 'exominer_run.log'}")
        sys.exit(1)

    # ── Merge scores ─────────────────────────────────────────────────────
    predictions_csv = run_dir / "output" / "predictions_outputs.csv"
    # Try alternative location
    if not predictions_csv.exists():
        hits = list((run_dir / "output").rglob("predictions_outputs.csv"))
        if hits:
            predictions_csv = hits[0]

    console.print()
    console.print(Rule("[bold]Merging scores[/bold]", style="bright_blue"))
    matched = merge_scores(bls_csv, predictions_csv, out_csv)

    if matched > 0:
        console.print(
            f"\n[bold green]Done![/bold green] Merged results saved to:\n"
            f"  [cyan]{out_csv}[/cyan]\n\n"
            "Load this CSV in the [bold]Candidate Browser[/bold] dashboard page "
            "to filter by ExoMiner score."
        )
    else:
        console.print(
            "[yellow]Warning: no ExoMiner scores were matched to BLS results.[/yellow]\n"
            f"Check {run_dir / 'output' / 'run_main.log'} for details."
        )


if __name__ == "__main__":
    main()
