#!/usr/bin/env python3
"""
hunt.py — Full 4-step planet-hunting pipeline for one TESS sector.

  Step 1 — Download:  Fetch lightcurves from NASA MAST
  Step 2 — Detect:   BLS transit search across all stars
  Step 3 — Score:    ExoMiner++ neural network scoring
  Step 4 — Report:   Summarise results

Usage:
    python hunt.py --sector 10 [--workers 16] [--limit 0] [--no-score]

Writes a clean, human-readable log to:
    /opt/exoplanet/data/results/sectorNN/hunt.log
"""
import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
TESS_DIR      = DATA_DIR / "tess"
RESULTS_DIR   = DATA_DIR / "results"
SCRIPTS_DIR   = Path(__file__).parent
EXOMINER_IMAGE = "ghcr.io/nasa/exominer:latest"
EXOMINER_SCORE_THRESHOLD = 9.0

_ANSI = re.compile(r'\x1b\[[0-9;]*[mK]|\r')


def sector_dates(n: int) -> str:
    start = datetime(2018, 7, 25) + timedelta(days=(n - 1) * 27.4)
    end   = start + timedelta(days=27.4)
    return f"{start.strftime('%b %d')}–{end.strftime('%b %d, %Y')}"


def planet_size(depth_ppm: float) -> str:
    if depth_ppm <= 0:
        return "unknown size"
    rp = math.sqrt(depth_ppm / 1_000_000) * 109.0  # Earth radii
    if rp < 1.5:   return "roughly Earth-sized"
    elif rp < 4:   return "roughly Super-Earth / mini-Neptune"
    elif rp < 8:   return "roughly Neptune-sized"
    elif rp < 15:  return "roughly Saturn-sized"
    else:          return "roughly Jupiter-sized"


def log(fh, msg: str = ""):
    fh.write(msg + "\n")
    fh.flush()


def read_progress(sector: int) -> dict:
    try:
        path = RESULTS_DIR / f"sector{sector:02d}" / "scan_progress.json"
        return json.loads(path.read_text())
    except Exception:
        return {}


def run_quiet(cmd, cwd=None, keep_re=None):
    """Run subprocess, return (returncode, kept_lines). Strips ANSI, filters noise."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=cwd, env=env,
    )
    kept = []
    for line in proc.stdout:
        clean = _ANSI.sub("", line).rstrip()
        if clean and keep_re and keep_re.search(clean):
            kept.append(clean)
    proc.wait()
    return proc.returncode, kept


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_cmd(cmd) -> str:
    return shlex.join(str(part) for part in cmd)


def choose_exominer_rows(all_rows: list[dict], candidates: list[dict]) -> tuple[list[dict], dict]:
    eligible = [
        row for row in all_rows
        if safe_float(row.get("bls_power", 0.0)) >= EXOMINER_SCORE_THRESHOLD
    ]
    verified_rows = [
        row for row in all_rows
        if str(row.get("verified", "")).strip().lower() == "true"
    ]
    use_verified = bool(verified_rows)
    selected_rows = verified_rows if use_verified else list(candidates)
    return selected_rows, {
        "total_rows": len(all_rows),
        "eligible_rows": len(eligible),
        "verified_rows": len(verified_rows),
        "used_verified_only": use_verified,
    }


def count_tics_csv_rows(path: Path) -> int:
    try:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def run_logged(cmd, log_path: Path, cwd=None) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb"}
    with open(log_path, "w", buffering=1) as run_log:
        run_log.write(f"ExoMiner image: {EXOMINER_IMAGE}\n")
        run_log.write(f"Command: {format_cmd(cmd)}\n")
        run_log.write(f"Working directory: {cwd or os.getcwd()}\n")
        run_log.write(f"Started: {datetime.now().isoformat()}\n")
        run_log.write("=" * 72 + "\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env,
        )
        for line in proc.stdout:
            run_log.write(_ANSI.sub("", line))
        proc.wait()
        run_log.write("=" * 72 + "\n")
        run_log.write(f"Exit code: {proc.returncode}\n")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector",    type=int, required=True)
    ap.add_argument("--workers",   type=int, default=28)
    ap.add_argument("--limit",     type=int, default=0)
    ap.add_argument("--threshold", type=float, default=7.0)
    ap.add_argument("--no-score",  action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip multi-sector verification step")
    ap.add_argument("--next-sector", type=int, default=None,
                    help="Sector to pre-warm in background while this sector scans")
    args = ap.parse_args()

    sector   = args.sector
    out_dir  = RESULTS_DIR / f"sector{sector:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "hunt.log"
    dl_dir   = TESS_DIR / f"sector{sector:02d}"

    # Pin BLAS/OMP to 1 thread per worker — prevents 28 mp.Pool workers each
    # spawning 32 OpenBLAS threads and causing severe oversubscription.
    for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_blas_var, "1")

    with open(log_path, "w", buffering=1) as lf:
        t0 = time.time()
        log(lf, f"🚀 Starting planet hunt — Sector {sector}")
        log(lf, f"   Observed: {sector_dates(sector)}")
        log(lf, f"   Workers: {args.workers}  |  Stars: {'all' if args.limit == 0 else args.limit}")
        log(lf)

        # ── Step 1: Download ──────────────────────────────────────────────────
        fits_files = list(dl_dir.glob("*_lc.fits")) if dl_dir.exists() else []
        if fits_files:
            log(lf, f"⬇️  Step 1/4 — Download")
            log(lf, f"   {len(fits_files):,} lightcurve files already on disk — skipping download")
        else:
            log(lf, f"⬇️  Step 1/4 — Downloading star data from NASA MAST...")
            log(lf, f"   (Each file is one star's brightness measurements over 27 days)")
            t1 = time.time()
            rc, _ = run_quiet([
                sys.executable, str(SCRIPTS_DIR / "prefetch_sector.py"),
                "--sector", str(sector),
            ])
            elapsed1 = time.time() - t1
            fits_files = list(dl_dir.glob("*_lc.fits")) if dl_dir.exists() else []
            if rc == 0 and fits_files:
                mb = sum(f.stat().st_size for f in fits_files) / 1_048_576
                log(lf, f"   ✅ Download complete — {len(fits_files):,} stars ({mb:.0f} MB in {elapsed1/60:.1f} min)")
            else:
                log(lf, f"   ❌ Download failed (exit {rc}) — cannot continue")
                return
        log(lf)

        # ── Page-cache pre-warm ──────────────────────────────────────────────
        log(lf, "🔥 Pre-loading sector FITS into RAM...")
        t_warm = time.time()
        subprocess.run(
            ["bash", "-c",
             f"find {dl_dir} -name '*_lc.fits' -print0"
             " | xargs -0 -P4 -n128 cat > /dev/null"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log(lf, f"   ✅ {len(fits_files):,} files warm in page cache ({time.time()-t_warm:.0f}s)")
        log(lf)

        # ── Step 2: BLS Detect ────────────────────────────────────────────────
        n_stars = min(len(fits_files), args.limit) if args.limit > 0 else len(fits_files)
        log(lf, f"🔍 Step 2/4 — Searching {n_stars:,} stars for transiting planets...")
        log(lf, f"   (Looking for repeating dips in brightness that could be a planet passing in front)")
        t2 = time.time()

        # Background pre-warm next sector while BLS scan runs
        _prewarm_proc = None
        if args.next_sector is not None:
            _next_dl = TESS_DIR / f"sector{args.next_sector:02d}"
            if _next_dl.exists() and list(_next_dl.glob("*_lc.fits")):
                log(lf, f"🔥 Background pre-warm: sector {args.next_sector} (runs in parallel with scan)")
                _prewarm_proc = subprocess.Popen(
                    ["bash", "-c",
                     f"find {_next_dl} -name '*_lc.fits' -print0"
                     " | xargs -0 -P4 -n128 cat > /dev/null"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

        bls_cmd = [
            sys.executable, str(SCRIPTS_DIR / "scan_sector.py"),
            "--sector", str(sector),
            "--workers", str(args.workers),
            "--prefetch",
        ]
        if args.limit > 0:
            bls_cmd += ["--limit", str(args.limit)]

        # Launch scan_sector.py; relay CANDIDATE lines + poll scan_progress.json
        scan_log   = out_dir / "scan_sector.log"
        env        = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb"}
        proc       = subprocess.Popen(bls_cmd, stdout=open(scan_log, "w"),
                                      stderr=subprocess.STDOUT, env=env)
        _CAND      = re.compile(r'CANDIDATE')
        _NOISE     = re.compile(r'Opening /|Detected filetype|cadences will be ignored|quality_bitmask')
        log_pos    = 0
        last_done  = -1

        while proc.poll() is None:
            time.sleep(2)
            # Relay new CANDIDATE lines
            try:
                with open(scan_log) as sf:
                    sf.seek(log_pos)
                    for line in sf:
                        clean = _ANSI.sub("", line).rstrip()
                        if clean and _CAND.search(clean) and not _NOISE.search(clean):
                            lf.write("   " + clean + "\n")
                            lf.flush()
                    log_pos = sf.tell()
            except Exception:
                pass
            # Write progress from scan_progress.json
            prog = read_progress(sector)
            done = prog.get("done", 0)
            if prog.get("sector") == sector and done != last_done:
                last_done = done
                total  = prog.get("total", n_stars)
                cands  = prog.get("candidates", 0)
                rate   = prog.get("rate", 0)
                pct    = 100 * done // max(total, 1)
                status = prog.get("status", "")
                if status:
                    log(lf, f"   ⏳ {status} ({done:,}/{total:,})")
                else:
                    eta_str = ""
                    if rate > 0 and done < total:
                        eta_min = (total - done) / rate / 60
                        eta_str = f" — ETA {eta_min:.0f}m" if eta_min >= 1 else " — ETA <1m"
                    log(lf, f"   🔍 {done:,}/{total:,} stars scanned ({pct}%)"
                            f" — {cands} signals found — {rate:.1f} stars/sec{eta_str}")
        proc.wait()
        elapsed2 = time.time() - t2
        if _prewarm_proc is not None:
            _prewarm_proc.wait()
            log(lf, f"   ✅ Background pre-warm of sector {args.next_sector} complete")

        # Drain remaining CANDIDATE lines
        try:
            with open(scan_log) as sf:
                sf.seek(log_pos)
                for line in sf:
                    clean = _ANSI.sub("", line).rstrip()
                    if clean and _CAND.search(clean) and not _NOISE.search(clean):
                        lf.write("   " + clean + "\n")
                        lf.flush()
        except Exception:
            pass

        bls_csv = out_dir / "bls_results.csv"
        if not bls_csv.exists():
            log(lf, f"   ❌ BLS scan produced no results file — something went wrong")
            return

        with open(bls_csv) as f:
            all_rows = list(csv.DictReader(f))
        # CSV already contains only filtered candidates; all rows pass SDE threshold
        candidates = all_rows

        # Read filter stats written by scan_sector.py
        try:
            stats_data  = json.loads((out_dir / "scan_filter_stats.json").read_text())
            n_raw       = stats_data.get("n_raw", len(all_rows))
            fc          = stats_data.get("filter_counts", {})
            n_filtered  = sum(fc.values())
            filt_parts  = []
            if fc.get("depth_hi"):   filt_parts.append(f"{fc['depth_hi']:,} eclipsing binaries")
            if fc.get("depth_lo"):   filt_parts.append(f"{fc['depth_lo']:,} too shallow")
            if fc.get("period_lo"):  filt_parts.append(f"{fc['period_lo']:,} bad periods")
            if fc.get("n_transits"): filt_parts.append(f"{fc['n_transits']:,} single-transit")
            if fc.get("sde"):        filt_parts.append(f"{fc['sde']:,} noise (SDE<{args.threshold})")
        except Exception:
            n_raw, n_filtered, filt_parts, fc = len(all_rows), 0, [], {}

        n_planet  = sum(1 for r in candidates if r.get("classification") == "Planet candidate")
        n_inspect = sum(1 for r in candidates if r.get("classification") == "Needs inspection")
        n_eb      = sum(1 for r in candidates if r.get("classification") == "Eclipsing binary")

        log(lf)
        log(lf, f"   🔍 Scan complete: {n_stars:,} stars searched in {elapsed2/60:.1f} min")
        log(lf, f"   📊 Raw BLS detections: {n_raw:,}")
        if filt_parts:
            log(lf, f"   🗑️  Filtered out: {n_filtered:,}  ({', '.join(filt_parts)})")
        log(lf, f"   🟢 Planet candidates: {n_planet}")
        log(lf, f"   🟡 Needs inspection:  {n_inspect}")
        if n_eb:
            log(lf, f"   🔴 Eclipsing binaries kept: {n_eb}")
        log(lf, f"   ✅ Results saved. {len(candidates)} candidates worth looking at.")
        log(lf)

        # ── Step 2.5: Multi-sector verification ──────────────────────────────
        if not args.no_verify and (n_planet + n_inspect) > 0:
            n_to_verify = n_planet + n_inspect
            log(lf, f"🔭 Step 2.5/4 — Verifying {n_to_verify} candidates across all TESS sectors...")
            log(lf, f"   (Checking if each signal repeats in other sector observations of this star)")
            t25 = time.time()
            verify_cmd = [
                sys.executable, str(SCRIPTS_DIR / "verify_batch.py"),
                "--csv",       str(bls_csv),
                "--out-dir",   str(out_dir / "verification"),
                "--threshold", "9.0",
                "--workers",   "16",
            ]
            rc_v, _ = run_quiet(verify_cmd,
                                keep_re=re.compile(r'TIC|verified|✅|❌|verify_batch'))
            # Re-read CSV (verify_batch updated it in-place)
            with open(bls_csv) as f:
                all_rows  = list(csv.DictReader(f))
            candidates = all_rows
            n_ver  = sum(1 for r in all_rows if r.get("verified") == "true")
            n_rej  = sum(1 for r in all_rows if r.get("verified") == "false")
            log(lf, f"   ✅ {n_ver} signals confirmed across multiple sectors, "
                    f"{n_rej} rejected")
            log(lf, f"   Verification took {(time.time()-t25)/60:.1f} min")
            log(lf)
        elif not args.no_verify:
            log(lf, f"🔭 Step 2.5/4 — Verification: no strong candidates to verify")
            log(lf)

        # ── Step 3: ExoMiner++ ────────────────────────────────────────────────
        scores = {}  # tic_id_str -> float
        if args.no_score:
            log(lf, f"🤖 Step 3/4 — Score: skipped (--no-score)")
        elif not candidates:
            log(lf, f"🤖 Step 3/4 — Score: no candidates above threshold — skipping ExoMiner++")
        else:
            score_rows, score_stats = choose_exominer_rows(all_rows, candidates)
            scoring_desc = (
                f"{len(score_rows)} verified signal(s)"
                if score_stats["used_verified_only"]
                else f"{len(score_rows)} candidate signal(s)"
            )
            log(lf, f"🤖 Step 3/4 — Asking NASA's AI (ExoMiner++) to evaluate {scoring_desc}...")
            log(lf, f"   ExoMiner++ was trained on thousands of confirmed planets and false positives.")
            log(lf, f"   Score close to 1.0 = looks like a real planet. Close to 0.0 = probably not.")
            log(lf, f"   Note: ExoMiner++ only scores stars that NASA's SPOC pipeline also flagged.")
            log(lf, f"   Total BLS rows: {score_stats['total_rows']}")
            log(lf, f"   Rows eligible for scoring (BLS power >= {EXOMINER_SCORE_THRESHOLD:.0f}): {score_stats['eligible_rows']}")
            log(lf, f"   Verified rows: {score_stats['verified_rows']}")
            if score_stats["used_verified_only"]:
                log(lf, f"   Scoring mode: verified-only (using rows explicitly marked verified=true)")
            else:
                log(lf, f"   Scoring mode: current candidate set (no verified=true rows found)")
            log(lf)

            run_dir   = out_dir / "exominer_run"
            tics_file = run_dir / "tics_tbl.csv"
            em_out    = run_dir / "output"
            em_log    = out_dir / "exominer_run.log"
            run_dir.mkdir(parents=True, exist_ok=True)
            em_out.mkdir(parents=True, exist_ok=True)

            with open(tics_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tic_id", "sector_run"])
                seen_tics = set()
                for r in score_rows:
                    tic_id = str(r["tic_id"]).strip()
                    if tic_id and tic_id not in seen_tics:
                        w.writerow([tic_id, f"{sector}-{sector}"])
                        seen_tics.add(tic_id)

            tic_count = count_tics_csv_rows(tics_file)
            log(lf, f"   ExoMiner image: {EXOMINER_IMAGE}")
            log(lf, f"   Input CSV mount: {tics_file} -> /tics_tbl.csv")
            log(lf, f"   Output mount:    {em_out} -> /outputs")
            log(lf, f"   Full run log:    {em_log}")
            log(lf, f"   tics_tbl.csv created: {'yes' if tics_file.exists() else 'no'} ({tic_count} unique TICs)")

            if tic_count == 0:
                log(lf, f"   ⚠️  No TICs were selected for ExoMiner++ after Step 3 filtering.")
                log(lf, f"   BLS and verification results are still available below.")
                log(lf)
            else:
                n_jobs = max(1, min(tic_count, args.workers // 4))
                podman_cmd = [
                    "podman", "run", "--rm",
                    "-v", f"{tics_file}:/tics_tbl.csv:Z",
                    "-v", f"{em_out}:/outputs:Z",
                    EXOMINER_IMAGE,
                    "--tic_ids_fp=/tics_tbl.csv",
                    "--output_dir=/outputs",
                    "--data_collection_mode=2min",
                    f"--num_processes={args.workers}",
                    f"--num_jobs={n_jobs}",
                    "--download_spoc_data_products=true",
                    "--stellar_parameters_source=ticv8",
                    "--ruwe_source=gaiadr2",
                    "--exominer_model=exominer++_single",
                ]
                log(lf, f"   Podman command: {format_cmd(podman_cmd)}")
                t3 = time.time()
                try:
                    rc3 = run_logged(podman_cmd, em_log, cwd=str(run_dir))
                    elapsed3 = time.time() - t3

                    pred_csv = em_out / "predictions_outputs.csv"
                    pred_exists = pred_csv.exists()
                    log(lf, f"   ExoMiner++ exit code: {rc3}")
                    log(lf, f"   predictions_outputs.csv present: {'yes' if pred_exists else 'no'}")

                    if pred_exists:
                        with open(pred_csv) as f:
                            for row in csv.DictReader(f):
                                tic = str(row.get("tic_id", "")).strip()
                                sc  = row.get("exominer_score", "")
                                if tic and sc:
                                    try:
                                        scores[tic] = float(sc)
                                    except ValueError:
                                        pass

                        log(lf, f"   ✅ ExoMiner++ complete in {elapsed3/60:.1f} min"
                                f" — scored {len(scores)} TCE(s)")
                        log(lf)

                        for r in sorted(candidates,
                                        key=lambda x: float(x.get("bls_power", 0)),
                                        reverse=True):
                            tic = str(r["tic_id"])
                            sc  = scores.get(tic)
                            if sc is not None:
                                if sc >= 0.8:   verdict = "🟢 likely planet"
                                elif sc >= 0.5: verdict = "🟡 possible planet"
                                else:           verdict = "🔴 probably not a planet"
                                log(lf, f"   🤖 TIC {tic}: ExoMiner++ score {sc:.2f} — {verdict}")

                        # Merge scores into all_rows and write combined CSV
                        for r in all_rows:
                            r["exominer_score"] = scores.get(str(r["tic_id"]), "")
                        em_csv = out_dir / "bls_exominer_results.csv"
                        with open(em_csv, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                            w.writeheader()
                            w.writerows(all_rows)
                    else:
                        if rc3 != 0:
                            log(lf, f"   ⚠️  ExoMiner++ failed before producing predictions.")
                            log(lf, f"   Check {em_log} for the exact Podman error output.")
                        else:
                            log(lf, f"   ⚠️  ExoMiner++ ran but produced no predictions file.")
                            log(lf, f"   Most likely none of the submitted TICs matched a SPOC TCE that ExoMiner++ can score.")
                        log(lf, f"   BLS and verification results are still available below.")
                except Exception as e:
                    log(lf, f"   ⚠️  ExoMiner++ failed: {e}")
                    log(lf, f"   Check {em_log} for the partial run log.")
                    log(lf, f"   BLS and verification results are still available below.")
        log(lf)

        # ── Step 4: Summary ───────────────────────────────────────────────────
        log(lf, f"📋 Step 4/4 — Results")
        log(lf)

        n_strong = len([s for s in scores.values() if s >= 0.5]) if scores else 0
        top = sorted(candidates, key=lambda x: float(x.get("bls_power", 0)), reverse=True)[:10]

        for r in top:
            tic     = r["tic_id"]
            period  = float(r.get("period", 0))
            depth   = float(r.get("depth_ppm", 0))
            power   = float(r.get("bls_power", 0))
            sc      = scores.get(str(tic))
            size    = planet_size(depth)
            dim_pct = depth / 10_000
            sc_str  = f"  |  ExoMiner++: {sc:.2f}" if sc is not None else ""
            log(lf, f"   ⚡ TIC {tic} — dims {dim_pct:.2f}% every {period:.3f} days"
                    f" — {size} (BLS power: {power:.1f}{sc_str})")

        total_t = time.time() - t0
        log(lf)
        if scores and n_strong > 0:
            log(lf, f"✅ Done! {n_strong} strong planet candidate(s) found.")
        elif candidates:
            log(lf, f"✅ Done! {len(candidates)} transit signal(s) found."
                    f" (ExoMiner++ scored 0 — BLS results still available)")
        else:
            log(lf, f"✅ Done! No transit signals found above threshold {args.threshold}.")
        log(lf, f"   Total time: {total_t/60:.1f} minutes")


if __name__ == "__main__":
    main()
