#!/usr/bin/env python3
"""
Detached background jobs for the Streamlit dashboard.

This script keeps long-running download/scan work alive after page changes by
persisting job state and logs to disk.
"""
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_new_sectors import get_available_sectors  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TESS_DIR = DATA_DIR / "tess"
RESULTS_DIR = DATA_DIR / "results"
JOBS_DIR = DATA_DIR / "jobs"
SCAN_STATE_FILE = DATA_DIR / ".scan_state.json"
CROSSMATCH_DIR = DATA_DIR / "crossmatch"
SHORTLIST_DIR = CROSSMATCH_DIR / "shortlists"
DEFAULT_SHORTLIST_PATH = SHORTLIST_DIR / "sector_85_99_shortlist_crossmatched.csv"
FAST_SHORTLIST_PATH = SHORTLIST_DIR / "sector_85_99_shortlist_fast_crossmatched.csv"


def now_iso() -> str:
    return datetime.now().isoformat()


def format_cmd(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_downloaded_sectors() -> list[int]:
    sectors = []
    if not TESS_DIR.exists():
        return sectors
    for d in TESS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"sector(\d+)", d.name)
        if m and any(d.glob("*_lc.fits")):
            sectors.append(int(m.group(1)))
    return sorted(sectors)


def tail_contains_done(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(size - 8192, 0))
            tail = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return False
    return "✅ Done!" in tail


def sector_hunt_complete(sector: int) -> bool:
    res_dir = RESULTS_DIR / f"sector{sector:02d}"
    return (res_dir / "bls_results.csv").exists() and tail_contains_done(res_dir / "hunt.log")


class JobRunner:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.meta_path = JOBS_DIR / f"{job_id}.json"
        self.log_path = JOBS_DIR / f"{job_id}.log"
        active_state = read_json(SCAN_STATE_FILE, {})
        if active_state.get("job_id") == job_id:
            self.state = active_state
        else:
            self.state = {"job_id": job_id}
        self.state["job_id"] = job_id
        self.state.setdefault("log_file", str(self.log_path))
        self.state.setdefault("sectors_total", 0)
        self.state.setdefault("sectors_done", 0)
        self.state.setdefault("sectors_failed", 0)
        self.state.setdefault("failed_sectors", [])
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def sync(self) -> None:
        write_json(self.meta_path, self.state)
        active = read_json(SCAN_STATE_FILE, {})
        if active.get("job_id") == self.job_id or self.state.get("running"):
            write_json(SCAN_STATE_FILE, self.state)

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        self.sync()

    def log(self, msg: str = "") -> None:
        with open(self.log_path, "a", buffering=1) as fh:
            if msg:
                fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            else:
                fh.write("\n")

    def run_subprocess(self, cmd: list[str], cwd: Path | None = None) -> int:
        self.log(f"Running: {format_cmd(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd or SCRIPTS_DIR,
        )
        with open(self.log_path, "a", buffering=1) as fh:
            for line in proc.stdout:
                fh.write(line)
        proc.wait()
        self.log(f"Exit code: {proc.returncode}")
        return proc.returncode

    def finish(self, status: str = "completed") -> None:
        self.update(
            running=False,
            status=status,
            finished_at=now_iso(),
            current_sector=None,
            current_log_file=None,
        )


def run_download_missing(job: JobRunner, min_sector: int, threads: int) -> int:
    job.update(
        running=True,
        status="running",
        mode="download_missing",
        label=f"Download missing sectors ({min_sector}+)",
        current_log_file=str(job.log_path),
        started_at=job.state.get("started_at", now_iso()),
    )
    job.log("Checking MAST for available sectors...")
    available = sorted(s for s in get_available_sectors() if s >= min_sector)
    downloaded = set(get_downloaded_sectors())
    missing = [s for s in available if s not in downloaded]
    job.update(sectors_total=len(missing), sectors_done=0, sectors_failed=0, target_sectors=missing)

    if not missing:
        job.log(f"No missing sectors found in target range {min_sector}+.")
        job.finish("completed")
        return 0

    job.log(f"Downloaded sectors on disk: {sorted(downloaded)}")
    job.log(f"Missing sectors to download: {missing}")

    for sector in missing:
        job.update(current_sector=sector, current_log_file=str(job.log_path))
        job.log(f"Starting download for sector {sector}...")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "prefetch_sector.py"),
            "--sector", str(sector),
            "--threads", str(threads),
        ]
        rc = job.run_subprocess(cmd)
        if rc == 0 and sector in set(get_downloaded_sectors()):
            job.state["sectors_done"] += 1
            job.log(f"Sector {sector} downloaded successfully.")
        else:
            job.state["sectors_failed"] += 1
            job.state.setdefault("failed_sectors", []).append(sector)
            job.log(f"Sector {sector} failed to download. Continuing.")
        job.sync()

    job.finish("completed")
    return 0


def run_scan_all(job: JobRunner, min_sector: int, workers: int, limit: int, no_score: bool) -> int:
    targets = [
        sector for sector in get_downloaded_sectors()
        if sector >= min_sector and not sector_hunt_complete(sector)
    ]
    job.update(
        running=True,
        status="running",
        mode="scan_all",
        label=f"Scan all sectors ({min_sector}+)",
        current_log_file=str(job.log_path),
        started_at=job.state.get("started_at", now_iso()),
        sectors_total=len(targets),
        sectors_done=0,
        sectors_failed=0,
        target_sectors=targets,
    )
    if not targets:
        job.log(f"No downloaded unfinished sectors found in target range {min_sector}+.")
        job.finish("completed")
        return 0

    job.log(f"Downloaded unfinished sectors to scan: {targets}")
    for i, sector in enumerate(targets):
        sector_log = RESULTS_DIR / f"sector{sector:02d}" / "hunt.log"
        job.update(current_sector=sector, current_log_file=str(sector_log))
        job.log(f"Starting hunt for sector {sector}...")
        job.log(f"Detailed sector log: {sector_log}")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "hunt.py"),
            "--sector", str(sector),
            "--workers", str(workers),
            "--limit", str(limit),
        ]
        if i + 1 < len(targets):
            cmd += ["--next-sector", str(targets[i + 1])]
        if no_score:
            cmd.append("--no-score")
        rc = job.run_subprocess(cmd)
        if rc == 0 and sector_hunt_complete(sector):
            job.state["sectors_done"] += 1
            job.log(f"Sector {sector} completed successfully.")
        else:
            job.state["sectors_failed"] += 1
            job.state.setdefault("failed_sectors", []).append(sector)
            job.log(f"Sector {sector} did not complete cleanly. Continuing.")
        job.sync()

    job.finish("completed")
    return 0


def summarize_shortlist(csv_path: Path) -> tuple[int, dict[str, int]]:
    counts = {
        "Confirmed planet": 0,
        "Known TOI": 0,
        "Caution": 0,
        "Unknown": 0,
        "Lookup failed": 0,
    }
    rows = 0
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            status = (row.get("status_label") or row.get("public_status") or "").strip()
            if status in counts:
                counts[status] += 1
    return rows, counts


def shortlist_metadata_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ".meta.json")


def run_build_shortlist(job: JobRunner, sectors: str, output_path: Path, fast: bool = False) -> int:
    SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = JOBS_DIR / f"{job.job_id}.progress.json"
    build_mode = "fast" if fast else "full"
    job.update(
        running=True,
        status="running",
        mode="build_shortlist",
        label=f"Build {'fast ' if fast else ''}cross-matched shortlist ({sectors})",
        current_log_file=str(job.log_path),
        started_at=job.state.get("started_at", now_iso()),
        output_file=str(output_path),
        progress_file=str(progress_path),
        build_mode=build_mode,
    )
    job.log(f"Building {'fast triage-first ' if fast else 'full '}cross-matched shortlist for sectors {sectors}...")
    job.log(f"Output path: {output_path}")
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "crossmatch_candidates.py"),
        "--sectors",
        sectors,
        "--out",
        str(output_path),
        "--progress-file",
        str(progress_path),
    ]
    if fast:
        cmd.append("--fast")
    rc = job.run_subprocess(cmd)
    if rc != 0 or not output_path.exists():
        prog = read_json(progress_path, {})
        if prog:
            job.update(
                failed_tics=prog.get("failed_tics", 0),
                skipped_tics=prog.get("skipped_tics", 0),
                last_error=prog.get("last_error"),
                build_mode=prog.get("build_mode", build_mode),
                cheap_pass_rows=prog.get("cheap_pass_rows", 0),
                deep_lookup_rows=prog.get("deep_lookup_rows", 0),
            )
        job.log("Shortlist build failed or did not produce an output CSV.")
        job.finish("failed")
        return 1

    row_count, status_counts = summarize_shortlist(output_path)
    prog = read_json(progress_path, {})
    meta = read_json(shortlist_metadata_path(output_path), {})
    job.update(
        row_count=row_count,
        status_counts=status_counts,
        built_at=now_iso(),
        failed_tics=prog.get("failed_tics", 0),
        skipped_tics=prog.get("skipped_tics", 0),
        last_error=prog.get("last_error"),
        build_mode=meta.get("build_mode", prog.get("build_mode", build_mode)),
        cheap_pass_rows=meta.get("cheap_pass_rows", prog.get("cheap_pass_rows", 0)),
        deep_lookup_rows=meta.get("deep_lookup_rows", prog.get("deep_lookup_rows", 0)),
    )
    job.log(f"{build_mode.title()} shortlist build completed with {row_count} rows.")
    for label, count in status_counts.items():
        job.log(f"{label}: {count}")
    job.finish("completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Detached dashboard jobs")
    sub = parser.add_subparsers(dest="mode", required=True)

    dl = sub.add_parser("download-missing-sectors")
    dl.add_argument("--job-id", required=True)
    dl.add_argument("--min-sector", type=int, default=68)
    dl.add_argument("--threads", type=int, default=8)

    scan = sub.add_parser("scan-all-sectors")
    scan.add_argument("--job-id", required=True)
    scan.add_argument("--min-sector", type=int, default=68)
    scan.add_argument("--workers", type=int, default=os.cpu_count() or 16)
    scan.add_argument("--limit", type=int, default=0)
    scan.add_argument("--no-score", action="store_true")

    shortlist = sub.add_parser("build-crossmatched-shortlist")
    shortlist.add_argument("--job-id", required=True)
    shortlist.add_argument("--sectors", default="85-99")
    shortlist.add_argument("--out", type=Path, default=DEFAULT_SHORTLIST_PATH)
    shortlist.add_argument("--fast", action="store_true")

    args = parser.parse_args()
    job = JobRunner(args.job_id)
    try:
        if args.mode == "download-missing-sectors":
            return run_download_missing(job, args.min_sector, args.threads)
        if args.mode == "scan-all-sectors":
            return run_scan_all(job, args.min_sector, args.workers, args.limit, args.no_score)
        if args.mode == "build-crossmatched-shortlist":
            return run_build_shortlist(job, args.sectors, args.out, args.fast)
        job.log(f"Unknown mode: {args.mode}")
        job.finish("failed")
        return 1
    except Exception as exc:
        job.log(f"Job failed: {exc}")
        job.finish("failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
