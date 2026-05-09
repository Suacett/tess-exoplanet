"""
Exoplanet Pipeline Dashboard — Streamlit app
Pages: Pipeline Status | Candidate Browser | Lightcurve Inspector | Scan Results | Run Scan
"""
import json
import os
import re
import shlex
import sys
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from crossmatch_candidates import annotate_candidate_rows, get_crossmatch

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
ZENODO_DIR     = DATA_DIR / "zenodo"
TESS_DIR       = DATA_DIR / "tess"
RESULTS_DIR    = DATA_DIR / "results"
CANDS_DIR      = DATA_DIR / "candidates"
KEPLER_DIR     = DATA_DIR / "kepler_results"
SHORTLIST_DIR  = DATA_DIR / "crossmatch" / "shortlists"
DEEP_SCAN_RUNS_DIR = RESULTS_DIR / "deep_scan_runs"
DEFAULT_SHORTLIST_PATH = SHORTLIST_DIR / "sector_85_99_shortlist_crossmatched.csv"
FAST_SHORTLIST_PATH = SHORTLIST_DIR / "sector_85_99_shortlist_fast_crossmatched.csv"
DEEP_SCAN_TARGET_CACHE = DATA_DIR / "deep_scan_targets.json"
SCRIPTS_DIR    = Path(__file__).parent
SCAN_STATE_FILE = DATA_DIR / ".scan_state.json"
JOBS_DIR       = DATA_DIR / "jobs"
EXOMINER_MAX_SECTOR = 67

DEEP_SCAN_PRESETS = {
    "Quick validation": {
        "ds_min_sectors": 5,
        "ds_limit": 5,
        "ds_workers": 4,
        "ds_min_period": 5.0,
        "ds_max_period": 30.0,
        "ds_target_order": "quick-first",
    },
    "Overnight long-period batch": {
        "ds_min_sectors": 5,
        "ds_limit": 50,
        "ds_workers": 6,
        "ds_min_period": 20.0,
        "ds_max_period": 120.0,
        "ds_target_order": "quick-first",
    },
    "Full campaign": {
        "ds_min_sectors": 5,
        "ds_limit": 0,
        "ds_workers": 8,
        "ds_min_period": 1.0,
        "ds_max_period": 200.0,
        "ds_target_order": "quick-first",
    },
}


@st.cache_data(ttl=60)
def get_downloaded_sectors() -> list[int]:
    """Return sorted list of sector numbers that have at least one FITS file on disk."""
    if not TESS_DIR.exists():
        return []
    sectors = []
    for d in TESS_DIR.iterdir():
        if d.is_dir():
            m = re.match(r'sector(\d+)', d.name)
            if m and any(d.glob("*_lc.fits")):
                sectors.append(int(m.group(1)))
    return sorted(sectors)


def deep_scan_target_rows() -> list[dict]:
    cache = read_json(DEEP_SCAN_TARGET_CACHE, {})
    rows = cache.get("targets") or []
    if isinstance(rows, list):
        return rows
    return []


def order_deep_scan_targets(rows: list[dict], target_order: str) -> list[dict]:
    if target_order == "quick-first":
        return sorted(
            rows,
            key=lambda row: (
                int(row.get("n_sectors_seed", 999999) or 999999),
                int(row.get("seed_sector_span", 999999) or 999999),
                int(row.get("seed_sector_max", 999999) or 999999),
                int(row.get("tic_id", 999999999) or 999999999),
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("n_sectors_seed", 0) or 0),
            -int(row.get("seed_sector_span", 0) or 0),
            int(row.get("tic_id", 999999999) or 999999999),
        ),
    )


def preview_deep_scan_targets(min_sectors: int, limit: int, target_order: str) -> tuple[list[dict], str]:
    rows = deep_scan_target_rows()
    if not rows:
        return [], "Target cache missing. The first Deep Scan run will build it."
    filtered = [row for row in rows if int(row.get("n_sectors_seed", 0) or 0) >= min_sectors]
    ordered = order_deep_scan_targets(filtered, target_order)
    if limit > 0:
        ordered = ordered[:limit]
    source = read_json(DEEP_SCAN_TARGET_CACHE, {}).get("source_used", "unknown")
    return ordered, f"Target preview uses cached discovery from {source}."


def deep_scan_cost_warning(selected_rows: list[dict], min_period_days: float, max_period_days: float, limit: int) -> tuple[str, str]:
    if not selected_rows:
        if limit == 0:
            return "warning", "Target cache is not ready yet. Full campaign sizing is unknown until the first target-discovery cache build completes."
        return "info", "No cached target preview available yet."

    count = len(selected_rows)
    sectors = [int(row.get("n_sectors_seed", 0) or 0) for row in selected_rows]
    spans = [int(row.get("seed_sector_span", 0) or 0) for row in selected_rows]
    median_sectors = int(np.median(sectors)) if sectors else 0
    median_span = int(np.median(spans)) if spans else 0
    period_span = float(max_period_days) - float(min_period_days)

    if count <= 10 and period_span <= 40:
        return "success", (
            f"This looks like a quick validation run: {count} targets, "
            f"{min_period_days:.0f}–{max_period_days:.0f} d, median seed coverage {median_sectors} sectors."
        )
    if count <= 50 and period_span <= 120 and median_sectors <= 8 and median_span <= 20:
        return "info", (
            f"This looks like a light overnight batch: {count} targets, "
            f"{min_period_days:.0f}–{max_period_days:.0f} d, median seed coverage {median_sectors} sectors."
        )
    if count <= 100 and (period_span > 120 or median_sectors > 8 or median_span > 20):
        return "warning", (
            f"This run may be heavy because it searches {count} multi-sector targets over "
            f"{min_period_days:.0f}–{max_period_days:.0f} d "
            f"(median seed coverage {median_sectors} sectors, median span {median_span} sectors)."
        )
    if limit == 0 or count > 100 or period_span > 150:
        return "error", (
            f"Full campaign likely takes multiple days: {count} cached targets selected over "
            f"{min_period_days:.0f}–{max_period_days:.0f} d."
        )
    return "warning", (
        f"This run is likely heavier than a quick overnight batch: {count} targets, "
        f"{min_period_days:.0f}–{max_period_days:.0f} d."
    )


def list_deep_scan_dirs() -> list[Path]:
    dirs: list[Path] = []
    legacy = RESULTS_DIR / "deep_scan"
    if legacy.exists():
        dirs.append(legacy)
    if DEEP_SCAN_RUNS_DIR.exists():
        dirs.extend(sorted((p for p in DEEP_SCAN_RUNS_DIR.iterdir() if p.is_dir()), reverse=True))
    return dirs


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _deep_period_token(value) -> str:
    number = _safe_float(value, 0.0)
    return str(int(number)) if float(number).is_integer() else f"{number:g}"


def _deep_order_short(value: str) -> str:
    return "quick" if str(value or "") == "quick-first" else "coverage"


def _deep_limit_text(limit: int, target_count: int) -> str:
    if limit <= 0:
        return "all"
    return str(target_count if target_count > 0 else limit)


def deep_scan_dir_summary(path: Path) -> str | None:
    if path.name == "deep_scan":
        return "Legacy Deep Scan output directory."

    cfg = read_json(path / "campaign_config.json", {})
    prog = read_json(path / "deep_scan_progress.json", {})
    if not cfg and not prog:
        return None

    if cfg.get("display_name"):
        return str(cfg.get("display_name"))

    run_cfg = dict(prog.get("run_config") or {})
    if run_cfg.get("display_name"):
        return str(run_cfg.get("display_name"))
    min_period = cfg.get("min_period_days", run_cfg.get("min_period_days"))
    max_period = cfg.get("max_period_days", run_cfg.get("max_period_days"))
    min_sectors = _safe_int(cfg.get("min_sectors", run_cfg.get("min_sectors")), 0)
    target_order = str(cfg.get("target_order", run_cfg.get("target_order", "")) or "")
    target_count = _safe_int(cfg.get("total_targets", run_cfg.get("target_count")), 0)
    limit = _safe_int(cfg.get("limit", run_cfg.get("limit")), 0)

    if path.name.startswith("campaign_"):
        workers_cfg = cfg.get("workers") or {}
        ct_workers = _safe_int(workers_cfg.get("ct100"), 0)
        fedora_workers = _safe_int(workers_cfg.get("fedora"), 0)
        return (
            f"Deep Scan {_deep_period_token(min_period)}-{_deep_period_token(max_period)}d"
            f" · min{min_sectors}"
            f" · {_deep_order_short(target_order)}"
            f" · {_deep_limit_text(limit, target_count)} targets"
            f" · ct{ct_workers}/fd{fedora_workers}"
        )

    workers = _safe_int(run_cfg.get("workers"), 0)
    return (
        f"Deep Scan {_deep_period_token(min_period)}-{_deep_period_token(max_period)}d"
        f" · min{min_sectors}"
        f" · {_deep_order_short(target_order)}"
        f" · {_deep_limit_text(limit, target_count)} targets"
        f" · w{workers}"
    )


def deep_scan_dir_label(path: Path) -> str:
    if path.name == "deep_scan":
        return "legacy / deep_scan"
    summary = deep_scan_dir_summary(path)
    if path.name.startswith("campaign_"):
        prefix = "campaign"
        stamp = path.name.removeprefix("campaign_")
    elif path.name.startswith("run_"):
        prefix = "run"
        stamp = path.name.removeprefix("run_")
    else:
        prefix = "deep-scan"
        stamp = path.name
    return f"{prefix} / {summary or stamp} / {stamp}"


def ensure_deep_scan_plot(tic_id: int, period: float, t0: float | None, depth_ppm: float, plot_dir: Path) -> tuple[Path | None, str | None]:
    png_path = plot_dir / f"tic_{tic_id}_deep.png"
    if png_path.exists():
        return png_path, None
    try:
        import lightkurve as lk
        from plot_candidate import make_4panel

        sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS", author="SPOC")
        if len(sr) == 0:
            return None, "No SPOC lightcurves found for this TIC."

        lcs = sr.download_all(quality_bitmask="default")
        if lcs is None or len(lcs) == 0:
            return None, "Deep-scan plot download returned no lightcurves."

        lc_raw = lcs.stitch() if len(lcs) > 1 else lcs[0]
        lc_flat = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)
        epoch = float(t0) if t0 is not None and not pd.isna(t0) else float(lc_flat.time[0].value)
        plot_dir.mkdir(parents=True, exist_ok=True)
        make_4panel(
            lc_raw=lc_raw,
            lc_flat=lc_flat,
            period_d=float(period),
            t0_btjd=epoch,
            tic_id=int(tic_id),
            sector=f"deep/{len(lcs)} sectors",
            depth_ppm=float(depth_ppm),
            bls_power=0.0,
            out_path=png_path,
        )
        return png_path, None
    except Exception as exc:
        text = str(exc).strip().replace("\n", " ")
        return None, text[:220] if text else exc.__class__.__name__

# ── Scan state helpers ────────────────────────────────────────────────────────

def read_scan_state() -> dict:
    try:
        if SCAN_STATE_FILE.exists():
            return json.loads(SCAN_STATE_FILE.read_text())
    except Exception:
        pass
    return {"running": False}


def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def write_scan_state(state: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SCAN_STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def parse_iso_dt(text: str | None) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def detect_active_deep_scan_campaign(max_age_minutes: int = 20) -> dict | None:
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    if not DEEP_SCAN_RUNS_DIR.exists():
        return None

    now_utc = datetime.now(timezone.utc)
    best: tuple[datetime, Path, dict] | None = None
    for run_dir in DEEP_SCAN_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        prog_path = run_dir / "deep_scan_progress.json"
        if not prog_path.exists():
            continue
        prog = read_json(prog_path, {})
        if not prog or prog.get("complete"):
            continue
        updated = parse_iso_dt(prog.get("updated_at"))
        if updated is None:
            continue
        age_seconds = (now_utc - updated.astimezone(timezone.utc)).total_seconds()
        if age_seconds > max_age_minutes * 60:
            continue
        item = (updated, run_dir, prog)
        if best is None or updated > best[0]:
            best = item

    if best is None:
        return None

    updated, run_dir, prog = best
    total = _safe_int(prog.get("total"), 0)
    done = _safe_int(prog.get("done"), 0)
    created_at = prog.get("updated_at", "")
    cfg_path = run_dir / "campaign_config.json"
    label = f"Deep Scan campaign ({done}/{total})"
    if cfg_path.exists():
        cfg = read_json(cfg_path, {})
        created_at = cfg.get("created_at", created_at)
        if cfg.get("display_name"):
            label = str(cfg.get("display_name"))
    return {
        "job_id": f"deep_scan_campaign_{run_dir.name}",
        "running": True,
        "mode": "deep_scan",
        "status": "running",
        "label": label,
        "output_dir": str(run_dir),
        "log_file": str(run_dir / "deep_scan.log"),
        "current_log_file": str(run_dir / "deep_scan.log"),
        "started_at": created_at,
        "pid": 0,
        "synthetic": True,
    }


def scan_is_running() -> tuple[bool, dict]:
    """Return (running, state). Updates state file if PID has died."""
    state = read_scan_state()
    if not state.get("running"):
        campaign_state = detect_active_deep_scan_campaign()
        if campaign_state:
            return True, campaign_state
        return False, state
    pid = state.get("pid", 0)
    if pid and is_pid_alive(pid):
        return True, state
    # PID died — mark as finished
    state["running"] = False
    state.setdefault("finished_at", datetime.now().isoformat())
    if state.get("status") not in {"completed", "failed"}:
        if state.get("mode") == "hunt" and state.get("sector") and sector_hunt_complete(int(state["sector"])):
            state["status"] = "completed"
            state["sectors_done"] = 1
        elif state.get("mode") == "deep_scan":
            _deep_dir = Path(state.get("output_dir") or (RESULTS_DIR / "deep_scan"))
            _prog = _deep_dir / "deep_scan_progress.json"
            _prog_data = read_json(_prog, {})
            if _prog_data.get("complete"):
                state["status"] = "completed"
            else:
                state["status"] = "failed"
        else:
            state["status"] = "failed"
    write_scan_state(state)
    write_job_meta(state)
    campaign_state = detect_active_deep_scan_campaign()
    if campaign_state:
        return True, campaign_state
    return False, state


def current_state_log(state: dict) -> Path | None:
    for key in ("current_log_file", "log_file"):
        val = state.get(key)
        if val and Path(val).exists():
            log_path = Path(val)
            if log_path.name == "hunt.log":
                detail_log = log_path.with_name("scan_sector.log")
                if detail_log.exists():
                    try:
                        hunt_done = "✅ Done!" in log_path.read_text(errors="replace")
                    except Exception:
                        hunt_done = False
                    try:
                        if not hunt_done and detail_log.stat().st_mtime >= log_path.stat().st_mtime:
                            return detail_log
                    except Exception:
                        return detail_log
            return log_path
    for key in ("current_log_file", "log_file"):
        val = state.get(key)
        if val:
            return Path(val)
    return None


def format_cmd(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def make_job_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_job_meta(state: dict) -> None:
    job_id = state.get("job_id")
    if not job_id:
        return
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def load_recent_jobs(limit: int = 5) -> list[dict]:
    jobs = []
    if not JOBS_DIR.exists():
        return jobs
    for meta_path in sorted(JOBS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            jobs.append(json.loads(meta_path.read_text()))
        except Exception:
            continue
    return jobs


def job_mode_label(state: dict) -> str:
    mode = state.get("mode", "")
    return {
        "hunt": "Hunting",
        "deep_scan": "Deep Scan",
        "download_missing": "Downloading Sectors",
        "scan_all": "Scan All",
        "build_shortlist": "Shortlist Build",
    }.get(mode, mode or "Job")


def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[mK]', '', text)


def tail_log(log_path: Path, n_chars: int = 8000) -> str:
    try:
        text = strip_ansi(log_path.read_text(errors="replace"))
    except Exception:
        return ""

    # hunt.log is already clean — return as-is
    if log_path.name == "hunt.log":
        return text[-n_chars:] if len(text) > n_chars else text

    # Filter raw subprocess logs — drop lightkurve/astropy noise
    _KEEP = re.compile(
        r'CANDIDATE|ERROR|error|WARNING|warn|sector|Sector|'
        r'stars/s|stars/sec|SDE|BLS|done|Done|complete|Complete|'
        r'Phase [12]|Prefetch|prefetch|Scanning|scan|'
        r'TIC|tic|summary|Summary|candidates|Candidates|'
        r'^\s*[=─╭╰│]'
    )
    _DROP = re.compile(
        r'Opening /|Detected filetype|cadences will be ignored|'
        r'quality_bitmask|NullHandler|^\s*$'
    )
    filtered = [l for l in text.splitlines()
                if _KEEP.search(l) and not _DROP.search(l)]
    out = "\n".join(filtered)
    return out[-n_chars:] if len(out) > n_chars else out


def extract_corrupt_cache_path(error_text: str) -> Path | None:
    match = re.search(
        r"Data product (.+?) of type .*?This file may be corrupt due to an interrupted download",
        error_text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    try:
        return Path(match.group(1).strip())
    except Exception:
        return None


def fetch_inspector_lightcurves(lk, tic_str: str, author: str):
    tic_id = int(tic_str.replace("TIC", "").strip())
    pattern = f"*-{tic_id:016d}-*_lc.fits"
    local_lcs = []
    for fits_path in sorted(TESS_DIR.glob(f"sector*/{pattern}")):
        try:
            lc = lk.read(str(fits_path), quality_bitmask="default")
            local_lcs.append(lc)
        except Exception:
            pass
    if local_lcs:
        lcs = lk.LightCurveCollection(local_lcs)
        return None, lcs
    results = lk.search_lightcurve(tic_str, mission="TESS", author=author, exptime=120)
    if len(results) == 0:
        results = lk.search_lightcurve(tic_str, mission="TESS")
    if len(results) == 0:
        raise FileNotFoundError(f"No TESS data found for {tic_str}")
    lcs = results.download_all(quality_bitmask="default")
    if lcs is None or len(lcs) == 0:
        raise RuntimeError(f"Download returned no lightcurves for {tic_str}")
    return results, lcs


def planet_size_estimate(depth_ppm: float) -> str:
    import math
    if depth_ppm <= 0:
        return "unknown size"
    rp = math.sqrt(depth_ppm / 1_000_000) * 109.0
    if rp < 1.5:   return "~Earth-sized"
    elif rp < 4:   return "~Super-Earth"
    elif rp < 8:   return "~Neptune-sized"
    elif rp < 15:  return "~Saturn-sized"
    else:          return "~Jupiter-sized"


def build_bls_attempts(period_max: float, stitched: bool, baseline_days: float | None = None):
    if not stitched:
        return [(np.arange(0.5, period_max, 0.001), np.arange(0.02, 0.20, 0.003), None)]

    # Interactive stitched BLS needs a hard cap and a much coarser grid or it explodes.
    capped_max = min(period_max, 120.0)
    if baseline_days is not None and baseline_days > 0:
        capped_max = min(capped_max, max(20.0, baseline_days * 0.60))

    attempts = [
        (
            np.arange(0.5, capped_max, max(0.05, capped_max / 3000.0)),
            np.arange(0.05, 0.25, 0.02),
            10,
        ),
        (
            np.arange(0.5, min(capped_max, 80.0), max(0.08, capped_max / 1800.0)),
            np.arange(0.06, 0.22, 0.03),
            20,
        ),
        (
            np.arange(0.5, min(capped_max, 50.0), 0.12),
            np.arange(0.08, 0.20, 0.04),
            30,
        ),
    ]
    return attempts


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def periods_agree(period_a: float | None, period_b: float | None, rel_tol: float = 0.02) -> bool:
    if period_a is None or period_b is None or period_a <= 0 or period_b <= 0:
        return False
    return abs(period_a - period_b) / max(period_a, period_b) <= rel_tol


def auto_detect_epoch(lc_flat, period: float) -> float | None:
    try:
        arb_t0 = float(lc_flat.time[0].value)
        lc_fold = lc_flat.fold(period=period, epoch_time=arb_t0)
        lc_bin = lc_fold.bin(time_bin_size=0.01)
        phase = np.asarray(lc_bin.phase.value, dtype=float)
        flux = np.ma.filled(np.asarray(lc_bin.flux.value), np.nan).astype(float)
        valid = np.isfinite(phase) & np.isfinite(flux)
        if not np.any(valid):
            return arb_t0
        mid_idx = int(np.nanargmin(flux[valid]))
        return arb_t0 + float(phase[valid][mid_idx])
    except Exception:
        return None


def run_bls_search(lc_for_bls, stitched: bool) -> dict:
    period_max = 200.0 if stitched else 14.0
    baseline_days = None
    if stitched:
        try:
            time_vals = np.asarray(lc_for_bls.time.value, dtype=float)
            if time_vals.size >= 2:
                baseline_days = float(np.nanmax(time_vals) - np.nanmin(time_vals))
        except Exception:
            baseline_days = None

    blsm = None
    last_err = None
    fallback_attempt = 1
    for idx, (periods, durations, freq_factor) in enumerate(
        build_bls_attempts(period_max, stitched, baseline_days),
        start=1,
    ):
        try:
            if freq_factor is None:
                blsm = lc_for_bls.to_periodogram(method="bls", period=periods, duration=durations)
            else:
                blsm = lc_for_bls.to_periodogram(
                    method="bls",
                    period=periods,
                    duration=durations,
                    frequency_factor=freq_factor,
                )
            fallback_attempt = idx
            break
        except Exception as exc:
            last_err = exc
            continue

    if blsm is None and last_err is not None:
        raise last_err
    return {
        "period": float(blsm.period_at_max_power.value),
        "t0": float(blsm.transit_time_at_max_power.value),
        "sde": float(blsm.max_power),
        "blsm": blsm,
        "attempt": fallback_attempt,
        "stitched": stitched,
    }


def fold_signal_summary(lc_flat, period: float, t0: float | None) -> dict:
    if period is None or period <= 0:
        return {}
    if t0 is None:
        t0 = auto_detect_epoch(lc_flat, period)
    if t0 is None:
        return {}

    try:
        lc_fold = lc_flat.fold(period=period, epoch_time=t0)
        lc_bin = lc_fold.bin(time_bin_size=0.01)
    except Exception:
        return {}

    phase_days = np.asarray(lc_bin.phase.value, dtype=float)
    flux = np.ma.filled(np.asarray(lc_bin.flux.value), np.nan).astype(float)
    valid = np.isfinite(phase_days) & np.isfinite(flux)
    if not np.any(valid):
        return {}

    phase_days = phase_days[valid]
    flux = flux[valid]
    order = np.argsort(phase_days)
    phase_days = phase_days[order]
    flux = flux[order]
    phase_frac = phase_days / period

    search_mask = np.abs(phase_frac) < 0.25
    if not np.any(search_mask):
        return {}
    center_idx = int(np.nanargmin(flux[search_mask]))
    center_phase = float(phase_frac[search_mask][center_idx])
    primary_flux = float(np.nanmin(flux[search_mask]))

    out_mask = np.abs(phase_frac - center_phase) > 0.15
    if np.sum(out_mask) < 5:
        out_mask = np.abs(phase_frac) > 0.15
    out_flux = flux[out_mask]
    if out_flux.size < 5:
        return {}

    baseline = float(np.nanmedian(out_flux))
    noise_ppm = float(np.nanstd(out_flux) / baseline * 1e6) if baseline else 0.0
    depth_frac = max(0.0, (baseline - primary_flux) / baseline) if baseline else 0.0
    depth_ppm = depth_frac * 1e6

    threshold_flux = baseline - depth_frac * baseline * 0.5
    dip_mask = (np.abs(phase_frac - center_phase) < 0.30) & (flux <= threshold_flux)
    broad_frac = 0.0
    if np.any(dip_mask):
        dip_phase = phase_frac[dip_mask]
        broad_frac = float(np.nanmax(dip_phase) - np.nanmin(dip_phase))

    score = depth_ppm / max(broad_frac, 0.02)
    return {
        "phase_days": phase_days,
        "phase_frac": phase_frac,
        "flux": flux,
        "baseline": baseline,
        "noise_ppm": noise_ppm,
        "depth_ppm": depth_ppm,
        "center_phase": center_phase,
        "broad_frac": broad_frac,
        "score": score,
        "period": period,
        "t0": t0,
    }


def odd_even_depth_check(lc_flat, period: float, t0: float | None, width_frac: float) -> dict:
    if t0 is None or period <= 0:
        return {"flag": False, "valid": False, "reason": "missing_period_or_epoch"}
    try:
        times = np.asarray(lc_flat.time.value, dtype=float)
        flux = np.ma.filled(np.asarray(lc_flat.flux.value), np.nan).astype(float)
    except Exception:
        return {"flag": False, "valid": False, "reason": "bad_lightcurve_arrays"}
    valid = np.isfinite(times) & np.isfinite(flux)
    if np.sum(valid) < 50:
        return {"flag": False, "valid": False, "reason": "too_few_points"}
    times = times[valid]
    flux = flux[valid]
    phase = ((times - t0 + 0.5 * period) % period) - 0.5 * period
    transit_num = np.round((times - t0) / period).astype(int)
    half_window_days = max(0.02, min(0.10, width_frac * period * 0.75 if width_frac else period * 0.06))
    in_mask = np.abs(phase) < half_window_days
    out_mask = np.abs(phase) > max(half_window_days * 2.0, min(0.25, 0.20 * period))
    if np.sum(in_mask) < 10 or np.sum(out_mask) < 20:
        return {"flag": False, "valid": False, "reason": "insufficient_in_out_points"}
    baseline = float(np.nanmedian(flux[out_mask]))
    odd_mask = in_mask & ((transit_num % 2) != 0)
    even_mask = in_mask & ((transit_num % 2) == 0)
    if np.sum(odd_mask) < 5 or np.sum(even_mask) < 5 or baseline <= 0:
        return {"flag": False, "valid": False, "reason": "insufficient_odd_even_points"}
    odd_depth = max(0.0, (baseline - float(np.nanmedian(flux[odd_mask]))) / baseline) * 1e6
    even_depth = max(0.0, (baseline - float(np.nanmedian(flux[even_mask]))) / baseline) * 1e6
    diff = abs(odd_depth - even_depth)
    flag = max(odd_depth, even_depth) > 0 and diff > max(0.25 * max(odd_depth, even_depth), 300.0)
    return {
        "valid": True,
        "flag": flag,
        "odd_depth_ppm": odd_depth,
        "even_depth_ppm": even_depth,
        "difference_ppm": diff,
        "fractional_difference": (diff / max(odd_depth, even_depth)) if max(odd_depth, even_depth) > 0 else 0.0,
    }


def secondary_eclipse_check(summary: dict) -> dict:
    phase_frac = summary.get("phase_frac")
    flux = summary.get("flux")
    baseline = summary.get("baseline")
    noise_ppm = summary.get("noise_ppm", 0.0)
    primary_depth = summary.get("depth_ppm", 0.0)
    if phase_frac is None or flux is None or baseline is None or baseline <= 0:
        return {"flag": False}
    sec_mask = np.abs(np.abs(phase_frac) - 0.5) < 0.05
    if np.sum(sec_mask) < 5:
        return {"flag": False}
    secondary_flux = float(np.nanmin(flux[sec_mask]))
    secondary_depth = max(0.0, (baseline - secondary_flux) / baseline) * 1e6
    secondary_snr = secondary_depth / max(noise_ppm, 1.0)
    flag = secondary_depth > max(primary_depth * 0.30, 300.0) and secondary_snr > 3.0
    return {
        "flag": flag,
        "secondary_depth_ppm": secondary_depth,
        "secondary_snr": secondary_snr,
    }


def alias_harmonic_check(lc_flat, period: float, t0: float | None, current_summary: dict) -> dict:
    candidates = []
    for alt_period, label in ((period / 2.0, "P/2"), (period * 2.0, "2P")):
        if alt_period <= 0.1:
            continue
        alt_summary = fold_signal_summary(lc_flat, alt_period, t0)
        if not alt_summary:
            continue
        better_score = alt_summary.get("score", 0.0) > current_summary.get("score", 0.0) * 1.35
        narrower = alt_summary.get("broad_frac", 1.0) < current_summary.get("broad_frac", 1.0) * 0.80
        if better_score and narrower:
            candidates.append((label, alt_period))
    if candidates:
        label, alt_period = candidates[0]
        return {"flag": True, "label": label, "period": alt_period}
    return {"flag": False}


def build_suspicion_checks(lc_flat, period: float, t0: float | None) -> dict:
    summary = fold_signal_summary(lc_flat, period, t0)
    if not summary:
        return {"summary": {}, "messages": [], "major_flags": [], "flags": {}}

    depth_ppm = summary.get("depth_ppm", 0.0)
    messages = []
    major_flags = []
    flags = {}

    if depth_ppm > 50000:
        messages.append("Almost certainly eclipsing binary: the dip is deeper than 50,000 ppm.")
        major_flags.append("deep")
        flags["deep"] = True
    elif depth_ppm > 20000:
        messages.append("Very deep signal: much more likely an eclipsing binary than a clean planet transit.")
        major_flags.append("deep")
        flags["deep"] = True
    elif depth_ppm > 2000:
        messages.append("Deeper than a typical clean planet candidate: treat this as EB-risky until proven otherwise.")
        major_flags.append("deep")
        flags["deep"] = True

    odd_even = odd_even_depth_check(lc_flat, period, summary.get("t0"), summary.get("broad_frac", 0.0))
    if odd_even.get("flag"):
        messages.append(
            f"Odd/even transit mismatch: odd {odd_even['odd_depth_ppm']:.0f} ppm vs even {odd_even['even_depth_ppm']:.0f} ppm."
        )
        major_flags.append("odd_even")
        flags["odd_even"] = True

    secondary = secondary_eclipse_check(summary)
    if secondary.get("flag"):
        messages.append(
            f"Possible secondary eclipse near phase 0.5: secondary depth about {secondary['secondary_depth_ppm']:.0f} ppm."
        )
        major_flags.append("secondary")
        flags["secondary"] = True

    alias = alias_harmonic_check(lc_flat, period, summary.get("t0"), summary)
    if alias.get("flag"):
        messages.append(
            f"Alias/harmonic warning: {alias['label']} ({alias['period']:.4f} d) folds more cleanly than the current period."
        )
        major_flags.append("alias")
        flags["alias"] = True

    if summary.get("broad_frac", 0.0) > 0.15:
        messages.append("Broad or wave-like dip: this looks more like variability/systematics than one compact transit.")
        major_flags.append("broad")
        flags["broad"] = True

    return {
        "summary": summary,
        "messages": messages,
        "major_flags": major_flags,
        "flags": flags,
    }


def verification_status(verdict: str | None) -> str:
    if not verdict:
        return "not_run"
    if "Likely real" in verdict:
        return "real"
    if "Inconclusive" in verdict:
        return "inconclusive"
    return "reject"


def build_confidence_ladder(mode: str, has_search_hit: bool, verify_result: dict | None, suspicion: dict) -> list[str]:
    stages = []
    vstatus = verification_status((verify_result or {}).get("verdict"))
    if has_search_hit:
        stages.append("Search hit only")
    if vstatus == "real":
        stages.append("Verified across sectors")
    if suspicion.get("major_flags"):
        stages.append("EB-risky")
    if mode == "Validate Known Target":
        stages.append("Known-object match")
    if has_search_hit or vstatus in {"real", "inconclusive"}:
        stages.append("Worth manual review")
    if (
        mode == "Search for Candidate"
        and vstatus == "real"
        and not suspicion.get("major_flags")
        and safe_float((verify_result or {}).get("consistency_score"), 0.0) >= 0.70
        and safe_float(suspicion.get("summary", {}).get("depth_ppm"), 0.0) <= 2000
    ):
        stages.append("Strong candidate")
    return stages


def render_badges(labels: list[str]) -> None:
    if not labels:
        return
    pills = []
    for label in labels:
        pills.append(
            f"<span style='display:inline-block;background:#1f6feb;color:#f0f6fc;"
            f"padding:4px 10px;border-radius:999px;margin:0 6px 6px 0;font-size:0.85rem;'>"
            f"{label}</span>"
        )
    st.markdown("".join(pills), unsafe_allow_html=True)


def append_period_history(history: list[dict] | None, label: str, period: float | None, source: str) -> list[dict]:
    items = [dict(item) for item in (history or []) if isinstance(item, dict)]
    p = safe_float(period)
    if p is None or p <= 0:
        return items[-5:]
    record = {
        "label": str(label),
        "period": float(p),
        "source": str(source),
    }
    deduped = [item for item in items if not periods_agree(safe_float(item.get("period")), p, rel_tol=0.0005)]
    deduped.append(record)
    return deduped[-5:]


def harmonic_source_key(action: str, base_source: str) -> str:
    return f"harmonic:{action}:{base_source}"


def odd_even_result_text(result: dict | None) -> tuple[str, str]:
    info = dict(result or {})
    if not info.get("valid"):
        return "info", "Odd/even check inconclusive at the current period."
    odd_depth = safe_float(info.get("odd_depth_ppm"), 0.0) or 0.0
    even_depth = safe_float(info.get("even_depth_ppm"), 0.0) or 0.0
    diff = safe_float(info.get("difference_ppm"), 0.0) or 0.0
    frac = safe_float(info.get("fractional_difference"), 0.0) or 0.0
    if diff > max(1000.0, 0.45 * max(odd_depth, even_depth, 1.0)):
        return "error", (
            f"Strong odd/even mismatch, EB-risky: odd {odd_depth:.0f} ppm vs even {even_depth:.0f} ppm."
        )
    if info.get("flag") or diff > max(400.0, 0.20 * max(odd_depth, even_depth, 1.0)):
        return "warning", (
            f"Possible odd/even mismatch: odd {odd_depth:.0f} ppm vs even {even_depth:.0f} ppm."
        )
    return "success", (
        f"Odd/even depths look similar: odd {odd_depth:.0f} ppm vs even {even_depth:.0f} ppm "
        f"({frac * 100:.0f}% difference)."
    )


def period_helper_suggestion(current_period: float | None, matched_period: float | None) -> str:
    current = safe_float(current_period)
    matched = safe_float(matched_period)
    suggestions = ["compare current fold with P/2 and 2P"]
    if matched is not None and current is not None and not periods_agree(current, matched, rel_tol=0.005):
        suggestions.append("try the catalogue period")
    return "Suggestion: " + ", then ".join(suggestions) + "."


def period_source_label(source: str | None) -> str:
    raw = str(source or "")
    if raw.startswith("harmonic:"):
        parts = raw.split(":", 2)
        if len(parts) == 3:
            _, action, base = parts
            return f"harmonic helper ({action} from {period_source_label(base)})"
    return {
        "known": "known input",
        "current_selected": "current selected period",
        "single_bls": "single-sector BLS",
        "single_bls_loaded": "single-sector BLS (from earlier scan)",
        "stitched_bls": "stitched BLS",
        "harmonic_p_half": "harmonic helper (P/2)",
        "harmonic_2p": "harmonic helper (2P)",
        "harmonic_3p": "harmonic helper (3P)",
        "harmonic_p_third": "harmonic helper (P/3)",
        "catalogue_period": "public catalogue period",
        "base_period": "current base period",
        "manual": "manual adjustment",
    }.get(source or "", source or "unknown")


def candidate_key(tic_id, period) -> tuple[str, str]:
    try:
        tic = str(int(float(tic_id)))
    except (TypeError, ValueError):
        tic = ""
    p = safe_float(period)
    return tic, (f"{p:.6f}" if p is not None else "")


def build_public_match_map(df: pd.DataFrame, limit: int = 100, detailed: bool = False) -> dict[tuple[str, str], dict]:
    if df is None or df.empty or "tic_id" not in df.columns:
        return {}
    work = df.copy()
    if "bls_power" in work.columns:
        work = work.sort_values("bls_power", ascending=False)
    rows = []
    for _, row in work.head(limit).iterrows():
        rows.append(
            {
                "tic_id": row.get("tic_id"),
                "period": row.get("period"),
                "depth_ppm": row.get("depth_ppm"),
                "sector": row.get("sector"),
            }
        )
    out = {}
    for item in annotate_candidate_rows(rows, detailed=detailed):
        out[candidate_key(item.get("tic_id"), item.get("local_period"))] = item
    return out


def public_status_chip(match: dict | None) -> str:
    if not match:
        return ""
    emoji = {
        "Confirmed planet": "🟦",
        "Known TOI": "🟨",
        "Caution": "🟥",
        "Unknown": "⬜",
        "Lookup failed": "⚪",
    }.get(match.get("status_label", ""), "⬜")
    return f"{emoji} {match.get('status_label', 'Unknown')}"


def render_public_match_box(match: dict | None) -> None:
    st.markdown("### Public Catalogue Match")
    if not match:
        st.info("No public catalogue match has been checked yet.")
        return

    summary = match.get("plain_english_summary", "No public catalogue summary available.")
    status = match.get("match_status")
    if status == "confirmed_planet":
        st.success(summary)
    elif status == "known_toi":
        st.info(summary)
    elif status == "suspicious_alert_object":
        st.warning(summary)
    elif status == "lookup_failed":
        st.warning(summary)
    else:
        st.info(summary)

    details = [
        f"- **Public status:** {match.get('status_label', 'Unknown')}",
        f"- **Matched name:** {match.get('matched_name') or 'none'}",
        f"- **Source used:** {match.get('source_used') or 'n/a'}",
    ]
    matched_period = safe_float(match.get("matched_period"))
    if matched_period is not None:
        details.append(f"- **Matched period:** {matched_period:.5f} d")
    diff_pct = safe_float(match.get("period_difference_pct"))
    if diff_pct is not None:
        details.append(f"- **Local vs catalogue period difference:** {diff_pct:.2f}%")
    st.markdown("\n".join(details))

    notes = match.get("caution_notes") or []
    if notes:
        st.markdown("**Public caution notes**")
        for note in notes[:5]:
            st.warning(note)

    st.caption(
        "Local 'planet candidate' labels come from this pipeline. "
        "This public match box tells you whether the object already appears in official public catalogues."
    )


def build_agreement_messages(known_period, single_period, stitched_period, current_period, verify_period, base_source):
    messages = []
    if known_period and single_period:
        messages.append(
            "Known period and single-sector BLS agree." if periods_agree(known_period, single_period)
            else "Single-sector BLS disagrees with the supplied known period."
        )
    if known_period and stitched_period:
        messages.append(
            "Known period and stitched BLS agree." if periods_agree(known_period, stitched_period)
            else "Stitched BLS disagrees with the supplied known period."
        )
    if current_period and single_period and base_source != "single_bls" and not periods_agree(current_period, single_period):
        messages.append("Manual/current period differs from the single-sector BLS best period.")
    if current_period and stitched_period and base_source != "stitched_bls" and not periods_agree(current_period, stitched_period):
        messages.append("Manual/current period differs from the stitched BLS best period.")
    if verify_period and current_period:
        messages.append(
            "Verification is running at the current selected period." if periods_agree(verify_period, current_period)
            else "Verification period differs from the current selected period."
        )
    if not messages:
        messages.append("No important disagreement detected among the periods you ran.")
    return messages


def load_validation_report() -> dict | None:
    report_path = DATA_DIR / "validation" / "last_validation_report.json"
    try:
        if report_path.exists():
            return json.loads(report_path.read_text())
    except Exception:
        pass
    return None


def shortlist_files() -> list[Path]:
    if not SHORTLIST_DIR.exists():
        return []
    return sorted(SHORTLIST_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def shortlist_metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def shortlist_metadata(path: Path) -> dict:
    return read_json(shortlist_metadata_path(path), {})


def shortlist_kind(path: Path, meta: dict | None = None) -> str:
    meta = meta or {}
    name = path.name.lower()
    if "manual_review" in name:
        return "manual-review"
    if meta.get("build_mode") == "fast":
        return "fast"
    return "full"


def latest_scan_result_mtime(min_sector: int = 85, max_sector: int = 99) -> float | None:
    mtimes = []
    for sector in range(min_sector, max_sector + 1):
        sector_dir = RESULTS_DIR / f"sector{sector}"
        for name in ("bls_exominer_results.csv", "bls_results.csv"):
            path = sector_dir / name
            if path.exists():
                mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else None


def shortlist_is_stale(path: Path, min_sector: int = 85, max_sector: int = 99) -> bool:
    if not path.exists():
        return True
    latest_scan = latest_scan_result_mtime(min_sector=min_sector, max_sector=max_sector)
    if latest_scan is None:
        return False
    return latest_scan > path.stat().st_mtime


@st.cache_data
def load_shortlist_csv(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def shortlist_sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "status_label" in work.columns and "public_status" not in work.columns:
        work["public_status"] = work["status_label"]
    status_rank = {
        "Unknown": 0,
        "Caution": 1,
        "Known TOI": 2,
        "Confirmed planet": 3,
        "Lookup failed": 4,
    }
    work["_status_rank"] = work.get("public_status", pd.Series(["Unknown"] * len(work))).map(status_rank).fillna(9)
    if "verified" in work.columns:
        work["_verified_rank"] = work["verified"].astype(str).str.lower().eq("true").astype(int)
    else:
        work["_verified_rank"] = 0
    if "consistency_score" in work.columns:
        work["_consistency_rank"] = pd.to_numeric(work["consistency_score"], errors="coerce").fillna(-1.0)
    else:
        work["_consistency_rank"] = -1.0
    if "depth_ppm" in work.columns:
        work["_depth_rank"] = pd.to_numeric(work["depth_ppm"], errors="coerce").fillna(1e12)
    else:
        work["_depth_rank"] = 1e12
    if "eb_warning" in work.columns:
        work["_eb_rank"] = work["eb_warning"].fillna("").astype(str).str.len().gt(0).astype(int)
    else:
        work["_eb_rank"] = 0
    work = work.sort_values(
        by=["_status_rank", "_verified_rank", "_consistency_rank", "_eb_rank", "_depth_rank"],
        ascending=[True, False, False, True, True],
    )
    return work.drop(columns=[c for c in work.columns if c.startswith("_")], errors="ignore")


def read_job_progress(state: dict) -> dict:
    progress_file = state.get("progress_file")
    if not progress_file:
        return {}
    return read_json(Path(progress_file), {})


def render_shortlist_progress(progress: dict) -> None:
    if not progress:
        return
    total = int(progress.get("progress_total") or 0)
    done = int(progress.get("progress_done") or 0)
    rows_total = int(progress.get("rows_total") or 0)
    rows_done = int(progress.get("rows_processed") or 0)
    tics_total = int(progress.get("unique_tics_total") or 0)
    tics_ready = int(progress.get("unique_tics_ready") or 0)
    cache_hits = int(progress.get("cache_hits") or 0)
    cache_misses = int(progress.get("cache_misses") or 0)
    failed_tics = int(progress.get("failed_tics") or 0)
    skipped_tics = int(progress.get("skipped_tics") or 0)
    build_mode = str(progress.get("build_mode") or "full")
    cheap_rows = int(progress.get("cheap_pass_rows") or 0)
    deep_rows = int(progress.get("deep_lookup_rows") or 0)
    deep_done = int(progress.get("deep_lookup_done") or 0)
    phase = progress.get("phase", "Running")
    current_tic = progress.get("current_tic")
    current_source = progress.get("current_source")
    if total > 0:
        st.progress(min(max(done / total, 0.0), 1.0), text=f"{phase} — {done}/{total} work units")
    metrics = []
    metrics.append(f"Mode: {'Fast triage-first' if build_mode == 'fast' else 'Full enrichment'}")
    if rows_total:
        metrics.append(f"Rows: {rows_done}/{rows_total}")
    if tics_total:
        metrics.append(f"TICs ready: {tics_ready}/{tics_total}")
    if cheap_rows:
        metrics.append(f"Cheap pass rows: {cheap_rows}")
    if deep_rows:
        metrics.append(f"Deep lookup: {deep_done}/{deep_rows}")
    metrics.append(f"Cache hits: {cache_hits}")
    metrics.append(f"Cache misses: {cache_misses}")
    if failed_tics:
        metrics.append(f"Lookup failed: {failed_tics}")
    if skipped_tics and skipped_tics != failed_tics:
        metrics.append(f"Skipped: {skipped_tics}")
    if current_tic:
        metrics.append(f"Current TIC: {current_tic}")
    if current_source:
        metrics.append(f"Source: {current_source}")
    st.caption(" | ".join(metrics))
    if progress.get("last_updated"):
        st.caption(f"Last progress update: {str(progress['last_updated']).replace('T', ' ')[:19]} UTC")
    if progress.get("last_error"):
        st.caption(f"Last error: {progress['last_error']}")


@st.cache_data(ttl=120)
def _cached_dir_file_count(path_str: str) -> int:
    return sum(1 for f in Path(path_str).rglob("*") if f.is_file())


@st.cache_data(ttl=120)
def _cached_tess_download_status() -> list[dict]:
    rows = []
    if not TESS_DIR.exists():
        return rows
    for d in sorted(TESS_DIR.iterdir()):
        if d.is_dir():
            m_dl = re.match(r'sector(\d+)', d.name)
            if m_dl:
                fits = list(d.glob("*_lc.fits"))
                if fits:
                    size_mb = sum(f.stat().st_size for f in fits) / 1_048_576
                    rows.append({
                        "Sector": int(m_dl.group(1)),
                        "Files": len(fits),
                        "Size (MB)": round(size_mb, 1),
                    })
    return rows


@st.cache_data(ttl=120)
def _discover_candidate_files() -> list:
    return (
        sorted(ZENODO_DIR.glob("*.csv")) +
        sorted(ZENODO_DIR.glob("*.parquet")) +
        sorted(CANDS_DIR.glob("*.csv")) +
        sorted(RESULTS_DIR.rglob("bls_exominer_results.csv"))
    )


@st.cache_data(ttl=60)
def _cached_scan_dirs() -> list:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [d for d in RESULTS_DIR.iterdir()
         if d.is_dir() and (d / "bls_results.csv").exists()],
        reverse=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_verify(tic_id: int, period: float, t0) -> dict:
    from verify_candidate import verify
    return verify(tic_id=tic_id, period=period, t0=t0)


def sector_hunt_complete(sector: int) -> bool:
    res_dir = RESULTS_DIR / f"sector{sector:02d}"
    csv_path = res_dir / "bls_results.csv"
    log_path = res_dir / "hunt.log"
    if not csv_path.exists() or not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return False
    return "✅ Done!" in text


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet Pipeline",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Pipeline Status",
    "Benchmarks",
    "Candidate Browser",
    "Cross-Matched Shortlist",
    "Lightcurve Inspector",
    "Scan Results",
    "Run Scan",
    "Deep Scan",
]

st.sidebar.title("🔭 Exoplanet Pipeline")
_jump_page = st.session_state.pop("_jump_page", None)
if _jump_page and _jump_page in PAGES:
    st.session_state["nav_radio"] = _jump_page
page = st.sidebar.radio("Navigate", PAGES, key="nav_radio")

# Sidebar: scan status indicator (always visible regardless of page)
running, _scan_state = scan_is_running()
if running:
    label = _scan_state.get("label", "?")
    _job_step = ""
    _lf = current_state_log(_scan_state)
    if _lf and _lf.name == "hunt.log":
        try:
            _lines = [l for l in _lf.read_text(errors="replace").splitlines() if l.strip()]
            if _lines:
                _job_step = "\n" + _lines[-1].strip()[:60]
        except Exception:
            pass
    st.sidebar.markdown(f"🟢 **{job_mode_label(_scan_state)}** {label}{_job_step}")
else:
    last = _scan_state.get("label")
    fin  = _scan_state.get("finished_at", "")
    status = _scan_state.get("status", "")
    if last and fin:
        fin_short = fin[11:16] if len(fin) > 16 else fin
        tail = f" *(last: {last} @ {fin_short})*"
        if status == "failed":
            st.sidebar.markdown(f"🟠 Idle{tail}")
        else:
            st.sidebar.markdown(f"🔴 Idle{tail}")
    else:
        st.sidebar.markdown("🔴 Idle")

st.sidebar.markdown("---")
st.sidebar.markdown("**Paths**")
st.sidebar.code(f"data:    {DATA_DIR}\nresults: {RESULTS_DIR}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def edu_expander(title: str, content_md: str):
    with st.expander(f"📖 {title}", expanded=False):
        st.markdown(content_md)


def exofop_footer():
    st.markdown("---")
    st.caption(
        "Found something interesting? If a candidate has high BLS power, "
        "reasonable depth (<20,000 ppm), and a clean U-shaped transit, you can "
        "[submit it as a Community TOI](https://exofop.ipac.caltech.edu/tess/) at ExoFOP."
    )


# ==============================================================================
# Page 1 — Pipeline Status
# ==============================================================================
if page == "Pipeline Status":
    st.title("Pipeline Status")

    edu_expander("What am I looking at?", """
**TESS** (Transiting Exoplanet Survey Satellite) stares at patches of sky for ~27 days at a time,
measuring the brightness of hundreds of thousands of stars. These patches are called **sectors**
(Sector 1 = July 2018, increasing to the present day).

When a planet passes in front of its star relative to Earth, it blocks a tiny fraction of the
starlight — a **transit**. This pipeline:
1. Downloads TESS lightcurves (brightness vs. time measurements)
2. Runs a **BLS** (Box Least Squares) period search to find repeating dips
3. Optionally scores candidates with **ExoMiner++** — NASA's deep learning classifier
4. Plots the best ones for human review

Sectors 1–67 have published ExoMiner++ scores. **Sectors 68+** are fresh — your scan may find
signals that no automated catalog has flagged yet.

**By the numbers**: TESS has observed over 400,000 stars and enabled confirmation of 700+ planets
as of 2024, with thousands more candidates awaiting follow-up. Each sector covers ~24° × 96° of
sky (about 2,300 square degrees), observed continuously for ~27 days.

**This hardware**: Your Threadripper PRO 3955WX runs 16 BLS workers in parallel, processing
roughly 5–15 stars/sec depending on lightcurve length. A typical sector of ~20,000 stars
completes in 20–60 minutes.
""")

    def svc(name):
        try:
            r = subprocess.run(["systemctl", "is-active", name],
                               capture_output=True, text=True, timeout=3)
            return r.stdout.strip()
        except Exception:
            return "unknown"

    jl = svc("jupyterlab")
    db = svc("exoplanet-dashboard")
    ip = subprocess.run(["hostname", "-I"], capture_output=True,
                        text=True).stdout.split()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("JupyterLab  :8888", jl.upper())
    c2.metric("Dashboard   :8501", db.upper())
    c3.metric("Container IP", ip)

    st.markdown("---")
    st.subheader("Data Storage")
    dirs = {
        "Zenodo catalog":   ZENODO_DIR,
        "TESS lightcurves": TESS_DIR,
        "Scan results":     RESULTS_DIR,
    }
    cols = st.columns(len(dirs))
    for col, (label, path) in zip(cols, dirs.items()):
        with col:
            if path.is_dir():
                n = _cached_dir_file_count(str(path))
                st.metric(label, f"{n} files")
            else:
                st.metric(label, "not found")

    st.markdown("---")
    st.subheader("Download Status")
    if TESS_DIR.exists():
        dl_rows = _cached_tess_download_status()
        if dl_rows:
            _dl_df = pd.DataFrame(dl_rows)
            st.dataframe(_dl_df, use_container_width=True, hide_index=True)
            # Check for running prefetch
            try:
                _pgrep = subprocess.run(["pgrep", "-af", "prefetch_sector"],
                                        capture_output=True, text=True, timeout=3)
                if _pgrep.stdout.strip():
                    st.info(f"⬇️ Download in progress: `{_pgrep.stdout.strip()[:80]}`")
            except Exception:
                pass
        else:
            st.info("No TESS lightcurves downloaded yet.")
    else:
        st.info("TESS data directory not found.")

    st.markdown("---")
    st.subheader("Recent Scan Activity")
    if RESULTS_DIR.exists():
        scans = _cached_scan_dirs()
        if scans:
            for s in scans[:5]:
                csv_f = s / "bls_results.csv"
                html  = s / "scan_report.html"
                if csv_f.exists():
                    try:
                        df_scan = pd.read_csv(csv_f)
                        n_cands = sum(df_scan["bls_power"] >= 7)
                    except Exception:
                        n_cands = "?"
                    col_txt, col_btn = st.columns([4, 1])
                    with col_txt:
                        st.write(f"**{s.name}** — {len(df_scan) if isinstance(df_scan, pd.DataFrame) else '?'} results, {n_cands} candidates")
                    with col_btn:
                        if html.exists():
                            st.download_button(
                                label="Download report",
                                data=html.read_bytes(),
                                file_name=f"{s.name}_report.html",
                                mime="text/html",
                                key=f"dl_{s.name}",
                            )
        else:
            st.info("No scans yet. Use the **Run Scan** page to start.")
    else:
        st.info("No results directory found.")

    st.markdown("---")
    st.subheader("Setup Log")
    log_path = Path("/var/log/exoplanet-setup.log")
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        st.code("\n".join(lines[-25:]), language="text")

    st.subheader("ExoMiner++")
    em = Path(__file__).resolve().parent.parent / "ExoMiner"
    if em.is_dir():
        st.success(f"Cloned at {em}")
    else:
        st.warning("Not cloned yet.")

    exofop_footer()


# ==============================================================================
# Page 2 — Benchmarks
# ==============================================================================
elif page == "Benchmarks":
    st.title("Benchmarks")
    st.markdown(
        "Known targets you can use to sanity-check the pipeline. "
        "This page reads the latest validation harness output and lets you jump straight into inspection."
    )

    _bench = load_validation_report()
    if not _bench:
        st.info(
            "No benchmark report found yet. Run `python /opt/exoplanet/scripts/validate_pipeline.py` "
            "from the shell, then refresh this page."
        )
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall", _bench.get("overall_status", "?"))
    c2.metric("Warnings", _bench.get("warnings_total", 0))
    c3.metric("Failures", _bench.get("failures_total", 0))
    st.caption(
        f"Last validation run: {_bench.get('finished_at', _bench.get('started_at', 'unknown'))}"
    )

    def _benchmark_block(title: str, rows: list[dict]):
        st.markdown(f"### {title}")
        if not rows:
            st.info("No rows in this section.")
            return
        for row in rows:
            _tic = int(row["tic_id"])
            _period = float(row["period"])
            _status = row.get("status", "?")
            _verdict = row.get("verdict", "")
            _cons = safe_float(row.get("consistency_score"))
            _reason = row.get("reason", "")
            col_a, col_b, col_c = st.columns([4, 2, 1])
            with col_a:
                st.markdown(f"**TIC {_tic}** — {row.get('name', '')}")
                st.caption(f"Expected: {row.get('expected_kind', '')} | Period: {_period:.5f} d")
                if _verdict:
                    st.write(_verdict)
                if _reason:
                    st.caption(_reason)
            with col_b:
                st.metric("Result", _status)
                if _cons is not None:
                    st.metric("Consistency", f"{_cons:.0%}")
            with col_c:
                if st.button("Inspect", key=f"bench_insp_{_tic}"):
                    st.session_state["jump_tic"] = _tic
                    st.session_state["jump_period"] = _period
                    st.session_state["insp_mode"] = "Validate Known Target"
                    st.session_state["_jump_page"] = "Lightcurve Inspector"
                    st.rerun()

    _benchmark_block("Known Planets", _bench.get("verify_planets", []))
    st.markdown("---")
    _benchmark_block("Known False Positives", _bench.get("verify_fps", []))

    exofop_footer()


# ==============================================================================
# Page 3 — Candidate Browser
# ==============================================================================
elif page == "Candidate Browser":
    st.title("Candidate Browser")
    st.markdown("Load an ExoMiner++ scored catalog to explore candidates.")

    edu_expander("Understanding the catalog columns", """
| Column | Meaning |
|--------|---------|
| **ExoMiner Score** | 0–1 probability of being a real planet. **≥ 0.9** is the standard "planet candidate" threshold. |
| **Orbital Period [day]** | How long one orbit takes. Short periods (< 10 days) are overrepresented because TESS only watches for 27 days. |
| **Transit Duration [hour]** | How long each dip lasts. Typical range: 1–5 hours (0.04–0.2 days). Very long durations may indicate an eclipsing binary. |
| **Transit Epoch [BTJD]** | When the first transit occurred. BTJD 0 = Dec 8, 2014; BTJD 1687 ≈ July 2019. |
| **Gaia RUWE** | Quality flag: **> 1.4** may indicate an unresolved companion (possible false positive). |
| **Planet Radius [Earth Radii]** | Estimated planet size. Earth = 1, Neptune ≈ 3.9, Jupiter ≈ 11.2. Values > 15 are almost certainly EBs. |
| **MES** | Multiple Event Statistic — TESS's internal SNR proxy. Detection threshold: **7.1**. |
| **Label** | KP = Kepler Planet, CP = Confirmed Planet, FP = False Positive, EB = Eclipsing Binary. |
| **Sector Run** | Which TESS sector(s) contributed. Sector 1 = July 2018; higher = more recent. |

**Common false positives to watch for:**
- **Eclipsing Binary (EB)**: Two stars orbiting each other produce deep (>20,000 ppm), often V-shaped dips. Secondary eclipses at half the period are a tell-tale sign.
- **Background blend**: A faint EB behind the target star dilutes the signal — the transit looks shallow and planet-like, but it's not. RUWE > 1.4 or nearby Gaia sources increase suspicion.
- **Stellar variability**: Starspots and pulsations create quasi-periodic signals that BLS can mistake for transits. These typically show up as very short periods (<1 day) or irregular shapes.

**RUWE explained**: Gaia's Renormalised Unit Weight Error measures how well a single-star model fits the astrometry. A well-behaved single star has RUWE ≈ 1.0. Values > 1.4 suggest either a binary companion or a crowded field — both increase false-positive probability.
""")

    if "catalog_df" not in st.session_state:
        st.session_state.catalog_df = None

    candidates = _discover_candidate_files()

    uploaded = st.file_uploader("Upload catalog CSV", type=["csv", "parquet"])

    @st.cache_data
    def load_catalog(path: str) -> pd.DataFrame:
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    if uploaded is not None:
        st.session_state.catalog_df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(st.session_state.catalog_df):,} rows")
    elif candidates:
        chosen = st.selectbox("Auto-detected file:", [str(p) for p in candidates])
        if st.button("Load"):
            st.session_state.catalog_df = load_catalog(chosen)
            st.success(f"Loaded {len(st.session_state.catalog_df):,} rows")
    else:
        st.info("No catalog found. Place CSV files in /opt/exoplanet/data/zenodo/")

    df = st.session_state.catalog_df
    if df is not None:
        st.markdown("---")

        score_col = next(
            (c for c in df.columns if "score" in c.lower() or "bls_power" in c.lower()), None
        )
        tic_col = next(
            (c for c in df.columns if c.lower().startswith("tic")), df.columns[0]
        )
        period_col = next(
            (c for c in df.columns if "period" in c.lower()), None
        )

        st.caption(
            f"Detected columns — TIC: `{tic_col}` | Score: `{score_col}` | Period: `{period_col}`"
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Score Distribution")
            if score_col:
                fig = px.histogram(df, x=score_col, nbins=100,
                                   color_discrete_sequence=["#58a6ff"],
                                   title=f"{score_col}  (n={len(df):,})")
                fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                  font_color="#c9d1d9")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if score_col:
                st.subheader("Filter")
                lo = float(df[score_col].dropna().min())
                hi = float(df[score_col].dropna().max())
                threshold = st.slider("Min score / power", lo, hi,
                                      min(0.9, hi * 0.9), (hi - lo) / 100)
                top = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)
                st.metric("Above threshold", len(top))
                if len(top) > 0:
                    st.dataframe(top[[tic_col, score_col]].head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Full Catalog")
        st.dataframe(df.head(500), use_container_width=True)

        if tic_col and period_col:
            with st.expander("Public catalogue spot-check (first 50 rows)", expanded=False):
                st.caption(
                    "This cross-check uses official public catalogues to separate confirmed objects, known TOIs, suspicious public targets, and objects that are not obviously known."
                )
                preview_n = min(50, len(df))
                preview_rows = []
                for _, row in df.head(preview_n).iterrows():
                    preview_rows.append({"tic_id": row.get(tic_col), "period": row.get(period_col)})
                preview_hits = pd.DataFrame(annotate_candidate_rows(preview_rows, detailed=False))
                if not preview_hits.empty:
                    cols = [c for c in ["tic_id", "local_period", "status_label", "matched_name", "plain_english_summary"] if c in preview_hits.columns]
                    preview_hits = preview_hits.rename(columns={"local_period": "local_period_d", "status_label": "public_status"})
                    st.dataframe(preview_hits[[c for c in ["tic_id", "local_period_d", "public_status", "matched_name", "plain_english_summary"] if c in preview_hits.columns]], use_container_width=True)
                else:
                    st.info("No TIC/period rows were available to cross-match.")

        if period_col and score_col:
            st.subheader("Score vs Period")
            _tmp = df[[period_col, score_col]].dropna()
            plot_df = _tmp.sample(min(5000, len(_tmp)), random_state=42)
            fig2 = px.scatter(plot_df, x=period_col, y=score_col,
                              opacity=0.4, color=score_col,
                              color_continuous_scale="RdYlGn",
                              title="Score / Power vs Orbital Period")
            fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                               font_color="#c9d1d9")
            st.plotly_chart(fig2, use_container_width=True)

    exofop_footer()


# ==============================================================================
# Page 4 — Cross-Matched Shortlist
# ==============================================================================
elif page == "Cross-Matched Shortlist":
    st.title("Cross-Matched Shortlist")
    st.markdown(
        "This page shows every saved shortlist file for sectors 85–99, including full cross-matched runs, fast triage-first runs, and manual-review cuts."
    )

    active_running, active_state = scan_is_running()
    latest_shortlist = shortlist_files()[0] if shortlist_files() else None

    if active_running and active_state.get("mode") == "build_shortlist":
        st.warning(
            f"Shortlist build is running in the background. Started {active_state.get('started_at', '')[:16]}."
        )
        _progress = read_job_progress(active_state)
        if _progress:
            render_shortlist_progress(_progress)
        else:
            st.info("This shortlist run does not have live progress data yet. That usually means it started before progress reporting was added.")
        if active_state.get("log_file"):
            st.caption(f"Log: {active_state['log_file']}")
        slog = current_state_log(active_state)
        if slog and slog.exists():
            with st.expander("Live job log", expanded=True):
                st.code(tail_log(slog), language="text")

    files = shortlist_files()
    if files:
        labels = []
        for p in files:
            meta = shortlist_metadata(p)
            kind = shortlist_kind(p, meta)
            labels.append(f"{p.name} — {kind} — {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
        chosen_label = st.selectbox("Available shortlist files", labels, index=0)
        shortlist_path = files[labels.index(chosen_label)]
    else:
        shortlist_path = latest_shortlist

    if shortlist_path and shortlist_path.exists():
        built_at = datetime.fromtimestamp(shortlist_path.stat().st_mtime)
        meta = shortlist_metadata(shortlist_path)
        kind = shortlist_kind(shortlist_path, meta)
        st.caption(f"File: {shortlist_path}")
        st.caption(f"Built: {built_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if kind == "manual-review":
            st.caption("Shortlist type: Manual review")
        else:
            st.caption(f"Build mode: {'Fast triage-first' if kind == 'fast' else 'Full enrichment'}")
        if meta:
            st.caption(
                f"Metadata: cheap pass rows {meta.get('cheap_pass_rows', 0)} | deep lookup rows {meta.get('deep_lookup_rows', 0)} | total rows {meta.get('total_rows', 0)}"
            )
        if shortlist_is_stale(shortlist_path):
            st.warning("This shortlist may be stale. Newer scan results exist in sectors 85–99. Rebuild it from the Run Scan page when convenient.")
        else:
            st.success("This shortlist is newer than the current sector 85–99 result CSVs.")

        sdf = load_shortlist_csv(str(shortlist_path), shortlist_path.stat().st_mtime)
        if "status_label" in sdf.columns and "public_status" not in sdf.columns:
            sdf["public_status"] = sdf["status_label"]
        if "review_score" in sdf.columns:
            sorted_df = sdf.copy()
            sorted_df["_review_score"] = pd.to_numeric(sorted_df["review_score"], errors="coerce").fillna(-1.0)
            bucket_rank = {"best_current_unknowns": 0, "review_with_caution": 1, "long_shot": 2}
            sorted_df["_review_bucket"] = sorted_df.get("review_bucket", pd.Series(dtype=str)).map(bucket_rank).fillna(9)
            sorted_df = sorted_df.sort_values(by=["_review_bucket", "_review_score"], ascending=[True, False])
            sorted_df = sorted_df.drop(columns=["_review_score", "_review_bucket"], errors="ignore")
        else:
            sorted_df = shortlist_sort_frame(sdf)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        counts = sorted_df["public_status"].value_counts() if "public_status" in sorted_df.columns else pd.Series(dtype=int)
        c1.metric("Rows", len(sorted_df))
        if "review_bucket" in sorted_df.columns:
            bucket_counts = sorted_df["review_bucket"].value_counts()
            c2.metric("Best", int(bucket_counts.get("best_current_unknowns", 0)))
            c3.metric("Caution", int(bucket_counts.get("review_with_caution", 0)))
            c4.metric("Long shot", int(bucket_counts.get("long_shot", 0)))
            c5.metric("Unknown", int(counts.get("Unknown", 0)))
            c6.metric("Lookup failed", int(counts.get("Lookup failed", 0)))
        else:
            c2.metric("Confirmed", int(counts.get("Confirmed planet", 0)))
            c3.metric("Known TOI", int(counts.get("Known TOI", 0)))
            c4.metric("Caution", int(counts.get("Caution", 0)))
            c5.metric("Unknown", int(counts.get("Unknown", 0)))
            c6.metric("Lookup failed", int(counts.get("Lookup failed", 0)))

        col_filter, col_dl = st.columns([3, 1])
        with col_filter:
            if "review_bucket" in sorted_df.columns:
                bucket_opts = [s for s in ["best_current_unknowns", "review_with_caution", "long_shot"] if s in set(sorted_df.get("review_bucket", pd.Series(dtype=str)).dropna())]
                selected_status = st.multiselect(
                    "Review bucket filter",
                    options=bucket_opts,
                    default=bucket_opts,
                    help="Manual-review shortlists are already cut down. The review bucket tells you how hard to look.",
                )
            else:
                status_opts = [s for s in ["Unknown", "Caution", "Known TOI", "Confirmed planet", "Lookup failed"] if s in set(sorted_df.get("public_status", pd.Series(dtype=str)).dropna())]
                selected_status = st.multiselect(
                    "Public status filter",
                    options=status_opts,
                    default=status_opts,
                    help="The default sort already pushes Unknown and stronger locally-verified rows toward the top.",
                )
        with col_dl:
            st.download_button(
                "Download CSV",
                shortlist_path.read_bytes(),
                file_name=shortlist_path.name,
                mime="text/csv",
                disabled=not shortlist_path.exists(),
            )

        filtered_df = sorted_df
        if "review_bucket" in filtered_df.columns and selected_status:
            filtered_df = filtered_df[filtered_df["review_bucket"].isin(selected_status)]
        elif selected_status and "public_status" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["public_status"].isin(selected_status)]

        table_columns = (
            [
                "review_score",
                "review_bucket",
                "review_reasons",
                "review_cautions",
                "tic_id",
                "sector",
                "period",
                "depth_ppm",
                "verified",
                "consistency_score",
                "public_status",
                "matched_name",
                "plain_english_summary",
            ]
            if "review_score" in filtered_df.columns
            else
            [
                "tic_id",
                "sector",
                "period",
                "public_status",
                "matched_name",
                "verified",
                "consistency_score",
                "eb_warning",
                "depth_ppm",
                "plain_english_summary",
            ]
        )
        st.dataframe(
            filtered_df[
                [c for c in table_columns if c in filtered_df.columns]
            ],
            use_container_width=True,
            height=520,
        )
    else:
        st.info("No cross-matched shortlist CSV exists yet. Build it from the Run Scan page.")

    exofop_footer()


# ==============================================================================
# Page 5 — Lightcurve Inspector
# ==============================================================================
elif page == "Lightcurve Inspector":
    st.title("Lightcurve Inspector")
    st.markdown(
        "Use this as a guided triage console: first separate the search result, then the verification result, then the final interpretation."
    )

    inspector_mode = st.radio(
        "Workflow",
        ["Validate Known Target", "Search for Candidate"],
        horizontal=True,
        index=0 if st.session_state.get("insp_mode") == "Validate Known Target" else 1,
    )
    st.session_state["insp_mode"] = inspector_mode

    if inspector_mode == "Validate Known Target":
        st.info(
            "Use this mode when you already know the TIC and period and want to see whether the pipeline recovers that known signal."
        )
    else:
        st.info(
            "Use this mode when you do not know the period yet. BLS proposes a candidate period, verification checks it, and the final interpretation tells you how seriously to take it."
        )

    default_tic = str(st.session_state.get("jump_tic", st.session_state.get("insp_tic", "")))
    default_period = str(
        st.session_state.get(
            "jump_period",
            st.session_state.get("insp_known_period" if inspector_mode == "Validate Known Target" else "insp_base_period", ""),
        )
    )
    default_t0 = str(
        st.session_state.get(
            "jump_t0",
            st.session_state.get("insp_known_t0" if inspector_mode == "Validate Known Target" else "insp_base_t0", ""),
        )
    )
    loaded_period = safe_float(st.session_state.get("jump_period"))
    loaded_t0 = safe_float(st.session_state.get("jump_t0"))
    loaded_source = st.session_state.get("jump_period_source", "single_bls_loaded")

    with st.form("inspect_form"):
        if inspector_mode == "Validate Known Target":
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                tic_input = st.text_input("TIC ID", value=default_tic, placeholder="e.g. 402026209")
            with c2:
                author = st.selectbox("Pipeline", ["SPOC", "QLP", "TESS-SPOC"], key="insp_author_validate")
            with c3:
                period_in = st.text_input("Known period (d)", value=default_period, placeholder="required")
            with c4:
                t0_in = st.text_input("Epoch (BTJD)", value=default_t0, placeholder="optional")
            submitted = st.form_submit_button("Validate This Target")
            use_loaded_candidate = False
            search_scope = "Most recent sector"
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                tic_input = st.text_input("TIC ID", value=default_tic, placeholder="e.g. 402026209")
            with c2:
                author = st.selectbox("Pipeline", ["SPOC", "QLP", "TESS-SPOC"], key="insp_author_search")
            use_loaded_candidate = False
            if loaded_period is not None and default_tic:
                use_loaded_candidate = st.checkbox(
                    "Use loaded candidate period from earlier results",
                    value=True,
                    help="Keeps the earlier search result visible instead of rerunning BLS immediately.",
                )
                st.caption(
                    f"Loaded candidate period: {loaded_period:.5f} d from {period_source_label(loaded_source)}"
                )
            if not use_loaded_candidate:
                search_scope = st.radio(
                    "Search scope",
                    ["Most recent sector", "All sectors stitched (slower)"],
                    horizontal=True,
                )
            else:
                search_scope = "Loaded candidate"
            period_in = ""
            t0_in = ""
            submitted = st.form_submit_button("Search For Candidate")

    if submitted:
        for key in ("jump_tic", "jump_period", "jump_t0", "jump_period_source"):
            st.session_state.pop(key, None)
        st.session_state.pop("insp_verify_result", None)
        st.session_state.pop("insp_verify_period", None)

    have_data = (
        "insp_lc_raw" in st.session_state
        and "insp_lc_flat" in st.session_state
        and "insp_tic" in st.session_state
        and st.session_state.get("insp_tic") == tic_input.strip()
        and st.session_state.get("insp_mode_used") == inspector_mode
    )

    if submitted and tic_input.strip():
        import lightkurve as lk

        tic_clean = tic_input.strip().lstrip("TIC").strip()
        tic_str = f"TIC {tic_clean}"
        st.markdown(f"**Fetching** `{tic_str}` from MAST...")

        with st.spinner("Downloading lightcurves..."):
            try:
                results, lcs = fetch_inspector_lightcurves(lk, tic_str, author)
                st.success(f"Found {len(results)} sector(s). Downloading...")
                lc_raw = lcs.stitch() if len(lcs) > 1 else lcs[0]
                lc_bls_src = lcs[-1]
                lc_flat_all = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)
                lc_flat_bls = lc_bls_src.normalize().flatten(window_length=401).remove_outliers(sigma=4)
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    st.error(str(e))
                    st.stop()
                err_text = str(e)
                corrupt_path = extract_corrupt_cache_path(err_text)
                if corrupt_path:
                    removed = False
                    try:
                        corrupt_path.unlink(missing_ok=True)
                        removed = True
                    except Exception:
                        pass
                    if removed:
                        st.warning("A cached FITS file looked corrupt, so I removed it and retried once.")
                        try:
                            results, lcs = fetch_inspector_lightcurves(lk, tic_str, author)
                            st.success(f"Found {len(results)} sector(s). Downloading...")
                            lc_raw = lcs.stitch() if len(lcs) > 1 else lcs[0]
                            lc_bls_src = lcs[-1]
                            lc_flat_all = lc_raw.normalize().flatten(window_length=401).remove_outliers(sigma=4)
                            lc_flat_bls = lc_bls_src.normalize().flatten(window_length=401).remove_outliers(sigma=4)
                        except Exception as retry_err:
                            st.error(f"Download failed after retry: {retry_err}")
                            st.stop()
                    else:
                        st.error(
                            "Download failed because a cached FITS file looks corrupt. "
                            f"Please remove it and retry:\n{corrupt_path}"
                        )
                        st.stop()
                st.error(f"Download failed: {e}")
                st.stop()

        eval_period = None
        eval_t0 = None
        base_source = None
        sde = None
        blsm = None
        single_bls_period = None
        single_bls_t0 = None
        stitched_bls_period = None
        stitched_bls_t0 = None
        known_period = None
        known_t0 = None

        if inspector_mode == "Validate Known Target":
            known_period = safe_float(period_in)
            known_t0 = safe_float(t0_in)
            if known_period is None or known_period <= 0:
                st.error("Validate Known Target mode needs a real known period.")
                st.stop()
            eval_period = known_period
            eval_t0 = known_t0 if known_t0 is not None else auto_detect_epoch(lc_flat_all, known_period)
            if eval_t0 is None:
                st.error("Could not estimate an epoch for the supplied period.")
                st.stop()
            base_source = "known"
        else:
            if use_loaded_candidate and loaded_period is not None:
                eval_period = loaded_period
                eval_t0 = loaded_t0 if loaded_t0 is not None else auto_detect_epoch(lc_flat_all, loaded_period)
                if eval_t0 is None:
                    st.error("Could not estimate an epoch for the loaded candidate period.")
                    st.stop()
                base_source = loaded_source or "single_bls_loaded"
                if base_source.startswith("stitched"):
                    stitched_bls_period = eval_period
                    stitched_bls_t0 = eval_t0
                else:
                    single_bls_period = eval_period
                    single_bls_t0 = eval_t0
            else:
                search_stitched = search_scope.startswith("All sectors stitched")
                with st.spinner("Searching all sectors (stitched)..." if search_stitched else "Running BLS period search on the most recent sector..."):
                    try:
                        search_res = run_bls_search(lc_flat_all if search_stitched else lc_flat_bls, search_stitched)
                    except Exception as e:
                        if search_stitched:
                            st.error(
                                "BLS failed on the stitched multi-sector search. "
                                "That usually means even the emergency coarse grid was still too large for this target. "
                                "Try inspecting a single sector instead, or use Deep Scan for long-period work. "
                                f"Details: {e}"
                            )
                        else:
                            st.error(f"BLS failed: {e}")
                        st.stop()
                eval_period = search_res["period"]
                eval_t0 = search_res["t0"]
                sde = search_res["sde"]
                blsm = search_res["blsm"]
                if search_stitched:
                    base_source = "stitched_bls"
                    stitched_bls_period = eval_period
                    stitched_bls_t0 = eval_t0
                else:
                    base_source = "single_bls"
                    single_bls_period = eval_period
                    single_bls_t0 = eval_t0

        depth_summary = fold_signal_summary(lc_flat_all, eval_period, eval_t0)
        depth_ppm = depth_summary.get("depth_ppm", 0.0) if depth_summary else 0.0

        st.session_state.update({
            "insp_mode_used": inspector_mode,
            "insp_tic": tic_clean,
            "insp_lc_raw": lc_raw,
            "insp_lc_flat": lc_flat_all,
            "insp_lc_flat_bls": lc_flat_bls,
            "insp_sector": getattr(lcs[-1], "sector", "?"),
            "insp_base_period": eval_period,
            "insp_base_t0": eval_t0,
            "insp_base_source": base_source,
            "insp_depth": depth_ppm,
            "insp_sde": sde,
            "insp_blsm": blsm,
            "insp_known_period": known_period,
            "insp_known_t0": known_t0,
            "insp_single_bls_period": single_bls_period,
            "insp_single_bls_t0": single_bls_t0,
            "insp_stitched_bls_period": stitched_bls_period,
            "insp_stitched_bls_t0": stitched_bls_t0,
        })
        have_data = True

    if have_data:
        from plot_candidate import make_4panel

        tic_id = st.session_state["insp_tic"]
        lc_raw = st.session_state["insp_lc_raw"]
        lc_flat = st.session_state["insp_lc_flat"]
        lc_flat_bls = st.session_state.get("insp_lc_flat_bls", lc_flat)
        base_period = float(st.session_state["insp_base_period"])
        base_t0 = float(st.session_state["insp_base_t0"])
        base_source = st.session_state.get("insp_base_source")
        known_period = safe_float(st.session_state.get("insp_known_period"))
        sector = st.session_state["insp_sector"]
        sde = safe_float(st.session_state.get("insp_sde"))
        single_bls_period = safe_float(st.session_state.get("insp_single_bls_period"))
        stitched_bls_period = safe_float(st.session_state.get("insp_stitched_bls_period"))

        inspector_context = f"{inspector_mode}:{tic_id}:{base_period:.8f}:{base_t0:.8f}:{base_source}:{sector}"
        if st.session_state.get("insp_context_key") != inspector_context:
            st.session_state["insp_context_key"] = inspector_context
            st.session_state["insp_selected_period"] = float(base_period)
            st.session_state["insp_selected_t0"] = float(base_t0)
            st.session_state["insp_period_source"] = str(base_source or "manual")
            st.session_state["insp_helper_base_source"] = "current_selected"
            st.session_state["insp_last_helper_action"] = "Use P"
            st.session_state["insp_period_history"] = append_period_history(
                [],
                period_source_label(base_source),
                base_period,
                str(base_source or ""),
            )
            st.session_state["_insp_prev_selected_period"] = float(base_period)
            st.session_state["_insp_prev_selected_t0"] = float(base_t0)
            st.session_state["insp_odd_even_result"] = None

        pending_selected_period = safe_float(st.session_state.pop("_insp_pending_selected_period", None))
        pending_selected_t0 = safe_float(st.session_state.pop("_insp_pending_selected_t0", None))
        pending_history_label = st.session_state.pop("_insp_pending_history_label", None)
        pending_history_source = st.session_state.pop("_insp_pending_history_source", None)
        pending_helper_action = st.session_state.pop("_insp_pending_helper_action", None)
        if pending_selected_period is not None and pending_selected_period > 0:
            st.session_state["insp_selected_period"] = float(pending_selected_period)
            if pending_selected_t0 is not None:
                st.session_state["insp_selected_t0"] = float(pending_selected_t0)
            if pending_helper_action:
                st.session_state["insp_last_helper_action"] = str(pending_helper_action)
            if pending_history_label and pending_history_source:
                st.session_state["insp_period_history"] = append_period_history(
                    st.session_state.get("insp_period_history"),
                    str(pending_history_label),
                    pending_selected_period,
                    str(pending_history_source),
                )
            st.session_state["insp_odd_even_result"] = None

        selected_period = safe_float(st.session_state.get("insp_selected_period"), base_period) or float(base_period)
        selected_t0 = safe_float(st.session_state.get("insp_selected_t0"), base_t0) or float(base_t0)
        pending_source = st.session_state.pop("_insp_pending_period_source", None)
        prev_period = safe_float(st.session_state.get("_insp_prev_selected_period"), selected_period) or selected_period
        prev_t0 = safe_float(st.session_state.get("_insp_prev_selected_t0"), selected_t0) or selected_t0

        slider_period_candidates = [
            float(base_period),
            float(selected_period),
            float(base_period * 2.0),
            float(base_period * 3.0),
            max(0.1, float(base_period / 2.0)),
            max(0.1, float(base_period / 3.0)),
        ]
        max_period_slider = max(30.0, max(slider_period_candidates) * 1.1)
        adj_period = st.slider(
            "Selected period (days)",
            min_value=0.1,
            max_value=float(max_period_slider),
            step=0.0001,
            format="%.4f",
            key="insp_selected_period",
        )
        adj_t0 = st.slider(
            "Selected epoch (BTJD)",
            float(base_t0) - float(max(base_period, adj_period)),
            float(base_t0) + float(max(base_period, adj_period)),
            float(selected_t0),
            0.001,
            "%.3f",
            key="insp_selected_t0",
        )

        manual_changed = (
            not periods_agree(adj_period, prev_period, rel_tol=0.0005)
            or abs(adj_t0 - prev_t0) > 0.002
        )
        if pending_source:
            st.session_state["insp_period_source"] = pending_source
        elif manual_changed:
            st.session_state["insp_period_source"] = "manual"
            st.session_state["insp_odd_even_result"] = None
        current_source = st.session_state.get("insp_period_source", base_source)
        st.session_state["_insp_prev_selected_period"] = float(adj_period)
        st.session_state["_insp_prev_selected_t0"] = float(adj_t0)

        _fold_cache_key = "insp_last_fold_result"
        _fold_period_key = "insp_last_fold_period"
        _fold_t0_key = "insp_last_fold_t0"
        _last_fp = st.session_state.get(_fold_period_key)
        _last_ft = st.session_state.get(_fold_t0_key)
        _needs_fold = (
            _last_fp is None
            or not periods_agree(adj_period, _last_fp, rel_tol=0.0005)
            or abs(adj_t0 - _last_ft) > 0.002
        )
        if _needs_fold:
            folded_summary = fold_signal_summary(lc_flat, adj_period, adj_t0) or {}
            st.session_state[_fold_cache_key] = folded_summary
            st.session_state[_fold_period_key] = adj_period
            st.session_state[_fold_t0_key] = adj_t0
        else:
            folded_summary = st.session_state.get(_fold_cache_key, {})
        depth_adj = folded_summary.get("depth_ppm", st.session_state.get("insp_depth", 0.0))
        public_match = get_crossmatch(tic_id, adj_period, detailed=True)
        matched_period = safe_float((public_match or {}).get("matched_period"))

        st.markdown("### Period Helper")
        helper_base_options = {
            "current_selected": ("current selected period", float(adj_period)),
        }
        if stitched_bls_period is not None and stitched_bls_period > 0:
            helper_base_options["stitched_bls"] = ("stitched BLS", float(stitched_bls_period))
        if single_bls_period is not None and single_bls_period > 0:
            helper_base_options["single_bls"] = ("single-sector BLS", float(single_bls_period))
        if matched_period is not None and matched_period > 0:
            helper_base_options["catalogue_period"] = ("public catalogue period", float(matched_period))

        helper_base_source = st.selectbox(
            "Helper base period",
            list(helper_base_options.keys()),
            key="insp_helper_base_source",
            format_func=lambda key: (
                f"{helper_base_options[key][0]} ({helper_base_options[key][1]:.5f} d)"
            ),
            help="Harmonic helper buttons transform this base period, not necessarily the original BLS seed.",
        )
        helper_base_label, helper_base_period = helper_base_options[helper_base_source]

        helper_specs = [
            ("Use P/2", max(0.1, helper_base_period / 2.0), harmonic_source_key("P/2", helper_base_source), f"P/2 from {helper_base_label}"),
            ("Use 2P", helper_base_period * 2.0, harmonic_source_key("2P", helper_base_source), f"2P from {helper_base_label}"),
        ]
        if matched_period is not None and matched_period > 0:
            helper_specs.append(("Use catalogue period", matched_period, "catalogue_period", "catalogue period"))
        helper_specs.extend(
            [
                ("Use P", helper_base_period, helper_base_source, f"P from {helper_base_label}"),
                ("Use 3P", helper_base_period * 3.0, harmonic_source_key("3P", helper_base_source), f"3P from {helper_base_label}"),
                ("Use P/3", max(0.1, helper_base_period / 3.0), harmonic_source_key("P/3", helper_base_source), f"P/3 from {helper_base_label}"),
            ]
        )
        helper_cols = st.columns(len(helper_specs) + 1)
        for idx, (label, helper_period, helper_source, helper_action) in enumerate(helper_specs):
            with helper_cols[idx]:
                if st.button(label, key=f"insp_helper_{helper_action}_{idx}", use_container_width=True):
                    st.session_state["_insp_pending_selected_period"] = float(helper_period)
                    st.session_state["_insp_pending_selected_t0"] = float(adj_t0)
                    st.session_state["_insp_pending_helper_action"] = helper_action
                    st.session_state["_insp_pending_history_label"] = helper_action
                    st.session_state["_insp_pending_history_source"] = helper_source
                    st.session_state["_insp_pending_period_source"] = helper_source
                    st.rerun()
        with helper_cols[-1]:
            if st.button("Run odd/even check", key="insp_odd_even_check", use_container_width=True):
                odd_even_now = odd_even_depth_check(
                    lc_flat,
                    adj_period,
                    adj_t0,
                    folded_summary.get("broad_frac", 0.0),
                )
                odd_even_now["period"] = float(adj_period)
                odd_even_now["t0"] = float(adj_t0)
                st.session_state["insp_odd_even_result"] = odd_even_now
                st.rerun()

        history = st.session_state.get("insp_period_history") or []
        last_helper_action = st.session_state.get("insp_last_helper_action", "none yet")
        history_text = (
            "Last tried: "
            + ", ".join(
                f"{item.get('label')} ({safe_float(item.get('period'), 0.0):.5f} d)"
                for item in history
            )
            if history
            else "Last tried: none yet"
        )
        helper_lines = [
            f"- **Current period:** {adj_period:.5f} d",
            f"- **Helper base period:** {helper_base_period:.5f} d",
            f"- **Helper base source:** {period_source_label(helper_base_source)}",
            f"- **Last helper action:** {last_helper_action}",
            f"- **Public catalogue period available:** {matched_period:.5f} d" if matched_period is not None else "- **Public catalogue period available:** none",
            f"- **{period_helper_suggestion(adj_period, matched_period)}**",
            f"- **{history_text}**",
        ]
        st.info("\n".join(helper_lines))

        odd_even_result = st.session_state.get("insp_odd_even_result")
        if odd_even_result and periods_agree(safe_float(odd_even_result.get("period")), adj_period, rel_tol=0.0005):
            odd_even_level, odd_even_text = odd_even_result_text(odd_even_result)
            if odd_even_level == "success":
                st.success(odd_even_text)
            elif odd_even_level == "warning":
                st.warning(odd_even_text)
            elif odd_even_level == "error":
                st.error(odd_even_text)
            else:
                st.info(odd_even_text)

        suspicion = build_suspicion_checks(lc_flat, adj_period, adj_t0)
        folded_summary = suspicion.get("summary") or folded_summary or fold_signal_summary(lc_flat, adj_period, adj_t0)
        depth_adj = folded_summary.get("depth_ppm", st.session_state.get("insp_depth", 0.0))

        with st.expander("Optional comparison search", expanded=False):
            st.caption("Use these only as comparison tools. They do not replace the selected evaluation period.")
            col_cmp1, col_cmp2 = st.columns(2)
            with col_cmp1:
                if st.button("Run single-sector comparison BLS", key="cmp_single_bls"):
                    try:
                        _cmp = run_bls_search(lc_flat_bls, False)
                        st.session_state["insp_single_bls_period"] = _cmp["period"]
                        st.session_state["insp_single_bls_t0"] = _cmp["t0"]
                        st.success(f"Single-sector comparison found {_cmp['period']:.5f} d")
                    except Exception as _e:
                        st.warning(f"Single-sector comparison BLS failed: {_e}")
            with col_cmp2:
                if st.button("Run stitched comparison BLS", key="cmp_stitched_bls"):
                    try:
                        _cmp = run_bls_search(lc_flat, True)
                        st.session_state["insp_stitched_bls_period"] = _cmp["period"]
                        st.session_state["insp_stitched_bls_t0"] = _cmp["t0"]
                        st.success(f"Stitched comparison found {_cmp['period']:.5f} d")
                    except Exception as _e:
                        st.warning(f"Stitched comparison BLS failed: {_e}")
            if safe_float(st.session_state.get("insp_single_bls_period")):
                st.caption(f"Single-sector comparison: {safe_float(st.session_state.get('insp_single_bls_period')):.5f} d")
            if safe_float(st.session_state.get("insp_stitched_bls_period")):
                st.caption(f"Stitched comparison: {safe_float(st.session_state.get('insp_stitched_bls_period')):.5f} d")

        verify_result = st.session_state.get("insp_verify_result")
        verify_period = safe_float(st.session_state.get("insp_verify_period"))
        current_verify = None
        if (
            verify_result
            and str(st.session_state.get("insp_tic", "")) == str(tic_id)
            and verify_period is not None
            and periods_agree(verify_period, adj_period, rel_tol=0.0005)
        ):
            current_verify = verify_result

        verify_verdict = (current_verify or {}).get("verdict")
        verify_state = verification_status(verify_verdict)

        agreement_messages = build_agreement_messages(
            known_period,
            safe_float(st.session_state.get("insp_single_bls_period")),
            safe_float(st.session_state.get("insp_stitched_bls_period")),
            adj_period,
            verify_period,
            current_source,
        )

        caution_messages = list(suspicion.get("messages", []))
        if current_verify and current_verify.get("eb_warning"):
            caution_messages.append(current_verify["eb_warning"])

        if inspector_mode == "Validate Known Target":
            if verify_state == "real":
                final_text = "Known target recovered cleanly."
                final_box = st.success
            elif verify_state == "inconclusive":
                final_text = "Known target recovered weakly."
                final_box = st.warning
            elif verify_state == "reject":
                final_text = "Did not recover clearly at the supplied known period."
                final_box = st.error
            else:
                final_text = "Using the supplied known period as the thing being tested. Run verification to judge the recovery."
                final_box = st.info
        else:
            if verify_state == "real" and not suspicion.get("major_flags"):
                final_text = "Strong multi-sector candidate worth manual review."
                final_box = st.success
            elif verify_state == "real":
                final_text = "Real periodic signal, but EB-like warnings mean it is not a clean planet candidate."
                final_box = st.warning
            elif verify_state == "inconclusive":
                final_text = "Real periodic signal possible, but not a strong planet candidate yet."
                final_box = st.warning
            elif verify_state == "reject":
                final_text = "BLS found a period, but multi-sector verification does not support it."
                final_box = st.error
            else:
                final_text = "Search hit only. BLS found a period, but that is not the same as a likely planet."
                final_box = st.info

        st.markdown("### Final Interpretation")
        final_box(final_text)
        summary_lines = [
            f"- **Mode:** {inspector_mode}",
            f"- **TIC:** {tic_id}",
            f"- **Period being evaluated:** {adj_period:.5f} d",
            f"- **Period source:** {period_source_label(current_source)}",
        ]
        if known_period:
            summary_lines.append(f"- **Known/reference period supplied:** {known_period:.5f} d")
        if safe_float(st.session_state.get("insp_single_bls_period")) is not None:
            summary_lines.append(
                f"- **Single-sector BLS best period:** {safe_float(st.session_state.get('insp_single_bls_period')):.5f} d"
            )
        if safe_float(st.session_state.get("insp_stitched_bls_period")) is not None:
            summary_lines.append(
                f"- **Stitched BLS best period:** {safe_float(st.session_state.get('insp_stitched_bls_period')):.5f} d"
            )
        summary_lines.append(f"- **Verification run:** {'yes' if current_verify else 'no'}")
        summary_lines.append(f"- **Verification verdict:** {verify_verdict or 'not run yet'}")
        st.markdown("\n".join(summary_lines))

        render_public_match_box(public_match)

        if caution_messages:
            st.markdown("**Major cautions**")
            for msg in caution_messages:
                st.warning(msg)

        st.markdown("### Confidence Ladder")
        render_badges(build_confidence_ladder(inspector_mode, base_source != "known", current_verify, suspicion))

        st.markdown("### Agreement / Disagreement")
        for msg in agreement_messages:
            st.write(f"- {msg}")

        with st.expander("What the plots mean", expanded=False):
            st.markdown(
                "- **Raw lightcurve:** brightness over time; useful for gaps and weird behavior, not the main proof.\n"
                "- **Flattened & cleaned:** the same data after slow trends are removed.\n"
                "- **Phase-folded:** the most important panel; if the period is right, real transits stack into one dip.\n"
                "- **Transit detail:** zoomed-in folded dip; look for one compact dip, not a broad wave.\n"
                "- **BLS periodogram:** the periods the search liked best; a big peak does not automatically mean planet."
            )

        fig = make_4panel(
            lc_raw=lc_raw,
            lc_flat=lc_flat,
            period_d=adj_period,
            t0_btjd=adj_t0,
            tic_id=tic_id,
            sector=sector,
            depth_ppm=depth_adj,
            bls_power=sde or 0.0,
        )
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected period", f"{adj_period:.5f} d")
        m2.metric("Transit depth", f"{depth_adj:.0f} ppm")
        m3.metric("Search strength", f"{sde:.1f}" if sde is not None else "loaded/manual")
        m4.metric("Radius-equivalent hint only", planet_size_estimate(depth_adj))
        st.caption("Planet-size wording here is only a radius-equivalent hint if the transit interpretation is real. It is not a classification.")

        blsm = st.session_state.get("insp_blsm")
        if blsm is not None and st.checkbox("Show BLS periodogram", value=False):
            fig_bls = go.Figure()
            fig_bls.add_trace(go.Scatter(x=blsm.period.value, y=blsm.power.value,
                                         mode="lines", line=dict(color="#58a6ff", width=1)))
            fig_bls.add_vline(x=adj_period, line_dash="dash", line_color="#ff7b72",
                              annotation_text=f"P={adj_period:.4f}d")
            fig_bls.update_layout(xaxis_title="Period (days)", yaxis_title="BLS Power",
                                  height=280, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                  font_color="#c9d1d9")
            st.plotly_chart(fig_bls, use_container_width=True)

        st.caption(f"Current raw data shown here uses {len(lc_raw):,} stitched points. Search strength and verification are separate from the final interpretation.")

        st.markdown("---")
        if st.button(
            "Verify Across All Sectors At The Selected Period",
            key="verify_all_sectors",
            help="This checks whether the currently selected period repeats across all available sectors.",
        ):
            with st.spinner("Downloading all sectors and checking consistency..."):
                _vr = _cached_verify(int(tic_id), round(adj_period, 6), round(adj_t0, 4) if adj_t0 else None)
            st.session_state["insp_verify_result"] = _vr
            st.session_state["insp_verify_period"] = adj_period
            st.rerun()

        if verify_result and not current_verify and verify_period is not None:
            st.info(
                f"The saved verification result is for {verify_period:.5f} d, not the current selected period. "
                "Run verification again if you want the verdict for the current setting."
            )

        if current_verify:
            _vverdict = current_verify["verdict"]
            _vcs = current_verify["consistency_score"]
            _vn = f"{current_verify['n_sectors_consistent']}/{current_verify['n_sectors_checked']} sectors"
            st.markdown("### Verification Result")
            if "Likely real" in _vverdict:
                st.success(f"Pipeline verification verdict: **{_vverdict}** — {_vn}")
            elif "Inconclusive" in _vverdict:
                st.warning(f"Pipeline verification verdict: **{_vverdict}** — {_vcs:.0%} consistency")
            else:
                st.error(f"Pipeline verification verdict: **{_vverdict}** — {_vcs:.0%} consistency")
            st.caption("This is a pipeline verdict, not astrophysical proof or planet confirmation.")
            if current_verify.get("figure"):
                st.pyplot(current_verify["figure"], use_container_width=True)
            if current_verify.get("sector_results"):
                st.dataframe(
                    pd.DataFrame(current_verify["sector_results"]).drop(
                        columns=["skip_reason", "has_signal"], errors="ignore"
                    ),
                    use_container_width=True,
                )
    elif not submitted:
        st.info("Choose a workflow above, enter a TIC, and click the action button.")

    exofop_footer()


# ==============================================================================
# Page 4 — Scan Results
# ==============================================================================
elif page == "Scan Results":
    st.title("Scan Results")
    st.markdown(
        "All completed sector scans, ranked by signal strength. "
        "🔴 = BLS power > 12 (high priority). "
        "Click **Inspect** to open in the Lightcurve Inspector."
    )

    edu_expander("Understanding the results", """
**BLS Power** measures how well a box-shaped transit model fits the data:
- **< 7**: Background noise
- **7–9**: Worth a look
- **9–12**: Solid candidate — inspect carefully
- **> 12**: High priority — strong signal

**Transit depth** interpretation:
- **100–500 ppm**: Earth-to-Neptune size planet
- **1,000–10,000 ppm**: Neptune-to-Jupiter size
- **10,000–50,000 ppm**: Hot Jupiter or inflated planet
- **> 50,000 ppm** (5%+): Almost certainly an **eclipsing binary**

**Is it a real planet? Checklist:**
- ✅ BLS power ≥ 9 and clean U-shaped transit in the phase-folded plot
- ✅ Depth < 20,000 ppm (rules out most EBs)
- ✅ No secondary eclipse at half-period
- ✅ Gaia RUWE < 1.4 (no unresolved companion)
- ✅ ExoMiner++ score ≥ 0.5 (if available)
- ⚠️ Period < 1 day → likely noise or EB (but hot Jupiters do exist)
- ⚠️ Very long duration (>8 h) relative to period → suspect EB

**Depth reference table:**
| Planet size | Depth around Sun-like star |
|-------------|---------------------------|
| Earth (1 R⊕) | ~84 ppm |
| Super-Earth (1.5 R⊕) | ~190 ppm |
| Neptune (3.9 R⊕) | ~1,270 ppm |
| Saturn (9.1 R⊕) | ~7,000 ppm |
| Jupiter (11.2 R⊕) | ~10,600 ppm |
| Hot Jupiter (1.5 R_J) | ~24,000 ppm |
""")

    scan_dirs = _cached_scan_dirs()

    if not scan_dirs:
        st.info("No completed scans. Use the **Run Scan** page to start.")
        st.stop()

    sector_names  = [d.name for d in scan_dirs]
    chosen_sector = st.selectbox("Select scan:", sector_names)
    scan_dir  = RESULTS_DIR / chosen_sector
    csv_path  = scan_dir / "bls_results.csv"
    html_path = scan_dir / "scan_report.html"
    plot_dir  = scan_dir / "plots"

    # Sector date banner
    m = re.search(r'(\d+)', chosen_sector)
    if m:
        sector_num   = int(m.group(1))
        sector_start = datetime(2018, 7, 25) + timedelta(days=(sector_num - 1) * 27.4)
        sector_end   = sector_start + timedelta(days=27.4)
        date_str = (
            f"Sector {sector_num} observed approximately "
            f"{sector_start.strftime('%b %d')}–{sector_end.strftime('%b %d, %Y')}"
        )
        if sector_num <= EXOMINER_MAX_SECTOR:
            st.success(f"{date_str}   ✅ ExoMiner++ has published scores for this sector.")
        else:
            st.info(
                f"{date_str}   🔬 Sector {sector_num} is **not yet in ExoMiner++ catalog** "
                f"(coverage: 1–{EXOMINER_MAX_SECTOR}) — fresh data!"
            )

    em_csv = scan_dir / "bls_exominer_results.csv"
    if em_csv.exists():
        csv_path = em_csv

    _scan_key = f"scan_df_{csv_path}"
    if _scan_key not in st.session_state:
        st.session_state[_scan_key] = pd.read_csv(csv_path)
    df = st.session_state[_scan_key]

    # Load filter stats if available
    _stats_file = scan_dir / "scan_filter_stats.json"
    _fstats = {}
    if _stats_file.exists():
        try:
            _fstats = json.loads(_stats_file.read_text())
        except Exception:
            pass

    n_total   = len(df)
    n_planet  = int((df.get("classification", pd.Series()) == "Planet candidate").sum()) if "classification" in df.columns else int((df["bls_power"] >= 9).sum())
    n_inspect = int((df.get("classification", pd.Series()) == "Needs inspection").sum()) if "classification" in df.columns else 0
    n_hot     = int((df["bls_power"] >= 12).sum())

    # Summary banner with filtering context
    _n_raw = _fstats.get("n_raw", 0)
    _fc    = _fstats.get("filter_counts", {})
    _n_filtered = sum(_fc.values())
    if _n_raw:
        _pct_filt = 100 * _n_filtered / max(_n_raw, 1)
        st.info(
            f"**{_n_raw:,}** raw BLS detections → "
            f"🗑️ **{_n_filtered:,}** filtered ({_pct_filt:.1f}% — binaries, noise, bad periods) → "
            f"🟢 **{n_total}** candidates in CSV"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates in CSV",    n_total)
    c2.metric("🟢 Planet candidates", n_planet)
    c3.metric("🟡 Needs inspection",  n_inspect)
    c4.metric("🔴 High SDE (≥12)",    n_hot)

    if html_path.exists():
        col_dl, col_prev = st.columns([1, 3])
        with col_dl:
            st.download_button("Download HTML report", html_path.read_bytes(),
                               f"{chosen_sector}_report.html", "text/html")
        with col_prev:
            if st.checkbox("Preview report inline", value=False):
                import streamlit.components.v1 as components
                components.html(html_path.read_text(encoding="utf-8", errors="replace"),
                                height=600, scrolling=True)

    st.markdown("---")
    st.subheader("Candidates")
    cand_df = df.sort_values("bls_power", ascending=False).copy()
    public_match_map = build_public_match_map(cand_df, limit=min(100, len(cand_df)), detailed=False)
    if public_match_map:
        cand_df["public_status"] = cand_df.apply(
            lambda row: public_status_chip(public_match_map.get(candidate_key(row.get("tic_id"), row.get("period")))),
            axis=1,
        )
    if cand_df.empty:
        st.info("No candidates in results.")
    else:
        def highlight_power(row):
            if row["bls_power"] >= 12:
                return ["background-color: rgba(255,123,114,0.15)"] * len(row)
            elif row["bls_power"] >= 9:
                return ["background-color: rgba(88,166,255,0.08)"] * len(row)
            return [""] * len(row)
        cols_to_show = [c for c in ["tic_id", "period", "depth_ppm", "duration_hours", "bls_power", "snr", "n_transits", "classification", "public_status"] if c in cand_df.columns]
        if public_match_map:
            st.caption("Public catalogue status uses official public cross-matches and helps separate known objects from locally-found pipeline candidates.")
        st.dataframe(cand_df[cols_to_show].style.apply(highlight_power, axis=1),
                     use_container_width=True, height=300)

    st.markdown("---")
    st.subheader("Transit Plots")
    strong_cands = df[df["bls_power"] >= 9].sort_values("bls_power", ascending=False)
    if strong_cands.empty:
        st.info("No candidates with BLS power ≥ 9.")
    else:
        for _, row in strong_cands.iterrows():
            tic_id = int(row["tic_id"])
            power  = row["bls_power"]
            period = row["period"]
            depth  = row["depth_ppm"]
            t0_val = row.get("t0", None)
            priority = "🔴 HIGH PRIORITY" if power >= 12 else "⚡ Candidate"
            label    = f"{priority}  ·  TIC {tic_id}  ·  P={period:.4f} d  ·  depth={depth:.0f} ppm  ·  BLS={power:.2f}"
            with st.expander(label, expanded=(power >= 12)):
                col_plot, col_info = st.columns([3, 1])
                png_path = plot_dir / f"tic_{tic_id}_s{chosen_sector.replace('sector','')}.png"
                if not png_path.exists():
                    hits = list(plot_dir.glob(f"tic_{tic_id}_*.png")) if plot_dir.exists() else []
                    if hits:
                        png_path = hits[0]
                with col_plot:
                    if png_path.exists():
                        st.image(str(png_path), use_container_width=True)
                    else:
                        st.warning("Plot not found.")
                with col_info:
                    st.metric("BLS Power", f"{power:.2f}")
                    st.metric("Period",    f"{period:.4f} d")
                    st.metric("Depth",     f"{depth:.0f} ppm")
                    _public = public_match_map.get(candidate_key(tic_id, period))
                    if _public:
                        st.caption(public_status_chip(_public))
                        st.caption(_public.get("plain_english_summary", ""))
                    if "duration_hours" in row:
                        st.metric("Duration", f"{row['duration_hours']:.2f} h")
                    if "snr" in row:
                        st.metric("SNR", f"{row['snr']:.2f}")
                    if st.button(f"Inspect TIC {tic_id}", key=f"insp_{tic_id}"):
                        st.session_state["jump_tic"]    = tic_id
                        st.session_state["jump_period"] = period
                        st.session_state["jump_period_source"] = "single_bls_loaded"
                        if t0_val is not None and not pd.isna(t0_val):
                            st.session_state["jump_t0"] = float(t0_val)
                        st.session_state["insp_mode"] = "Search for Candidate"
                        st.session_state["_jump_page"] = "Lightcurve Inspector"
                        st.rerun()
                    # Verification badge / button
                    _cs_raw  = row.get("consistency_score", "")
                    _ver     = row.get("verified", "")
                    _eb_warn = row.get("eb_warning", "")
                    if _ver == "true":
                        st.markdown("✅ **Multi-sector verified**")
                    elif _ver == "false":
                        st.markdown("❌ **Not consistent across sectors**")
                    elif _cs_raw not in ("", None):
                        try:
                            _cs = float(_cs_raw)
                            _badge = "✅" if _cs >= 0.7 else "❓" if _cs >= 0.3 else "❌"
                            st.markdown(f"{_badge} Consistency: {_cs:.0%}")
                        except (ValueError, TypeError):
                            pass
                    else:
                        _vk = f"verify_result_{tic_id}"
                        if st.button("🔭 Verify", key=f"verify_{tic_id}",
                                     help="Check this signal across all available TESS sectors"):
                            from verify_candidate import verify as _vfy
                            with st.spinner(f"Verifying TIC {tic_id}…"):
                                _res = _vfy(tic_id, period,
                                            t0=float(t0_val) if t0_val is not None and not pd.isna(t0_val) else None)
                            st.session_state[_vk] = _res
                            st.rerun()
                    _vr2 = st.session_state.get(f"verify_result_{tic_id}")
                    if _vr2:
                        _vv = _vr2["verdict"]
                        _vn = f"{_vr2['n_sectors_consistent']}/{_vr2['n_sectors_checked']}"
                        st.caption(f"**{_vv[:40]}** ({_vn})")
                        if _vr2.get("eb_warning"):
                            st.warning(_vr2["eb_warning"])
                        if _vr2.get("figure"):
                            st.pyplot(_vr2["figure"], use_container_width=True)

    exofop_footer()


# ==============================================================================
# Page 5 — Hunt for Planets
# ==============================================================================
elif page == "Run Scan":
    st.title("🪐 Hunt for Planets")

    for _k in [k for k in st.session_state if k.startswith("insp_")]:
        del st.session_state[_k]

    st.markdown(
        "One button. Four steps. Download the data, search for signals, run NASA's AI, show you the results."
    )

    # ── Sector selector ───────────────────────────────────────────────────
    dl_sectors = get_downloaded_sectors()

    def sector_label(n: int) -> str:
        start = datetime(2018, 7, 25) + timedelta(days=(n - 1) * 27.4)
        date_str = start.strftime("%b %Y")
        scanned = sector_hunt_complete(n)
        if n > EXOMINER_MAX_SECTOR:
            tag = "🔬 FRESH — not yet in ExoMiner++ catalog"
        else:
            tag = "already scored by ExoMiner++"
        done_tag = " ✅ already scanned" if scanned else ""
        return f"Sector {n} — {date_str} — {tag}{done_tag}"

    # Default: newest downloaded unscanned sector
    unscanned = [s for s in dl_sectors
                 if not sector_hunt_complete(s)]
    default_sector = unscanned[-1] if unscanned else (dl_sectors[-1] if dl_sectors else None)

    if dl_sectors:
        default_idx = dl_sectors.index(default_sector) if default_sector in dl_sectors else len(dl_sectors) - 1
        sector_n = st.selectbox(
            "Choose a sector to hunt",
            dl_sectors,
            index=default_idx,
            format_func=sector_label,
            key="hunt_sector",
        )
    else:
        st.warning("No downloaded sectors found. Sectors are downloaded automatically in Step 1.")
        sector_n = st.number_input("Or enter a sector number to download + scan", 1, 200, 10, 1, key="hunt_sector_num")

    # Sector info banner
    if sector_n:
        _s_start = datetime(2018, 7, 25) + timedelta(days=(sector_n - 1) * 27.4)
        _s_end   = _s_start + timedelta(days=27.4)
        _date    = f"{_s_start.strftime('%b %d')}–{_s_end.strftime('%b %d, %Y')}"
        if sector_n > EXOMINER_MAX_SECTOR:
            st.info(
                f"**Sector {sector_n}** was observed {_date}.  \n"
                f"🔬 This sector is **NOT in ExoMiner++'s published catalog** "
                f"(coverage ends at sector {EXOMINER_MAX_SECTOR}). "
                f"Any candidates you find here are **fresh** — you may be the first to identify them."
            )
        else:
            st.info(
                f"**Sector {sector_n}** was observed {_date}.  \n"
                f"This sector is already in the ExoMiner++ catalog — you're re-checking their published work."
            )

    # Advanced options
    with st.expander("⚙️ Advanced options"):
        c1, c2 = st.columns(2)
        with c1:
            workers = st.slider("CPU workers", 1, 32, 28, key="hunt_workers",
                                help="28 workers recommended for this 32-vCPU machine")
        with c2:
            limit = st.number_input("Star limit (0 = all stars)", 0, 100000, 0, 10, key="hunt_limit",
                                    help="Set to e.g. 20 for a quick test run")
        no_score = st.checkbox("Skip ExoMiner++ scoring", value=False, key="hunt_no_score",
                               help="Skip the AI scoring step and just show BLS results")

    st.markdown("---")

    # ── Job controls / running state ──────────────────────────────────────
    active_running, active_state = scan_is_running()
    active_log = current_state_log(active_state)

    if active_running:
        started = active_state.get("started_at", "")[:16]
        col_status, col_stop = st.columns([4, 1])
        with col_status:
            _msg = (
                f"🟢 **{job_mode_label(active_state)} running**: "
                f"{active_state.get('label', '?')} — started {started}"
            )
            if active_state.get("current_sector"):
                _msg += f"  \nCurrent sector: **{active_state['current_sector']}**"
            if active_state.get("sectors_total"):
                _msg += (
                    f"  \nProgress: **{active_state.get('sectors_done', 0)}"
                    f"/{active_state.get('sectors_total', 0)}** done"
                )
                if active_state.get("sectors_failed", 0):
                    _msg += f", **{active_state['sectors_failed']}** failed"
            st.warning(_msg)
            if active_state.get("mode") == "build_shortlist":
                _progress = read_job_progress(active_state)
                if _progress:
                    render_shortlist_progress(_progress)
                else:
                    st.caption("Progress details are not available for this shortlist run yet.")
        with col_stop:
            if st.button("⏹ Stop Job", type="secondary"):
                _pid = active_state.get("pid", 0)
                if _pid:
                    try:
                        import signal
                        os.killpg(os.getpgid(_pid), signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        pass
                active_state["running"] = False
                active_state["status"] = "failed"
                active_state["finished_at"] = datetime.now().isoformat()
                write_scan_state(active_state)
                write_job_meta(active_state)
                st.rerun()
    else:
        if st.button("🪐 Hunt for Planets", type="primary", disabled=active_running):
            _s = int(sector_n)
            _log_path = RESULTS_DIR / f"sector{_s:02d}" / "hunt.log"
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            _cmd = [
                sys.executable, str(SCRIPTS_DIR / "hunt.py"),
                "--sector",  str(_s),
                "--workers", str(workers),
                "--limit",   str(int(limit)),
            ]
            if no_score:
                _cmd.append("--no-score")
            with open(_log_path, "w") as _lf:
                _proc = subprocess.Popen(_cmd, stdout=_lf, stderr=subprocess.STDOUT,
                                         cwd=SCRIPTS_DIR, start_new_session=True)
            _state = {
                "job_id":      make_job_id(f"hunt_s{_s:02d}"),
                "running":     True,
                "mode":        "hunt",
                "status":      "running",
                "label":       f"Sector {_s}",
                "sector":      _s,
                "current_sector": _s,
                "pid":         _proc.pid,
                "log_file":    str(_log_path),
                "current_log_file": str(_log_path),
                "command":     format_cmd(_cmd),
                "started_at":  datetime.now().isoformat(),
                "sectors_total": 1,
                "sectors_done": 0,
                "sectors_failed": 0,
            }
            write_scan_state(_state)
            write_job_meta(_state)
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.subheader("Sector Operations")
    st.caption(
        "Run durable background jobs that keep going if you leave the page. "
        "Only one dashboard job runs at a time."
    )

    col_dl, col_all, col_short = st.columns(3)
    with col_dl:
        if st.button("⬇️ Check + Download Missing Sectors (68+)", disabled=active_running):
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            _job_id = make_job_id("download_missing")
            _job_log = JOBS_DIR / f"{_job_id}.log"
            _cmd = [
                sys.executable, str(SCRIPTS_DIR / "dashboard_jobs.py"),
                "download-missing-sectors",
                "--job-id", _job_id,
                "--min-sector", "68",
                "--threads", "8",
            ]
            _proc = subprocess.Popen(
                _cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=SCRIPTS_DIR,
                start_new_session=True,
            )
            _state = {
                "job_id": _job_id,
                "running": True,
                "mode": "download_missing",
                "status": "queued",
                "label": "Download missing sectors (68+)",
                "pid": _proc.pid,
                "log_file": str(_job_log),
                "command": format_cmd(_cmd),
                "started_at": datetime.now().isoformat(),
                "sectors_total": 0,
                "sectors_done": 0,
                "sectors_failed": 0,
            }
            write_scan_state(_state)
            write_job_meta(_state)
            time.sleep(0.5)
            st.rerun()
    with col_all:
        if st.button("🛰️ Scan All Downloaded Unfinished Sectors (68+)", disabled=active_running):
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            _job_id = make_job_id("scan_all")
            _job_log = JOBS_DIR / f"{_job_id}.log"
            _cmd = [
                sys.executable, str(SCRIPTS_DIR / "dashboard_jobs.py"),
                "scan-all-sectors",
                "--job-id", _job_id,
                "--min-sector", "68",
                "--workers", str(workers),
                "--limit", str(int(limit)),
            ]
            if no_score:
                _cmd.append("--no-score")
            _proc = subprocess.Popen(
                _cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=SCRIPTS_DIR,
                start_new_session=True,
            )
            _state = {
                "job_id": _job_id,
                "running": True,
                "mode": "scan_all",
                "status": "queued",
                "label": "Scan all sectors (68+)",
                "pid": _proc.pid,
                "log_file": str(_job_log),
                "command": format_cmd(_cmd),
                "started_at": datetime.now().isoformat(),
                "sectors_total": 0,
                "sectors_done": 0,
                "sectors_failed": 0,
            }
            write_scan_state(_state)
            write_job_meta(_state)
            time.sleep(0.5)
            st.rerun()
    with col_short:
        if st.button("⚡ Build Fast Cross-Matched Shortlist (85–99)", disabled=active_running):
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)
            _job_id = make_job_id("build_shortlist")
            _job_log = JOBS_DIR / f"{_job_id}.log"
            _job_progress = JOBS_DIR / f"{_job_id}.progress.json"
            _cmd = [
                sys.executable, str(SCRIPTS_DIR / "dashboard_jobs.py"),
                "build-crossmatched-shortlist",
                "--job-id", _job_id,
                "--sectors", "85-99",
                "--out", str(FAST_SHORTLIST_PATH),
                "--fast",
            ]
            _proc = subprocess.Popen(
                _cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=SCRIPTS_DIR,
                start_new_session=True,
            )
            _state = {
                "job_id": _job_id,
                "running": True,
                "mode": "build_shortlist",
                "status": "queued",
                "label": "Build fast cross-matched shortlist (85–99)",
                "pid": _proc.pid,
                "log_file": str(_job_log),
                "current_log_file": str(_job_log),
                "progress_file": str(_job_progress),
                "command": format_cmd(_cmd),
                "started_at": datetime.now().isoformat(),
                "output_file": str(FAST_SHORTLIST_PATH),
                "build_mode": "fast",
            }
            write_scan_state(_state)
            write_job_meta(_state)
            time.sleep(0.5)
            st.rerun()
        if st.button("📋 Build Full Cross-Matched Shortlist (85–99)", disabled=active_running):
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)
            _job_id = make_job_id("build_shortlist")
            _job_log = JOBS_DIR / f"{_job_id}.log"
            _job_progress = JOBS_DIR / f"{_job_id}.progress.json"
            _cmd = [
                sys.executable, str(SCRIPTS_DIR / "dashboard_jobs.py"),
                "build-crossmatched-shortlist",
                "--job-id", _job_id,
                "--sectors", "85-99",
                "--out", str(DEFAULT_SHORTLIST_PATH),
            ]
            _proc = subprocess.Popen(
                _cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=SCRIPTS_DIR,
                start_new_session=True,
            )
            _state = {
                "job_id": _job_id,
                "running": True,
                "mode": "build_shortlist",
                "status": "queued",
                "label": "Build full cross-matched shortlist (85–99)",
                "pid": _proc.pid,
                "log_file": str(_job_log),
                "current_log_file": str(_job_log),
                "progress_file": str(_job_progress),
                "command": format_cmd(_cmd),
                "started_at": datetime.now().isoformat(),
                "output_file": str(DEFAULT_SHORTLIST_PATH),
                "build_mode": "full",
            }
            write_scan_state(_state)
            write_job_meta(_state)
            time.sleep(0.5)
            st.rerun()

    if FAST_SHORTLIST_PATH.exists():
        _built = datetime.fromtimestamp(FAST_SHORTLIST_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last fast shortlist build: {_built} — {FAST_SHORTLIST_PATH}")
        if shortlist_is_stale(FAST_SHORTLIST_PATH):
            st.warning("The existing fast shortlist may be stale because newer sector 85–99 scan results exist.")
    if DEFAULT_SHORTLIST_PATH.exists():
        _built = datetime.fromtimestamp(DEFAULT_SHORTLIST_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Last full shortlist build: {_built} — {DEFAULT_SHORTLIST_PATH}")
        if shortlist_is_stale(DEFAULT_SHORTLIST_PATH):
            st.warning("The existing full shortlist may be stale because newer sector 85–99 scan results exist.")

    # ── Live log display ──────────────────────────────────────────────────
    active_running, active_state = scan_is_running()
    display_log = current_state_log(active_state)
    if not display_log and active_log and active_log.exists():
        display_log = active_log

    import streamlit.components.v1 as components

    def _log_box(text: str):
        import html as _html
        escaped = _html.escape(text)
        st.markdown(
            f'<div id="hunt-log-box" style="height:300px;overflow-y:auto;background:#0e1117;color:#e0e0e0;'
            f'font-family:monospace;font-size:12px;padding:10px;border-radius:6px;'
            f'border:1px solid #333;white-space:pre-wrap;">{escaped}</div>',
            unsafe_allow_html=True,
        )
        components.html(
            '<script>var e=parent.document.getElementById("hunt-log-box");'
            'if(e)e.scrollTop=e.scrollHeight;</script>',
            height=0,
        )

    @st.fragment(run_every=2)
    def live_log_panel():
        _running, _state = scan_is_running()
        _dlog = current_state_log(_state)
        if not _dlog and display_log and display_log.exists():
            _dlog = display_log
        if _running and _dlog and _dlog.exists():
            if _state.get("log_file"):
                st.caption(f"Master log: {_state.get('log_file')}")
            if _dlog and str(_dlog) != _state.get("log_file"):
                st.caption(f"Showing current detail log: {_dlog}")
            _log_box(tail_log(_dlog))
        elif _dlog and _dlog.exists():
            fin = _state.get("finished_at", "")
            status = _state.get("status", "")
            if fin and status == "failed":
                st.error(f"⚠️ Job failed at {fin[:16]}")
            elif fin:
                st.success(f"✅ Job finished at {fin[:16]}")
            if _state.get("log_file"):
                st.caption(f"Master log: {_state.get('log_file')}")
            if _dlog and str(_dlog) != _state.get("log_file"):
                st.caption(f"Showing current detail log: {_dlog}")
            with st.expander("Show full log", expanded=False):
                _log_box(tail_log(_dlog))

    if display_log and display_log.exists():
        live_log_panel()

    recent_jobs = load_recent_jobs()
    if recent_jobs:
        st.markdown("---")
        st.subheader("Recent Jobs")
        for _job in recent_jobs[:5]:
            _started = (_job.get("started_at") or "")[:16]
            _finished = (_job.get("finished_at") or "")[:16]
            _status = _job.get("status", "unknown")
            _line = f"**{_job.get('label', _job.get('mode', 'Job'))}** — {_status}"
            if _started:
                _line += f" — started {_started}"
            if _finished:
                _line += f" — finished {_finished}"
            st.markdown(_line)
            if _job.get("sectors_total"):
                st.caption(
                    f"Progress: {_job.get('sectors_done', 0)}/{_job.get('sectors_total', 0)} "
                    f"done, {_job.get('sectors_failed', 0)} failed"
                )
            if _job.get("row_count") is not None:
                st.caption(f"Rows: {_job.get('row_count')}")
            if _job.get("output_file"):
                st.caption(f"Output: {_job.get('output_file')}")
            if _job.get("log_file"):
                st.caption(f"Log: {_job.get('log_file')}")

    # ── Results (shown after hunt completes) ──────────────────────────────
    if not active_running and sector_n:
        _s = int(sector_n)
        _res_dir = RESULTS_DIR / f"sector{_s:02d}"
        _csv = _res_dir / "bls_exominer_results.csv"
        if not _csv.exists():
            _csv = _res_dir / "bls_results.csv"

        if _csv.exists():
            st.markdown("---")
            st.subheader(f"Results — Sector {_s}")
            try:
                _df = pd.read_csv(_csv)
                _has_scores = (
                    "exominer_score" in _df.columns
                    and pd.to_numeric(_df["exominer_score"], errors="coerce").notna().any()
                )
                _cands = _df[_df["bls_power"] >= 7.0].copy()
                _cands = _cands.sort_values("bls_power", ascending=False)

                if _has_scores:
                    _cands["_sc"] = pd.to_numeric(_cands["exominer_score"], errors="coerce")
                    _n_strong = int((_cands["_sc"] >= 0.5).sum())
                    st.markdown(
                        f"Scanned **{len(_df):,} stars**. "
                        f"Found **{len(_cands)} transit signals**. "
                        f"ExoMiner++ says **{_n_strong}** look like real planet candidates."
                    )
                else:
                    st.markdown(
                        f"Scanned **{len(_df):,} stars**. "
                        f"Found **{len(_cands)} transit signals** (BLS power ≥ 7)."
                    )
                    st.caption("ExoMiner++ scores not available — "
                               "SPOC may not have detected TCEs for these targets.")

                if _s > EXOMINER_MAX_SECTOR:
                    st.info(
                        f"🔬 Sector {_s} is **not** in ExoMiner++'s published catalog "
                        f"(sectors 1–{EXOMINER_MAX_SECTOR}). "
                        "If any of these are real, you may be among the first to find them."
                    )

                if not _cands.empty:
                    st.markdown("#### Top Candidates")
                    for _, _row in _cands.head(20).iterrows():
                        _tic    = int(_row["tic_id"])
                        _period = float(_row.get("period", 0))
                        _depth  = float(_row.get("depth_ppm", 0))
                        _power  = float(_row.get("bls_power", 0))
                        _sc_raw = pd.to_numeric(_row.get("exominer_score", ""), errors="coerce")
                        _sc     = float(_sc_raw) if pd.notna(_sc_raw) else None
                        _size   = planet_size_estimate(_depth)
                        _dim    = _depth / 10_000

                        col_info, col_score, col_btn = st.columns([4, 2, 1])
                        with col_info:
                            st.markdown(
                                f"**TIC {_tic}** — dims **{_dim:.2f}%** every **{_period:.3f} days** "
                                f"— {_size}  *(BLS power: {_power:.1f})*"
                            )
                        with col_score:
                            if _sc is not None:
                                color   = "green" if _sc >= 0.8 else ("orange" if _sc >= 0.5 else "red")
                                verdict = "likely planet" if _sc >= 0.8 else ("possible" if _sc >= 0.5 else "unlikely")
                                st.markdown(f"ExoMiner++: :{color}[**{_sc:.2f}**] — {verdict}")
                            else:
                                st.caption("not scored by ExoMiner++")
                        with col_btn:
                            if st.button("Inspect", key=f"hunt_insp_{_tic}"):
                                st.session_state["jump_tic"]    = _tic
                                st.session_state["jump_period"] = _period
                                st.session_state["jump_period_source"] = "single_bls_loaded"
                                _t0_num = pd.to_numeric(_row.get("t0", ""), errors="coerce")
                                if pd.notna(_t0_num):
                                    st.session_state["jump_t0"] = float(_t0_num)
                                st.session_state["insp_mode"] = "Search for Candidate"
                                st.session_state["_jump_page"]  = "Lightcurve Inspector"
                                st.rerun()
                else:
                    st.info("No transit signals found above BLS power 7.")

                st.markdown("---")
                if st.button("📊 View all results with transit plots", key="go_scan_results"):
                    st.session_state["_jump_page"] = "Scan Results"
                    st.rerun()

            except Exception as _e:
                st.error(f"Could not load results: {_e}")

    exofop_footer()


# ==============================================================================
# Page 6 — Deep Scan (multi-sector, long-period)
# ==============================================================================
elif page == "Deep Scan":
    st.title("🌌 Deep Scan — Long-Period Planets")
    st.markdown(
        "Stitches all available TESS sectors per star to find planets with "
        "**20–200 day orbits** invisible in single-sector scans."
    )

    edu_expander("Why Deep Scan?", """
Standard BLS scans one 27-day sector at a time, so planets with periods > ~13 days
show fewer than 2 transits and are filtered out. By stitching years of TESS observations,
we can detect:
- **Warm Jupiters** (30–100 day periods)
- **Temperate mini-Neptunes** (50–200 day periods)
- Any planet observed in 5+ sectors

**Trade-offs**: Deep scan takes ~30–60 seconds per star (vs 2 sec for single-sector),
and the search space is much larger. Use `--min-sectors 5` to focus on the best-observed
targets and reduce false positives.
""")

    ds_preset = st.selectbox(
        "Deep Scan preset",
        list(DEEP_SCAN_PRESETS.keys()),
        index=1,
        key="ds_preset",
        help="Presets only populate the controls below. They do not create a separate run type.",
    )
    if st.button("Apply preset", key="ds_apply_preset"):
        for _key, _value in DEEP_SCAN_PRESETS[ds_preset].items():
            st.session_state[_key] = _value
        st.rerun()

    # ── Config ───────────────────────────────────────────────────────────
    with st.expander("⚙️ Deep Scan options"):
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            ds_min_sec = st.slider("Minimum sectors", 3, 15, 5, key="ds_min_sectors",
                                   help="Only process stars observed in this many+ TESS sectors")
        with dc2:
            ds_limit = st.number_input("Star limit (0=all)", 0, 50000, 500, 50, key="ds_limit",
                                       help="0 = all qualifying stars from catalog")
        with dc3:
            ds_workers = st.slider("Workers", 1, 32, 4, key="ds_workers",
                                   help="Auto-raised to 32 when GPU BLS is active (BLS RAM freed).")
        dc4, dc5, dc6 = st.columns(3)
        with dc4:
            ds_min_period = st.number_input(
                "Minimum period (days)", 1.0, 199.0, 20.0, 1.0,
                key="ds_min_period",
                help="Quick validation runs should use a narrower range than the full 1–200 d search.",
            )
        with dc5:
            ds_max_period = st.number_input(
                "Maximum period (days)", 2.0, 200.0, 120.0, 1.0,
                key="ds_max_period",
                help="Wider period ranges are materially slower on stitched multi-sector lightcurves.",
            )
        with dc6:
            ds_target_order = st.selectbox(
                "Target order",
                ["quick-first", "coverage-first"],
                key="ds_target_order",
                format_func=lambda x: "Quick first" if x == "quick-first" else "Coverage first",
                help="Quick first favors cheaper targets so you see completed stars sooner.",
            )
        _gpu_default = os.environ.get("GPU_BLS_URL", "http://192.168.1.163:9876")
        ds_gpu_url = st.text_input(
            "GPU BLS service URL (leave blank to use CPU only)",
            value=_gpu_default,
            key="ds_gpu_url",
            help="Set GPU_BLS_URL to the gpu_bls_service.py host. Workers auto-raise to 32 when active.",
        )
        if ds_gpu_url:
            try:
                import requests as _req
                _gr = _req.get(ds_gpu_url.rstrip("/") + "/health", timeout=2)
                if _gr.ok:
                    st.success(f"GPU service online — {_gr.json()}")
                else:
                    st.warning(f"GPU service returned {_gr.status_code} — will fall back to CPU")
            except Exception:
                st.warning("GPU service unreachable — will fall back to CPU BLS")

    ds_preview_rows, ds_preview_note = preview_deep_scan_targets(ds_min_sec, int(ds_limit), ds_target_order)
    ds_period_valid = ds_min_period < ds_max_period
    if not ds_period_valid:
        st.error("Minimum period must be smaller than maximum period.")
    ds_warning_level, ds_warning_text = deep_scan_cost_warning(
        ds_preview_rows,
        ds_min_period,
        ds_max_period,
        int(ds_limit),
    )
    if ds_warning_level == "success":
        st.success(ds_warning_text)
    elif ds_warning_level == "info":
        st.info(ds_warning_text)
    elif ds_warning_level == "warning":
        st.warning(ds_warning_text)
    else:
        st.error(ds_warning_text)
    if int(ds_limit) == 0 and ds_target_order == "coverage-first":
        st.warning(
            "Coverage-first full campaigns are the heaviest launch mode. "
            "The backend now warms up with lighter targets first, but this is still a more aggressive choice."
        )
    st.caption(ds_preview_note)

    if ds_preview_rows:
        _preview = pd.DataFrame(ds_preview_rows[:10])
        _preview_cols = [c for c in ["tic_id", "n_sectors_seed", "seed_sector_min", "seed_sector_max", "seed_sector_span"] if c in _preview.columns]
        if _preview_cols:
            with st.expander("Preview selected targets", expanded=False):
                st.dataframe(_preview[_preview_cols], use_container_width=True, height=240)

    st.markdown("---")

    # ── Check if running ─────────────────────────────────────────────────
    _global_running, _global_state = scan_is_running()
    _ds_running = _global_running and _global_state.get("mode") == "deep_scan"
    _ds_state   = _global_state if _ds_running else {}
    _active_deep_dir = Path(_ds_state.get("output_dir")) if _ds_state.get("output_dir") else None

    _deep_dir_options = []
    _seen_deep_dirs = set()
    if _active_deep_dir:
        _deep_dir_options.append(_active_deep_dir)
        _seen_deep_dirs.add(str(_active_deep_dir))
    for _cand_dir in list_deep_scan_dirs():
        _key = str(_cand_dir)
        if _key in _seen_deep_dirs:
            continue
        _deep_dir_options.append(_cand_dir)
        _seen_deep_dirs.add(_key)
    if not _deep_dir_options:
        _deep_dir_options = [RESULTS_DIR / "deep_scan"]

    _default_deep_idx = 0
    _chosen_deep_dir = st.selectbox(
        "Deep Scan run directory",
        _deep_dir_options,
        index=_default_deep_idx,
        format_func=deep_scan_dir_label,
        key="ds_results_dir",
        help="New dashboard-launched runs are stored separately so previous deep-scan campaigns are not overwritten.",
    )
    deep_dir = Path(_chosen_deep_dir)
    deep_csv  = deep_dir / "deep_scan_results.csv"
    deep_prog = deep_dir / "deep_scan_progress.json"
    deep_log  = deep_dir / "deep_scan.log"
    deep_plot_dir = deep_dir / "plots"
    st.caption(f"Viewing Deep Scan directory: `{deep_dir}`")
    _deep_summary = deep_scan_dir_summary(deep_dir)
    if _deep_summary:
        st.caption(f"Run label: {_deep_summary}")
    _progress_dir = _active_deep_dir if (_ds_running and _active_deep_dir) else deep_dir
    _progress_prog = _progress_dir / "deep_scan_progress.json"
    _progress_log = _progress_dir / "deep_scan.log"

    if _ds_running:
        _started = _ds_state.get("started_at", "")[:16]
        if _ds_state.get("synthetic"):
            st.warning(f"🟢 **Deep Scan campaign active** — started {_started}")
            st.caption(
                "This campaign is being tracked from merged progress files. "
                "Stop the shard scripts from the shells running them if you need to halt it."
            )
        else:
            col_ds_status, col_ds_stop = st.columns([4, 1])
            with col_ds_status:
                st.warning(f"🟢 **Deep Scan running** — started {_started}")
            with col_ds_stop:
                if st.button("⏹ Stop", key="ds_stop"):
                    _pid = _ds_state.get("pid", 0)
                    if _pid:
                        try:
                            import signal as _sig
                            os.killpg(os.getpgid(_pid), _sig.SIGTERM)
                        except (OSError, ProcessLookupError):
                            pass
                    _ds_state["running"] = False
                    _ds_state["status"] = "failed"
                    _ds_state["finished_at"] = datetime.now().isoformat()
                    write_scan_state(_ds_state)
                    write_job_meta(_ds_state)
                    st.rerun()
    else:
        if _global_running:
            st.info(f"Another job is already running: {_global_state.get('label', _global_state.get('mode', 'job'))}")
        if st.button("🌌 Launch Deep Scan", type="primary", key="ds_launch", disabled=_global_running or not ds_period_valid):
            DEEP_SCAN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
            _run_name = (
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_p{_deep_period_token(ds_min_period)}-{_deep_period_token(ds_max_period)}"
                f"_s{int(ds_min_sec)}"
                f"_{'quick' if ds_target_order == 'quick-first' else 'coverage'}"
                f"_n{'all' if int(ds_limit) <= 0 else int(ds_limit)}"
                f"_w{int(ds_workers)}"
            )
            _run_dir = DEEP_SCAN_RUNS_DIR / _run_name
            _run_dir.mkdir(parents=True, exist_ok=True)
            deep_dir = _run_dir
            _ds_log = deep_dir / "deep_scan.log"
            _ds_cmd = [
                sys.executable, str(SCRIPTS_DIR / "deep_scan.py"),
                "--min-sectors", str(ds_min_sec),
                "--workers",     str(ds_workers),
                "--min-period-days", str(ds_min_period),
                "--max-period-days", str(ds_max_period),
                "--target-order", ds_target_order,
                "--output", str(deep_dir),
            ]
            if ds_limit > 0:
                _ds_cmd += ["--limit", str(ds_limit)]
            _ds_env = {**os.environ}
            if ds_gpu_url:
                _ds_env["GPU_BLS_URL"] = ds_gpu_url.strip()
            elif "GPU_BLS_URL" in _ds_env:
                del _ds_env["GPU_BLS_URL"]
            with open(_ds_log, "w") as _lf:
                _ds_proc = subprocess.Popen(
                    _ds_cmd, stdout=_lf, stderr=subprocess.STDOUT,
                    cwd=SCRIPTS_DIR, start_new_session=True, env=_ds_env,
                )
            write_scan_state({
                "job_id":      make_job_id("deep_scan"),
                "running":    True,
                "mode":       "deep_scan",
                "status":     "running",
                "label":      (
                    f"Deep Scan {_deep_period_token(ds_min_period)}-{_deep_period_token(ds_max_period)}d"
                    f" · min{int(ds_min_sec)}"
                    f" · {'quick' if ds_target_order == 'quick-first' else 'coverage'}"
                ),
                "pid":        _ds_proc.pid,
                "log_file":   str(_ds_log),
                "current_log_file": str(_ds_log),
                "output_dir": str(deep_dir),
                "command":    format_cmd(_ds_cmd),
                "started_at": datetime.now().isoformat(),
            })
            write_job_meta(read_scan_state())
            time.sleep(0.5)
            st.rerun()

    # ── Progress monitor ─────────────────────────────────────────────────
    @st.fragment(run_every=5)
    def ds_progress():
        _p = {}
        if _progress_prog.exists():
            try:
                _p = json.loads(_progress_prog.read_text())
            except Exception:
                pass
        if _p.get("total", 0) > 0:
            done  = _p["done"]
            total = _p["total"]
            st.progress(done / total, text=f"{done}/{total} targets processed — {_p.get('candidates', 0)} signals found")
            if _p.get("current_tic"):
                st.caption(
                    f"Current target: TIC {_p.get('current_tic')}  ·  "
                    f"phase: {_p.get('current_status', 'running')}"
                )
            active = _p.get("active_targets") or []
            if active:
                _active_df = pd.DataFrame(active)
                _active_cols = [c for c in ["tic_id", "phase", "n_sectors", "total_days", "n_points", "bls_sde", "updated_at"] if c in _active_df.columns]
                with st.expander("Active Deep Scan targets", expanded=False):
                    st.dataframe(_active_df[_active_cols], use_container_width=True, height=220)
        if _progress_log.exists():
            _lines = _progress_log.read_text(errors="replace").splitlines()
            _cand_lines = [l for l in _lines if "⚡" in l or "🪐" in l or "TIC" in l or "deep_scan" in l][-20:]
            if _cand_lines:
                with st.expander("Live log", expanded=True):
                    st.code("\n".join(_cand_lines), language="text")

    if _ds_running or _progress_prog.exists():
        ds_progress()

    # ── Results ──────────────────────────────────────────────────────────
    if deep_csv.exists():
        st.markdown("---")
        st.subheader("Deep Scan Results")
        try:
            _ds_df = pd.read_csv(deep_csv)
            n_long = int((_ds_df["period"] >= 30).sum()) if "period" in _ds_df.columns else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Candidates found", len(_ds_df))
            c2.metric("🪐 Long-period (≥30d)", n_long)
            c3.metric("Sectors data used", _ds_df["n_sectors"].max() if "n_sectors" in _ds_df.columns else "—")

            def _ds_highlight(row):
                if row.get("period", 0) >= 30:
                    return ["background-color: rgba(255,215,0,0.12)"] * len(row)
                return [""] * len(row)

            _ds_cols = [c for c in ["tic_id", "n_sectors", "total_days", "period",
                                     "depth_ppm", "bls_power", "snr", "n_transits",
                                     "classification"] if c in _ds_df.columns]
            st.dataframe(
                _ds_df[_ds_cols].style.apply(_ds_highlight, axis=1),
                use_container_width=True,
            )

            # Individual candidate cards
            _ds_strong = _ds_df[_ds_df["bls_power"] >= 8].sort_values("bls_power", ascending=False)
            if not _ds_strong.empty:
                st.markdown("#### Candidates")
                for _, _dr in _ds_strong.iterrows():
                    _dtic    = int(_dr["tic_id"])
                    _dp      = float(_dr["period"])
                    _dd      = float(_dr["depth_ppm"])
                    _dsde    = float(_dr["bls_power"])
                    _dn      = int(_dr.get("n_sectors", 0))
                    _dtot    = float(_dr.get("total_days", 0))
                    _dcls    = str(_dr.get("classification", ""))
                    _long    = _dp >= 30
                    _tag     = "🪐 Long-period" if _long else "⚡"
                    _label   = f"{_tag}  TIC {_dtic}  ·  P={_dp:.3f} d  ·  depth={_dd:.0f} ppm  ·  SDE={_dsde:.1f}"
                    with st.expander(_label, expanded=_long):
                        if _long:
                            st.info(
                                f"This star was observed in **{_dn} TESS sectors** "
                                f"spanning **{_dtot:.0f} days**. "
                                f"A planet orbiting every **{_dp:.1f} days** would transit "
                                f"approximately **{int(_dtot/_dp)} times** in this dataset."
                            )
                        else:
                            st.caption(
                                f"Observed in {_dn} sectors over {_dtot:.0f} days. "
                                f"Period: {_dp:.3f} d."
                            )
                        _dt0 = _dr.get("t0")
                        col_b, col_i = st.columns([3, 1])
                        _deep_png = deep_plot_dir / f"tic_{_dtic}_deep.png"
                        with col_i:
                            st.metric("SDE", f"{_dsde:.1f}")
                            st.metric("Period", f"{_dp:.3f} d")
                            st.metric("Depth", f"{_dd:.0f} ppm")
                            st.metric("Sectors", _dn)
                            if st.button(f"Inspect TIC {_dtic}", key=f"ds_insp_{_dtic}"):
                                st.session_state["jump_tic"]    = _dtic
                                st.session_state["jump_period"] = _dp
                                st.session_state["jump_period_source"] = "stitched_bls"
                                if _dt0 is not None and not pd.isna(_dt0):
                                    st.session_state["jump_t0"] = float(_dt0)
                                st.session_state["insp_mode"] = "Search for Candidate"
                                st.session_state["_jump_page"] = "Lightcurve Inspector"
                                st.rerun()
                        with col_b:
                            if _deep_png.exists():
                                st.image(str(_deep_png), use_container_width=True)
                            else:
                                st.caption("Deep-scan plot not generated yet.")
                                if st.button(f"Render plot for TIC {_dtic}", key=f"ds_plot_{_dtic}"):
                                    with st.spinner(f"Rendering deep-scan plot for TIC {_dtic}..."):
                                        _plot_path, _plot_err = ensure_deep_scan_plot(
                                            tic_id=_dtic,
                                            period=_dp,
                                            t0=safe_float(_dt0),
                                            depth_ppm=_dd,
                                            plot_dir=deep_plot_dir,
                                        )
                                    if _plot_path and _plot_path.exists():
                                        st.image(str(_plot_path), use_container_width=True)
                                    else:
                                        st.warning(_plot_err or "Could not generate the deep-scan plot.")

        except Exception as _e:
            st.error(f"Could not load results: {_e}")

    exofop_footer()
