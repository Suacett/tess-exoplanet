#!/usr/bin/env python3
"""
deep_scan.py — Multi-sector stitched BLS for long-period planets.

Stitches all available TESS sectors for each target, then runs BLS
with a wider period search (1–200 d) to find planets invisible in
single-sector scans.

Usage:
    python deep_scan.py --min-sectors 5 --limit 100 --workers 4
    python deep_scan.py --target-tics 25155310,149603524 --workers 4
"""
import argparse
import base64
from collections import deque
import csv
import hashlib
import json
import multiprocessing as mp
import os
import re
import signal
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import requests as _requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZENODO_DIR = DATA_DIR / "zenodo"
RESULTS_DIR = DATA_DIR / "results"
SCRIPTS_DIR = Path(__file__).parent

LABELED_CATALOG = ZENODO_DIR / "exominerplusplus_catalog_labeled_tces.csv"
UNLABELED_CATALOG = ZENODO_DIR / "exominerplusplus_catalog_unk_tces.csv"
TARGET_CACHE = DATA_DIR / "deep_scan_targets.json"

# GPU BLS service — opt-in. Set GPU_BLS_URL=http://<host>:9876 to enable.
_GPU_URL = os.environ.get("GPU_BLS_URL")  # None = CPU-only (default)

# BLS parameters — wider than single-sector scan
BLS_PERIOD_MIN = 1.0      # days
BLS_PERIOD_MAX = 200.0    # days
BLS_DUR_MIN = 0.05        # days
BLS_DUR_MAX = 0.30        # days
BLS_DUR_STEP = 0.01       # days
BLS_FREQ_FACTORS = (2, 5, 10, 20)

# Quality filters
SDE_THRESHOLD = 8.0
PLOT_THRESHOLD = 8.0
MAX_DEPTH_PPM = 20_000
MIN_DEPTH_PPM = 50
MIN_N_TRANSITS = 3
MIN_POINTS = 500
MIN_TOTAL_DAYS = 30
DEFAULT_WORKERS = 4
DEFAULT_TARGET_ORDER = "quick-first"

TESS_LOCAL_DIR = DATA_DIR / "tess"


def _load_local_lcs(tic_id: int):
    """Load all locally cached TESS FITS files for a TIC.
    Returns (LightCurveCollection, [sector_ints]) or (None, []).
    """
    import lightkurve as lk
    pattern = f"*-{tic_id:016d}-*_lc.fits"
    lcs, sectors = [], []
    for fits_path in sorted(TESS_LOCAL_DIR.glob(f"sector*/{pattern}")):
        try:
            lc = lk.read(str(fits_path), quality_bitmask="default")
            sec = getattr(lc, "sector", None)
            if sec is None:
                sec = lc.meta.get("SECTOR") or lc.meta.get("sector")
            if sec is None:
                sec = int(fits_path.parent.name.replace("sector", ""))
            lcs.append(lc)
            sectors.append(int(sec))
        except Exception:
            pass
    if not lcs:
        return None, []
    return lk.LightCurveCollection(lcs), sorted(sectors)
DEFAULT_MAX_TARGET_MINUTES = 0.0

CANDIDATE_FIELDS = [
    "tic_id",
    "n_sectors",
    "total_days",
    "period",
    "depth_ppm",
    "duration_hours",
    "bls_sde",
    "bls_power",
    "snr",
    "t0",
    "n_transits",
    "classification",
]

UID_SECTOR_RE = re.compile(r"-S(\d+)(?:-(\d+))?$")
LOCAL_FITS_RE = re.compile(r"-s(\d{4})-(\d{16})-.*_lc\.fits$")
HEARTBEAT_SECONDS = 5.0
SESSION_POISON_RETRY_LIMIT = 1


def log(msg: str) -> None:
    print(f"[deep_scan] {msg}", flush=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def short_exc(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:220] if text else exc.__class__.__name__


def is_session_poison_error_text(text: str) -> bool:
    msg = str(text or "").lower()
    return (
        "session is in the kill state" in msg
        or "severe error occurred on the current command" in msg
    )


class TargetTimeoutError(RuntimeError):
    pass


class target_timeout:
    def __init__(self, seconds: float):
        self.seconds = max(0.0, float(seconds))
        self._enabled = hasattr(signal, "SIGALRM") and self.seconds > 0
        self._prev_handler = None

    def _handle(self, signum, frame):
        raise TargetTimeoutError("target runtime limit exceeded")

    def __enter__(self):
        if not self._enabled:
            return self
        self._prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._handle)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return False
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._prev_handler)
        return False


def write_worker_status(status_dir: Path | None, tic_id: int, **payload) -> None:
    if status_dir is None:
        return
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{tic_id}.json"
    payload = dict(payload)
    payload["tic_id"] = tic_id
    payload["updated_at"] = now_iso()
    write_json(path, payload)


def read_worker_statuses(status_dir: Path) -> list[dict]:
    if not status_dir.exists():
        return []
    rows = []
    for path in sorted(status_dir.glob("*.json")):
        data = load_json(path, {})
        if data:
            rows.append(data)
    return rows


def parse_sector_run(value: str) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = text.split("-")
    try:
        if len(parts) == 1:
            sec = int(parts[0])
            return [sec]
        start = int(parts[0])
        end = int(parts[1])
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))
    except ValueError:
        return []


def discover_targets_from_labeled() -> dict | None:
    if not LABELED_CATALOG.exists():
        return None

    log(f"Target discovery source: {LABELED_CATALOG}")
    sector_counts: dict[int, set[int]] = {}
    bad_rows = 0
    total_rows = 0

    with LABELED_CATALOG.open(newline="") as fh:
        reader = csv.DictReader(fh)
        log(f"Labeled catalog columns: {', '.join(reader.fieldnames or [])}")
        for row in reader:
            total_rows += 1
            tic = parse_int(row.get("target_id"))
            uid = str(row.get("uid", "")).strip()
            match = UID_SECTOR_RE.search(uid)
            if tic <= 0 or not match:
                bad_rows += 1
                continue
            start = int(match.group(1))
            end = int(match.group(2) or start)
            sector_counts.setdefault(tic, set()).update(range(start, end + 1))

    targets = []
    for tic, sectors in sector_counts.items():
        ordered = sorted(sectors)
        targets.append({
            "tic_id": tic,
            "n_sectors_seed": len(ordered),
            "seed_sector_min": ordered[0],
            "seed_sector_max": ordered[-1],
            "seed_sector_span": ordered[-1] - ordered[0],
        })
    targets.sort(key=lambda row: row["n_sectors_seed"], reverse=True)
    log(
        f"Labeled catalog parsed: {len(targets)} TICs, {bad_rows} rows skipped "
        f"out of {total_rows} rows."
    )
    return {
        "generated_at": now_iso(),
        "source_used": "zenodo_labeled_uid",
        "catalog_path": str(LABELED_CATALOG),
        "total_rows": total_rows,
        "bad_rows": bad_rows,
        "targets": targets,
    }


def discover_targets_from_unlabeled() -> dict | None:
    if not UNLABELED_CATALOG.exists():
        return None

    log(f"Target discovery fallback: {UNLABELED_CATALOG}")
    sector_counts: dict[int, set[int]] = {}
    bad_rows = 0
    total_rows = 0

    with UNLABELED_CATALOG.open(newline="") as fh:
        reader = csv.DictReader(fh)
        log(f"Unlabeled catalog columns: {', '.join(reader.fieldnames or [])}")
        for row in reader:
            total_rows += 1
            tic = parse_int(row.get("TIC ID"))
            sectors = parse_sector_run(row.get("Sector Run", ""))
            if tic <= 0 or not sectors:
                bad_rows += 1
                continue
            sector_counts.setdefault(tic, set()).update(sectors)

    targets = []
    for tic, sectors in sector_counts.items():
        ordered = sorted(sectors)
        targets.append({
            "tic_id": tic,
            "n_sectors_seed": len(ordered),
            "seed_sector_min": ordered[0],
            "seed_sector_max": ordered[-1],
            "seed_sector_span": ordered[-1] - ordered[0],
        })
    targets.sort(key=lambda row: row["n_sectors_seed"], reverse=True)
    log(
        f"Unlabeled catalog parsed: {len(targets)} TICs, {bad_rows} rows skipped "
        f"out of {total_rows} rows."
    )
    return {
        "generated_at": now_iso(),
        "source_used": "zenodo_unlabeled_sector_run",
        "catalog_path": str(UNLABELED_CATALOG),
        "total_rows": total_rows,
        "bad_rows": bad_rows,
        "targets": targets,
    }


def discover_targets_from_local() -> dict | None:
    tess_dir = DATA_DIR / "tess"
    if not tess_dir.exists():
        return None

    log(f"Target discovery last fallback: {tess_dir}")
    sector_counts: dict[int, set[int]] = {}
    files_seen = 0

    for sector_dir in sorted(tess_dir.glob("sector*")):
        if not sector_dir.is_dir():
            continue
        for fits_path in sector_dir.glob("*_lc.fits"):
            files_seen += 1
            match = LOCAL_FITS_RE.search(fits_path.name)
            if not match:
                continue
            sector = int(match.group(1))
            tic = int(match.group(2))
            sector_counts.setdefault(tic, set()).add(sector)

    targets = []
    for tic, sectors in sector_counts.items():
        ordered = sorted(sectors)
        targets.append({
            "tic_id": tic,
            "n_sectors_seed": len(ordered),
            "seed_sector_min": ordered[0],
            "seed_sector_max": ordered[-1],
            "seed_sector_span": ordered[-1] - ordered[0],
        })
    targets.sort(key=lambda row: row["n_sectors_seed"], reverse=True)
    log(f"Local FITS parsed: {len(targets)} TICs from {files_seen} files.")
    return {
        "generated_at": now_iso(),
        "source_used": "local_downloaded_fits",
        "catalog_path": str(tess_dir),
        "total_rows": files_seen,
        "bad_rows": 0,
        "targets": targets,
    }


def load_or_build_target_cache(refresh_targets: bool = False) -> dict:
    if TARGET_CACHE.exists() and not refresh_targets:
        cache = load_json(TARGET_CACHE, {})
        targets = cache.get("targets") or []
        cache_ok = bool(targets) and all(
            "seed_sector_span" in row and "n_sectors_seed" in row
            for row in targets[:10]
        )
        if cache_ok:
            log(
                f"Using cached target list from {TARGET_CACHE} "
                f"({cache.get('source_used', 'unknown source')})."
            )
            return cache
        if targets:
            log(
                f"Rebuilding target cache at {TARGET_CACHE} because the cached rows "
                "are missing cost metadata needed for Deep Scan ordering."
            )

    for builder in (
        discover_targets_from_labeled,
        discover_targets_from_unlabeled,
        discover_targets_from_local,
    ):
        cache = builder()
        if cache and cache.get("targets"):
            write_json(TARGET_CACHE, cache)
            log(
                f"Cached {len(cache['targets'])} target seeds to {TARGET_CACHE} "
                f"from {cache['source_used']}."
            )
            return cache

    raise RuntimeError("Unable to discover deep-scan targets from Zenodo or local FITS.")


def parse_target_tics(raw: str) -> list[int]:
    tics: list[int] = []
    for chunk in str(raw or "").split(","):
        tic = parse_int(chunk.strip())
        if tic > 0:
            tics.append(tic)
    # Preserve order while deduplicating.
    return list(dict.fromkeys(tics))


def target_list_signature(tics: list[int]) -> str:
    payload = "\n".join(str(parse_int(tic)) for tic in tics if parse_int(tic) > 0)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_target_file(path: Path) -> tuple[list[int], str]:
    tics: list[int] = []
    for line in path.read_text().splitlines():
        tic = parse_int(line.strip())
        if tic > 0:
            tics.append(tic)
    ordered = list(dict.fromkeys(tics))
    return ordered, target_list_signature(ordered)


def normalize_run_config(config: dict | None) -> dict:
    cfg = dict(config or {})
    cfg.setdefault("min_sectors", 5)
    cfg.setdefault("limit", 0)
    cfg.setdefault("target_tics", [])
    cfg.setdefault("target_file_sha256", "")
    cfg.setdefault("target_count", 0)
    cfg.setdefault("worker_name", "")
    cfg.setdefault("workers", 0)
    cfg.setdefault("display_name", "")
    cfg.setdefault("min_period_days", BLS_PERIOD_MIN)
    cfg.setdefault("max_period_days", BLS_PERIOD_MAX)
    cfg.setdefault("max_target_minutes", DEFAULT_MAX_TARGET_MINUTES)
    cfg.setdefault("target_order", DEFAULT_TARGET_ORDER)
    cfg.setdefault("source_used", "unknown")
    cfg["min_sectors"] = parse_int(cfg.get("min_sectors"), 5)
    cfg["limit"] = parse_int(cfg.get("limit"), 0)
    cfg["target_file_sha256"] = str(cfg.get("target_file_sha256", "") or "")
    cfg["target_count"] = parse_int(cfg.get("target_count"), 0)
    cfg["worker_name"] = str(cfg.get("worker_name", "") or "")
    cfg["workers"] = parse_int(cfg.get("workers"), 0)
    cfg["display_name"] = str(cfg.get("display_name", "") or "")
    cfg["min_period_days"] = parse_float(cfg.get("min_period_days"), BLS_PERIOD_MIN)
    cfg["max_period_days"] = parse_float(cfg.get("max_period_days"), BLS_PERIOD_MAX)
    cfg["max_target_minutes"] = parse_float(
        cfg.get("max_target_minutes"), DEFAULT_MAX_TARGET_MINUTES
    )
    cfg["target_order"] = str(
        cfg.get("target_order", DEFAULT_TARGET_ORDER) or DEFAULT_TARGET_ORDER
    )
    cfg["source_used"] = str(cfg.get("source_used", "unknown") or "unknown")
    cfg["target_tics"] = parse_target_tics(",".join(str(t) for t in (cfg.get("target_tics") or [])))
    return cfg


def order_targets(targets: list[dict], target_order: str) -> list[dict]:
    if target_order == "quick-first":
        return sorted(
            targets,
            key=lambda row: (
                parse_int(row.get("n_sectors_seed"), 999999),
                parse_int(row.get("seed_sector_span"), 999999),
                parse_int(row.get("seed_sector_max"), 999999),
                parse_int(row.get("tic_id"), 999999999),
            ),
        )
    return sorted(
        targets,
        key=lambda row: (
            -parse_int(row.get("n_sectors_seed")),
            -parse_int(row.get("seed_sector_span")),
            parse_int(row.get("tic_id")),
        ),
    )


def apply_startup_guardrail(
    targets: list[dict],
    target_order: str,
    *,
    explicit_tics: list[int] | None,
    workers: int,
    limit: int | None,
) -> list[dict]:
    if explicit_tics:
        return targets
    if target_order != "coverage-first":
        return targets
    if len(targets) <= max(2, workers):
        return targets
    if limit is not None and limit > 0 and limit <= max(2, workers):
        return targets

    startup_count = min(len(targets), max(workers * 2, 32))
    warm_rows = order_targets(list(targets), "quick-first")[:startup_count]
    warm_ids = {parse_int(row.get("tic_id")) for row in warm_rows}
    remaining = [row for row in targets if parse_int(row.get("tic_id")) not in warm_ids]
    log(
        f"Startup guardrail: warming with {len(warm_rows)} lighter targets first "
        f"before continuing coverage-first ordering."
    )
    return warm_rows + remaining


def find_multi_sector_targets(
    min_sectors: int = 5,
    limit: int | None = None,
    target_tics: list[int] | None = None,
    refresh_targets: bool = False,
    target_order: str = DEFAULT_TARGET_ORDER,
) -> tuple[list[dict], dict]:
    if target_tics:
        seed_map: dict[int, int] = {}
        cache = {
            "generated_at": now_iso(),
            "source_used": "target_file",
            "catalog_path": "",
            "targets": [],
        }
        if TARGET_CACHE.exists():
            try:
                cache = load_json(TARGET_CACHE, cache)
                seed_map = {
                    int(row["tic_id"]): int(row.get("n_sectors_seed", 0))
                    for row in cache.get("targets", [])
                }
            except Exception:
                seed_map = {}
        targets = [
            {
                "tic_id": tic,
                "n_sectors_seed": parse_int(seed_map.get(tic, 0)),
            }
            for tic in target_tics
        ]
        if limit:
            targets = targets[:limit]
        log(
            f"Using explicit target override for {len(targets)} TICs; "
            f"skipping catalog-based ranking."
        )
        return targets, cache

    cache = load_or_build_target_cache(refresh_targets=refresh_targets)
    seed_map = {
        int(row["tic_id"]): int(row.get("n_sectors_seed", 0))
        for row in cache.get("targets", [])
    }

    targets = [
        row
        for row in cache.get("targets", [])
        if parse_int(row.get("n_sectors_seed")) >= min_sectors
    ]
    targets = order_targets(targets, target_order)

    if limit:
        targets = targets[:limit]

    log(
        f"Target filter: {len(targets)} TICs meet min_sectors >= {min_sectors} "
        f"from {len(seed_map)} discovered TICs "
        f"(order: {target_order})."
    )
    return targets, cache


def select_best_spoc_products(search_result):
    if len(search_result) == 0:
        return search_result, []

    table = search_result.table
    colnames = set(table.colnames)
    best_by_sector: dict[int, tuple[float, int]] = {}
    sectorless: list[int] = []

    for idx, row in enumerate(table):
        sector = row["sequence_number"]
        if sector in (None, "", np.ma.masked):
            sectorless.append(idx)
            continue
        try:
            sector_num = int(sector)
        except (TypeError, ValueError):
            sectorless.append(idx)
            continue

        if "exptime" in colnames:
            exptime = parse_float(row["exptime"], default=1e9)
        elif "t_exptime" in colnames:
            exptime = parse_float(row["t_exptime"], default=1e9)
        else:
            exptime = 1e9
        current = best_by_sector.get(sector_num)
        if current is None or exptime < current[0]:
            best_by_sector[sector_num] = (exptime, idx)

    indices = [best_by_sector[sec][1] for sec in sorted(best_by_sector)]
    if not indices:
        indices = sectorless or list(range(len(search_result)))
        sectors = []
    else:
        sectors = sorted(best_by_sector)

    return search_result[indices], sectors


def should_retry_bls(exc: Exception) -> bool:
    text = short_exc(exc).lower()
    return any(
        needle in text
        for needle in (
            "periodogram is too large",
            "period contains",
            "too large to evaluate",
            "too many periods",
            "memory",
        )
    )


def run_bls(lc_flat, durations, min_period_days: float, max_period_days: float):
    base_kwargs = dict(
        method="bls",
        minimum_period=min_period_days,
        maximum_period=max_period_days,
        duration=durations,
    )
    last_exc = None
    for i, freq_factor in enumerate(BLS_FREQ_FACTORS):
        kwargs = dict(base_kwargs)
        kwargs["frequency_factor"] = freq_factor
        try:
            return lc_flat.to_periodogram(**kwargs), freq_factor
        except Exception as exc:
            last_exc = exc
            if i == len(BLS_FREQ_FACTORS) - 1 or not should_retry_bls(exc):
                raise
    raise last_exc


def process_deep(task: dict) -> dict:
    tic_id = int(task["tic_id"])
    seed_sectors = int(task.get("n_sectors_seed", 0))
    min_sectors = int(task.get("min_sectors", 5))
    min_period_days = parse_float(task.get("min_period_days"), BLS_PERIOD_MIN)
    max_period_days = parse_float(task.get("max_period_days"), BLS_PERIOD_MAX)
    max_target_minutes = parse_float(task.get("max_target_minutes"), DEFAULT_MAX_TARGET_MINUTES)
    retry_count = int(task.get("retry_count", 0))
    status_dir = Path(task["status_dir"]) if task.get("status_dir") else None
    current_phase = "starting"

    def heartbeat(phase: str, **extra) -> None:
        nonlocal current_phase
        current_phase = phase
        write_worker_status(
            status_dir,
            tic_id,
            phase=phase,
            n_sectors_seed=seed_sectors,
            retry_count=retry_count,
            **extra,
        )

    try:
        with target_timeout(max_target_minutes * 60.0):
            warnings.filterwarnings("ignore")
            import lightkurve as lk
            from astropy import units as u

            heartbeat("searching")
            lcs, sectors = _load_local_lcs(tic_id)
            n_sectors = len(lcs) if lcs is not None else 0

            if lcs is None or n_sectors < min_sectors:
                # Not enough local data — fall back to MAST
                sr = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS",
                    author="SPOC",
                )
                if len(sr) == 0:
                    return {
                        "tic_id": tic_id,
                        "status": "skipped",
                        "reason": "no_spoc_lightcurves",
                        "n_sectors_seed": seed_sectors,
                        "retry_count": retry_count,
                    }

                sr, sectors = select_best_spoc_products(sr)
                n_available = len(sr)
                heartbeat("found_search_products", actual_sectors=sectors, n_available=n_available)
                if n_available < min_sectors:
                    return {
                        "tic_id": tic_id,
                        "status": "skipped",
                        "reason": f"actual_sectors_below_min ({n_available} < {min_sectors})",
                        "n_sectors_seed": seed_sectors,
                        "n_sectors": n_available,
                        "retry_count": retry_count,
                    }

                heartbeat("downloading", n_available=n_available)
                lcs = sr.download_all(quality_bitmask="default")
                if lcs is None or len(lcs) == 0:
                    return {
                        "tic_id": tic_id,
                        "status": "error",
                        "error": "download_all returned no lightcurves",
                        "n_sectors_seed": seed_sectors,
                        "retry_count": retry_count,
                    }

                n_sectors = len(lcs)
                heartbeat("downloaded", n_sectors=n_sectors)
                if n_sectors < min_sectors:
                    return {
                        "tic_id": tic_id,
                        "status": "skipped",
                        "reason": f"downloaded_sectors_below_min ({n_sectors} < {min_sectors})",
                        "n_sectors_seed": seed_sectors,
                        "n_sectors": n_sectors,
                        "retry_count": retry_count,
                    }
            else:
                heartbeat("found_local", actual_sectors=sectors, n_available=n_sectors)

            heartbeat("stitching", n_sectors=n_sectors)
            lc_raw = lcs.stitch() if len(lcs) > 1 else lcs[0]
            total_days = float(lc_raw.time[-1].value - lc_raw.time[0].value)
            heartbeat("stitched", n_sectors=n_sectors, total_days=round(total_days, 1))
            if total_days < MIN_TOTAL_DAYS:
                return {
                    "tic_id": tic_id,
                    "status": "skipped",
                    "reason": f"time_span_too_short ({total_days:.1f} d)",
                    "n_sectors_seed": seed_sectors,
                    "n_sectors": n_sectors,
                    "total_days": round(total_days, 1),
                    "retry_count": retry_count,
                }

            heartbeat("flattening", n_sectors=n_sectors, total_days=round(total_days, 1))
            lc_flat = (
                lc_raw.normalize()
                .flatten(window_length=401)
                .remove_outliers(sigma=4)
            )
            n_points = len(lc_flat)
            heartbeat("flattened", n_points=n_points, n_sectors=n_sectors, total_days=round(total_days, 1))
            if n_points < MIN_POINTS:
                return {
                    "tic_id": tic_id,
                    "status": "skipped",
                    "reason": f"too_few_points ({n_points})",
                    "n_sectors_seed": seed_sectors,
                    "n_sectors": n_sectors,
                    "total_days": round(total_days, 1),
                    "retry_count": retry_count,
                }

            # --- GPU BLS path (opt-in via GPU_BLS_URL) ---
            _gpu_bls_result = None
            if _GPU_URL:
                try:
                    _resp = _requests.post(
                        _GPU_URL + "/bls_deep",
                        json={
                            "tic_id":     tic_id,
                            "time":       lc_flat.time.value.tolist(),
                            "flux":       lc_flat.flux.value.tolist(),
                            "period_min": min_period_days,
                            "period_max": max_period_days,
                            "oversample": 50,
                        },
                        timeout=120,
                    )
                    if _resp.ok:
                        _gpu_bls_result = _resp.json()
                except Exception:
                    pass  # silent fallback to CPU BLS
            # -----------------------------------------------

            if _gpu_bls_result is not None:
                heartbeat("bls_complete", frequency_factor=0, n_points=n_points,
                          n_sectors=n_sectors, total_days=round(total_days, 1))
                best_p      = float(_gpu_bls_result["period"])
                best_t0     = float(_gpu_bls_result["t0"])
                sde         = float(_gpu_bls_result["sde"])
                freq_factor = 0  # sentinel: GPU used
            else:
                durations = np.arange(BLS_DUR_MIN, BLS_DUR_MAX + 1e-9, BLS_DUR_STEP)
                heartbeat("running_bls", n_points=n_points, n_sectors=n_sectors,
                          total_days=round(total_days, 1))
                blsm, freq_factor = run_bls(lc_flat, durations, min_period_days, max_period_days)
                heartbeat(
                    "bls_complete",
                    frequency_factor=freq_factor,
                    n_points=n_points,
                    n_sectors=n_sectors,
                    total_days=round(total_days, 1),
                )
                best_p  = float(blsm.period_at_max_power.value)
                best_t0 = float(blsm.transit_time_at_max_power.value)
                power_arr = np.asarray(blsm.power.value, dtype=float)
                p_mean    = float(np.nanmean(power_arr))
                p_std     = float(np.nanstd(power_arr))
                sde       = (float(blsm.max_power) - p_mean) / p_std if p_std > 0 else 0.0

            if sde < SDE_THRESHOLD:
                return {
                    "tic_id": tic_id,
                    "status": "scanned_no_candidate",
                    "reason": f"sde_below_threshold ({sde:.2f} < {SDE_THRESHOLD:.1f})",
                    "n_sectors_seed": seed_sectors,
                    "n_sectors": n_sectors,
                    "total_days": round(total_days, 1),
                    "n_points": n_points,
                    "frequency_factor": freq_factor,
                    "retry_count": retry_count,
                }

            heartbeat("folding", period=round(best_p, 5), bls_sde=round(sde, 4))
            lc_fold = lc_flat.fold(period=best_p * u.day, epoch_time=best_t0 * u.day)
            lc_bin = lc_fold.bin(time_bin_size=0.005)
            flux = np.ma.filled(np.asarray(lc_bin.flux.value), fill_value=np.nan).astype(float)
            phase = np.asarray(lc_bin.phase.value, dtype=float)

            search_half = min(0.20, best_p * 0.15)
            cen_mask = np.abs(phase) < search_half
            if not cen_mask.any():
                return {
                    "tic_id": tic_id,
                    "status": "scanned_no_candidate",
                    "reason": "no_phase_points_near_transit",
                    "n_sectors": n_sectors,
                    "total_days": round(total_days, 1),
                    "n_points": n_points,
                    "frequency_factor": freq_factor,
                    "retry_count": retry_count,
                }

            transit_min = float(np.nanmin(flux[cen_mask]))
            baseline_mask = np.abs(phase) > min(0.25, best_p * 0.20)
            baseline_flux = flux[baseline_mask] if baseline_mask.any() else flux
            baseline = float(np.nanmedian(baseline_flux))
            depth_ppm = (
                max(0.0, (baseline - transit_min) / baseline * 1e6)
                if baseline > 0
                else 0.0
            )
            if depth_ppm > MAX_DEPTH_PPM or depth_ppm < MIN_DEPTH_PPM:
                return {
                    "tic_id": tic_id,
                    "status": "scanned_no_candidate",
                    "reason": f"depth_out_of_range ({depth_ppm:.1f} ppm)",
                    "n_sectors": n_sectors,
                    "total_days": round(total_days, 1),
                    "n_points": n_points,
                    "frequency_factor": freq_factor,
                    "retry_count": retry_count,
                }

            half_lev = baseline - (baseline - transit_min) * 0.5
            below_half = np.sum(flux[cen_mask] < half_lev) * 0.005
            dur_hours = below_half * best_p * 24

            n_transits = max(1, int(total_days / best_p))
            if n_transits < MIN_N_TRANSITS:
                return {
                    "tic_id": tic_id,
                    "status": "scanned_no_candidate",
                    "reason": f"too_few_transits ({n_transits})",
                    "n_sectors": n_sectors,
                    "total_days": round(total_days, 1),
                    "n_points": n_points,
                    "frequency_factor": freq_factor,
                    "retry_count": retry_count,
                }

            flux_std = float(np.nanstd(flux))
            snr = depth_ppm / (flux_std * 1e6) if flux_std > 0 else 0.0

            if depth_ppm <= 10_000 and sde >= 9 and best_p >= 20:
                classification = "Long-period candidate"
            elif sde >= 9:
                classification = "Planet candidate"
            else:
                classification = "Needs inspection"

            heartbeat(
                "candidate",
                classification=classification,
                period=round(best_p, 5),
                depth_ppm=round(depth_ppm, 1),
                bls_sde=round(sde, 4),
                n_transits=n_transits,
            )

            return {
                "tic_id": tic_id,
                "status": "candidate",
                "n_sectors_seed": seed_sectors,
                "n_sectors": n_sectors,
                "total_days": round(total_days, 1),
                "period": round(best_p, 5),
                "depth_ppm": round(depth_ppm, 1),
                "duration_hours": round(dur_hours, 3),
                "bls_sde": round(sde, 4),
                "bls_power": round(sde, 4),
                "snr": round(snr, 2),
                "t0": round(best_t0, 5),
                "n_transits": n_transits,
                "classification": classification,
                "frequency_factor": freq_factor,
                "n_points": n_points,
                "actual_sectors": sectors,
                "retry_count": retry_count,
            }
    except TargetTimeoutError:
        error = (
            f"target_timeout ({max_target_minutes:.1f} min cap exceeded during {current_phase})"
        )
        heartbeat("timeout", error=error)
        return {
            "tic_id": tic_id,
            "status": "error",
            "error": error,
            "n_sectors_seed": seed_sectors,
            "retry_count": retry_count,
        }
    except Exception as exc:
        heartbeat("error", error=short_exc(exc))
        return {
            "tic_id": tic_id,
            "status": "error",
            "error": short_exc(exc),
            "n_sectors_seed": seed_sectors,
            "retry_count": retry_count,
        }


def load_candidate_rows(out_csv: Path) -> list[dict]:
    if not out_csv.exists():
        return []

    rows: list[dict] = []
    with out_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["tic_id"] = parse_int(row.get("tic_id"))
            row["n_sectors"] = parse_int(row.get("n_sectors"))
            row["total_days"] = parse_float(row.get("total_days"))
            row["period"] = parse_float(row.get("period"))
            row["depth_ppm"] = parse_float(row.get("depth_ppm"))
            row["duration_hours"] = parse_float(row.get("duration_hours"))
            row["bls_sde"] = parse_float(row.get("bls_sde", row.get("bls_power")))
            row["bls_power"] = parse_float(row.get("bls_power", row.get("bls_sde")))
            row["snr"] = parse_float(row.get("snr"))
            row["t0"] = parse_float(row.get("t0"))
            row["n_transits"] = parse_int(row.get("n_transits"))
            row["classification"] = str(row.get("classification", "")).strip()
            rows.append(row)
    return rows


def write_candidate_rows(out_csv: Path, rows: list[dict]) -> None:
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_candidate_row(out_csv: Path, row: dict) -> None:
    exists = out_csv.exists() and out_csv.stat().st_size > 0
    with out_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CANDIDATE_FIELDS})


def load_resume_state(prog_file: Path, out_csv: Path, run_config: dict) -> tuple[dict, list[dict]] | None:
    progress = load_json(prog_file, {})
    if not progress or progress.get("complete"):
        return None
    progress_cfg = normalize_run_config(progress.get("run_config"))
    current_cfg = normalize_run_config(run_config)
    # Timeout policy is operational, not scientific; allow resume across cap changes.
    progress_cfg.pop("max_target_minutes", None)
    current_cfg.pop("max_target_minutes", None)
    if progress_cfg != current_cfg:
        return None
    processed = progress.get("processed_tics")
    if not isinstance(processed, list):
        return None
    return progress, load_candidate_rows(out_csv)


def write_progress(
    prog_file: Path,
    *,
    done: int,
    total: int,
    candidates: int,
    rate: float,
    complete: bool,
    processed_tics: set[int],
    failed_tics: dict[str, str],
    current_tic: int | None,
    current_status: str,
    last_error: str,
    target_source: str,
    target_cache_path: Path,
    run_config: dict,
    active_targets: list[dict] | None = None,
) -> None:
    payload = {
        "done": done,
        "total": total,
        "candidates": candidates,
        "errors": len(failed_tics),
        "rate": round(rate, 2),
        "complete": complete,
        "processed_tics": sorted(processed_tics),
        "failed_tics": failed_tics,
        "current_tic": current_tic,
        "current_status": current_status,
        "last_error": last_error,
        "target_source": target_source,
        "target_cache": str(target_cache_path),
        "run_config": run_config,
        "updated_at": now_iso(),
    }
    if active_targets is not None:
        payload["active_targets"] = active_targets
    write_json(prog_file, payload)


def generate_plots(candidates: list[dict], plot_dir: Path) -> list[Path]:
    if not candidates:
        return []

    sys.path.insert(0, str(SCRIPTS_DIR))
    from plot_candidate import make_4panel
    import lightkurve as lk
    import warnings as warn_mod

    warn_mod.filterwarnings("ignore")
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    strong = [row for row in candidates if parse_float(row.get("bls_sde")) >= PLOT_THRESHOLD]
    if not strong:
        return generated

    log(f"Generating {len(strong)} deep-scan plots (SDE >= {PLOT_THRESHOLD:.1f})...")
    for row in strong:
        tic_id = int(row["tic_id"])
        out_png = plot_dir / f"tic_{tic_id}_deep.png"
        try:
            lcs, _ = _load_local_lcs(tic_id)
            if lcs is None:
                sr = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS",
                    author="SPOC",
                )
                if len(sr) == 0:
                    continue
                sr, _ = select_best_spoc_products(sr)
                lcs = sr.download_all(quality_bitmask="default")
                if lcs is None or len(lcs) == 0:
                    continue
            lc_raw = lcs.stitch() if len(lcs) > 1 else lcs[0]
            lc_flat = (
                lc_raw.normalize()
                .flatten(window_length=401)
                .remove_outliers(sigma=4)
            )
            make_4panel(
                lc_raw=lc_raw,
                lc_flat=lc_flat,
                period_d=parse_float(row["period"]),
                t0_btjd=parse_float(row["t0"]),
                tic_id=tic_id,
                sector=f"stitched ({parse_int(row['n_sectors'])} sectors)",
                depth_ppm=parse_float(row["depth_ppm"]),
                bls_power=parse_float(row.get("bls_sde", row.get("bls_power"))),
                out_path=str(out_png),
            )
            generated.append(out_png)
            log(f"  ✓ TIC {tic_id} -> {out_png.name}")
        except Exception as exc:
            log(f"  ✗ TIC {tic_id}: plot generation failed ({short_exc(exc)})")

    return generated


def generate_html_report(
    candidates: list[dict],
    n_targets: int,
    elapsed: float,
    plot_dir: Path,
    out_html: Path,
) -> None:
    def b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode()

    rows = ""
    for i, row in enumerate(candidates[:100], 1):
        sde = parse_float(row.get("bls_sde", row.get("bls_power")))
        power_class = "high" if sde >= 12 else "med" if sde >= PLOT_THRESHOLD else "low"
        long_tag = "🪐" if parse_float(row.get("period")) > 20 else "⚡"
        rows += f"""
        <tr class="{power_class}">
          <td>{i}</td>
          <td><a href="#tic{row['tic_id']}">TIC {row['tic_id']}</a></td>
          <td>{long_tag}</td>
          <td>{parse_float(row['period']):.5f}</td>
          <td>{parse_float(row['depth_ppm']):.1f}</td>
          <td>{parse_float(row['duration_hours']):.2f}</td>
          <td><b>{sde:.2f}</b></td>
          <td>{parse_float(row['snr']):.2f}</td>
          <td>{row['classification']}</td>
        </tr>"""

    cards = ""
    for row in candidates:
        sde = parse_float(row.get("bls_sde", row.get("bls_power")))
        if sde < PLOT_THRESHOLD:
            continue
        png = plot_dir / f"tic_{row['tic_id']}_deep.png"
        img_tag = (
            f'<img src="data:image/png;base64,{b64(png)}" width="100%">'
            if png.exists() else "<p><i>Plot not generated</i></p>"
        )
        priority = "🪐 LONG-PERIOD" if parse_float(row["period"]) > 20 else "⚡ CANDIDATE"
        cards += f"""
        <div class="card" id="tic{row['tic_id']}">
          <h2>{priority} &nbsp; TIC {row['tic_id']}</h2>
          <p>Period: <b>{parse_float(row['period']):.5f} d</b> &nbsp;|&nbsp;
             Depth: <b>{parse_float(row['depth_ppm']):.0f} ppm</b> &nbsp;|&nbsp;
             SDE: <b>{sde:.2f}</b> &nbsp;|&nbsp;
             Duration: <b>{parse_float(row['duration_hours']):.2f} h</b> &nbsp;|&nbsp;
             Sectors: <b>{parse_int(row['n_sectors'])}</b></p>
          {img_tag}
        </div>"""

    n_long = sum(1 for row in candidates if parse_float(row["period"]) > 20)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Deep Scan Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif;
          padding: 24px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  .meta {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 16px 24px; text-align: center; min-width: 160px; }}
  .stat .val {{ font-size: 32px; font-weight: bold; color: #58a6ff; }}
  .stat .lab {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  h2.sec {{ color: #c9d1d9; margin: 32px 0 12px; border-bottom: 1px solid #30363d;
             padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 600; padding: 10px 12px;
        text-align: left; border-bottom: 1px solid #30363d; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr.high td {{ background: rgba(255,123,114,0.08); }}
  tr.med td {{ background: rgba(88,166,255,0.06); }}
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
<h1>🌌 Deep Scan Report</h1>
<p class="meta">Generated on scan completion &nbsp;|&nbsp; {n_targets:,} TICs processed in {elapsed/60:.1f} min</p>
<div class="stats">
  <div class="stat"><div class="val">{n_targets:,}</div><div class="lab">Targets Processed</div></div>
  <div class="stat"><div class="val">{len(candidates):,}</div><div class="lab">Candidates</div></div>
  <div class="stat"><div class="val">{n_long:,}</div><div class="lab">Period &gt; 20 d</div></div>
  <div class="stat"><div class="val">{elapsed/60:.1f} min</div><div class="lab">Runtime</div></div>
</div>

<h2 class="sec">Candidate Table (ranked by SDE)</h2>
<table>
  <thead>
    <tr><th>#</th><th>TIC ID</th><th>Type</th><th>Period (d)</th>
        <th>Depth (ppm)</th><th>Duration (h)</th><th>SDE</th><th>SNR</th><th>Classification</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<h2 class="sec">Candidate Plots (SDE ≥ {PLOT_THRESHOLD:.1f})</h2>
{cards if cards else '<p style="color:#8b949e">No strong deep-scan signals detected.</p>'}

</body>
</html>
"""
    out_html.write_text(html)
    log(f"HTML report: {out_html}")


def summarise_result(result: dict) -> str:
    status = result["status"]
    tic = result["tic_id"]
    if status == "candidate":
        tag = "🪐" if "Long-period" in result["classification"] else "⚡"
        return (
            f"{tag} TIC {tic}  P={parse_float(result['period']):.3f}d  "
            f"depth={parse_float(result['depth_ppm']):.0f}ppm  "
            f"SDE={parse_float(result.get('bls_sde', result.get('bls_power'))):.1f}  "
            f"{result['classification']}"
        )
    if status == "error":
        return f"✗ TIC {tic}  error={result.get('error', 'unknown error')}"
    return f"· TIC {tic}  skipped ({result.get('reason', status)})"


def summarize_active_targets(status_dir: Path, pending_ids: set[int]) -> list[dict]:
    rows = []
    for data in read_worker_statuses(status_dir):
        tic_id = parse_int(data.get("tic_id"))
        if tic_id <= 0 or tic_id not in pending_ids:
            continue
        rows.append({
            "tic_id": tic_id,
            "phase": str(data.get("phase", "working")),
            "updated_at": str(data.get("updated_at", "")),
            "n_sectors": parse_int(data.get("n_sectors")),
            "n_points": parse_int(data.get("n_points")),
            "total_days": parse_float(data.get("total_days")),
            "period": parse_float(data.get("period")),
            "bls_sde": parse_float(data.get("bls_sde")),
        })
    rows.sort(key=lambda row: (row["phase"], row["tic_id"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Multi-sector deep BLS scan")
    ap.add_argument("--min-sectors", type=int, default=5)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max targets to process (0=all qualifying targets)",
    )
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--min-period-days", type=float, default=BLS_PERIOD_MIN)
    ap.add_argument("--max-period-days", type=float, default=BLS_PERIOD_MAX)
    ap.add_argument(
        "--max-target-minutes",
        type=float,
        default=DEFAULT_MAX_TARGET_MINUTES,
        help="Per-target runtime cap in minutes. Use 0 to disable timeouts.",
    )
    ap.add_argument(
        "--target-order",
        choices=("coverage-first", "quick-first"),
        default=DEFAULT_TARGET_ORDER,
        help="Coverage-first favors the deepest multi-sector targets; quick-first favors cheaper targets.",
    )
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--refresh-targets", action="store_true")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any incomplete progress file and start this output directory fresh",
    )
    ap.add_argument(
        "--target-tics",
        type=str,
        default="",
        help="Comma-separated TIC IDs for a deterministic smoke test",
    )
    ap.add_argument(
        "--target-file",
        type=str,
        default="",
        help="Path to a plain-text target list (one TIC ID per line).",
    )
    ap.add_argument(
        "--worker-name",
        type=str,
        default="",
        help="Optional worker label recorded in run metadata (for shard/campaign runs).",
    )
    args = ap.parse_args()
    if args.min_period_days < 1.0 or args.max_period_days > BLS_PERIOD_MAX:
        raise SystemExit(f"Period range must stay within 1.0-{BLS_PERIOD_MAX:.0f} days.")
    if args.min_period_days >= args.max_period_days:
        raise SystemExit("Minimum period must be smaller than maximum period.")
    if args.target_tics and args.target_file:
        raise SystemExit("Use either --target-tics or --target-file, not both.")

    out_dir = Path(args.output) if args.output else RESULTS_DIR / "deep_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "deep_scan_results.csv"
    prog_file = out_dir / "deep_scan_progress.json"
    out_html = out_dir / "deep_scan_report.html"
    plot_dir = out_dir / "plots"
    status_dir = out_dir / "worker_status"

    target_file_sha = ""
    explicit_tics = parse_target_tics(args.target_tics)
    target_source = ""
    if args.target_file:
        target_file_path = Path(args.target_file)
        explicit_tics, target_file_sha = load_target_file(target_file_path)
        target_source = "target_file"
        log(
            f"Using target file {target_file_path} with {len(explicit_tics)} TICs "
            "as the explicit Deep Scan target list."
        )
    elif explicit_tics:
        target_source = "target_tics"
    log(f"Finding targets with >= {args.min_sectors} sectors...")
    targets, cache = find_multi_sector_targets(
        min_sectors=args.min_sectors,
        limit=args.limit if args.limit > 0 else None,
        target_tics=explicit_tics,
        refresh_targets=args.refresh_targets,
        target_order=args.target_order,
    )
    if not target_source:
        target_source = cache.get("source_used", "unknown")
    order_label = "quick" if str(args.target_order) == "quick-first" else "coverage"
    limit_label = "all" if int(args.limit) <= 0 else str(int(args.limit))
    display_name = (
        f"Deep Scan {args.min_period_days:g}-{args.max_period_days:g}d"
        f" · min{int(args.min_sectors)}"
        f" · {order_label}"
        f" · {limit_label} targets"
        f" · w{int(args.workers)}"
    )
    run_config = {
        "min_sectors": int(args.min_sectors),
        "limit": int(args.limit),
        "target_tics": explicit_tics,
        "target_file_sha256": target_file_sha,
        "target_count": len(targets),
        "worker_name": str(args.worker_name or ""),
        "workers": int(args.workers),
        "display_name": display_name,
        "min_period_days": float(args.min_period_days),
        "max_period_days": float(args.max_period_days),
        "max_target_minutes": float(args.max_target_minutes),
        "target_order": str(args.target_order),
        "source_used": target_source,
    }
    log(f"{len(targets)} targets to process")

    if not targets:
        log("No targets found. Check Zenodo catalog / target cache.")
        write_progress(
            prog_file,
            done=0,
            total=0,
            candidates=0,
            rate=0.0,
            complete=True,
            processed_tics=set(),
            failed_tics={},
            current_tic=None,
            current_status="no_targets",
            last_error="",
            target_source=target_source,
            target_cache_path=TARGET_CACHE,
            run_config=run_config,
        )
        return

    resume = None if args.fresh else load_resume_state(prog_file, out_csv, run_config)
    if resume:
        prev_progress, candidates = resume
        processed_tics = {parse_int(t) for t in prev_progress.get("processed_tics", [])}
        failed_tics = {
            str(parse_int(tic)): str(reason)
            for tic, reason in (prev_progress.get("failed_tics") or {}).items()
        }
        if float(args.max_target_minutes) <= 0:
            timed_out = {
                parse_int(tic)
                for tic, reason in failed_tics.items()
                if str(reason).startswith("target_timeout")
            }
            if timed_out:
                processed_tics.difference_update(timed_out)
                for tic in timed_out:
                    failed_tics.pop(str(tic), None)
                log(
                    f"Requeueing {len(timed_out)} previously timed-out targets because "
                    "target timeouts are disabled for this run."
                )
        log(
            f"Resuming incomplete run: {len(processed_tics)} targets already processed, "
            f"{len(candidates)} candidate rows already written."
        )
    else:
        processed_tics = set()
        failed_tics = {}
        candidates = []
        write_candidate_rows(out_csv, [])
        if status_dir.exists():
            shutil.rmtree(status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)

    pending_targets = []
    for row in targets:
        tic_id = int(row["tic_id"])
        if tic_id in processed_tics:
            continue
        pending_targets.append(
            {
                "tic_id": tic_id,
                "n_sectors_seed": int(row.get("n_sectors_seed", 0)),
                "min_sectors": int(args.min_sectors),
                "min_period_days": float(args.min_period_days),
                "max_period_days": float(args.max_period_days),
                "max_target_minutes": float(args.max_target_minutes),
                "retry_count": 0,
                "status_dir": str(status_dir),
            }
        )

    pending_targets = apply_startup_guardrail(
        pending_targets,
        args.target_order,
        explicit_tics=explicit_tics,
        workers=max(1, min(int(args.workers), os.cpu_count() or DEFAULT_WORKERS)),
        limit=int(args.limit),
    )

    n_total = len(targets)
    log(
        f"Pending targets this run: {len(pending_targets)} "
        f"(source: {target_source})."
    )
    if not pending_targets:
        log("All requested targets are already complete for this output directory.")
        candidates.sort(key=lambda row: parse_float(row.get("bls_sde", row.get("bls_power"))), reverse=True)
        write_candidate_rows(out_csv, candidates)
        if not args.no_plots:
            generate_plots(candidates, plot_dir)
        generate_html_report(candidates, len(processed_tics), 0.0, plot_dir, out_html)
        write_progress(
            prog_file,
            done=n_total,
            total=n_total,
            candidates=len(candidates),
            rate=0.0,
            complete=True,
            processed_tics=processed_tics,
            failed_tics=failed_tics,
            current_tic=None,
            current_status="complete",
            last_error="",
            target_source=target_source,
            target_cache_path=TARGET_CACHE,
            run_config=run_config,
            active_targets=[],
        )
        return

    write_progress(
        prog_file,
        done=len(processed_tics),
        total=n_total,
        candidates=len(candidates),
        rate=0.0,
        complete=False,
        processed_tics=processed_tics,
        failed_tics=failed_tics,
        current_tic=None,
        current_status="starting",
        last_error="",
        target_source=target_source,
        target_cache_path=TARGET_CACHE,
        run_config=run_config,
        active_targets=[],
    )

    workers = max(1, min(int(args.workers), os.cpu_count() or DEFAULT_WORKERS))

    _gpu_alive = False
    if _GPU_URL:
        try:
            _r = _requests.get(_GPU_URL + "/health", timeout=3)
            _gpu_alive = _r.ok
        except Exception:
            pass
        log(f"GPU BLS service: {'online — BLS offloaded to GPU' if _gpu_alive else 'offline — using CPU BLS'}")
        if _gpu_alive and int(args.workers) <= DEFAULT_WORKERS:
            workers = min(32, os.cpu_count() or 32)
            log(f"GPU active: auto-raising workers to {workers} (BLS RAM freed)")

    log(f"Using {workers} workers")
    log("Worker/session isolation enabled: each TIC runs in a fresh worker process.")

    t0_run = time.time()
    pending_ids = {int(row["tic_id"]) for row in pending_targets}
    task_queue = deque(pending_targets)

    def submit_one(pool, inflight):
        if not task_queue:
            return False
        task = task_queue.popleft()
        inflight.append((pool.apply_async(process_deep, (task,)), task))
        return True

    with mp.Pool(processes=workers, maxtasksperchild=1) as pool:
        inflight: list[tuple[object, dict]] = []
        while len(inflight) < workers and submit_one(pool, inflight):
            pass

        last_progress_write = 0.0
        while inflight or task_queue:
            completed: list[tuple[object, dict]] = []
            for async_result, task in inflight:
                if async_result.ready():
                    completed.append((async_result, task))

            for async_result, task in completed:
                inflight.remove((async_result, task))
                try:
                    result = async_result.get()
                except Exception as exc:
                    result = {
                        "tic_id": int(task["tic_id"]),
                        "status": "error",
                        "error": short_exc(exc),
                        "retry_count": int(task.get("retry_count", 0)),
                    }
                tic_id = int(result["tic_id"])
                retry_count = int(task.get("retry_count", 0))
                last_error = ""
                error_text = str(result.get("error", ""))
                poison_error = result["status"] == "error" and is_session_poison_error_text(error_text)

                if poison_error and retry_count < SESSION_POISON_RETRY_LIMIT:
                    retried_task = dict(task)
                    retried_task["retry_count"] = retry_count + 1
                    task_queue.appendleft(retried_task)
                    write_worker_status(
                        status_dir,
                        tic_id,
                        phase="retry_queued",
                        error=error_text,
                        retry_count=retried_task["retry_count"],
                    )
                    log(
                        f"✗ TIC {tic_id}  worker/session poisoned; replacing worker and retrying once "
                        "on a fresh worker."
                    )
                    while len(inflight) < workers and submit_one(pool, inflight):
                        pass
                    continue

                pending_ids.discard(tic_id)
                processed_tics.add(tic_id)
                failed_tics.pop(str(tic_id), None)

                if poison_error:
                    result["error"] = f"{error_text} (failed after fresh-worker retry)"
                    failed_tics[str(tic_id)] = result["error"]
                    last_error = result["error"]
                    log(
                        f"✗ TIC {tic_id}  worker/session poisoned again after retry; marking failed."
                    )
                elif result["status"] == "candidate":
                    candidate_row = {field: result.get(field) for field in CANDIDATE_FIELDS}
                    candidates.append(candidate_row)
                    append_candidate_row(out_csv, candidate_row)
                elif result["status"] == "error":
                    failed_tics[str(tic_id)] = error_text or "unknown error"
                    last_error = failed_tics[str(tic_id)]

                log(summarise_result(result))

                elapsed = time.time() - t0_run
                rate = len(processed_tics) / elapsed if elapsed > 0 else 0.0
                active_targets = summarize_active_targets(status_dir, pending_ids)
                write_progress(
                    prog_file,
                    done=len(processed_tics),
                    total=n_total,
                    candidates=len(candidates),
                    rate=rate,
                    complete=False,
                    processed_tics=processed_tics,
                    failed_tics=failed_tics,
                    current_tic=tic_id,
                    current_status=result["status"],
                    last_error=last_error,
                    target_source=target_source,
                    target_cache_path=TARGET_CACHE,
                    run_config=run_config,
                    active_targets=active_targets,
                )

                while len(inflight) < workers and submit_one(pool, inflight):
                    pass

            now = time.time()
            if (inflight or task_queue) and now - last_progress_write >= HEARTBEAT_SECONDS:
                elapsed = now - t0_run
                rate = len(processed_tics) / elapsed if elapsed > 0 else 0.0
                active_targets = summarize_active_targets(status_dir, pending_ids)
                current_tic = active_targets[0]["tic_id"] if active_targets else None
                current_status = active_targets[0]["phase"] if active_targets else "running"
                write_progress(
                    prog_file,
                    done=len(processed_tics),
                    total=n_total,
                    candidates=len(candidates),
                    rate=rate,
                    complete=False,
                    processed_tics=processed_tics,
                    failed_tics=failed_tics,
                    current_tic=current_tic,
                    current_status=current_status,
                    last_error="",
                    target_source=target_source,
                    target_cache_path=TARGET_CACHE,
                    run_config=run_config,
                    active_targets=active_targets,
                )
                last_progress_write = now

            if inflight or task_queue:
                time.sleep(1.0)

    elapsed_total = time.time() - t0_run

    candidates.sort(
        key=lambda row: parse_float(row.get("bls_sde", row.get("bls_power"))),
        reverse=True,
    )
    write_candidate_rows(out_csv, candidates)

    if not args.no_plots:
        generate_plots(candidates, plot_dir)
    generate_html_report(candidates, len(processed_tics), elapsed_total, plot_dir, out_html)

    log(
        f"Complete: {len(candidates)} candidates from {len(processed_tics)} processed targets "
        f"in {elapsed_total / 60:.1f} min"
    )
    log(f"Results: {out_csv}")

    write_progress(
        prog_file,
        done=len(processed_tics),
        total=n_total,
        candidates=len(candidates),
        rate=(len(processed_tics) / elapsed_total) if elapsed_total > 0 else 0.0,
        complete=True,
        processed_tics=processed_tics,
        failed_tics=failed_tics,
        current_tic=None,
        current_status="complete",
        last_error="",
        target_source=target_source,
        target_cache_path=TARGET_CACHE,
        run_config=run_config,
        active_targets=[],
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
