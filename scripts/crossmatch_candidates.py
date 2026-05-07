#!/usr/bin/env python3
"""Small public-catalog cross-match helper for exoplanet candidates.

Primary sources, in order:
1. NASA Exoplanet Archive confirmed planets (pscomppars)
2. NASA Exoplanet Archive TOI table (updated from ExoFOP-TESS)
3. ExoFOP-TESS target page text for alias resolution and caution notes
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import signal
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "crossmatch"
SHORTLIST_DIR = CACHE_DIR / "shortlists"
CACHE_TTL_DAYS = 7
CACHE_VERSION = 1
HTTP_TIMEOUT_S = 10
PER_TIC_TIMEOUT_S = 25
TOI_BATCH_TIMEOUT_S = 20
TOI_BATCH_SPLIT_THRESHOLD = 10
FAST_TOP_N_PER_SECTOR = 25
FAST_GLOBAL_LIMIT = 1200
PARTIAL_WRITE_EVERY = 500

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOFOP_URL = "https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"

STATUS_LABELS = {
    "confirmed_planet": "Confirmed planet",
    "known_toi": "Known TOI",
    "suspicious_alert_object": "Caution",
    "not_obviously_known": "Unknown",
    "lookup_failed": "Lookup failed",
}

SUSPICIOUS_DISPOSITIONS = {"FP", "FA", "V", "IS", "EB", "O"}
KNOWN_CANDIDATE_DISPOSITIONS = {"PC", "CP", "KP", "APC"}
PUBLIC_DISPOSITION_NOTES = {
    "FP": "Public TOI disposition is false positive.",
    "FA": "Public TOI disposition flags this as a false alarm.",
    "V": "Public TOI disposition says the signal looks like stellar variability.",
    "IS": "Public TOI disposition says the signal looks instrumental.",
    "EB": "Public TOI disposition says this looks like an eclipsing binary.",
    "O": "Public TOI disposition says this is another known non-planet case.",
    "PC": "Public TOI disposition is planet candidate.",
    "CP": "Public TOI disposition marks this as a confirmed planet.",
    "KP": "Public TOI disposition marks this as a known planet.",
    "APC": "Public TOI disposition says this is an active planet candidate.",
}


def safe_print(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except (ValueError, OSError):
        pass


class ProgressTracker:
    def __init__(self, path: Path | None, rows_total: int = 0, unique_tics_total: int = 0, output_path: Path | None = None):
        self.path = path
        self.state = {
            "status": "running",
            "phase": "Starting",
            "started_at": now_iso(),
            "last_updated": now_iso(),
            "rows_total": int(rows_total),
            "rows_processed": 0,
            "unique_tics_total": int(unique_tics_total),
            "unique_tics_ready": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "current_tic": None,
            "current_source": None,
            "output_path": str(output_path) if output_path else None,
            "progress_total": int(rows_total) + int(unique_tics_total),
            "progress_done": 0,
            "failed_tics": 0,
            "skipped_tics": 0,
            "last_error": None,
            "build_mode": "full",
            "cheap_pass_rows": 0,
            "deep_lookup_rows": 0,
            "deep_lookup_done": 0,
        }
        self._flush()

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_updated"] = now_iso()
        self.path.write_text(json.dumps(self.state, indent=2))

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        if "progress_total" not in self.state:
            self.state["progress_total"] = self.state.get("rows_total", 0) + self.state.get("unique_tics_total", 0)
        self._flush()

    def finish(self, status: str = "completed", **kwargs) -> None:
        self.state.update(kwargs)
        self.state["status"] = status
        self.state["finished_at"] = now_iso()
        if status == "completed":
            self.state["progress_done"] = self.state.get("progress_total", self.state.get("progress_done", 0))
            self.state["rows_processed"] = self.state.get("rows_total", self.state.get("rows_processed", 0))
            self.state["unique_tics_ready"] = self.state.get("unique_tics_total", self.state.get("unique_tics_ready", 0))
        self._flush()

    def log(self, message: str) -> None:
        safe_print(f"[crossmatch] {message}")


def safe_float(value):
    try:
        if value in ("", None):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def safe_int(value):
    try:
        if value in ("", None):
            return None
        return int(value)
    except Exception:
        return None



def parse_tic_id(value) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return int(str(value).split("-")[0])
        except (ValueError, TypeError):
            return None


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sql_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def cache_path_for_tic(tic_id: int | str) -> Path:
    return CACHE_DIR / f"tic_{parse_tic_id(tic_id)}.json"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SHORTLIST_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(tic_id: int | str) -> dict | None:
    path = cache_path_for_tic(tic_id)
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def cache_is_fresh(cache: dict | None, max_age_days: int = CACHE_TTL_DAYS) -> bool:
    if not cache:
        return False
    try:
        fetched = datetime.fromisoformat(cache["fetched_at"].replace("Z", "+00:00"))
    except Exception:
        return False
    return (datetime.now(fetched.tzinfo) - fetched) <= timedelta(days=max_age_days)


def save_cache(payload: dict) -> None:
    ensure_dirs()
    path = cache_path_for_tic(payload["tic_id"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def metadata_path_for_output(path: Path | None) -> Path | None:
    if not path:
        return None
    return path.with_name(path.name + ".meta.json")


def http_get_text(url: str, timeout: int = HTTP_TIMEOUT_S) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "exoplanet-homelab/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="ignore")


def tap_query(sql: str, timeout: int = HTTP_TIMEOUT_S) -> list[dict]:
    url = TAP_URL + "?query=" + urllib.parse.quote(sql) + "&format=json"
    text = http_get_text(url, timeout=timeout)
    return json.loads(text)


class TicResolutionTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TicResolutionTimeout("per-TIC public lookup timed out")


class tic_timeout:
    def __init__(self, seconds: int):
        self.seconds = int(seconds)
        self.previous_handler = None
        self._active = False

    def __enter__(self):
        if self.seconds > 0 and threading.current_thread() is threading.main_thread():
            self.previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self.previous_handler is not None:
                signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", raw_html)
    text = re.sub(r"(?i)</li>", "\n", text)
    text = re.sub(r"(?i)</div>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def extract_confirmed_planet_names(raw_html: str, text: str) -> list[str]:
    names: list[str] = []
    patterns = [
        r"Confirmed Planet\(s\):\s*([^<\n]+)",
        r"Confirmed Planet:\s*([^<\n]+)",
    ]
    for pat in patterns:
        for match in re.finditer(pat, raw_html, flags=re.IGNORECASE):
            chunk = match.group(1).strip()
            for part in re.split(r",|/|;", chunk):
                name = " ".join(part.split())
                if name:
                    names.append(name)
    if not names:
        for line in text.splitlines():
            if "Confirmed Planet" in line:
                maybe = line.split(":", 1)[-1].strip()
                for part in re.split(r",|/|;", maybe):
                    name = " ".join(part.split())
                    if name and name.lower() not in {"confirmed planet(s)", "confirmed planet"}:
                        names.append(name)
    seen = set()
    out = []
    for name in names:
        key = name.lower()
        if key not in seen:
            out.append(name)
            seen.add(key)
    return out


def extract_host_candidates(confirmed_names: Iterable[str]) -> list[str]:
    hosts = []
    for name in confirmed_names:
        host = re.sub(r"\s+[b-z]\Z", "", name.strip(), flags=re.IGNORECASE)
        if host and host != name:
            hosts.append(host)
    seen = set()
    out = []
    for host in hosts:
        key = host.lower()
        if key not in seen:
            out.append(host)
            seen.add(key)
    return out


def extract_caution_notes(raw_html: str, text: str) -> list[str]:
    notes: list[str] = []
    comment_hits = re.findall(r'"comment":"([^"]{1,220})"', raw_html, flags=re.IGNORECASE)
    keyword_map = [
        ("v-shaped", "Public note mentions a V-shaped signal."),
        ("odd-even", "Public note mentions odd/even mismatch."),
        ("odd even", "Public note mentions odd/even mismatch."),
        ("secondary eclipse", "Public note mentions a possible secondary eclipse."),
        ("centroid", "Public note mentions a centroid concern."),
        ("contamination", "Public note mentions contamination or blending risk."),
        ("nearby star", "Public note mentions a nearby-star contamination risk."),
        ("eclipsing binary", "Public note mentions eclipsing-binary risk."),
    ]
    for raw_comment in comment_hits:
        comment = html.unescape(raw_comment).strip()
        lower = comment.lower()
        matched = False
        for needle, note in keyword_map:
            if needle in lower:
                notes.append(note)
                matched = True
        if matched:
            notes.append(f"Public note: {comment}")
    seen = set()
    out = []
    for note in notes:
        key = note.lower()
        if key not in seen:
            out.append(note)
            seen.add(key)
    return out


def choose_best_match(rows: list[dict], local_period: float | None, period_key: str) -> dict | None:
    if not rows:
        return None
    if local_period is None:
        return rows[0]
    with_period = []
    for row in rows:
        p = safe_float(row.get(period_key))
        if p and p > 0:
            with_period.append((abs(p - local_period), row))
    if with_period:
        with_period.sort(key=lambda item: item[0])
        return with_period[0][1]
    return rows[0]


def period_diff(local_period: float | None, matched_period: float | None) -> tuple[float | None, float | None]:
    if local_period is None or matched_period is None or matched_period == 0:
        return None, None
    diff_days = abs(local_period - matched_period)
    diff_pct = 100.0 * diff_days / abs(matched_period)
    return diff_pct, diff_days


def fetch_toi_rows(tic_ids: list[int]) -> dict[int, list[dict]]:
    rows, _ = fetch_toi_rows_resilient(tic_ids)
    return rows


def fetch_toi_rows_resilient(
    tic_ids: list[int],
    tracker: ProgressTracker | None = None,
) -> tuple[dict[int, list[dict]], set[int]]:
    if not tic_ids:
        return {}, set()

    out = {tic: [] for tic in tic_ids}
    failed: set[int] = set()

    def fetch_chunk(chunk: list[int]) -> None:
        if not chunk:
            return
        in_list = ",".join(str(int(t)) for t in chunk)
        sql = (
            "select toi,tid,tfopwg_disp,pl_orbper,ctoi_alias "
            f"from toi where tid in ({in_list}) order by tid,toi"
        )
        try:
            for row in tap_query(sql, timeout=TOI_BATCH_TIMEOUT_S):
                tid = int(row["tid"])
                out.setdefault(tid, []).append(row)
        except Exception as exc:
            short_reason = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__
            if len(chunk) <= TOI_BATCH_SPLIT_THRESHOLD:
                failed.update(int(t) for t in chunk)
                if tracker:
                    tracker.update(last_error=f"TOI query failed for {len(chunk)} TICs: {short_reason}")
                    tracker.log(
                        f"TOI query failed for TICs {chunk[0]}-{chunk[-1]} ({len(chunk)} TICs): {short_reason}. "
                        "Continuing with ExoFOP-only resolution for these TICs."
                    )
                return
            mid = len(chunk) // 2
            if tracker:
                tracker.update(last_error=f"TOI query retry split after error: {short_reason}")
                tracker.log(
                    f"TOI query failed for TICs {chunk[0]}-{chunk[-1]} ({len(chunk)} TICs): {short_reason}. "
                    "Retrying in smaller chunks."
                )
            fetch_chunk(chunk[:mid])
            fetch_chunk(chunk[mid:])

    for start in range(0, len(tic_ids), 100):
        fetch_chunk(tic_ids[start:start + 100])

    return out, failed


def fetch_confirmed_rows_for_names(planet_names: list[str], host_names: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for name in planet_names:
        sql = (
            "select pl_name,hostname,pl_orbper "
            f"from pscomppars where pl_name = {sql_quote(name)}"
        )
        for row in tap_query(sql):
            key = (row.get("pl_name"), row.get("hostname"), row.get("pl_orbper"))
            if key not in seen:
                rows.append(row)
                seen.add(key)
    for host in host_names:
        sql = (
            "select pl_name,hostname,pl_orbper "
            f"from pscomppars where hostname = {sql_quote(host)}"
        )
        for row in tap_query(sql):
            key = (row.get("pl_name"), row.get("hostname"), row.get("pl_orbper"))
            if key not in seen:
                rows.append(row)
                seen.add(key)
    return rows


def build_remote_payload(
    tic_id: int,
    toi_rows: list[dict] | None = None,
    need_exofop: bool = False,
) -> dict:
    payload = {
        "cache_version": CACHE_VERSION,
        "tic_id": int(tic_id),
        "fetched_at": now_iso(),
        "toi_rows": toi_rows or [],
        "confirmed_rows": [],
        "confirmed_names": [],
        "host_names": [],
        "caution_notes": [],
        "errors": [],
        "toi_lookup_done": toi_rows is not None,
        "exofop_lookup_done": False,
        "confirmed_lookup_done": False,
    }
    if not need_exofop:
        return payload
    try:
        raw_html = http_get_text(EXOFOP_URL.format(tic_id=int(tic_id)))
        text = html_to_text(raw_html)
        confirmed_names = extract_confirmed_planet_names(raw_html, text)
        host_names = extract_host_candidates(confirmed_names)
        caution_notes = extract_caution_notes(raw_html, text)
        confirmed_rows = fetch_confirmed_rows_for_names(confirmed_names, host_names) if confirmed_names or host_names else []
        payload.update(
            {
                "confirmed_rows": confirmed_rows,
                "confirmed_names": confirmed_names,
                "host_names": host_names,
                "caution_notes": caution_notes,
                "exofop_lookup_done": True,
                "confirmed_lookup_done": True,
            }
        )
    except Exception as exc:
        payload["errors"].append(f"ExoFOP/confirmed lookup failed: {exc}")
    return payload


def build_failure_payload(tic_id: int, reason: str, kind: str, toi_rows: list[dict] | None = None) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "tic_id": int(tic_id),
        "fetched_at": now_iso(),
        "toi_rows": toi_rows or [],
        "confirmed_rows": [],
        "confirmed_names": [],
        "host_names": [],
        "caution_notes": [],
        "errors": [reason],
        "toi_lookup_done": toi_rows is not None,
        "exofop_lookup_done": False,
        "confirmed_lookup_done": False,
        "forced_lookup_failed": True,
        "failure_kind": kind,
    }


def needs_exofop_lookup(toi_rows: list[dict], detailed: bool) -> bool:
    if detailed:
        return True
    for row in toi_rows:
        disp = str(row.get("tfopwg_disp", "")).upper()
        if disp in {"CP", "KP"}:
            return True
    return False


def ensure_cache_for_tics(
    tic_ids: Iterable[int],
    detailed: bool = False,
    max_age_days: int = CACHE_TTL_DAYS,
    tracker: ProgressTracker | None = None,
) -> dict:
    ensure_dirs()
    tics = sorted({t for t in (parse_tic_id(raw) for raw in tic_ids if raw not in ("", None)) if t is not None})
    stale = []
    fresh_hits = 0
    total_tics = len(tics)
    for idx, tic in enumerate(tics, start=1):
        cache = load_cache(tic)
        if cache_is_fresh(cache, max_age_days=max_age_days):
            fresh_hits += 1
        else:
            stale.append(tic)
        if tracker and (idx == total_tics or idx % 500 == 0):
            tracker.update(
                phase="Checking cache",
                current_tic=tic,
                cache_hits=fresh_hits,
                cache_misses=len(stale),
                unique_tics_ready=fresh_hits,
                progress_done=fresh_hits,
            )
            tracker.log(
                f"Checking cache: {idx}/{total_tics} TICs scanned, {fresh_hits} fresh cache hits, {len(stale)} need public lookup."
            )
    if tracker:
        tracker.update(
            phase="Checking cache",
            cache_hits=fresh_hits,
            cache_misses=len(stale),
            unique_tics_ready=fresh_hits,
            progress_done=fresh_hits,
            current_tic=None,
            current_source="cache",
        )
        tracker.log(f"Cache scan complete: {fresh_hits} cache hits, {len(stale)} TICs need public lookup.")
    if not stale:
        return {"cache_hits": fresh_hits, "cache_misses": 0, "unique_tics_total": len(tics)}
    chunk_size = 100
    total_chunks = math.ceil(len(stale) / chunk_size)
    completed_stale = 0
    failed_tics = 0
    for chunk_idx, start in enumerate(range(0, len(stale), chunk_size), start=1):
        chunk = stale[start:start + chunk_size]
        chunk_failures = 0
        if tracker:
            tracker.update(
                phase=f"Fetching public matches (chunk {chunk_idx}/{total_chunks})",
                current_tic=chunk[0] if chunk else None,
                current_source="TOI batch query",
            )
            tracker.log(
                f"Fetching TOI chunk {chunk_idx}/{total_chunks} for TICs {chunk[0]}-{chunk[-1]} ({len(chunk)} TICs)."
            )
        toi_map, toi_batch_failed = fetch_toi_rows_resilient(chunk, tracker=tracker)
        toi_batch_failed = {int(t) for t in toi_batch_failed}
        if tracker:
            if toi_batch_failed:
                tracker.log(
                    f"TOI chunk {chunk_idx}/{total_chunks} fetched with {len(toi_batch_failed)} TICs falling back "
                    "to ExoFOP-only resolution."
                )
            else:
                tracker.log(f"TOI chunk {chunk_idx}/{total_chunks} fetched. Resolving and caching TICs...")
        for tic in chunk:
            toi_rows = toi_map.get(tic, [])
            try:
                if tracker:
                    tracker.update(current_tic=tic, current_source="ExoFOP / confirmed resolution")
                with tic_timeout(PER_TIC_TIMEOUT_S):
                    payload = build_remote_payload(tic, toi_rows=toi_rows, need_exofop=needs_exofop_lookup(toi_rows, detailed))
                if tic in toi_batch_failed:
                    payload.setdefault("errors", []).append(
                        "TOI batch query failed; resolved using ExoFOP/confirmed sources only."
                    )
                    payload["toi_lookup_failed"] = True
                save_cache(payload)
            except TicResolutionTimeout as exc:
                payload = build_failure_payload(tic, f"Timeout during public resolution: {exc}", "timeout", toi_rows=toi_rows)
                save_cache(payload)
                failed_tics += 1
                chunk_failures += 1
                if tracker:
                    tracker.update(
                        failed_tics=failed_tics,
                        skipped_tics=failed_tics,
                        last_error=f"TIC {tic}: timeout during public resolution",
                    )
                    tracker.log(f"Skipping TIC {tic} after timeout during ExoFOP resolution.")
            except Exception as exc:
                short_reason = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__
                payload = build_failure_payload(tic, f"Public resolution failed: {short_reason}", "lookup_failed", toi_rows=toi_rows)
                save_cache(payload)
                failed_tics += 1
                chunk_failures += 1
                if tracker:
                    tracker.update(
                        failed_tics=failed_tics,
                        skipped_tics=failed_tics,
                        last_error=f"TIC {tic}: {short_reason}",
                    )
                    tracker.log(f"Skipping TIC {tic} after parse/lookup error: {short_reason}")
            completed_stale += 1
            if tracker and (completed_stale == len(stale) or completed_stale % 25 == 0):
                tracker.update(
                    phase=f"Fetching public matches (chunk {chunk_idx}/{total_chunks})",
                    current_tic=tic,
                    unique_tics_ready=fresh_hits + completed_stale,
                    progress_done=fresh_hits + completed_stale,
                    current_source="ExoFOP / confirmed resolution",
                )
                tracker.log(
                    f"Public matches ready for {completed_stale}/{len(stale)} uncached TICs. Current TIC {tic}."
                )
        if tracker:
            chunk_successes = len(chunk) - chunk_failures
            tracker.log(
                f"Finished TOI chunk {chunk_idx}/{total_chunks}: {chunk_successes} resolved, {chunk_failures} skipped/lookup-failed."
            )
    if tracker:
        tracker.update(current_tic=None, current_source=None, failed_tics=failed_tics, skipped_tics=failed_tics)
    return {"cache_hits": fresh_hits, "cache_misses": len(stale), "unique_tics_total": len(tics), "failed_tics": failed_tics}


def maybe_upgrade_cache_for_detail(tic_id: int, max_age_days: int = CACHE_TTL_DAYS) -> None:
    cache = load_cache(tic_id)
    if cache_is_fresh(cache, max_age_days=max_age_days) and cache.get("exofop_lookup_done"):
        return
    toi_rows = (cache or {}).get("toi_rows")
    if not toi_rows:
        toi_map, toi_failed = fetch_toi_rows_resilient([tic_id])
        toi_rows = toi_map.get(tic_id, [])
        if tic_id in toi_failed:
            payload = build_remote_payload(tic_id, toi_rows=toi_rows, need_exofop=True)
            payload.setdefault("errors", []).append(
                "TOI batch query failed; resolved using ExoFOP/confirmed sources only."
            )
            payload["toi_lookup_failed"] = True
            save_cache(payload)
            return
    payload = build_remote_payload(tic_id, toi_rows=toi_rows, need_exofop=True)
    save_cache(payload)


def build_result_from_cache(cache: dict | None, local_period: float | None) -> dict:
    if not cache:
        return {
            "tic_id": None,
            "local_period": local_period,
            "match_status": "lookup_failed",
            "status_label": STATUS_LABELS["lookup_failed"],
            "matched_name": "",
            "matched_period": None,
            "period_difference_pct": None,
            "period_difference_days": None,
            "source_used": "none",
            "caution_notes": [],
            "plain_english_summary": "Public catalogue lookup failed before any usable result was cached.",
            "discovery_value": "Retry lookup later.",
        }

    if cache.get("forced_lookup_failed"):
        reason = "; ".join(cache.get("errors") or []) or "Public catalogue lookup failed or timed out; requires manual check."
        return {
            "tic_id": int(cache["tic_id"]),
            "local_period": local_period,
            "match_status": "lookup_failed",
            "status_label": STATUS_LABELS["lookup_failed"],
            "matched_name": "",
            "matched_period": None,
            "period_difference_pct": None,
            "period_difference_days": None,
            "source_used": cache.get("failure_kind", "public lookup"),
            "caution_notes": [],
            "plain_english_summary": "Public catalogue lookup failed or timed out; requires manual check.",
            "discovery_value": "Still usable locally, but public identity needs manual checking.",
            "cached_at": cache.get("fetched_at"),
            "failure_kind": cache.get("failure_kind"),
            "failure_reason": reason,
        }

    tic_id = int(cache["tic_id"])
    toi_rows = cache.get("toi_rows") or []
    confirmed_rows = cache.get("confirmed_rows") or []
    caution_notes = list(cache.get("caution_notes") or [])
    best_confirmed = choose_best_match(confirmed_rows, local_period, "pl_orbper")
    best_toi = choose_best_match(toi_rows, local_period, "pl_orbper")

    if best_confirmed:
        matched_period = safe_float(best_confirmed.get("pl_orbper"))
        diff_pct, diff_days = period_diff(local_period, matched_period)
        summary = f"Confirmed planet: {best_confirmed.get('pl_name') or best_confirmed.get('hostname')}."
        if local_period is not None and matched_period is not None:
            if diff_pct is not None and diff_pct <= 2.0:
                summary += f" Local period {local_period:.4f} d is close to archive period {matched_period:.4f} d."
            else:
                summary += f" Local period {local_period:.4f} d differs from archive period {matched_period:.4f} d."
        return {
            "tic_id": tic_id,
            "local_period": local_period,
            "match_status": "confirmed_planet",
            "status_label": STATUS_LABELS["confirmed_planet"],
            "matched_name": best_confirmed.get("pl_name") or best_confirmed.get("hostname") or "",
            "matched_period": matched_period,
            "period_difference_pct": diff_pct,
            "period_difference_days": diff_days,
            "source_used": "NASA Exoplanet Archive pscomppars",
            "caution_notes": caution_notes,
            "plain_english_summary": summary,
            "discovery_value": "Known object, good pipeline sanity check.",
            "tfopwg_disp": (best_toi or {}).get("tfopwg_disp", ""),
            "toi_number": (best_toi or {}).get("toi", ""),
            "cached_at": cache.get("fetched_at"),
        }

    if best_toi:
        disp = str(best_toi.get("tfopwg_disp", "")).upper()
        matched_period = safe_float(best_toi.get("pl_orbper"))
        diff_pct, diff_days = period_diff(local_period, matched_period)
        if disp in PUBLIC_DISPOSITION_NOTES:
            caution_notes.append(PUBLIC_DISPOSITION_NOTES[disp])
        seen = set()
        caution_notes = [n for n in caution_notes if not (n.lower() in seen or seen.add(n.lower()))]

        if disp in SUSPICIOUS_DISPOSITIONS or any(
            needle in " ".join(caution_notes).lower()
            for needle in ("v-shaped", "odd/even", "secondary eclipse", "eclipsing-binary", "eb catalog", "centroid")
        ):
            summary = f"Known alert-style object: TOI-{best_toi.get('toi')}. Public notes make this look suspicious."
            return {
                "tic_id": tic_id,
                "local_period": local_period,
                "match_status": "suspicious_alert_object",
                "status_label": STATUS_LABELS["suspicious_alert_object"],
                "matched_name": f"TOI-{best_toi.get('toi')}",
                "matched_period": matched_period,
                "period_difference_pct": diff_pct,
                "period_difference_days": diff_days,
                "source_used": "NASA Exoplanet Archive TOI table / ExoFOP-TESS",
                "caution_notes": caution_notes,
                "plain_english_summary": summary,
                "discovery_value": "Low priority until it survives EB checks.",
                "tfopwg_disp": disp,
                "toi_number": best_toi.get("toi", ""),
                "cached_at": cache.get("fetched_at"),
            }

        summary = f"Known TOI: TOI-{best_toi.get('toi')}. This is a pre-known TESS candidate, not a new discovery."
        return {
            "tic_id": tic_id,
            "local_period": local_period,
            "match_status": "known_toi",
            "status_label": STATUS_LABELS["known_toi"],
            "matched_name": f"TOI-{best_toi.get('toi')}",
            "matched_period": matched_period,
            "period_difference_pct": diff_pct,
            "period_difference_days": diff_days,
            "source_used": "NASA Exoplanet Archive TOI table",
            "caution_notes": caution_notes,
            "plain_english_summary": summary,
            "discovery_value": "Not new, but still useful for pipeline validation.",
            "tfopwg_disp": disp,
            "toi_number": best_toi.get("toi", ""),
            "cached_at": cache.get("fetched_at"),
        }

    if cache.get("errors"):
        return {
            "tic_id": tic_id,
            "local_period": local_period,
            "match_status": "lookup_failed",
            "status_label": STATUS_LABELS["lookup_failed"],
            "matched_name": "",
            "matched_period": None,
            "period_difference_pct": None,
            "period_difference_days": None,
            "source_used": "lookup failed",
            "caution_notes": [],
            "plain_english_summary": "Public catalogue lookup failed. No cached match could be confirmed.",
            "discovery_value": "Retry lookup later.",
            "cached_at": cache.get("fetched_at"),
        }

    return {
        "tic_id": tic_id,
        "local_period": local_period,
        "match_status": "not_obviously_known",
        "status_label": STATUS_LABELS["not_obviously_known"],
        "matched_name": "",
        "matched_period": None,
        "period_difference_pct": None,
        "period_difference_days": None,
        "source_used": "NASA Exoplanet Archive TOI table / ExoFOP-TESS",
        "caution_notes": caution_notes,
        "plain_english_summary": "No obvious confirmed-planet or TOI match found. Still requires vetting.",
        "discovery_value": "Worth triage if local verification is strong.",
        "cached_at": cache.get("fetched_at"),
    }


def row_public_status(row: dict) -> str:
    return str(row.get("status_label") or row.get("public_status") or "").strip() or "Unknown"


def triage_rank_tuple(row: dict) -> tuple:
    verified = str(row.get("verified", "")).strip().lower() == "true"
    consistency = safe_float(row.get("consistency_score"))
    depth = safe_float(row.get("depth_ppm"))
    eb_warning = str(row.get("eb_warning", "")).strip()
    power = safe_float(row.get("bls_power"))
    status = row_public_status(row)
    status_rank = {
        "Unknown": 0,
        "Lookup failed": 1,
        "Caution": 2,
        "Known TOI": 3,
        "Confirmed planet": 4,
    }.get(status, 5)
    return (
        0 if verified else 1,
        -(consistency if consistency is not None else -1.0),
        status_rank,
        1 if eb_warning else 0,
        depth if depth is not None else 1e12,
        -(power if power is not None else -1.0),
    )


def write_rows_csv_atomic(path: Path, rows: list[dict]) -> None:
    ensure_dirs()
    tmp_path = path.with_name(path.name + ".tmp")
    if not rows:
        tmp_path.write_text("")
        tmp_path.replace(path)
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with tmp_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_shortlist_metadata(path: Path | None, metadata: dict) -> None:
    meta_path = metadata_path_for_output(path)
    if not meta_path:
        return
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    meta_tmp.replace(meta_path)


def snapshot_shortlist_output(path: Path | None, rows: list[dict], metadata: dict) -> None:
    if not path:
        return
    write_rows_csv_atomic(path, rows)
    write_shortlist_metadata(path, metadata)


def merge_rows_from_cache(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        cache = load_cache(row["tic_id"]) if row.get("tic_id") not in ("", None) else None
        result = build_result_from_cache(cache, safe_float(row.get("period")))
        merged = dict(row)
        merged.update(result)
        merged["public_badge"] = result["status_label"]
        merged["public_summary"] = result["plain_english_summary"]
        out.append(merged)
    return out


def select_fast_detail_tics(rows: list[dict]) -> list[int]:
    if not rows:
        return []
    by_sector: dict[int, list[dict]] = {}
    must_include: set[int] = set()
    for row in rows:
        tic = safe_int(row.get("tic_id"))
        if tic is None:
            continue
        sector = safe_int(row.get("sector")) or -1
        by_sector.setdefault(sector, []).append(row)
        verified = str(row.get("verified", "")).strip().lower() == "true"
        consistency = safe_float(row.get("consistency_score"))
        status = row_public_status(row)
        if verified or (consistency is not None and consistency >= 0.70):
            must_include.add(tic)
        elif status in {"Unknown", "Lookup failed"}:
            pass

    selected: list[int] = []
    seen: set[int] = set()
    for tic in sorted(must_include):
        if tic not in seen:
            selected.append(tic)
            seen.add(tic)

    for sector in sorted(by_sector):
        ranked = sorted(by_sector[sector], key=triage_rank_tuple)
        added = 0
        for row in ranked:
            tic = safe_int(row.get("tic_id"))
            if tic is None or tic in seen:
                continue
            status = row_public_status(row)
            if status not in {"Unknown", "Lookup failed", "Known TOI"}:
                continue
            selected.append(tic)
            seen.add(tic)
            added += 1
            if added >= FAST_TOP_N_PER_SECTOR or len(selected) >= FAST_GLOBAL_LIMIT:
                break
        if len(selected) >= FAST_GLOBAL_LIMIT:
            break
    return selected[:FAST_GLOBAL_LIMIT]


def ensure_detailed_cache_for_tics(
    tic_ids: Iterable[int],
    tracker: ProgressTracker | None = None,
    max_age_days: int = CACHE_TTL_DAYS,
) -> dict:
    tics = [int(t) for t in tic_ids]
    done = 0
    failed = 0
    for idx, tic in enumerate(tics, start=1):
        cache = load_cache(tic)
        if cache_is_fresh(cache, max_age_days=max_age_days) and cache.get("exofop_lookup_done"):
            done += 1
            if tracker:
                tracker.update(
                    phase="Deep lookup",
                    current_tic=tic,
                    current_source="cached detailed",
                    deep_lookup_done=done,
                    progress_done=tracker.state.get("unique_tics_total", 0) + done,
                )
            continue
        toi_rows = (cache or {}).get("toi_rows")
        if toi_rows is None:
            toi_map, toi_failed = fetch_toi_rows_resilient([tic], tracker=tracker)
            toi_rows = toi_map.get(tic, [])
            if tic in toi_failed:
                toi_rows = toi_rows or []
        try:
            if tracker:
                tracker.update(
                    phase="Deep lookup",
                    current_tic=tic,
                    current_source="ExoFOP / confirmed resolution",
                    progress_done=tracker.state.get("unique_tics_total", 0) + done,
                )
            with tic_timeout(PER_TIC_TIMEOUT_S):
                payload = build_remote_payload(tic, toi_rows=toi_rows, need_exofop=True)
            save_cache(payload)
        except TicResolutionTimeout as exc:
            save_cache(build_failure_payload(tic, f"Timeout during public resolution: {exc}", "timeout", toi_rows=toi_rows))
            failed += 1
            if tracker:
                tracker.update(failed_tics=tracker.state.get("failed_tics", 0) + 1, skipped_tics=tracker.state.get("skipped_tics", 0) + 1, last_error=f"TIC {tic}: timeout during deep lookup")
                tracker.log(f"Skipping TIC {tic} after timeout during detailed lookup.")
        except Exception as exc:
            short_reason = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__
            save_cache(build_failure_payload(tic, f"Public resolution failed: {short_reason}", "lookup_failed", toi_rows=toi_rows))
            failed += 1
            if tracker:
                tracker.update(failed_tics=tracker.state.get("failed_tics", 0) + 1, skipped_tics=tracker.state.get("skipped_tics", 0) + 1, last_error=f"TIC {tic}: {short_reason}")
                tracker.log(f"Skipping TIC {tic} after detailed lookup error: {short_reason}")
        done += 1
        if tracker:
            tracker.update(
                deep_lookup_done=done,
                progress_done=tracker.state.get("unique_tics_total", 0) + done,
            )
            if idx == len(tics) or idx % 25 == 0:
                tracker.log(f"Deep lookup ready for {done}/{len(tics)} selected TICs. Current TIC {tic}.")
    return {"deep_lookup_done": done, "failed_tics": failed}


def get_crossmatch(tic_id: int | str, local_period: float | None = None, detailed: bool = True, max_age_days: int = CACHE_TTL_DAYS) -> dict:
    tic_int = parse_tic_id(tic_id)
    if tic_int is None:
        return build_result_from_cache(None, safe_float(local_period))
    ensure_cache_for_tics([tic_int], detailed=False, max_age_days=max_age_days)
    if detailed:
        maybe_upgrade_cache_for_detail(tic_int, max_age_days=max_age_days)
    cache = load_cache(tic_int)
    return build_result_from_cache(cache, safe_float(local_period))


def annotate_candidate_rows(
    rows: list[dict],
    detailed: bool = False,
    max_age_days: int = CACHE_TTL_DAYS,
    progress_file: Path | None = None,
    output_path: Path | None = None,
    build_mode: str = "full",
) -> list[dict]:
    tics = [t for t in (parse_tic_id(row["tic_id"]) for row in rows if row.get("tic_id") not in ("", None)) if t is not None]
    tracker = ProgressTracker(progress_file, rows_total=len(rows), unique_tics_total=len(sorted(set(tics))), output_path=output_path)
    tracker.update(build_mode=build_mode)
    tracker.log(
        f"Starting shortlist annotation for {len(rows)} rows across {len(sorted(set(tics)))} unique TICs."
    )
    try:
        cache_stats = ensure_cache_for_tics(tics, detailed=detailed, max_age_days=max_age_days, tracker=tracker)
        out = []
        unique_tics_total = cache_stats.get("unique_tics_total", len(sorted(set(tics))))
        for idx, row in enumerate(rows, start=1):
            current_tic = None
            try:
                current_tic = int(row["tic_id"])
            except Exception:
                pass
            tracker.update(
                phase="Annotating rows",
                current_tic=current_tic,
                rows_processed=idx - 1,
                progress_done=unique_tics_total + idx - 1,
            )
            cache = load_cache(row["tic_id"]) if row.get("tic_id") not in ("", None) else None
            result = build_result_from_cache(cache, safe_float(row.get("period")))
            merged = dict(row)
            merged.update(result)
            merged["public_badge"] = result["status_label"]
            merged["public_summary"] = result["plain_english_summary"]
            out.append(merged)
            tracker.update(
                rows_processed=idx,
                progress_done=unique_tics_total + idx,
            )
            if output_path and (idx == len(rows) or idx % PARTIAL_WRITE_EVERY == 0):
                snapshot_shortlist_output(
                    output_path,
                    out,
                    {
                        "build_mode": build_mode,
                        "generated_at": now_iso(),
                        "cheap_pass_rows": len(rows),
                        "deep_lookup_rows": len(sorted(set(tics))) if detailed else 0,
                        "total_rows": len(rows),
                        "rows_written": len(out),
                        "status": "running",
                    },
                )
            if idx == len(rows) or idx % 500 == 0:
                tracker.log(f"Annotated {idx}/{len(rows)} rows. Current TIC {current_tic}.")
        tracker.finish(
            "completed",
            phase="Completed",
            current_tic=None,
            current_source=None,
            cache_hits=cache_stats.get("cache_hits", 0),
            cache_misses=cache_stats.get("cache_misses", 0),
            row_count=len(out),
            failed_tics=cache_stats.get("failed_tics", tracker.state.get("failed_tics", 0)),
            skipped_tics=cache_stats.get("failed_tics", tracker.state.get("skipped_tics", 0)),
        )
        if output_path:
            snapshot_shortlist_output(
                output_path,
                out,
                {
                    "build_mode": build_mode,
                    "generated_at": now_iso(),
                    "cheap_pass_rows": len(rows),
                    "deep_lookup_rows": len(sorted(set(tics))) if detailed else 0,
                    "total_rows": len(rows),
                    "rows_written": len(out),
                    "status": "completed",
                },
            )
        tracker.log(f"Shortlist completed: {len(out)} rows written.")
        return out
    except Exception as exc:
        tracker.finish("failed", phase="Failed", error=str(exc))
        tracker.log(f"Shortlist failed: {exc}")
        raise


def parse_sector_spec(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))
    return [int(spec)]


def preferred_result_csv(sector: int) -> Path | None:
    sector_dir = RESULTS_DIR / f"sector{sector}"
    for name in ("bls_exominer_results.csv", "bls_results.csv"):
        path = sector_dir / name
        if path.exists():
            return path
    return None


def load_candidate_rows_from_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    write_rows_csv_atomic(path, rows)


def build_fast_shortlist_for_sectors(sectors: list[int], progress_file: Path | None = None, output_path: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for sector in sectors:
        csv_path = preferred_result_csv(sector)
        if not csv_path:
            continue
        for row in load_candidate_rows_from_csv(csv_path):
            row = dict(row)
            row.setdefault("sector", sector)
            rows.append(row)

    tics = [t for t in (parse_tic_id(row["tic_id"]) for row in rows if row.get("tic_id") not in ("", None)) if t is not None]
    tracker = ProgressTracker(progress_file, rows_total=len(rows), unique_tics_total=len(sorted(set(tics))), output_path=output_path)
    tracker.update(build_mode="fast")
    tracker.log(f"Starting fast triage-first shortlist for {len(rows)} rows across {len(sorted(set(tics)))} unique TICs.")

    cache_stats = ensure_cache_for_tics(tics, detailed=False, tracker=tracker)
    cheap_rows = merge_rows_from_cache(rows)
    deep_tics = select_fast_detail_tics(cheap_rows)
    tracker.update(
        phase="Selecting deep lookups",
        cheap_pass_rows=len(rows),
        deep_lookup_rows=len(deep_tics),
        progress_total=len(rows) + len(sorted(set(tics))) + len(deep_tics),
        progress_done=cache_stats.get("cache_hits", 0),
    )
    tracker.log(
        f"Fast mode cheap pass complete. Selected {len(deep_tics)} high-priority TICs for detailed lookup."
    )
    if output_path:
        snapshot_shortlist_output(
            output_path,
            cheap_rows,
            {
                "build_mode": "fast",
                "generated_at": now_iso(),
                "cheap_pass_rows": len(rows),
                "deep_lookup_rows": len(deep_tics),
                "total_rows": len(rows),
                "rows_written": len(cheap_rows),
                "status": "running",
                "phase": "cheap_pass_complete",
            },
        )

    detail_stats = ensure_detailed_cache_for_tics(deep_tics, tracker=tracker)
    out = []
    unique_tics_total = cache_stats.get("unique_tics_total", len(sorted(set(tics))))
    deep_lookup_rows = len(deep_tics)
    for idx, row in enumerate(rows, start=1):
        current_tic = safe_int(row.get("tic_id"))
        tracker.update(
            phase="Annotating rows",
            current_tic=current_tic,
            current_source="cache merge",
            rows_processed=idx - 1,
            progress_done=unique_tics_total + deep_lookup_rows + idx - 1,
        )
        cache = load_cache(row["tic_id"]) if row.get("tic_id") not in ("", None) else None
        result = build_result_from_cache(cache, safe_float(row.get("period")))
        merged = dict(row)
        merged.update(result)
        merged["public_badge"] = result["status_label"]
        merged["public_summary"] = result["plain_english_summary"]
        out.append(merged)
        tracker.update(rows_processed=idx, progress_done=unique_tics_total + deep_lookup_rows + idx)
        if output_path and (idx == len(rows) or idx % PARTIAL_WRITE_EVERY == 0):
            snapshot_shortlist_output(
                output_path,
                out,
                {
                    "build_mode": "fast",
                    "generated_at": now_iso(),
                    "cheap_pass_rows": len(rows),
                    "deep_lookup_rows": deep_lookup_rows,
                    "total_rows": len(rows),
                    "rows_written": len(out),
                    "status": "running",
                    "phase": "annotating_rows",
                },
            )
        if idx == len(rows) or idx % 500 == 0:
            tracker.log(f"Annotated {idx}/{len(rows)} rows. Current TIC {current_tic}.")

    tracker.finish(
        "completed",
        phase="Completed",
        current_tic=None,
        current_source=None,
        cache_hits=cache_stats.get("cache_hits", 0),
        cache_misses=cache_stats.get("cache_misses", 0),
        row_count=len(out),
        cheap_pass_rows=len(rows),
        deep_lookup_rows=deep_lookup_rows,
        deep_lookup_done=detail_stats.get("deep_lookup_done", deep_lookup_rows),
        failed_tics=tracker.state.get("failed_tics", 0),
        skipped_tics=tracker.state.get("skipped_tics", 0),
    )
    if output_path:
        snapshot_shortlist_output(
            output_path,
            out,
            {
                "build_mode": "fast",
                "generated_at": now_iso(),
                "cheap_pass_rows": len(rows),
                "deep_lookup_rows": deep_lookup_rows,
                "total_rows": len(rows),
                "rows_written": len(out),
                "status": "completed",
            },
        )
    tracker.log(f"Fast shortlist completed: {len(out)} rows written.")
    return out


def build_shortlist_for_sectors(sectors: list[int], progress_file: Path | None = None, output_path: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for sector in sectors:
        csv_path = preferred_result_csv(sector)
        if not csv_path:
            continue
        for row in load_candidate_rows_from_csv(csv_path):
            row = dict(row)
            row.setdefault("sector", sector)
            rows.append(row)
    return annotate_candidate_rows(rows, detailed=True, progress_file=progress_file, output_path=output_path, build_mode="full")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-match exoplanet candidates against public catalogs.")
    parser.add_argument("--tic", type=int, help="Single TIC to cross-match.")
    parser.add_argument("--period", type=float, help="Local period for a single TIC.")
    parser.add_argument("--csv", type=Path, help="Annotate one existing candidate CSV.")
    parser.add_argument("--sectors", type=str, help="Sector or range like 85-99 to merge and annotate.")
    parser.add_argument("--out", type=Path, help="Output file for annotated CSV or JSON.")
    parser.add_argument("--json", action="store_true", help="Print JSON for single-TIC mode.")
    parser.add_argument("--progress-file", type=Path, help="Optional JSON progress file for long shortlist builds.")
    parser.add_argument("--fast", action="store_true", help="Build a fast triage-first shortlist instead of a full enriched one.")
    args = parser.parse_args()

    ensure_dirs()

    if args.tic:
        result = get_crossmatch(args.tic, args.period, detailed=True)
        if args.out:
            args.out.write_text(json.dumps(result, indent=2))
        if args.json or not args.out:
            print(json.dumps(result, indent=2))
        return 0

    if args.csv:
        rows = load_candidate_rows_from_csv(args.csv)
        annotated = annotate_candidate_rows(rows, detailed=False, progress_file=args.progress_file, output_path=args.out, build_mode="fast" if args.fast else "full")
        out_path = args.out or SHORTLIST_DIR / f"{args.csv.stem}_crossmatched.csv"
        write_rows_csv(out_path, annotated)
        write_shortlist_metadata(
            out_path,
            {
                "build_mode": "fast" if args.fast else "full",
                "generated_at": now_iso(),
                "cheap_pass_rows": len(rows),
                "deep_lookup_rows": 0 if not args.fast else None,
                "total_rows": len(rows),
                "rows_written": len(annotated),
                "status": "completed",
            },
        )
        print(f"Wrote {len(annotated)} annotated rows to {out_path}")
        return 0

    if args.sectors:
        sectors = parse_sector_spec(args.sectors)
        suffix = "fast_crossmatched" if args.fast else "crossmatched"
        out_path = args.out or SHORTLIST_DIR / f"sector_{args.sectors.replace('-', '_')}_shortlist_{suffix}.csv"
        annotated = (
            build_fast_shortlist_for_sectors(sectors, progress_file=args.progress_file, output_path=out_path)
            if args.fast
            else build_shortlist_for_sectors(sectors, progress_file=args.progress_file, output_path=out_path)
        )
        write_rows_csv(out_path, annotated)
        print(f"Wrote {len(annotated)} annotated rows to {out_path}")
        return 0

    parser.error("choose one of --tic, --csv, or --sectors")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
