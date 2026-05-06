#!/usr/bin/env python3
"""
deep_scan_campaign.py — sharded Deep Scan campaigns across CT100 and a temp Fedora worker.
"""

import argparse
import fcntl
import os

# Path to the project on a remote worker — override per deployment
REMOTE_PROJECT_ROOT = os.environ.get("REMOTE_PROJECT_ROOT", "/opt/exoplanet")
import shlex
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from deep_scan import (  # noqa: E402
    CANDIDATE_FIELDS,
    DEFAULT_MAX_TARGET_MINUTES,
    DEFAULT_TARGET_ORDER,
    RESULTS_DIR,
    TARGET_CACHE,
    apply_startup_guardrail,
    find_multi_sector_targets,
    generate_html_report,
    generate_plots,
    load_candidate_rows,
    load_json,
    now_iso,
    order_targets,
    parse_float,
    parse_int,
    write_candidate_rows,
    write_json,
    write_progress,
)

DEEP_SCAN_RUNS_DIR = RESULTS_DIR / "deep_scan_runs"
DEFAULT_CT_HOST = "192.168.1.233"
DEFAULT_CT_USER = "root"
DEFAULT_FEDORA_KEY = "~/.ssh/id_ed25519_exoplanet_ct100"
DEFAULT_SCRATCH_ROOT = "/var/tmp/exoplanet-deep-scan"
DEFAULT_LOCAL_WORKERS = 6
DEFAULT_REMOTE_WORKERS = 10
MAX_REMOTE_WORKERS = 10
SYNC_INTERVAL_SECONDS = 300
FEDORA_PACKAGES = [
    "lightkurve==2.5.1",
    "astropy==7.2.0",
    "numpy==2.4.3",
    "pandas==2.3.3",
    "matplotlib==3.10.8",
    "astroquery==0.4.11",
    "beautifulsoup4==4.14.3",
    "scipy==1.17.1",
]


def log(msg: str) -> None:
    print(f"[deep_scan_campaign] {msg}", flush=True)


def period_token(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def short_order_name(target_order: str) -> str:
    return "quick" if str(target_order) == "quick-first" else "coverage"


def target_count_token(limit: int) -> str:
    return "all" if int(limit) <= 0 else str(int(limit))


def build_campaign_display_name(
    *,
    min_sectors: int,
    min_period_days: float,
    max_period_days: float,
    target_order: str,
    limit: int,
    workers_local: int,
    workers_remote: int,
) -> str:
    return (
        f"Deep Scan {period_token(min_period_days)}-{period_token(max_period_days)}d"
        f" · min{int(min_sectors)}"
        f" · {short_order_name(target_order)}"
        f" · {target_count_token(limit)} targets"
        f" · ct{int(workers_local)}/fd{int(workers_remote)}"
    )


def build_default_campaign_name(
    *,
    min_sectors: int,
    min_period_days: float,
    max_period_days: float,
    target_order: str,
    limit: int,
    workers_local: int,
    workers_remote: int,
) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"campaign_{stamp}"
        f"_p{period_token(min_period_days)}-{period_token(max_period_days)}"
        f"_s{int(min_sectors)}"
        f"_{short_order_name(target_order)}"
        f"_n{target_count_token(limit)}"
        f"_ct{int(workers_local)}"
        f"_fd{int(workers_remote)}"
    )


def campaign_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return DEEP_SCAN_RUNS_DIR / name


def shell_quote(value: str) -> str:
    return shlex.quote(str(value))


def ensure_executable(path: Path) -> None:
    path.chmod(0o755)


def append_campaign_log(campaign_dir: Path, msg: str) -> None:
    log_path = campaign_dir / "deep_scan.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] [deep_scan_campaign] {msg}\n")


@contextmanager
def campaign_lock(campaign_dir: Path):
    lock_path = campaign_dir / ".merge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield


def estimate_target_cost(row: dict) -> float:
    sectors = max(1, parse_int(row.get("n_sectors_seed"), 1))
    span = max(0, parse_int(row.get("seed_sector_span"), 0))
    return sectors * (1.0 + span / 10.0)


def order_shard_rows(rows: list[dict], target_order: str, workers: int, limit: int) -> list[dict]:
    ordered = order_targets(list(rows), target_order)
    guarded = apply_startup_guardrail(
        ordered,
        target_order,
        explicit_tics=[],
        workers=max(1, workers),
        limit=limit,
    )
    return guarded


def split_targets(rows: list[dict], local_workers: int, remote_workers: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=estimate_target_cost, reverse=True)
    local_rows: list[dict] = []
    remote_rows: list[dict] = []
    local_load = 0.0
    remote_load = 0.0
    local_capacity = max(1, local_workers)
    remote_capacity = max(1, remote_workers)

    for row in ordered:
        cost = estimate_target_cost(row)
        if (local_load / local_capacity) <= (remote_load / remote_capacity):
            local_rows.append(row)
            local_load += cost
        else:
            remote_rows.append(row)
            remote_load += cost

    return local_rows, remote_rows


def write_target_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{parse_int(row.get('tic_id'))}\n")


def read_target_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    values: list[int] = []
    for line in path.read_text().splitlines():
        tic = parse_int(line.strip())
        if tic > 0:
            values.append(tic)
    return values


def aggregate_run_config(config: dict) -> dict:
    return {
        "min_sectors": parse_int(config.get("min_sectors"), 5),
        "limit": parse_int(config.get("limit"), 0),
        "target_tics": [],
        "target_file_sha256": "",
        "target_count": parse_int(config.get("total_targets"), 0),
        "worker_name": "merged",
        "min_period_days": parse_float(config.get("min_period_days"), 1.0),
        "max_period_days": parse_float(config.get("max_period_days"), 200.0),
        "max_target_minutes": parse_float(
            config.get("max_target_minutes"), DEFAULT_MAX_TARGET_MINUTES
        ),
        "target_order": str(config.get("target_order", DEFAULT_TARGET_ORDER)),
        "source_used": "campaign_sharded",
    }


def parse_created_at(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        return time.time()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


def merge_candidate_rows(candidate_rows: list[dict]) -> list[dict]:
    by_tic: dict[int, dict] = {}
    for row in candidate_rows:
        tic_id = parse_int(row.get("tic_id"))
        if tic_id <= 0:
            continue
        current = by_tic.get(tic_id)
        score = parse_float(row.get("bls_sde", row.get("bls_power")))
        if current is None or score > parse_float(current.get("bls_sde", current.get("bls_power"))):
            by_tic[tic_id] = {field: row.get(field) for field in CANDIDATE_FIELDS}
    merged = list(by_tic.values())
    merged.sort(
        key=lambda row: parse_float(row.get("bls_sde", row.get("bls_power"))),
        reverse=True,
    )
    return merged


def worker_dir(campaign_dir: Path, worker_name: str) -> Path:
    return campaign_dir / "workers" / worker_name


def merge_campaign(campaign_dir: Path, *, finalize: bool = False) -> dict:
    campaign_dir = campaign_dir.resolve()
    config_path = campaign_dir / "campaign_config.json"
    if not config_path.exists():
        raise SystemExit(f"Campaign config not found: {config_path}")

    config = load_json(config_path, {})
    workers_cfg = config.get("workers", {})
    shard_counts = config.get("shard_counts", {})
    total_targets = parse_int(config.get("total_targets"), 0)
    processed_tics: set[int] = set()
    failed_tics: dict[str, str] = {}
    candidate_rows: list[dict] = []
    active_targets: list[dict] = []
    total_rate = 0.0
    complete_flags: list[bool] = []
    last_error = ""
    per_worker: dict[str, dict] = {}

    for worker_name in ("ct100", "fedora"):
        out_dir = worker_dir(campaign_dir, worker_name)
        progress = load_json(out_dir / "deep_scan_progress.json", {})
        rows = load_candidate_rows(out_dir / "deep_scan_results.csv")
        assigned = parse_int(shard_counts.get(worker_name), 0)

        if progress:
            processed_tics.update(parse_int(tic) for tic in progress.get("processed_tics", []))
            failed_tics.update(
                {
                    str(parse_int(tic)): str(reason)
                    for tic, reason in (progress.get("failed_tics") or {}).items()
                }
            )
            total_rate += parse_float(progress.get("rate"), 0.0)
            last_error = str(progress.get("last_error") or last_error or "")
            for active in progress.get("active_targets") or []:
                item = dict(active)
                item["worker_name"] = worker_name
                active_targets.append(item)
            complete_flags.append(bool(progress.get("complete")))
        else:
            complete_flags.append(assigned == 0)

        per_worker[worker_name] = {
            "assigned": assigned,
            "processed": len({parse_int(tic) for tic in (progress.get("processed_tics") or [])}),
            "candidates": len(rows),
            "errors": len(progress.get("failed_tics") or {}),
            "complete": bool(progress.get("complete")) if progress else (assigned == 0),
        }
        candidate_rows.extend(rows)

    merged_candidates = merge_candidate_rows(candidate_rows)
    aggregate_csv = campaign_dir / "deep_scan_results.csv"
    aggregate_progress = campaign_dir / "deep_scan_progress.json"
    aggregate_html = campaign_dir / "deep_scan_report.html"
    aggregate_plot_dir = campaign_dir / "plots"

    write_candidate_rows(aggregate_csv, merged_candidates)
    active_targets.sort(key=lambda row: (str(row.get("worker_name", "")), parse_int(row.get("tic_id"))))
    current_tic = parse_int(active_targets[0].get("tic_id")) if active_targets else None
    current_status = (
        f"{active_targets[0].get('worker_name')}:{active_targets[0].get('phase', 'running')}"
        if active_targets
        else ("complete" if total_targets > 0 and len(processed_tics) >= total_targets and all(complete_flags) else "idle")
    )
    write_progress(
        aggregate_progress,
        done=len(processed_tics),
        total=total_targets,
        candidates=len(merged_candidates),
        rate=total_rate,
        complete=bool(total_targets > 0 and len(processed_tics) >= total_targets and all(complete_flags)),
        processed_tics=processed_tics,
        failed_tics=failed_tics,
        current_tic=current_tic,
        current_status=current_status,
        last_error=last_error,
        target_source=str(config.get("source_used", "unknown")),
        target_cache_path=TARGET_CACHE,
        run_config=aggregate_run_config(config),
        active_targets=active_targets,
    )

    append_campaign_log(
        campaign_dir,
        f"Merged {len(processed_tics)}/{total_targets} targets, "
        f"{len(merged_candidates)} candidates, {len(failed_tics)} errors, "
        f"{len(active_targets)} active workers.",
    )

    if finalize:
        created_at = parse_created_at(config.get("created_at"))
        elapsed = max(0.0, time.time() - created_at)
        generate_plots(merged_candidates, aggregate_plot_dir)
        generate_html_report(merged_candidates, len(processed_tics), elapsed, aggregate_plot_dir, aggregate_html)
        append_campaign_log(
            campaign_dir,
            f"Finalized campaign report with {len(merged_candidates)} candidates.",
        )

    return {
        "campaign_dir": str(campaign_dir),
        "total_targets": total_targets,
        "done": len(processed_tics),
        "candidates": len(merged_candidates),
        "errors": len(failed_tics),
        "active_targets": len(active_targets),
        "workers": per_worker,
    }


def build_local_script(campaign_dir: Path, config: dict) -> str:
    campaign_id = campaign_dir.name
    python_bin = str(Path(__file__).resolve().parent.parent / "venv/bin/python")
    out_dir = campaign_dir / "workers" / "ct100"
    target_file = campaign_dir / "shards" / "ct100_targets.txt"
    return f"""#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN_ID={shell_quote(campaign_id)}
CAMPAIGN_DIR={shell_quote(str(campaign_dir))}
OUT_DIR={shell_quote(str(out_dir))}
LOG_FILE="$OUT_DIR/deep_scan.log"
PYTHON={shell_quote(python_bin)}
mkdir -p "$OUT_DIR"

merge_once() {{
  "$PYTHON" {str(Path(__file__).resolve().parent / "deep_scan_campaign.py")} merge --campaign "$CAMPAIGN_ID" >/dev/null 2>&1 || true
}}

merge_loop() {{
  while true; do
    merge_once
    sleep {SYNC_INTERVAL_SECONDS}
  done
}}

merge_loop &
MERGE_LOOP_PID=$!
cleanup() {{
  kill "$MERGE_LOOP_PID" 2>/dev/null || true
  merge_once
}}
trap cleanup EXIT

set +e
"$PYTHON" {str(Path(__file__).resolve().parent / "deep_scan.py")} \\
  --target-file {shell_quote(str(target_file))} \\
  --worker-name ct100 \\
  --min-sectors {parse_int(config.get("min_sectors"), 5)} \\
  --workers {parse_int(config.get("workers", {}).get("ct100"), DEFAULT_LOCAL_WORKERS)} \\
  --min-period-days {parse_float(config.get("min_period_days"), 1.0)} \\
  --max-period-days {parse_float(config.get("max_period_days"), 200.0)} \\
  --max-target-minutes {parse_float(config.get("max_target_minutes"), DEFAULT_MAX_TARGET_MINUTES)} \\
  --target-order {shell_quote(str(config.get("target_order", DEFAULT_TARGET_ORDER)))} \\
  --output "$OUT_DIR" \\
  --no-plots 2>&1 | tee -a "$LOG_FILE"
RC=${{PIPESTATUS[0]}}
set -e

merge_once
exit "$RC"
"""


def build_fedora_script(campaign_dir: Path, config: dict) -> str:
    campaign_id = campaign_dir.name
    ct_host = str(config.get("ct_host", DEFAULT_CT_HOST))
    ct_user = str(config.get("ct_user", DEFAULT_CT_USER))
    ct_key = str(config.get("ct_key", DEFAULT_FEDORA_KEY))
    if ct_key.startswith("~/"):
        ct_key_expr = f'$REAL_HOME/{ct_key[2:]}'
    else:
        ct_key_expr = ct_key
    scratch_root = f"{config.get('remote_scratch_root', DEFAULT_SCRATCH_ROOT)}/{campaign_id}"
    remote_workers = min(
        parse_int(config.get("workers", {}).get("fedora"), DEFAULT_REMOTE_WORKERS),
        MAX_REMOTE_WORKERS,
    )
    packages = " ".join(FEDORA_PACKAGES)
    remote_campaign_dir = campaign_dir
    remote_worker_dir = remote_campaign_dir / "workers" / "fedora"
    remote_target_file = remote_campaign_dir / "shards" / "fedora_targets.txt"
    return f"""#!/usr/bin/env bash
set -euo pipefail

REAL_HOME="$HOME"
PATH="$REAL_HOME/.local/bin:$PATH"
CT_HOST="${{CT_HOST:-{ct_host}}}"
CT_USER="${{CT_USER:-{ct_user}}}"
CT_KEY="${{CT_KEY:-{ct_key_expr}}}"
CAMPAIGN_ID={shell_quote(campaign_id)}
SCRATCH_ROOT="${{SCRATCH_ROOT:-{scratch_root}}}"
CODE_DIR="$SCRATCH_ROOT/code"
OUT_DIR="$SCRATCH_ROOT/output"
SCRATCH_HOME="$SCRATCH_ROOT/home"
XDG_CACHE="$SCRATCH_ROOT/xdg-cache"
MPL_DIR="$SCRATCH_ROOT/mpl"
VENV_DIR="$SCRATCH_ROOT/venv"
LOG_FILE="$OUT_DIR/deep_scan.log"
REMOTE_CAMPAIGN_DIR={shell_quote(str(remote_campaign_dir))}
REMOTE_WORKER_DIR={shell_quote(str(remote_worker_dir))}
REMOTE_DEEP_SCAN="{REMOTE_PROJECT_ROOT}/scripts/deep_scan.py"
REMOTE_TARGET_FILE={shell_quote(str(remote_target_file))}
SSH_CMD=(ssh -i "$CT_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5)

mkdir -p "$CODE_DIR" "$OUT_DIR" "$SCRATCH_HOME" "$XDG_CACHE" "$MPL_DIR"

if ! command -v uv >/dev/null 2>&1; then
  HOME="$REAL_HOME" python3 -m pip install --user uv
fi
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "$REAL_HOME/.local/bin/uv" ]; then
  UV_BIN="$REAL_HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
  echo "uv is not available on Fedora worker." >&2
  exit 1
fi

"$UV_BIN" python install 3.12
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv --python 3.12 "$VENV_DIR"
fi
if [ ! -f "$VENV_DIR/.deep_scan_worker_ready" ]; then
  "$VENV_DIR/bin/python" -m ensurepip --upgrade
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install {packages}
  touch "$VENV_DIR/.deep_scan_worker_ready"
fi

export HOME="$SCRATCH_HOME"
export XDG_CACHE_HOME="$XDG_CACHE"
export MPLCONFIGDIR="$MPL_DIR"

rsync -az -e "${{SSH_CMD[*]}}" "$CT_USER@$CT_HOST:$REMOTE_DEEP_SCAN" "$CODE_DIR/deep_scan.py"
rsync -az -e "${{SSH_CMD[*]}}" "$CT_USER@$CT_HOST:$REMOTE_TARGET_FILE" "$CODE_DIR/fedora_targets.txt"
"${{SSH_CMD[@]}}" "$CT_USER@$CT_HOST" "mkdir -p '$REMOTE_WORKER_DIR'"

sync_once() {{
  rsync -az --delete -e "${{SSH_CMD[*]}}" "$OUT_DIR"/ "$CT_USER@$CT_HOST:$REMOTE_WORKER_DIR/" >/dev/null 2>&1 || true
  "${{SSH_CMD[@]}}" "$CT_USER@$CT_HOST" "{REMOTE_PROJECT_ROOT}/venv/bin/python {REMOTE_PROJECT_ROOT}/scripts/deep_scan_campaign.py merge --campaign '$CAMPAIGN_ID'" >/dev/null 2>&1 || true
}}

sync_loop() {{
  while true; do
    sync_once
    sleep {SYNC_INTERVAL_SECONDS}
  done
}}

sync_loop &
SYNC_LOOP_PID=$!
cleanup() {{
  kill "$SYNC_LOOP_PID" 2>/dev/null || true
  sync_once
}}
trap cleanup EXIT

set +e
nice -n 10 "$VENV_DIR/bin/python" "$CODE_DIR/deep_scan.py" \\
  --target-file "$CODE_DIR/fedora_targets.txt" \\
  --worker-name fedora \\
  --min-sectors {parse_int(config.get("min_sectors"), 5)} \\
  --workers {remote_workers} \\
  --min-period-days {parse_float(config.get("min_period_days"), 1.0)} \\
  --max-period-days {parse_float(config.get("max_period_days"), 200.0)} \\
  --max-target-minutes {parse_float(config.get("max_target_minutes"), DEFAULT_MAX_TARGET_MINUTES)} \\
  --target-order {shell_quote(str(config.get("target_order", DEFAULT_TARGET_ORDER)))} \\
  --output "$OUT_DIR" \\
  --no-plots 2>&1 | tee -a "$LOG_FILE"
RC=${{PIPESTATUS[0]}}
set -e

sync_once
exit "$RC"
"""


def build_finalize_script(campaign_dir: Path) -> str:
    campaign_id = campaign_dir.name
    return f"""#!/usr/bin/env bash
set -euo pipefail
{str(Path(__file__).resolve().parent.parent / "venv/bin/python")} {str(Path(__file__).resolve().parent / "deep_scan_campaign.py")} finalize --campaign {shell_quote(campaign_id)}
"""


def write_script(path: Path, content: str) -> None:
    path.write_text(content)
    ensure_executable(path)


def create_campaign(args) -> int:
    DEEP_SCAN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    workers_remote = min(max(1, int(args.workers_remote)), MAX_REMOTE_WORKERS)
    workers_local = max(1, int(args.workers_local))
    raw_limit = int(args.limit)
    limit = raw_limit if raw_limit > 0 else None
    display_name = build_campaign_display_name(
        min_sectors=int(args.min_sectors),
        min_period_days=float(args.min_period_days),
        max_period_days=float(args.max_period_days),
        target_order=str(args.target_order),
        limit=raw_limit,
        workers_local=workers_local,
        workers_remote=workers_remote,
    )
    campaign_name = args.campaign_name or build_default_campaign_name(
        min_sectors=int(args.min_sectors),
        min_period_days=float(args.min_period_days),
        max_period_days=float(args.max_period_days),
        target_order=str(args.target_order),
        limit=raw_limit,
        workers_local=workers_local,
        workers_remote=workers_remote,
    )
    campaign_dir = campaign_path(campaign_name)
    if campaign_dir.exists():
        raise SystemExit(f"Campaign directory already exists: {campaign_dir}")

    targets, cache = find_multi_sector_targets(
        min_sectors=int(args.min_sectors),
        limit=limit,
        refresh_targets=bool(args.refresh_targets),
        target_order=str(args.target_order),
    )
    if not targets:
        raise SystemExit("No Deep Scan targets available for the requested campaign config.")

    local_rows, remote_rows = split_targets(targets, workers_local, workers_remote)
    local_rows = order_shard_rows(local_rows, str(args.target_order), workers_local, int(args.limit))
    remote_rows = order_shard_rows(remote_rows, str(args.target_order), workers_remote, int(args.limit))

    (campaign_dir / "shards").mkdir(parents=True, exist_ok=True)
    (campaign_dir / "workers" / "ct100").mkdir(parents=True, exist_ok=True)
    (campaign_dir / "workers" / "fedora").mkdir(parents=True, exist_ok=True)
    (campaign_dir / "commands").mkdir(parents=True, exist_ok=True)

    write_target_file(campaign_dir / "shards" / "ct100_targets.txt", local_rows)
    write_target_file(campaign_dir / "shards" / "fedora_targets.txt", remote_rows)

    config = {
        "campaign_id": campaign_dir.name,
        "display_name": display_name,
        "created_at": now_iso(),
        "source_used": cache.get("source_used", "unknown"),
        "target_cache": str(TARGET_CACHE),
        "min_sectors": int(args.min_sectors),
        "limit": int(args.limit),
        "min_period_days": float(args.min_period_days),
        "max_period_days": float(args.max_period_days),
        "max_target_minutes": float(args.max_target_minutes),
        "target_order": str(args.target_order),
        "workers": {
            "ct100": workers_local,
            "fedora": workers_remote,
        },
        "ct_host": str(args.ct_host),
        "ct_user": str(args.ct_user),
        "ct_key": str(args.ct_key),
        "remote_scratch_root": str(args.remote_scratch_root),
        "total_targets": len(targets),
        "shard_counts": {
            "ct100": len(local_rows),
            "fedora": len(remote_rows),
        },
    }
    write_json(campaign_dir / "campaign_config.json", config)
    write_candidate_rows(campaign_dir / "deep_scan_results.csv", [])
    write_progress(
        campaign_dir / "deep_scan_progress.json",
        done=0,
        total=len(targets),
        candidates=0,
        rate=0.0,
        complete=False,
        processed_tics=set(),
        failed_tics={},
        current_tic=None,
        current_status="campaign_created",
        last_error="",
        target_source=str(cache.get("source_used", "unknown")),
        target_cache_path=TARGET_CACHE,
        run_config=aggregate_run_config(config),
        active_targets=[],
    )
    append_campaign_log(
        campaign_dir,
        f"Created {display_name} with {len(targets)} targets "
        f"({len(local_rows)} ct100 / {len(remote_rows)} fedora).",
    )

    write_script(campaign_dir / "commands" / "run_local.sh", build_local_script(campaign_dir, config))
    write_script(campaign_dir / "commands" / "run_fedora.sh", build_fedora_script(campaign_dir, config))
    write_script(campaign_dir / "commands" / "finalize.sh", build_finalize_script(campaign_dir))

    log(
        f"Created {campaign_dir.name} ({display_name}): {len(targets)} total targets, "
        f"{len(local_rows)} local / {len(remote_rows)} Fedora shard."
    )
    log(f"Local shard script: {campaign_dir / 'commands' / 'run_local.sh'}")
    log(f"Fedora shard script: {campaign_dir / 'commands' / 'run_fedora.sh'}")
    return 0


def merge_command(args, *, finalize: bool = False) -> int:
    campaign_dir = campaign_path(args.campaign)
    with campaign_lock(campaign_dir):
        summary = merge_campaign(campaign_dir, finalize=finalize)
    log(
        f"{summary['done']}/{summary['total_targets']} processed, "
        f"{summary['candidates']} candidates, {summary['errors']} errors."
    )
    return 0


def status_command(args) -> int:
    campaign_dir = campaign_path(args.campaign)
    config = load_json(campaign_dir / "campaign_config.json", {})
    progress = load_json(campaign_dir / "deep_scan_progress.json", {})
    if not config:
        raise SystemExit(f"Campaign not found: {campaign_dir}")
    print(f"Campaign: {campaign_dir.name}")
    if config.get("display_name"):
        print(f"Label: {config.get('display_name')}")
    print(f"Directory: {campaign_dir}")
    print(
        f"Merged: {parse_int(progress.get('done'))}/{parse_int(progress.get('total'))} processed, "
        f"{parse_int(progress.get('candidates'))} candidates, "
        f"{parse_int(progress.get('errors'))} errors, "
        f"complete={bool(progress.get('complete'))}"
    )
    for worker_name in ("ct100", "fedora"):
        worker_progress = load_json(worker_dir(campaign_dir, worker_name) / "deep_scan_progress.json", {})
        assigned = parse_int(config.get("shard_counts", {}).get(worker_name), 0)
        print(
            f"  {worker_name}: assigned={assigned} "
            f"processed={len(worker_progress.get('processed_tics') or [])} "
            f"candidates={parse_int(worker_progress.get('candidates'))} "
            f"errors={parse_int(worker_progress.get('errors'))} "
            f"complete={bool(worker_progress.get('complete'))}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sharded Deep Scan campaigns across CT100 and Fedora")
    sub = ap.add_subparsers(dest="command", required=True)

    ap_create = sub.add_parser("create", help="Create a new sharded Deep Scan campaign")
    ap_create.add_argument("--campaign-name", type=str, default="")
    ap_create.add_argument("--min-sectors", type=int, default=5)
    ap_create.add_argument("--limit", type=int, default=0)
    ap_create.add_argument("--min-period-days", type=float, default=20.0)
    ap_create.add_argument("--max-period-days", type=float, default=120.0)
    ap_create.add_argument("--max-target-minutes", type=float, default=DEFAULT_MAX_TARGET_MINUTES)
    ap_create.add_argument(
        "--target-order",
        choices=("coverage-first", "quick-first"),
        default=DEFAULT_TARGET_ORDER,
    )
    ap_create.add_argument("--workers-local", type=int, default=DEFAULT_LOCAL_WORKERS)
    ap_create.add_argument("--workers-remote", type=int, default=DEFAULT_REMOTE_WORKERS)
    ap_create.add_argument("--ct-host", type=str, default=DEFAULT_CT_HOST)
    ap_create.add_argument("--ct-user", type=str, default=DEFAULT_CT_USER)
    ap_create.add_argument("--ct-key", type=str, default=DEFAULT_FEDORA_KEY)
    ap_create.add_argument("--remote-scratch-root", type=str, default=DEFAULT_SCRATCH_ROOT)
    ap_create.add_argument("--refresh-targets", action="store_true")
    ap_create.set_defaults(func=create_campaign)

    ap_merge = sub.add_parser("merge", help="Merge shard progress/results into the aggregate campaign output")
    ap_merge.add_argument("--campaign", type=str, required=True)
    ap_merge.set_defaults(func=lambda args: merge_command(args, finalize=False))

    ap_finalize = sub.add_parser("finalize", help="Generate final merged plots/report for a completed campaign")
    ap_finalize.add_argument("--campaign", type=str, required=True)
    ap_finalize.set_defaults(func=lambda args: merge_command(args, finalize=True))

    ap_status = sub.add_parser("status", help="Print merged and per-worker campaign status")
    ap_status.add_argument("--campaign", type=str, required=True)
    ap_status.set_defaults(func=status_command)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
