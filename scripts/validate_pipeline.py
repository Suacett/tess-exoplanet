#!/usr/bin/env python3
"""
validate_pipeline.py — Small regression harness for the exoplanet pipeline.

Run this after code changes when you want a plain-English answer to:
    "Is the pipeline still working, or did I break it?"
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
VALIDATION_DIR = DATA_DIR / "validation"
SCAN_STATE_FILE = DATA_DIR / ".scan_state.json"
SCRIPTS_DIR = Path(__file__).resolve().parent

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
STATUS_ORDER = {PASS: 0, WARN: 1, FAIL: 2}

try:
    from calibrate_verification import KNOWN_PLANETS, KNOWN_FPS
except Exception:
    KNOWN_PLANETS = [
        {"tic_id": 66818296, "period": 3.73548, "label": "KP", "name": "WASP-17 b"},
        {"tic_id": 36724087, "period": 0.7684, "label": "CP", "name": "CP (ExoMiner 0.998)"},
    ]
    KNOWN_FPS = [
        {"tic_id": 9033144, "period": 4.72, "label": "FP", "name": "FP TIC 9033144"},
        {"tic_id": 32830028, "period": 0.52, "label": "FP", "name": "FP TIC 32830028"},
    ]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def format_command(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([str(part)]) for part in cmd)


def pick_targets(quick: bool) -> tuple[list[dict], list[dict]]:
    planets_by_tic = {int(t["tic_id"]): dict(t) for t in KNOWN_PLANETS}
    fps_by_tic = {int(t["tic_id"]): dict(t) for t in KNOWN_FPS}

    planets = [
        {
            "tic_id": 402026209,
            "period": 1.338,
            "label": "KP",
            "name": "Operator benchmark TIC 402026209",
        }
    ]

    for tic in (36724087, 66818296):
        if tic in planets_by_tic:
            planets.append(planets_by_tic[tic])

    fps = []
    for tic in (9033144, 32830028):
        if tic in fps_by_tic:
            fps.append(fps_by_tic[tic])

    if not quick and 28873762 in fps_by_tic:
        fps.append(fps_by_tic[28873762])

    return planets, fps


def verdict_family(verdict: str) -> str:
    if "Likely real" in verdict:
        return "real"
    if "Inconclusive" in verdict:
        return "inconclusive"
    if "Likely not" in verdict:
        return "reject"
    return "unknown"


def interpret_verification(expected_kind: str, verdict: str) -> tuple[str, str]:
    family = verdict_family(verdict)
    if expected_kind == "planet":
        if family == "real":
            return PASS, "Verifier recovered a known planet."
        if family == "inconclusive":
            return WARN, "Verifier did not reject it, but it was not a clean recovery."
        return FAIL, "Verifier rejected a known planet."

    if family == "reject":
        return PASS, "Verifier rejected a known false positive."
    if family == "inconclusive":
        return WARN, "Verifier was cautious but not decisive on a known false positive."
    return FAIL, "Verifier treated a known false positive like a likely planet."


def run_verify_target(target: dict, run_dir: Path) -> dict:
    tic_id = int(target["tic_id"])
    period = float(target["period"])
    expected_kind = "planet" if target["label"] in {"KP", "CP"} else "false_positive"
    out_dir = run_dir / f"TIC_{tic_id}"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "verify_candidate.py"),
        "--tic",
        str(tic_id),
        "--period",
        str(period),
        "--out-dir",
        str(out_dir),
    ]

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=SCRIPTS_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started

    result_json = out_dir / "verification.json"
    result = read_json(result_json, None)
    if proc.returncode != 0:
        return {
            "name": target["name"],
            "tic_id": tic_id,
            "expected_kind": expected_kind,
            "period": period,
            "command": format_command(cmd),
            "status": FAIL,
            "reason": f"verify_candidate.py exited with code {proc.returncode}",
            "elapsed_s": round(elapsed, 1),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-10:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
        }
    if not result:
        return {
            "name": target["name"],
            "tic_id": tic_id,
            "expected_kind": expected_kind,
            "period": period,
            "command": format_command(cmd),
            "status": FAIL,
            "reason": "verification.json was not written",
            "elapsed_s": round(elapsed, 1),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-10:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
        }

    verdict = str(result.get("verdict", ""))
    consistency = result.get("consistency_score")
    status, reason = interpret_verification(expected_kind, verdict)
    return {
        "name": target["name"],
        "tic_id": tic_id,
        "expected_kind": expected_kind,
        "period": period,
        "command": format_command(cmd),
        "status": status,
        "reason": reason,
        "elapsed_s": round(elapsed, 1),
        "verdict": verdict,
        "consistency_score": consistency,
        "n_sectors_checked": result.get("n_sectors_checked"),
        "n_sectors_consistent": result.get("n_sectors_consistent"),
        "json_path": str(result_json),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-10:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-10:]),
    }


def summarise_status(items: list[dict]) -> str:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for item in items:
        counts[item["status"]] += 1
    if counts[FAIL]:
        return FAIL
    if counts[WARN]:
        return WARN
    return PASS


def read_active_job() -> dict:
    state = read_json(SCAN_STATE_FILE, {"running": False})
    pid = int(state.get("pid") or 0)
    if state.get("running") and pid and is_pid_alive(pid):
        return state
    return {"running": False}


def hunt_smoke_test(sector: int, workers: int, limit: int) -> list[dict]:
    active = read_active_job()
    if active.get("running"):
        label = active.get("label") or active.get("mode") or "another job"
        reason = f"Skipped because the pipeline is already busy: {label} (PID {active.get('pid')})"
        return [
            {"name": "HUNT SMOKE TEST", "status": WARN, "reason": reason},
            {"name": "EXOMINER LOGGING", "status": WARN, "reason": reason},
        ]

    out_dir = RESULTS_DIR / f"sector{sector:02d}"
    hunt_log = out_dir / "hunt.log"
    exominer_log = out_dir / "exominer_run.log"
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hunt.py"),
        "--sector",
        str(sector),
        "--workers",
        str(workers),
        "--limit",
        str(limit),
    ]

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPTS_DIR,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return [
            {
                "name": "HUNT SMOKE TEST",
                "status": FAIL,
                "reason": f"hunt.py timed out after 3600 seconds on sector {sector}",
                "command": format_command(cmd),
            },
            {
                "name": "EXOMINER LOGGING",
                "status": FAIL,
                "reason": "Could not validate ExoMiner logging because hunt.py timed out",
            },
        ]

    elapsed = time.time() - started
    hunt_text = hunt_log.read_text(errors="replace") if hunt_log.exists() else ""
    hunt_done = "✅ Done!" in hunt_text
    bls_csv = out_dir / "bls_results.csv"

    if not hunt_log.exists() or not bls_csv.exists():
        return [
            {
                "name": "HUNT SMOKE TEST",
                "status": FAIL,
                "reason": "hunt.py did not leave the expected core outputs (hunt.log and bls_results.csv)",
                "command": format_command(cmd),
                "elapsed_s": round(elapsed, 1),
                "returncode": proc.returncode,
            },
            {
                "name": "EXOMINER LOGGING",
                "status": FAIL,
                "reason": "Core hunt outputs were missing, so Step 3 logging could not be trusted",
            },
        ]

    smoke_status = PASS if hunt_done else WARN
    smoke_reason = (
        f"hunt.py completed sector {sector} and wrote the expected outputs in {elapsed/60:.1f} min."
        if hunt_done else
        f"hunt.py wrote outputs for sector {sector}, but the final done marker was missing."
    )
    if proc.returncode != 0:
        smoke_status = FAIL
        smoke_reason = f"hunt.py exited with code {proc.returncode}"

    has_step3 = "🤖 Step 3/4" in hunt_text
    has_counts = (
        "Total BLS rows:" in hunt_text and
        "Rows eligible for scoring" in hunt_text and
        "Verified rows:" in hunt_text
    )
    has_fallback = "BLS and verification results are still available below." in hunt_text
    ran_podman = "Podman command:" in hunt_text

    if not has_step3:
        exominer_status = FAIL
        exominer_reason = "Step 3 logging block was missing from hunt.log."
    elif ran_podman:
        if not exominer_log.exists():
            exominer_status = FAIL
            exominer_reason = "Step 3 says Podman ran, but exominer_run.log is missing."
        elif has_counts and ("ExoMiner++ exit code:" in hunt_text) and (
            "predictions_outputs.csv present:" in hunt_text or has_fallback
        ):
            exominer_status = PASS
            if "produced no predictions file" in hunt_text or "failed before producing predictions" in hunt_text:
                exominer_reason = "ExoMiner fallback was clean and the detailed run log was written."
            else:
                exominer_reason = "ExoMiner logging and diagnostics were present."
        else:
            exominer_status = WARN
            exominer_reason = "Step 3 ran, but some of the new diagnostics were missing or ambiguous."
    else:
        if "no candidates above threshold" in hunt_text:
            exominer_status = WARN
            exominer_reason = "Smoke sector produced no scoreable candidates, so the Podman path was not exercised."
        elif has_counts:
            exominer_status = PASS
            exominer_reason = "Step 3 diagnostics were present even though ExoMiner did not need to run."
        else:
            exominer_status = WARN
            exominer_reason = "Smoke test completed, but Step 3 diagnostics were thinner than expected."

    return [
        {
            "name": "HUNT SMOKE TEST",
            "status": smoke_status,
            "reason": smoke_reason,
            "command": format_command(cmd),
            "elapsed_s": round(elapsed, 1),
            "returncode": proc.returncode,
            "hunt_log": str(hunt_log),
            "results_dir": str(out_dir),
        },
        {
            "name": "EXOMINER LOGGING",
            "status": exominer_status,
            "reason": exominer_reason,
            "hunt_log": str(hunt_log),
            "exominer_log": str(exominer_log),
        },
    ]


def overall_status(checks: list[dict]) -> str:
    worst = PASS
    for check in checks:
        if STATUS_ORDER[check["status"]] > STATUS_ORDER[worst]:
            worst = check["status"]
    return worst


def overall_exit_code(checks: list[dict]) -> int:
    status = overall_status(checks)
    if status == FAIL:
        return 2
    if status == WARN:
        return 1
    return 0


def build_text_report(report: dict) -> str:
    lines = []
    lines.append("PIPELINE VALIDATION REPORT")
    lines.append(f"Started: {report['started_at']}")
    lines.append(f"Finished: {report['finished_at']}")
    lines.append("")

    lines.append("Section A — Known-target verification")
    for item in report["verify_planets"] + report["verify_fps"]:
        score = item.get("consistency_score")
        score_txt = f", consistency {score:.3f}" if isinstance(score, (int, float)) else ""
        verdict_txt = f", verdict: {item.get('verdict')}" if item.get("verdict") else ""
        lines.append(
            f"- {item['status']} — TIC {item['tic_id']} ({item['expected_kind']}, P={item['period']} d): "
            f"{item['reason']}{score_txt}{verdict_txt}"
        )
        lines.append(f"  Command: {item['command']}")
    lines.append("")

    lines.append("Section B — Tiny hunt smoke test")
    for item in report["smoke_checks"]:
        lines.append(f"- {item['name']}: {item['status']} — {item['reason']}")
        if item.get("command"):
            lines.append(f"  Command: {item['command']}")
    lines.append("")

    lines.append("Section C — Final summary")
    for item in report["summary_lines"]:
        lines.append(item)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Small confidence harness for the exoplanet pipeline")
    ap.add_argument("--quick", action="store_true", help="Run a smaller verification subset")
    ap.add_argument("--skip-verify", action="store_true", help="Skip known-target verification checks")
    ap.add_argument("--skip-hunt", action="store_true", help="Skip the hunt.py smoke test")
    ap.add_argument("--sector", type=int, default=95, help="Sector for the hunt smoke test (default: 95)")
    ap.add_argument("--workers", type=int, default=16, help="Workers for the hunt smoke test")
    ap.add_argument("--limit", type=int, default=20, help="Star limit for the hunt smoke test")
    args = ap.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = VALIDATION_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "started_at": now_iso(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "verify_planets": [],
        "verify_fps": [],
        "smoke_checks": [],
    }

    planets, fps = pick_targets(args.quick)

    if not args.skip_verify:
        verify_run_dir = run_dir / "verification"
        verify_run_dir.mkdir(parents=True, exist_ok=True)
        for target in planets:
            report["verify_planets"].append(run_verify_target(target, verify_run_dir))
        for target in fps:
            report["verify_fps"].append(run_verify_target(target, verify_run_dir))

    if not args.skip_hunt:
        report["smoke_checks"] = hunt_smoke_test(args.sector, args.workers, args.limit)

    checks = report["verify_planets"] + report["verify_fps"] + report["smoke_checks"]

    planet_counts = {PASS: 0, WARN: 0, FAIL: 0}
    for item in report["verify_planets"]:
        planet_counts[item["status"]] += 1
    fp_counts = {PASS: 0, WARN: 0, FAIL: 0}
    for item in report["verify_fps"]:
        fp_counts[item["status"]] += 1

    smoke_map = {item["name"]: item for item in report["smoke_checks"]}
    warnings_total = sum(1 for item in checks if item["status"] == WARN)
    failures_total = sum(1 for item in checks if item["status"] == FAIL)

    summary_lines = []
    if report["verify_planets"]:
        summary_lines.append(
            f"VERIFY KNOWN PLANETS: {summarise_status(report['verify_planets'])} "
            f"({planet_counts[PASS]}/{len(report['verify_planets'])} pass, "
            f"{planet_counts[WARN]} warning, {planet_counts[FAIL]} fail)"
        )
    if report["verify_fps"]:
        summary_lines.append(
            f"VERIFY FALSE POSITIVES: {summarise_status(report['verify_fps'])} "
            f"({fp_counts[PASS]}/{len(report['verify_fps'])} pass, "
            f"{fp_counts[WARN]} warning, {fp_counts[FAIL]} fail)"
        )
    if smoke_map.get("HUNT SMOKE TEST"):
        summary_lines.append(
            f"HUNT SMOKE TEST: {smoke_map['HUNT SMOKE TEST']['status']} "
            f"({smoke_map['HUNT SMOKE TEST']['reason']})"
        )
    if smoke_map.get("EXOMINER LOGGING"):
        summary_lines.append(
            f"EXOMINER LOGGING: {smoke_map['EXOMINER LOGGING']['status']} "
            f"({smoke_map['EXOMINER LOGGING']['reason']})"
        )

    overall = overall_status(checks) if checks else WARN
    if overall == FAIL:
        summary_lines.append(f"OVERALL: FAIL ({failures_total} failure, {warnings_total} warning)")
    elif overall == WARN:
        summary_lines.append(f"OVERALL: PASS WITH {warnings_total} WARNING{'S' if warnings_total != 1 else ''}")
    else:
        summary_lines.append("OVERALL: PASS")

    report["summary_lines"] = summary_lines
    report["finished_at"] = now_iso()
    report["overall_status"] = overall
    report["warnings_total"] = warnings_total
    report["failures_total"] = failures_total

    text_report = build_text_report(report)
    report_txt = run_dir / "validation_report.txt"
    report_json = run_dir / "validation_report.json"
    report_txt.write_text(text_report)
    report_json.write_text(json.dumps(report, indent=2))
    shutil.copyfile(report_txt, VALIDATION_DIR / "last_validation_report.txt")
    shutil.copyfile(report_json, VALIDATION_DIR / "last_validation_report.json")

    print(text_report)
    return overall_exit_code(checks)


if __name__ == "__main__":
    sys.exit(main())
