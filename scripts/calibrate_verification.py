"""
Calibration script for verify_candidate.py.

Runs verify() on 5 known planets and 5 known false positives, then generates
an HTML calibration report with per-target plots and a summary table.

Usage:
    python calibrate_verification.py [--out /path/to/report.html]

Output:
    /opt/exoplanet/data/verification/calibration_report.html   (default)
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# ── Make sure we can import from the scripts directory ───────────────────────
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from verify_candidate import verify

# ── Calibration targets ───────────────────────────────────────────────────────
KNOWN_PLANETS = [
    {"tic_id": 66818296,  "period": 3.73548, "label": "KP", "name": "WASP-17 b"},
    {"tic_id": 375506058, "period": 2.65022, "label": "KP", "name": "TOI-1431 b"},
    {"tic_id": 25155310,  "period": 3.2888,  "label": "KP", "name": "KP (ExoMiner 0.9985)"},
    {"tic_id": 19028197,  "period": 3.3366,  "label": "KP", "name": "KP (ExoMiner 0.9985)"},
    {"tic_id": 36724087,  "period": 0.7684,  "label": "CP", "name": "CP (ExoMiner 0.998)"},
]

KNOWN_FPS = [
    {"tic_id": 30312676, "period": 1.10, "label": "FP", "name": "FP TIC 30312676"},
    {"tic_id": 28873762, "period": 3.11, "label": "FP", "name": "FP TIC 28873762"},
    {"tic_id": 22843856, "period": 1.85, "label": "FP", "name": "FP TIC 22843856"},
    {"tic_id": 9033144,  "period": 4.72, "label": "FP", "name": "FP TIC 9033144"},
    {"tic_id": 32830028, "period": 0.52, "label": "FP", "name": "FP TIC 32830028"},
]

ALL_TARGETS = KNOWN_PLANETS + KNOWN_FPS

# ── Base output directory ─────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "verification"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Verdict → pass/fail logic ─────────────────────────────────────────────────
def is_correct(verdict: str, label: str) -> bool:
    """True when the verdict matches the expected label."""
    real   = "Likely real" in verdict
    not_pl = "Likely not" in verdict or "Inconclusive" in verdict
    if label in ("KP", "CP"):
        return real
    else:  # FP
        return not_pl


# ── HTML helpers ──────────────────────────────────────────────────────────────
CSS = """
body{background:#0d1117;color:#c9d1d9;font-family:monospace;margin:0;padding:20px}
h1,h2{color:#e6edf3}
table{border-collapse:collapse;width:100%;margin-bottom:30px}
th{background:#161b22;color:#58a6ff;padding:8px 12px;text-align:left;
   border-bottom:2px solid #30363d}
td{padding:7px 12px;border-bottom:1px solid #21262d}
tr:hover td{background:#1c2128}
.pass{color:#3fb950;font-weight:bold}
.fail{color:#ff7b72;font-weight:bold}
.inconc{color:#f0883e}
.accuracy{font-size:1.4em;font-weight:bold;margin:10px 0}
.target-block{background:#161b22;border:1px solid #30363d;border-radius:8px;
               padding:16px;margin-bottom:24px}
.target-block h3{color:#58a6ff;margin-top:0}
.metric{display:inline-block;background:#0d1117;border:1px solid #30363d;
        border-radius:4px;padding:4px 10px;margin:3px;font-size:0.85em}
img{max-width:100%;border-radius:4px;margin-top:12px}
.sector-table td:first-child{color:#8b949e}
"""


def _encode_png(path: Path) -> str:
    """Return base64-encoded PNG for inline embedding."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def verdict_class(verdict: str) -> str:
    if "Likely real" in verdict:
        return "pass"
    if "Likely not" in verdict:
        return "fail"
    return "inconc"


def build_html(results: list[dict]) -> str:
    n_total  = len(results)
    n_pass   = sum(1 for r in results if r["correct"])
    accuracy = n_pass / n_total * 100

    acc_color = "#3fb950" if n_pass >= 8 else "#f0883e" if n_pass >= 6 else "#ff7b72"

    rows_html = ""
    for r in results:
        vclass = verdict_class(r["verdict"])
        correct_str = '<span class="pass">PASS</span>' if r["correct"] else '<span class="fail">FAIL</span>'
        rows_html += (
            f"<tr>"
            f"<td>TIC {r['tic_id']}</td>"
            f"<td>{r['name']}</td>"
            f"<td>{r['label']}</td>"
            f"<td class='{vclass}'>{r['verdict']}</td>"
            f"<td>{r.get('consistency_score', 0):.0%}</td>"
            f"<td>{r.get('n_sectors_consistent', '?')}/{r.get('n_sectors_checked', '?')}</td>"
            f"<td>{correct_str}</td>"
            f"<td>{r.get('elapsed_s', 0):.0f}s</td>"
            f"</tr>\n"
        )

    # Per-target blocks
    blocks_html = ""
    for r in results:
        border_color = "#3fb950" if r["correct"] else "#ff7b72"
        img_tag = ""
        _pp = r.get("plot_path", "")
        if _pp:
            plot_path = Path(_pp)
            if plot_path.is_file():
                img_tag = f'<img src="data:image/png;base64,{_encode_png(plot_path)}" alt="verification plot">'

        # sector table
        sector_rows = ""
        for sr in r.get("sector_results", []):
            flag = "✅" if sr.get("has_signal") else "❌"
            sector_rows += (
                f"<tr>"
                f"<td>Sector {sr['sector']}</td>"
                f"<td>{sr.get('depth_ppm', 0):.0f} ppm</td>"
                f"<td>{sr.get('snr', 0):.1f}</td>"
                f"<td>{sr.get('n_in_transit', 0)}</td>"
                f"<td>{flag}</td>"
                f"</tr>\n"
            )
        sector_table = ""
        if sector_rows:
            sector_table = (
                "<table class='sector-table' style='margin-top:10px;width:auto'>"
                "<tr><th>Sector</th><th>Depth</th><th>SNR</th><th>In-transit pts</th><th>Signal</th></tr>"
                + sector_rows + "</table>"
            )

        vclass = verdict_class(r["verdict"])
        correct_lbl = "✅ PASS" if r["correct"] else "❌ FAIL"
        blocks_html += f"""
<div class="target-block" style="border-color:{border_color}">
  <h3>{r['name']} — TIC {r['tic_id']} &nbsp; ({r['label']}) &nbsp; {correct_lbl}</h3>
  <span class="metric">Period: {r.get('period','?')} d</span>
  <span class="metric">Sectors checked: {r.get('n_sectors_checked','?')}</span>
  <span class="metric">Sectors consistent: {r.get('n_sectors_consistent','?')}</span>
  <span class="metric">Consistency: {r.get('consistency_score',0):.0%}</span>
  <span class="metric">Median depth: {r.get('median_depth_ppm',0):.0f} ppm</span>
  <br><br>
  <b class="{vclass}">Verdict: {r['verdict']}</b>
  {sector_table}
  {img_tag}
  {'<p style="color:#ff7b72">Error: '+r['error']+'</p>' if r.get('error') else ''}
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Verification Calibration Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>Verification Calibration Report</h1>
<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>
<p class="accuracy" style="color:{acc_color}">
  Overall accuracy: {n_pass}/{n_total} ({accuracy:.0f}%)
  &nbsp;{'✅ ACCEPTABLE' if n_pass >= 8 else '❌ BELOW TARGET (need ≥ 8/10)'}
</p>

<h2>Summary Table</h2>
<table>
<tr>
  <th>TIC ID</th><th>Name</th><th>Label</th><th>Verdict</th>
  <th>Consistency</th><th>Sectors</th><th>Result</th><th>Time</th>
</tr>
{rows_html}
</table>

<h2>Per-Target Detail</h2>
{blocks_html}
</body>
</html>
"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Calibrate verify_candidate.py on 10 known targets")
    ap.add_argument("--out", default=str(DATA_DIR / "calibration_report.html"),
                    help="Output HTML path")
    ap.add_argument("--targets", default=None,
                    help="Comma-separated TIC IDs to run (default: all 10)")
    args = ap.parse_args()

    target_filter = None
    if args.targets:
        target_filter = set(int(x.strip()) for x in args.targets.split(","))

    results = []

    for i, target in enumerate(ALL_TARGETS):
        tic   = target["tic_id"]
        if target_filter and tic not in target_filter:
            continue

        label = target["label"]
        name  = target["name"]
        period = target["period"]

        print(f"\n[{i+1:02d}/{len(ALL_TARGETS)}] TIC {tic} — {name} (expected: {label})")
        print(f"  Period: {period} d")

        t_start = time.time()
        try:
            out_subdir = DATA_DIR / f"TIC_{tic}"
            result = verify(tic_id=tic, period=period, out_dir=out_subdir)
            elapsed = time.time() - t_start

            verdict = result["verdict"]
            cs      = result["consistency_score"]
            n_con   = result["n_sectors_consistent"]
            n_check = result["n_sectors_checked"]
            correct = is_correct(verdict, label)

            plot_path = out_subdir / "verification_plot.png"

            print(f"  Verdict: {verdict}")
            print(f"  Consistency: {n_con}/{n_check} ({cs:.0%})")
            print(f"  Result: {'✅ PASS' if correct else '❌ FAIL'}  ({elapsed:.0f}s)")

            results.append({
                "tic_id":               tic,
                "name":                 name,
                "label":                label,
                "period":               period,
                "verdict":              verdict,
                "consistency_score":    cs,
                "n_sectors_checked":    n_check,
                "n_sectors_consistent": n_con,
                "median_depth_ppm":     result.get("median_depth_ppm", 0),
                "sector_results":       result.get("sector_results", []),
                "plot_path":            str(plot_path) if plot_path.exists() else "",
                "correct":              correct,
                "elapsed_s":            elapsed,
                "error":                None,
            })

        except Exception as exc:
            elapsed = time.time() - t_start
            import traceback
            tb = traceback.format_exc()
            print(f"  ERROR: {exc}")
            print(tb)
            results.append({
                "tic_id":               tic,
                "name":                 name,
                "label":                label,
                "period":               period,
                "verdict":              "Error",
                "consistency_score":    0.0,
                "n_sectors_checked":    0,
                "n_sectors_consistent": 0,
                "median_depth_ppm":     0,
                "sector_results":       [],
                "plot_path":            "",
                "correct":              False,
                "elapsed_s":            elapsed,
                "error":                str(exc),
            })

    # ── Print final summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    n_pass = sum(1 for r in results if r["correct"])
    n_total = len(results)
    print(f"Overall: {n_pass}/{n_total} correct ({n_pass/n_total*100:.0f}%)")
    print()
    for r in results:
        flag = "✅" if r["correct"] else "❌"
        print(f"  {flag} TIC {r['tic_id']:>9}  {r['label']}  {r['verdict'][:50]}")

    # ── Write HTML report ─────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(results)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written to: {out_path}")

    # ── Write JSON summary (for programmatic use) ─────────────────────────────
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as fh:
        json.dump({
            "n_pass": n_pass,
            "n_total": n_total,
            "accuracy": n_pass / n_total if n_total else 0,
            "results": [{k: v for k, v in r.items() if k not in ("sector_results", "plot_path")}
                        for r in results],
        }, fh, indent=2)
    print(f"JSON  written to: {json_path}")

    return 0 if n_pass >= 8 else 1


if __name__ == "__main__":
    sys.exit(main())
