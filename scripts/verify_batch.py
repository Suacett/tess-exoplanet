"""
verify_batch.py — Batch multi-sector verification of BLS candidates.

Reads a BLS CSV, verifies candidates with SDE >= threshold using
ThreadPoolExecutor (I/O-bound MAST downloads), updates CSV in-place with
new columns: n_sectors_checked, n_sectors_consistent, consistency_score,
verified, eb_warning.

Usage:
    python verify_batch.py --csv /path/to/bls_results.csv \
                           --out-dir /opt/exoplanet/data/verification \
                           --threshold 9.0 --workers 4
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from verify_candidate import verify

VERIFY_COLS = ["n_sectors_checked", "n_sectors_consistent",
               "consistency_score", "verified", "eb_warning"]


def _verify_one(args):
    row, out_dir = args
    tic_id = int(row["tic_id"])
    period = float(row["period"])
    t0_raw = row.get("t0", "")
    try:
        t0 = float(t0_raw) if t0_raw and str(t0_raw).strip() else None
    except (ValueError, TypeError):
        t0 = None

    try:
        result = verify(
            tic_id=tic_id, period=period, t0=t0,
            out_dir=out_dir / f"TIC_{tic_id}",
        )
        verdict = result["verdict"]
        cs      = result["consistency_score"]
        n_chk   = result["n_sectors_checked"]
        n_con   = result["n_sectors_consistent"]

        if "Likely real" in verdict:
            verified = "true"
        elif n_chk == 0 or "Only 1 sector" in verdict or "No TESS" in verdict:
            verified = "insufficient"
        else:
            verified = "false"

        row.update({
            "n_sectors_checked":    n_chk,
            "n_sectors_consistent": n_con,
            "consistency_score":    round(cs, 3),
            "verified":             verified,
            "eb_warning":           result.get("eb_warning") or "",
        })
        badge = "✅" if verified == "true" else ("❌" if verified == "false" else "?")
        print(f"  {badge} TIC {tic_id}: {verdict[:70]}", flush=True)

    except Exception as exc:
        row.update({
            "n_sectors_checked": 0, "n_sectors_consistent": 0,
            "consistency_score": "", "verified": "error", "eb_warning": "",
        })
        print(f"  ⚠️  TIC {tic_id}: ERROR — {exc}", flush=True)

    return row


def verify_batch(csv_path: str, out_dir: str,
                 sde_threshold: float = 9.0, max_workers: int = 4):
    csv_path = Path(csv_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as fh:
        reader    = csv.DictReader(fh)
        all_rows  = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Ensure verification columns exist
    for col in VERIFY_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
        for row in all_rows:
            row.setdefault(col, "")

    # Only verify rows above threshold that haven't been verified yet
    to_verify = []
    for row in all_rows:
        try:
            sde = float(row.get("bls_power", 0) or 0)
        except (ValueError, TypeError):
            sde = 0.0
        already = str(row.get("verified", "")).strip()
        if sde >= sde_threshold and already in ("", "error"):
            to_verify.append(row)

    n_total = len(to_verify)
    print(f"verify_batch: verifying {n_total} candidates (SDE ≥ {sde_threshold})",
          flush=True)

    progress_file = out_dir / "verify_progress.json"
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_verify_one, (row, out_dir)): row
            for row in to_verify
        }
        for fut in as_completed(futures):
            done += 1
            try:
                progress_file.write_text(json.dumps({
                    "done": done, "total": n_total,
                    "pct": round(100 * done / max(n_total, 1)),
                    "complete": done >= n_total,
                }))
            except Exception:
                pass

    # Write updated CSV
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    n_ver   = sum(1 for r in all_rows if r.get("verified") == "true")
    n_false = sum(1 for r in all_rows if r.get("verified") == "false")
    n_insuf = sum(1 for r in all_rows if r.get("verified") == "insufficient")
    print(f"verify_batch complete: {n_ver} verified ✅  {n_false} rejected ❌  "
          f"{n_insuf} insufficient data", flush=True)

    # Mark complete
    try:
        progress_file.write_text(json.dumps({
            "done": n_total, "total": n_total, "pct": 100, "complete": True,
        }))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True)
    ap.add_argument("--out-dir",   default=str(Path(__file__).resolve().parent.parent / "data/verification"))
    ap.add_argument("--threshold", type=float, default=9.0)
    ap.add_argument("--workers",   type=int,   default=4)
    args = ap.parse_args()
    verify_batch(args.csv, args.out_dir, args.threshold, args.workers)


if __name__ == "__main__":
    main()
