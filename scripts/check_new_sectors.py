#!/usr/bin/env python3
"""
Check for newly available TESS sectors with SPOC 2-min lightcurve products.
Run daily to see if fresh data is ready to scan.

Usage:
    python check_new_sectors.py               # check and print report
    python check_new_sectors.py --mark 99     # manually mark a sector as scanned
    python check_new_sectors.py --json        # machine-readable JSON output
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import requests

TRACKING_FILE = Path(__file__).resolve().parent.parent / "data/scanned_sectors.json"
BULK_INDEX_URL = (
    "https://archive.stsci.edu/tess/bulk_downloads/"
    "bulk_downloads_ffi-tp-lc-dv.html"
)


# ── Tracking file helpers ─────────────────────────────────────────────────────

def load_tracking() -> dict:
    if TRACKING_FILE.exists():
        return json.loads(TRACKING_FILE.read_text())
    return {"scanned": [], "available": [], "last_checked": None}


def save_tracking(data: dict) -> None:
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKING_FILE.write_text(json.dumps(data, indent=2))


# ── MAST query ────────────────────────────────────────────────────────────────

def get_available_sectors() -> list:
    """
    Scrape the MAST bulk-download index to find all sectors that have
    SPOC 2-min lightcurve products (tesscurl_sector_NN_lc.sh links).
    """
    resp = requests.get(BULK_INDEX_URL, timeout=30)
    resp.raise_for_status()
    # Exclude the anomalous "sector 1751" entry (a Cycle label, not a sector)
    sectors = sorted(
        {int(m) for m in re.findall(r"sector_(\d+)_lc", resp.text) if int(m) < 500}
    )
    return sectors


def get_sector_release_date(sector: int) -> str | None:
    """
    Try to extract the release date for a sector from the first filename
    in its bulk-download script (tess<YYYYDDD>... encodes the observation date).
    Returns ISO date string or None.
    """
    try:
        url = (
            f"https://archive.stsci.edu/missions/tess/download_scripts/sector/"
            f"tesscurl_sector_{sector:02d}_lc.sh"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        m = re.search(r"tess(\d{4})(\d{3})\d+-s", resp.text)
        if m:
            year, doy = int(m.group(1)), int(m.group(2))
            from datetime import datetime
            d = datetime(year, 1, 1) + __import__("datetime").timedelta(days=doy - 1)
            return d.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check for new TESS sectors")
    parser.add_argument("--mark",  type=int, metavar="SECTOR",
                        help="Mark a sector as already scanned")
    parser.add_argument("--json",  action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--unmark", type=int, metavar="SECTOR",
                        help="Remove a sector from the scanned list")
    args = parser.parse_args()

    tracking = load_tracking()

    # ── Manual mark/unmark ────────────────────────────────────────────────────
    if args.mark:
        scanned = set(tracking.get("scanned", []))
        scanned.add(args.mark)
        tracking["scanned"] = sorted(scanned)
        save_tracking(tracking)
        print(f"Sector {args.mark} marked as scanned.")
        return

    if args.unmark:
        scanned = set(tracking.get("scanned", []))
        scanned.discard(args.unmark)
        tracking["scanned"] = sorted(scanned)
        save_tracking(tracking)
        print(f"Sector {args.unmark} removed from scanned list.")
        return

    # ── Fetch available sectors ───────────────────────────────────────────────
    if not args.json:
        print("Querying MAST for available TESS sectors...")
    try:
        available = get_available_sectors()
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"ERROR: Could not reach MAST: {e}", file=sys.stderr)
        sys.exit(1)

    scanned     = set(tracking.get("scanned", []))
    new_sectors = [s for s in available if s not in scanned]
    latest      = max(available) if available else None

    # Update tracking
    tracking["available"]     = available
    tracking["last_checked"]  = str(date.today())
    save_tracking(tracking)

    # ── Output ────────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps({
            "available":    available,
            "scanned":      sorted(scanned),
            "new_sectors":  new_sectors,
            "latest_sector": latest,
            "checked":      str(date.today()),
        }, indent=2))
        return

    print(f"\n{'─'*56}")
    print(f"  TESS Sector Status — {date.today()}")
    print(f"{'─'*56}")
    print(f"  Available on MAST : {len(available):>4}  (latest: S{latest})")
    print(f"  Already scanned   : {len(scanned):>4}")
    print(f"  New / unscanned   : {len(new_sectors):>4}")
    print(f"{'─'*56}")

    if new_sectors:
        print(f"\n  New sectors available to scan:")
        for s in new_sectors[-10:]:   # show at most last 10
            rel = get_sector_release_date(s)
            date_str = f"  (obs start: {rel})" if rel else ""
            print(f"    S{s:>3}{date_str}")
        if len(new_sectors) > 10:
            print(f"    ... and {len(new_sectors) - 10} more")

        latest_new = new_sectors[-1]
        print(f"\n  Scan the latest new sector:")
        print(f"    python /opt/exoplanet/scripts/scan_sector.py --sector {latest_new}")
        print(f"\n  Quick test (20 stars):")
        print(f"    python /opt/exoplanet/scripts/scan_sector.py --sector {latest_new} --limit 20")
    else:
        print("\n  No new sectors. All available sectors have been scanned.")
        if latest:
            print(f"  Check back after sector {latest + 1} is released (~27 days from last scan).")

    print()
    if scanned:
        print(f"  Scanned sectors: {sorted(scanned)}")
    print(f"  Tracking file: {TRACKING_FILE}")


if __name__ == "__main__":
    main()
