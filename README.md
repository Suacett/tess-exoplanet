# TESS Exoplanet Transit Search Pipeline

Automated pipeline for hunting exoplanet transit signals in NASA TESS photometry data.

It downloads TESS light curves from NASA MAST, runs a Box Least Squares (BLS) period search across every star in a sector, and pipes the strongest candidates through NASA's ExoMiner++ neural network classifier to separate genuine transits from astrophysical false positives.

---

## Screenshot

<!-- TODO: Add Streamlit dashboard screenshot -->
> *Screenshot coming soon — run `streamlit run scripts/dashboard.py` to see the live pipeline dashboard.*

---

## Installation

### Requirements
- Python 3.12
- [Podman](https://podman.io/docs/installation) — for ExoMiner++ scoring
- ~10 GB free disk per sector scanned (light curves + results)

### Setup

```bash
git clone https://github.com/Suacett/tess-exoplanet.git
cd tess-exoplanet
./bootstrap.sh
```

`bootstrap.sh` checks for Python 3.12, creates the virtualenv, installs all dependencies, creates the runtime data directories, and pulls the ExoMiner++ container image (one-time, ~4 GB). It prints install hints if Python 3.12 or Podman are missing.

> **ExoMiner++ weights** are bundled inside the official NASA container image — no separate download is needed.

---

## Usage

```bash
source venv/bin/activate

# Launch the full dashboard
streamlit run scripts/dashboard.py
# Open http://localhost:8501

# Run a full BLS + ExoMiner++ sector search
python scripts/hunt.py --sector 10

# Quick test (50 stars, no scoring)
python scripts/hunt.py --sector 10 --limit 50 --no-score
```

Results are written to `data/results/sector10/`.

### Other scripts

| Script | Purpose |
|--------|---------|
| `scripts/scan_sector.py` | BLS search only, no ExoMiner++ |
| `scripts/score_candidates.py` | Score an existing BLS results CSV with ExoMiner++ |
| `scripts/verify_candidate.py` | Multi-sector consistency check for a single candidate |
| `scripts/deep_scan.py` | Multi-sector stitched BLS for long-period planets (>27 days) |
| `scripts/prefetch_sector.py` | Pre-download FITS files for offline scanning |
| `scripts/check_new_sectors.py` | Check MAST for newly available TESS sectors |
| `scripts/monitor.sh` | Live system monitor (requires `lm-sensors` and `sysstat`: `sudo apt install lm-sensors sysstat`) |

For ExoMiner++ invocation details and output format see [EXOMINER_HOWTO.md](EXOMINER_HOWTO.md).

---

## How It Works

### 1 — Light curve download

[lightkurve](https://docs.lightkurve.org) queries NASA MAST for all targets in a TESS sector observed at 2-minute cadence by the SPOC pipeline. Each star's pixel data is downloaded as a FITS file and the pre-extracted PDCSAP flux (systematics-corrected) is used.

### 2 — BLS period search

For each target the pipeline runs a **Box Least Squares** (BLS) periodogram ([Kovács et al. 2002](https://doi.org/10.1051/0004-6361:20020704)) over a period grid from 0.5 to 13.5 days (the maximum detectable with a single 27-day TESS sector).

BLS folds the light curve at each trial period and fits a box-shaped dip, computing a Signal Detection Efficiency (SDE) score. Peaks in SDE space indicate periodic dimming events consistent with a transiting object. The pipeline retains candidates above a configurable SDE threshold (default 9.0) and records their period, epoch, depth, and duration.

This step runs across all targets in the sector in parallel using Python's `multiprocessing` — 16 workers on a Threadripper PRO can process a full sector (~20,000 stars) in a few hours.

### 3 — ExoMiner++ scoring

BLS finds *periodic signals* but can't distinguish planets from eclipsing binaries, systematic artefacts, or other astrophysical impostors. The pipeline feeds strong BLS candidates to **ExoMiner++**, a deep neural network trained by NASA Ames on the full TESS Object of Interest (TOI) catalogue.

ExoMiner++ runs via the official NASA container image (`ghcr.io/nasa/exominer:latest`) through Podman. Given a list of TIC IDs and sector runs, it:

1. Downloads SPOC Data Validation (DV) reports and light curve FITS files from MAST
2. Extracts ~50 features per Transit Candidate Event (TCE): phase-folded flux views (global and local), odd/even transit comparison, secondary eclipse search, centroid time series, Lomb-Scargle periodogram, and stellar parameters from TICv8 and Gaia DR2
3. Passes these features through the trained network
4. Outputs a score from 0 (false positive) to 1 (planet) per TCE

Scores above ~0.5 are worth inspecting; scores above ~0.9 are strong candidates.

### 4 — Verification

`verify_candidate.py` downloads all available TESS sectors for a candidate, phase-folds each one at the candidate period, and checks for consistent transit depth and SNR across sectors. A real planet shows a reproducible dip; instrumental systematics usually don't.

---

## Data Sources

| Source | What | Access |
|--------|------|--------|
| [NASA MAST](https://mast.stsci.edu) | TESS light curves (SPOC 2-min, PDCSAP) | Automatic via astroquery |
| [ExoFOP / TOI catalogue](https://tev.mit.edu/data/) | Known planet candidates for cross-matching | Manual CSV download |
| [Zenodo ExoMiner++ catalogue](https://zenodo.org/record/15466292) | Pre-scored TCEs sectors 1–67 | Manual CSV download |
| [Gaia DR2](https://www.cosmos.esa.int/web/gaia/dr2) | Stellar RUWE for contamination checks | Via ExoMiner++ pipeline |

All TESS data is provided by NASA/MIT and is freely available for scientific use.

---

## License

Original pipeline code: **MIT** — see [LICENSE](LICENSE).

Third-party components (ExoMiner++, lightkurve, astropy, TESS data) are subject to their own licenses — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

---

## Acknowledgements

- **NASA TESS mission** — data provided by the MIT/NASA TESS Science Team via MAST
- **ExoMiner++** — Valizadegan et al., NASA Ames Research Center ([arXiv:2502.09790](https://arxiv.org/abs/2502.09790), [GitHub](https://github.com/nasa/ExoMiner))
- **lightkurve** — Lightkurve Collaboration (2018), [docs.lightkurve.org](https://docs.lightkurve.org)
- **astropy** — Astropy Collaboration (2013, 2018, 2022), [astropy.org](https://www.astropy.org)
