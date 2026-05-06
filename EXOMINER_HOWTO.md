# ExoMiner++ — How to Run It

_Written from the actual docs in `/opt/exoplanet/ExoMiner/docs/`._

---

## What ExoMiner++ Does

ExoMiner++ is a NASA deep learning model that scores transit candidates from TESS. Given a list of TIC IDs, it:

1. Downloads their SPOC lightcurve FITS files and Data Validation (DV) XML reports from MAST
2. Extracts ~50 features from the lightcurve (global/local flux views, odd/even transit depths, centroid shift, secondary eclipse, stellar parameters, etc.)
3. Runs those features through a neural network
4. Outputs a **score from 0 to 1** per TCE (Transit Candidate Event): scores near 1 = likely planet, near 0 = junk/false positive

The pipeline does *not* find new transit candidates itself — it scores them. The transit candidates (TCEs) come from the SPOC pipeline at MAST. You tell ExoMiner which TIC IDs and sector runs to score.

---

## What You Feed It

**Input: a CSV file** with two columns:

```csv
tic_id, sector_run
402026209, 1-39
402026209, 56-56
103633434, 3-3
```

- `tic_id`: the TESS Input Catalog ID (the number in "TIC 402026209")
- `sector_run`: the sector range. For single sectors use `6-6`. For multi-sector searches use `1-39` (sectors 1 through 39). You need at least one TCE to have been detected by SPOC for that sector run — if SPOC found nothing, ExoMiner has nothing to score.

**Where to get TIC IDs and sector runs to score:**
- From the Zenodo catalog (`exominerplusplus_catalog_unk_tces.csv`) — these are all TCEs SPOC found in sectors 1–67 that ExoMiner has already scored
- From the TOI catalog at https://tev.mit.edu/data/ — confirmed/candidate planets
- From a fresh BLS search in the Phase 2 notebook — if you find a new signal, grab the TIC ID and sector and score it

---

## How to Run It (Podman)

The pipeline lives at `ghcr.io/nasa/exominer:latest` (already pulled on this machine).

### Quick Run Script

The wrapper script is at `/opt/exoplanet/ExoMiner/exominer_pipeline/run_podman_application.sh`. Here's the minimal working version for the P620:

```bash
#!/bin/bash
# Edit these four variables, then run this script

INPUTS_DIR="/opt/exoplanet/data/candidates/my-run"
TICS_FILE="$INPUTS_DIR/tics_tbl.csv"
RUN_DIR="/opt/exoplanet/data/candidates/my-run/output"

mkdir -p "$RUN_DIR"

# Create input CSV if it doesn't exist
# cat > "$TICS_FILE" << 'EOF'
# tic_id, sector_run
# 402026209, 56-56
# EOF

podman run \
  -v "$TICS_FILE:/tics_tbl.csv:Z" \
  -v "$RUN_DIR:/outputs:Z" \
  ghcr.io/nasa/exominer:latest \
  --tic_ids_fp=/tics_tbl.csv \
  --output_dir=/outputs \
  --data_collection_mode=2min \
  --num_processes=16 \
  --num_jobs=4 \
  --download_spoc_data_products=true \
  --stellar_parameters_source=ticv8 \
  --ruwe_source=gaiadr2 \
  --exominer_model=exominer++_single
```

**Key flags:**
- `--num_processes=16`: use all 16 Threadripper cores
- `--num_jobs=4`: split TICs across 4 parallel jobs (set this to `ceil(n_tics / desired_batch_size)`)
- `--data_collection_mode=2min`: SPOC 2-minute cadence (vs `ffi` for Full Frame Image)
- `--exominer_model`: choose one (see Models section below)

### Using Pre-downloaded Lightcurves

If you've already downloaded lightcurves to `/opt/exoplanet/data/tess/`, you can skip re-downloading by mounting that as `external_data_repository`. It must contain both the `_lc.fits` files AND the `_dv.xml` files organized by MAST's directory structure:

```bash
podman run \
  -v "$TICS_FILE:/tics_tbl.csv:Z" \
  -v "$RUN_DIR:/outputs:Z" \
  -v "/opt/exoplanet/data/tess:/external_data_repository:Z" \
  ghcr.io/nasa/exominer:latest \
  --tic_ids_fp=/tics_tbl.csv \
  --output_dir=/outputs \
  --data_collection_mode=2min \
  --external_data_repository=/external_data_repository \
  --num_processes=16 \
  --num_jobs=4 \
  --stellar_parameters_source=ticv8 \
  --ruwe_source=gaiadr2 \
  --exominer_model=exominer++_single
```

**Note:** The bulk-downloaded `.fits` files in `/opt/exoplanet/data/tess/sector99/` are lightcurve files only — they don't include the DV XML files. For Phase 3, let the pipeline download everything itself for your TIC list. Use the pre-downloaded files for lightkurve / BLS analysis.

---

## What Comes Out

After a successful run, `$RUN_DIR` contains:

```
my-run/
├── run_main.log                    ← check here first if something goes wrong
├── predictions_outputs.csv         ← THE MAIN RESULT — scores for all TCEs
├── dv_reports_all_jobs.csv         ← URLs to SPOC PDF reports for each TCE
├── pipeline_run_config.yaml
└── job_0/
    ├── predictions/
    │   └── ranked_predictions_predictset.csv   ← per-job scores
    ├── tce_table/
    │   └── tess-spoc-dv_tces_0_processed.csv  ← full TCE table with features
    └── mastDownload/               ← raw FITS + DV XML files
```

**`predictions_outputs.csv` columns:**
- `tic_id`, `sector_run`, `tce_plnt_num` — identifies the TCE
- `exominer_score` — the score (0–1)
- `tce_period`, `tce_duration`, `tce_time0bk` — orbital period, duration, epoch
- `tce_depth` — transit depth in ppm

Load this into the Dashboard → Candidate Browser to filter by score.

---

## The Three Models

| Model | Parameters | Speed | Use When |
|-------|-----------|-------|----------|
| `exominer++_single` | ~1M | Fastest | First pass, exploring a new sector |
| `exominer++_cviter-mean-ensemble` | ~10M | Medium | When you want better calibrated probabilities |
| `exominer++_cv-super-mean-ensemble` | ~100M | Slowest | Final candidates before follow-up |

The P620 with 16 cores can handle all three. Start with `_single` for new sector runs and use `_cv-super-mean-ensemble` for your best candidates.

---

## The Features ExoMiner Uses

These are the ~50 features extracted from each TCE (from `exominer-features.md`):

**Lightcurve views (time-series):**
- `global_flux_view_fluxnorm` [301 pts] — full orbit phase-folded
- `local_flux_view_fluxnorm` [31 pts] — zoomed into transit
- `local_flux_odd_view_fluxnorm` / `local_flux_even_view_fluxnorm` — odd/even transits separately (EB diagnostic)
- `local_weak_secondary_view_selfnorm` — phase-folded at 0.5 to look for secondary eclipse
- `local_centr_view_std_noclip` — centroid time series (does the star move during transit?)
- `flux_trend_global_norm` — long-term trend
- `pgram_smooth_norm` — Lomb-Scargle periodogram (stellar variability)
- `local_momentum_dump_view` — spacecraft reaction wheel dumps (systematics)
- `unfolded_local_flux_view_fluxnorm` [20×31] — individual transit snapshots

**Difference imaging:**
- `diff_imgs_std_trainset` [33×33×5] — pixel difference images during vs out of transit
- `oot_imgs_std_trainset` [33×33×5] — out-of-transit reference images
- `target_imgs` [33×33×5] — target star pixel images

**Scalar features:**
- `tce_maxmes_norm` — Multiple Event Statistic (signal strength)
- `boot_fap_norm` — bootstrap false alarm probability
- `tce_cap_stat_norm` / `tce_hap_stat_norm` — centroid offset statistics
- `tce_albedo_stat_norm` / `tce_ptemp_stat_norm` — planet temperature checks
- `tce_steff_norm`, `tce_sradius_norm`, `tce_smass_norm`, `tce_slogg_norm` — stellar params
- `ruwe_norm` — Gaia astrometric excess noise (nearby stars can dilute/contaminate)

If you want to modify ExoMiner's architecture or weights, you need to change what features it uses or add new ones. The custom model path option (`--exominer_model /path/to/model.keras`) lets you drop in a TensorFlow 2.13 Keras model that uses any subset of these features.

---

## Phase 3 Workflow

For scoring a fresh sector (e.g. sector 99):

1. **Find TICs with SPOC TCEs in sector 99** — query MAST for the DV results:
   ```python
   # In JupyterLab:
   from astroquery.mast import Observations
   obs = Observations.query_criteria(obs_collection="TESS", sequence_number=99,
                                      dataproduct_type="timeseries", calib_level=3)
   # Get TIC IDs from the results
   ```

2. **Build your input CSV:**
   ```csv
   tic_id, sector_run
   860419, 99-99
   821838, 99-99
   ...
   ```

3. **Run ExoMiner:**
   ```bash
   pct exec 100 -- /opt/exoplanet/ExoMiner/exominer_pipeline/run_podman_application.sh \
     --tics_tbl_fp /opt/exoplanet/data/candidates/sector99/tics_tbl.csv \
     --exominer_pipeline_run_dir /opt/exoplanet/data/candidates/sector99/output \
     --data_collection_mode 2min \
     --num_processes 16 \
     --num_jobs 8 \
     --exominer_model exominer++_single
   ```

4. **Browse results** in Dashboard → Candidate Browser, load `predictions_outputs.csv`

5. **Inspect top candidates** in Dashboard → Lightcurve Inspector

---

## Architecture Note

The image is tagged as `arm64` in the wrapper script but the actual image at `ghcr.io/nasa/exominer:latest` is the amd64 build. If you get architecture errors, explicitly use `ghcr.io/nasa/exominer:latest` (not `:arm64`). The image was built for `linux/amd64` which matches the P620's Threadripper PRO.

---

## Contacts / Issues

GitHub: https://github.com/nasa/ExoMiner/issues
Paper: https://doi.org/10.48550/arXiv.2502.09790
