# TESS Exoplanet Transit Search Pipeline

An automated pipeline for searching NASA TESS photometry data for exoplanet
transit signals, running BLS period analysis, and scoring candidates with
NASA's ExoMiner++ neural network.

> **Note:** This README is a work in progress. Full setup guide, screenshots,
> and BLS walkthrough coming soon.

---

## What this does

1. **Download** — Fetches TESS light curves from NASA MAST for a given sector
2. **Detect** — Runs Box Least Squares (BLS) transit search across all targets
3. **Score** — Pipes strong candidates through NASA's ExoMiner++ classifier
4. **Report** — Outputs a ranked candidate list with disposition probabilities

---

## Built on

This pipeline is a wrapper and orchestration layer around existing NASA and
community tools. **The underlying ML models are NASA's work, not mine.**

| Component | What it does | Credit |
|-----------|-------------|--------|
| **ExoMiner++** | Neural network transit classifier | NASA Ames Research Center |
| **lightkurve** | TESS light curve download & processing | Lightkurve Collaboration |
| **astropy** | Astronomy utilities | The Astropy Collaboration |
| **TESS data** | Space photometry | NASA/MIT TESS mission via MAST |

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full attribution and
license details for each component.

---

## Getting started

Full setup instructions are in [EXOMINER_HOWTO.md](EXOMINER_HOWTO.md).

Quick start (Ubuntu 24.04 LXC / bare metal):

```bash
# Run a BLS search on sector 10
python scripts/hunt.py --sector 10

# Launch the Streamlit dashboard
streamlit run dashboard.py

# Score candidates with ExoMiner++
python scripts/score_candidates.py --sector 10
```

The pipeline writes results to `data/results/sectorNN/`.

---

## Citation

If you use this pipeline in research, please also cite the underlying tools:

- ExoMiner++: Valizadegan et al. (NASA Ames) — https://github.com/nasa/ExoMiner
- lightkurve: Lightkurve Collaboration (2018) — https://docs.lightkurve.org
- astropy: Astropy Collaboration (2013, 2018, 2022)
- TESS: Ricker et al. (2015)

---

## License

Original pipeline code: **MIT** — see [LICENSE](LICENSE).  
Third-party components: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
