# Third-Party Licenses

This project integrates or depends on the following third-party components.

---

## ExoMiner++

- **Source:** https://github.com/nasa/ExoMiner
- **License:** NASA Open Source Agreement (NOSA) 1.3
- **Credit:** Developed by NASA Ames Research Center

The ExoMiner++ neural network models and associated code are the work of NASA,
not the author of this repository. This project clones the NASA repo at setup
time and invokes it as an external pipeline stage via Podman. No ExoMiner++
source code or model weights are redistributed here.

---

## lightkurve

- **Source:** https://docs.lightkurve.org / https://github.com/lightkurve/lightkurve
- **License:** MIT License
- **Credit:** Lightkurve Collaboration (NASA Kepler/TESS community tool)

---

## astropy

- **Source:** https://www.astropy.org / https://github.com/astropy/astropy
- **License:** BSD 3-Clause License
- **Credit:** The Astropy Collaboration

---

## TESS mission data

- **Source:** NASA/MIT TESS mission, retrieved via MAST (https://mast.stsci.edu)
- **License:** NASA public domain / no redistribution restrictions for science use
- **Credit:** NASA Ames Research Center, MIT, TESS Science Team

---

## Other Python dependencies

The following packages are used as dependencies (see `setup.sh` for the full
list) and are each subject to their own open-source licenses:

| Package     | License     |
|-------------|-------------|
| numpy       | BSD 3-Clause |
| scipy       | BSD 3-Clause |
| pandas      | BSD 3-Clause |
| matplotlib  | PSF / BSD   |
| plotly      | MIT         |
| streamlit   | Apache 2.0  |
| PyTorch     | BSD 3-Clause |
| scikit-learn | BSD 3-Clause |
| h5py        | BSD 3-Clause |
