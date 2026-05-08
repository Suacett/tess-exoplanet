#!/usr/bin/env python3
"""
GPU BLS (Box Least Squares) microservice for the TESS exoplanet pipeline.

Deploy on any machine with an NVIDIA GPU and cupy/fastapi installed:
    pip install cupy-cuda12x fastapi uvicorn

Configure via environment variables:
    BLS_GPU_DEVICE   GPU index to use (default: 0)
    BLS_PORT         HTTP port (default: 9876)
    BLS_HOST         Bind address (default: 0.0.0.0)

The scanner (scan_sector.py) uses this automatically when GPU_BLS_URL is set:
    export GPU_BLS_URL=http://<this-host>:9876
"""
import logging
import os

import numpy as np

GPU_DEVICE  = int(os.environ.get("BLS_GPU_DEVICE", "0"))
PORT        = int(os.environ.get("BLS_PORT", "9876"))
HOST        = os.environ.get("BLS_HOST", "0.0.0.0")

PERIOD_MIN  = 0.5
PERIOD_MAX  = 14.0
PERIOD_STEP = 0.02
DURATIONS_D = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20]
N_BINS      = 200
MIN_FREE_MB = 512

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _bls_star(cp, time_cp, flux_cp):
    """
    Vectorised BLS for one star over all trial periods using cupy.
    Returns (best_period, best_t0, sde, best_duration_d).
    """
    periods = cp.arange(PERIOD_MIN, PERIOD_MAX, PERIOD_STEP, dtype=cp.float32)
    N_p = len(periods)
    N_t = len(time_cp)

    flux_norm = flux_cp - cp.mean(flux_cp)

    # Phase matrix: shape (N_p, N_t)
    phase = (time_cp[None, :] % periods[:, None]) / periods[:, None]

    # Bin phase into N_BINS integer indices
    phase_bin = cp.clip((phase * N_BINS).astype(cp.int32), 0, N_BINS - 1)  # (N_p, N_t)

    # Flat scatter indices into a (N_p * N_BINS,) buffer
    row_off  = (cp.arange(N_p, dtype=cp.int32) * N_BINS)[:, None]   # (N_p, 1)
    flat_idx = (row_off + phase_bin).ravel()                          # (N_p * N_t,)

    bf = cp.zeros(N_p * N_BINS, dtype=cp.float32)
    bc = cp.zeros(N_p * N_BINS, dtype=cp.float32)
    cp.add.at(bf, flat_idx, cp.tile(flux_norm, N_p))
    cp.add.at(bc, flat_idx, cp.ones(N_p * N_t, dtype=cp.float32))

    bf = bf.reshape(N_p, N_BINS)   # (N_p, N_BINS)
    bc = bc.reshape(N_p, N_BINS)

    total_f = bf.sum(axis=1, keepdims=True)   # (N_p, 1)
    total_c = bc.sum(axis=1, keepdims=True)

    best_power    = cp.full(N_p, -cp.inf, dtype=cp.float32)
    best_t0_phase = cp.zeros(N_p, dtype=cp.float32)
    best_dur      = cp.zeros(N_p, dtype=cp.float32)
    all_max_pwr   = []

    med_period = (PERIOD_MIN + PERIOD_MAX) / 2  # ≈ 7.25 d — used for n_dur approximation

    for dur_d in DURATIONS_D:
        n_dur = max(1, round(dur_d / med_period * N_BINS))

        # Double arrays for circular (wraparound) window
        bf_ext = cp.concatenate([bf, bf], axis=1)   # (N_p, 2*N_BINS)
        bc_ext = cp.concatenate([bc, bc], axis=1)

        # Prefix sums (leading-zero padded)
        cs_f = cp.concatenate([cp.zeros((N_p, 1), dtype=cp.float32),
                                cp.cumsum(bf_ext, axis=1)], axis=1)  # (N_p, 2*N_BINS+1)
        cs_c = cp.concatenate([cp.zeros((N_p, 1), dtype=cp.float32),
                                cp.cumsum(bc_ext, axis=1)], axis=1)

        # Sliding window sums: s_in[:, i] = sum of n_dur bins starting at bin i
        s_in = cs_f[:, n_dur:N_BINS + n_dur] - cs_f[:, :N_BINS]   # (N_p, N_BINS)
        n_in = cs_c[:, n_dur:N_BINS + n_dur] - cs_c[:, :N_BINS]

        s_out = total_f - s_in
        n_out = total_c - n_in

        # Transit signal: depth × sqrt(n_in) — negative s_in = flux dip = transit
        signal = -(s_in / (n_in + 1e-6)) + (s_out / (n_out + 1e-6))
        power  = cp.nan_to_num(signal * cp.sqrt(cp.maximum(n_in, 0)), nan=0.0)

        best_center = cp.argmax(power, axis=1)                       # (N_p,)
        max_p       = power[cp.arange(N_p), best_center]             # (N_p,)

        improved      = max_p > best_power
        best_power    = cp.where(improved, max_p, best_power)
        best_t0_phase = cp.where(improved,
                                  best_center.astype(cp.float32) / N_BINS,
                                  best_t0_phase)
        best_dur      = cp.where(improved,
                                  cp.full(N_p, dur_d, dtype=cp.float32),
                                  best_dur)
        all_max_pwr.append(max_p)

    # SDE: normalise best_power across all periods
    all_pwr  = cp.stack(all_max_pwr, axis=0).max(axis=0)   # (N_p,)
    p_std    = cp.std(all_pwr) + 1e-9
    sde_arr  = (best_power - cp.mean(all_pwr)) / p_std

    best_idx    = int(cp.argmax(sde_arr).item())
    best_period = float(periods[best_idx].item())
    sde         = float(sde_arr[best_idx].item())
    t0_phase    = float(best_t0_phase[best_idx].item())
    dur_d_out   = float(best_dur[best_idx].item())

    # t0: time at which best transit centre occurs
    t0 = (float(time_cp[0].item()) // best_period) * best_period + t0_phase * best_period

    return best_period, t0, sde, dur_d_out


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    class StarData(BaseModel):
        tic_id: int
        time:   list[float]
        flux:   list[float]

    class BLSRequest(BaseModel):
        stars: list[StarData]

    app = FastAPI(title="GPU BLS Service", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "gpu_device": GPU_DEVICE}

    @app.post("/bls")
    def run_bls(req: BLSRequest):
        try:
            import cupy as cp
            with cp.cuda.Device(GPU_DEVICE):
                free_b, _ = cp.cuda.runtime.memGetInfo()
                if free_b < MIN_FREE_MB * 1024 * 1024:
                    raise HTTPException(status_code=503, detail="GPU memory low")
                results = []
                for star in req.stars:
                    t_cp = cp.array(star.time, dtype=cp.float32)
                    f_cp = cp.array(star.flux, dtype=cp.float32)
                    p, t0, sde, dur = _bls_star(cp, t_cp, f_cp)
                    results.append({
                        "tic_id":     star.tic_id,
                        "period":     round(p, 5),
                        "t0":         round(t0, 5),
                        "sde":        round(sde, 4),
                        "duration_d": round(dur, 3),
                    })
                return results
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("BLS computation error")
            raise HTTPException(status_code=500, detail=str(exc))

except ImportError:
    log.error("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn cupy-cuda12x")
    app = None  # type: ignore


if __name__ == "__main__":
    if app is None:
        raise SystemExit("Missing dependencies — see error above.")
    log.info(f"GPU BLS service starting on {HOST}:{PORT} (GPU device {GPU_DEVICE})")
    import uvicorn as _uv
    _uv.run(app, host=HOST, port=PORT, log_level="warning")
