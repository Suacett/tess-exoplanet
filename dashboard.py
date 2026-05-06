"""
Exoplanet Pipeline Dashboard — Streamlit app
Pages: Pipeline Status | Candidate Browser | Lightcurve Inspector
"""
import os
from pathlib import Path
import glob
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet Pipeline",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR   = str(Path(__file__).resolve().parent / "data")
ZENODO_DIR = os.path.join(DATA_DIR, "zenodo")
TESS_DIR   = os.path.join(DATA_DIR, "tess")

# ── Sidebar nav ────────────────────────────────────────────────────────────────
st.sidebar.title("🔭 Exoplanet Pipeline")
page = st.sidebar.radio(
    "Navigate",
    ["Pipeline Status", "Candidate Browser", "Lightcurve Inspector"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data directories**")
st.sidebar.code(f"zenodo: {ZENODO_DIR}\ntess:   {TESS_DIR}")

# ==============================================================================
# Page 1 — Pipeline Status
# ==============================================================================
if page == "Pipeline Status":
    st.title("Pipeline Status")

    # Service health checks
    import subprocess

    def svc_status(name: str) -> str:
        try:
            r = subprocess.run(
                ["systemctl", "is-active", name],
                capture_output=True, text=True, timeout=3,
            )
            return r.stdout.strip()
        except Exception:
            return "unknown"

    col1, col2, col3 = st.columns(3)
    jl  = svc_status("jupyterlab")
    db  = svc_status("exoplanet-dashboard")

    def badge(status: str):
        color = "green" if status == "active" else "red"
        return f":{color}[{status}]"

    with col1:
        st.metric("JupyterLab (8888)", jl.upper())
    with col2:
        st.metric("Dashboard (8501)", db.upper())
    with col3:
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
        st.metric("Container IP", ip)

    st.markdown("---")
    st.subheader("Data Storage")
    cols = st.columns(3)
    dirs = {
        "Zenodo catalog": ZENODO_DIR,
        "TESS lightcurves": TESS_DIR,
        "Candidates": os.path.join(DATA_DIR, "candidates"),
    }
    for col, (label, path) in zip(cols, dirs.items()):
        with col:
            if os.path.isdir(path):
                files = glob.glob(os.path.join(path, "**", "*"), recursive=True)
                n = len([f for f in files if os.path.isfile(f)])
                st.metric(label, f"{n} files")
            else:
                st.metric(label, "not found")

    st.markdown("---")
    st.subheader("Recent Log Output")
    log_path = "/var/log/exoplanet-setup.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        st.code("".join(lines[-30:]), language="text")
    else:
        st.info("No setup log found. Run setup.sh first.")

    # Check ExoMiner repo
    st.markdown("---")
    st.subheader("ExoMiner++ Repository")
    exominer = str(Path(__file__).resolve().parent / "ExoMiner")
    if os.path.isdir(exominer):
        docs = glob.glob(os.path.join(exominer, "docs", "*.md"))
        st.success(f"Cloned at {exominer}")
        if docs:
            st.write("Documentation files:", [os.path.basename(d) for d in docs])
    else:
        st.warning("ExoMiner++ not cloned yet.")


# ==============================================================================
# Page 2 — Candidate Browser
# ==============================================================================
elif page == "Candidate Browser":
    st.title("Candidate Browser")
    st.markdown("Load the ExoMiner++ TESS catalog from Zenodo to browse scored candidates.")

    # Auto-detect catalog files
    candidates = (
        glob.glob(os.path.join(ZENODO_DIR, "*.csv")) +
        glob.glob(os.path.join(ZENODO_DIR, "*.parquet")) +
        glob.glob(os.path.join(DATA_DIR, "candidates", "*.csv"))
    )

    uploaded = st.file_uploader(
        "Upload catalog CSV (or place file in /opt/exoplanet/data/zenodo/)",
        type=["csv", "parquet"],
    )

    @st.cache_data
    def load_catalog(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    df = None

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows from upload")
    elif candidates:
        chosen = st.selectbox("Auto-detected catalog file:", candidates)
        if st.button("Load"):
            df = load_catalog(chosen)
            st.success(f"Loaded {len(df):,} rows")
    else:
        st.info(
            "No catalog found. Download the Zenodo archive (DOI: 10.5281/zenodo.15466292) "
            "and place CSV files in `/opt/exoplanet/data/zenodo/`."
        )

    if df is not None:
        st.markdown("---")

        # Detect score column
        score_col = next(
            (c for c in df.columns if "score" in c.lower() or "exominer" in c.lower()),
            df.select_dtypes(include=np.number).columns[0] if len(df.select_dtypes(include=np.number).columns) else None,
        )
        tic_col = next(
            (c for c in df.columns if "tic" in c.lower() or "toi" in c.lower()),
            df.columns[0],
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Score Distribution")
            if score_col:
                fig = px.histogram(
                    df, x=score_col, nbins=100,
                    color_discrete_sequence=["steelblue"],
                    labels={score_col: "ExoMiner++ Score"},
                    title=f"Distribution of {score_col} (n={len(df):,})",
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                              annotation_text="0.5 threshold")
                fig.add_vline(x=0.9, line_dash="dash", line_color="green",
                              annotation_text="High confidence (0.9)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not detect score column.")

        with col2:
            if score_col:
                st.subheader("Threshold Filter")
                threshold = st.slider("Min score", 0.0, 1.0, 0.9, 0.01)
                top = df[df[score_col] >= threshold].sort_values(score_col, ascending=False)
                st.metric("Candidates above threshold", len(top))
                if len(top) > 0:
                    st.dataframe(top[[tic_col, score_col]].head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Full Catalog")
        st.dataframe(df.head(500), use_container_width=True)

        # Score scatter vs period if available
        period_col = next(
            (c for c in df.columns if "period" in c.lower() or "per" in c.lower()),
            None,
        )
        if period_col and score_col:
            st.subheader("Score vs Period")
            fig2 = px.scatter(
                df.sample(min(5000, len(df))),
                x=period_col, y=score_col,
                opacity=0.4,
                color=score_col,
                color_continuous_scale="RdYlGn",
                labels={period_col: "Period (days)", score_col: "ExoMiner++ Score"},
                title="Score vs Orbital Period",
            )
            st.plotly_chart(fig2, use_container_width=True)


# ==============================================================================
# Page 3 — Lightcurve Inspector
# ==============================================================================
elif page == "Lightcurve Inspector":
    st.title("Lightcurve Inspector")
    st.markdown(
        "Enter a TIC ID to fetch the TESS lightcurve from MAST, run BLS, and inspect transits."
    )

    with st.form("lc_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            tic_input = st.text_input("TIC ID", value="402026209", placeholder="e.g. 402026209")
        with col2:
            author = st.selectbox("Pipeline", ["SPOC", "QLP", "TESS-SPOC"])
        with col3:
            cadence = st.selectbox("Cadence", ["short", "long", "fast"])
        submitted = st.form_submit_button("Fetch Lightcurve")

    if submitted and tic_input.strip():
        import lightkurve as lk

        tic = f"TIC {tic_input.strip().lstrip('TIC').strip()}"
        st.markdown(f"**Fetching:** `{tic}` from MAST (author={author})…")

        with st.spinner("Downloading from MAST..."):
            try:
                results = lk.search_lightcurve(tic, mission="TESS", author=author,
                                                exptime=cadence if cadence != "short" else 120)
                if len(results) == 0:
                    # Try any author
                    results = lk.search_lightcurve(tic, mission="TESS")

                if len(results) == 0:
                    st.error(f"No TESS data found for {tic}")
                    st.stop()

                st.success(f"Found {len(results)} sector(s). Downloading all...")
                lcs = results.download_all(quality_bitmask='default')
                lc_raw = lcs.stitch() if len(lcs) > 1 else lcs[0]

            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

        st.markdown(f"**Downloaded:** {len(lc_raw)} data points across "
                    f"{lc_raw.time.value.min():.1f}–{lc_raw.time.value.max():.1f} BTJD")

        # ── Raw lightcurve plot ──────────────────────────────────────────────
        st.subheader("Raw Lightcurve")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scattergl(
            x=lc_raw.time.value,
            y=lc_raw.flux.value,
            mode="markers",
            marker=dict(size=2, opacity=0.5, color="steelblue"),
            name="Flux",
        ))
        fig_raw.update_layout(
            xaxis_title="BTJD",
            yaxis_title="Flux (e-/s)",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_raw, use_container_width=True)

        # ── Flatten & BLS ────────────────────────────────────────────────────
        with st.spinner("Flattening and running BLS period search..."):
            try:
                lc_flat = lc_raw.flatten(window_length=401).remove_outliers(sigma=4)
                import numpy as np
                periods = np.arange(0.5, 14.0, 0.001)
                durations = np.arange(0.02, 0.2, 0.003)
                blsm = lc_flat.to_periodogram(
                    method='bls', period=periods, duration=durations,
                    frequency_factor=2,
                )
                best_p = blsm.period_at_max_power
                best_t0 = blsm.transit_time_at_max_power
                sde = float(blsm.max_power)
            except Exception as e:
                st.error(f"BLS failed: {e}")
                st.stop()

        st.subheader("BLS Periodogram")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Best Period", f"{best_p.value:.4f} d")
        col_b.metric("Transit Epoch", f"{best_t0.value:.3f} BTJD")
        col_c.metric("BLS SDE", f"{sde:.1f}", delta="⚠ SDE > 9 is interesting" if sde > 9 else None)

        fig_bls = go.Figure()
        fig_bls.add_trace(go.Scatter(
            x=blsm.period.value,
            y=blsm.power.value,
            mode="lines",
            line=dict(color="steelblue", width=1),
            name="BLS power",
        ))
        fig_bls.add_vline(x=best_p.value, line_dash="dash", line_color="red",
                          annotation_text=f"P={best_p.value:.3f}d")
        fig_bls.update_layout(
            xaxis_title="Period (days)",
            yaxis_title="BLS Power",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_bls, use_container_width=True)

        # ── Phase fold ───────────────────────────────────────────────────────
        st.subheader("Phase-folded Lightcurve")
        user_period = st.number_input("Override period (days):", value=float(best_p.value),
                                       min_value=0.1, max_value=100.0, step=0.001, format="%.4f")

        try:
            from astropy import units as u_ast
            lc_fold = lc_flat.fold(
                period=user_period * u_ast.day,
                epoch_time=best_t0,
            )
            lc_bin = lc_fold.bin(time_bin_size=0.01)

            fig_fold = go.Figure()
            fig_fold.add_trace(go.Scattergl(
                x=lc_fold.phase.value,
                y=lc_fold.flux.value,
                mode="markers",
                marker=dict(size=2, opacity=0.3, color="gray"),
                name="All data",
            ))
            fig_fold.add_trace(go.Scatter(
                x=lc_bin.phase.value,
                y=lc_bin.flux.value,
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=4),
                name="10-min bins",
            ))
            fig_fold.update_layout(
                xaxis_title="Phase",
                yaxis_title="Normalized Flux",
                height=350,
                xaxis=dict(range=[-0.5, 0.5]),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_fold, use_container_width=True)

            # Transit diagnostics
            st.subheader("Quick Diagnostics")
            bin_arr = lc_bin.flux.value
            baseline = np.nanmedian(bin_arr[np.abs(lc_bin.phase.value) > 0.1])
            in_transit = np.nanmin(bin_arr[np.abs(lc_bin.phase.value) < 0.05])
            depth_ppm = (baseline - in_transit) / baseline * 1e6

            d1, d2, d3 = st.columns(3)
            d1.metric("Transit depth", f"{depth_ppm:.0f} ppm")
            d2.metric("Period", f"{user_period:.4f} d")
            d3.metric(
                "Classification hint",
                "Hot Jupiter?" if depth_ppm > 5000
                else "Sub-Sat/Neptune?" if depth_ppm > 500
                else "Earth-size?" if depth_ppm > 50
                else "Flat / no signal",
            )

        except Exception as e:
            st.error(f"Phase-fold error: {e}")

    elif submitted:
        st.warning("Please enter a TIC ID.")
