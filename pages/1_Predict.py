import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from datetime import datetime

from model_backend import (
    init_global_state,
    set_active_sample_from_row,
    render_persistent_header,
    load_pytorch_artifacts,
    get_site_dataframe,
    unscale_row,
    predict_with_pytorch
)

st.set_page_config(page_title="Live Sample Analysis | AquaGuard", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .result-card-safe {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        padding: 24px;
        border-radius: 12px;
        margin-top: 15px;
    }
    .result-card-unsafe {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        padding: 24px;
        border-radius: 12px;
        margin-top: 15px;
    }
    .violation-tag {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Initialize Global State & Persistent Header
init_global_state()
render_persistent_header()

_, _, _, df_full, feat_cols = load_pytorch_artifacts()
available_sites = df_full['site_name'].unique().tolist() if 'site_name' in df_full.columns else ["Tank A (Main Reservoir)", "Industrial Basin B", "Effluent Plant C", "Coastal Bio-Swale"]

# Page Header
st.title("🔬 Live Sample Analysis & PyTorch Model Inference")
st.caption("Direct stream ingestion from `processed_fishpond_train.csv` (1,722 Rows). Moving the slider instantly runs PyTorch forward pass inference and updates all pages!")

site_col, mode_col = st.columns([2, 2])

with site_col:
    selected_site_predict = st.selectbox(
        "Select Active Facility Site:",
        available_sites,
        index=available_sites.index(st.session_state["selected_site"]) if st.session_state["selected_site"] in available_sites else 0,
        key="pred_site_select"
    )

df_site = get_site_dataframe(selected_site_predict)

st.markdown("---")

# 2. Input Section & Reactive Slider
st.subheader(f"📥 Telemetry Sample Selection ({selected_site_predict})")
st.caption(f"Filtered **{len(df_site)} real dataset rows** for {selected_site_predict}. Slider movement automatically updates PyTorch inference state across all pages.")

# Row Slider (Immediate Reactive State Update)
row_idx = st.slider(
    f"Select Dataset Row Index (0 to {len(df_site)-1})",
    0, len(df_site)-1,
    min(st.session_state.get("selected_row_idx", 0), len(df_site)-1),
    key="pred_row_slider"
)

selected_row = df_site.iloc[row_idx]
phys = unscale_row(selected_row)

st.write("Loaded Scaled Model Input Vector from Dataset Row:", selected_row[feat_cols].to_dict())

# Automatically update active sample and run PyTorch inference whenever row_idx or site changes
set_active_sample_from_row(
    row=selected_row,
    row_idx=row_idx,
    site_name=selected_site_predict
)

pred_res = st.session_state["active_prediction"]
active_sample = st.session_state["active_sample"]

# 3. Prediction Result Card
st.subheader("🎯 PyTorch PINN Model Inference Result")
st.caption("Output calculated by PyTorch Neural Network forward pass on scaled dataset features.")

if pred_res["prediction"] == "SAFE":
    st.success(
        f"### SAFE\n\n"
        f"**PyTorch NN Classification:** {pred_res['status_tier']}\n\n"
        f"**Model Confidence:** {pred_res['confidence'] * 100:.1f}%\n\n"
        f"**Explanation:** {pred_res['explanation']}"
    )
else:
    if pred_res["domain_violation"]:
        st.warning("**DOMAIN VIOLATION — PHYSICALLY INVALID READING**")
    st.error(
        f"### UNSAFE\n\n"
        f"**PyTorch NN Classification:** {pred_res['status_tier']}\n\n"
        f"**Model Confidence:** {pred_res['confidence'] * 100:.1f}%\n\n"
        f"**Explanation:** {pred_res['explanation']}"
    )

st.progress(int(pred_res['confidence'] * 100))

# Quick Navigation Buttons
st.markdown("<br>", unsafe_allow_html=True)
nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 2])
with nav_c1:
    if st.button("🔍 Explain This Result →"):
        st.switch_page("pages/2_Explain.py")
with nav_c2:
    if st.button("🛠️ View Recourse Action Plan →"):
        st.switch_page("pages/3_Recourse.py")

st.markdown("---")

# 4. Live Rolling Telemetry Trend Chart
st.subheader("📈 Real Dataset Rolling Telemetry Stream")
st.caption(f"Real unscaled parameter trajectory from dataset rows up to Row {row_idx}.")

current_r_idx = min(st.session_state.get("selected_row_idx", 0), len(df_site)-1)
win_start = max(0, current_r_idx - 25)
df_sub = df_site.iloc[win_start:current_r_idx+1].copy()

hist_rows = []
for idx, r in df_sub.iterrows():
    p_unscaled = unscale_row(r)
    hist_rows.append({
        "Sample Point": f"Row-{idx:04d}",
        "pH Level": p_unscaled["ph"],
        "Dissolved Oxygen (mg/L)": p_unscaled["do"],
        "BOD (mg/L)": p_unscaled["bod"]
    })

df_chart = pd.DataFrame(hist_rows)

fig_trend = px.line(
    df_chart,
    x="Sample Point",
    y=["pH Level", "Dissolved Oxygen (mg/L)", "BOD (mg/L)"],
    markers=True,
    color_discrete_sequence=["#0ea5e9", "#22c55e", "#f59e0b"],
    template="plotly_white"
)
fig_trend.add_hrect(y0=6.5, y1=8.5, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="pH Safe Envelope", annotation_position="top left")
fig_trend.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_trend, width='stretch')

st.markdown("---")

# 5. Dynamic Audit Log Table
st.subheader("📋 Real Dataset Prediction Audit Log")
st.caption("Growing log of all PyTorch neural network evaluations performed in this session.")

df_audit = pd.DataFrame(st.session_state["audit_logs"])
display_audit = df_audit.drop(columns=["_raw_features"], errors="ignore")
st.dataframe(display_audit, width='stretch', hide_index=True)

exp_col1, exp_col2 = st.columns([1, 1])
with exp_col1:
    csv_bytes = display_audit.to_csv(index=False).encode('utf-8')
    st.download_button("📊 Export Session Audit Log (CSV)", data=csv_bytes, file_name="Session_Prediction_Audit_Log.csv", mime="text/csv")
