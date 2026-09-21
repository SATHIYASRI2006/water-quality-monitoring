import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from model_backend import (
    init_global_state,
    set_active_sample_from_row,
    render_persistent_header,
    render_global_sidebar,
    load_pytorch_artifacts,
    get_site_dataframe,
    unscale_row,
    predict_with_pytorch
)

st.set_page_config(page_title="Live Monitoring Dashboard | AquaGuard", layout="wide", initial_sidebar_state="expanded")

# Inject Custom Design System CSS
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    .badge-safe { background-color: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; }
    .badge-medium { background-color: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; }
    .badge-high { background-color: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; }
    .alert-banner {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        color: #7f1d1d;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stMetric { background: #ffffff; padding: 14px 18px; border-radius: 10px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# 1. Initialize Global State, Sidebar & Persistent Header
init_global_state()
render_global_sidebar()
render_persistent_header()

_, _, _, df_full, _ = load_pytorch_artifacts()
available_sites = df_full['site_name'].unique().tolist() if 'site_name' in df_full.columns else ["Tank A (Main Reservoir)", "Industrial Basin B", "Effluent Plant C", "Coastal Bio-Swale", "Metro Reservoir A"]

# Top Control Bar
h_col1, h_col2, h_col3 = st.columns([3, 2, 2])

with h_col1:
    st.title("Live Monitoring Dashboard & Command Hub")
    st.caption("Facility-wide IoT telemetry streams, safe envelope monitoring, and multi-site status overview.")

with h_col2:
    current_site_sel = st.selectbox(
        "Active Reservoir / Site:",
        available_sites,
        index=available_sites.index(st.session_state["selected_site"]) if st.session_state["selected_site"] in available_sites else 0,
        key="dash_site_select"
    )
    
    if current_site_sel != st.session_state["selected_site"]:
        df_site = get_site_dataframe(current_site_sel)
        first_row = df_site.iloc[0]
        set_active_sample_from_row(
            row=first_row,
            row_idx=0,
            site_name=current_site_sel
        )
        st.rerun()

with h_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Refresh Stream"):
            st.toast("Telemetry stream refreshed successfully!")
            st.rerun()
    with c_btn2:
        st.markdown("<span style='background:#dcfce7; color:#166534; padding:8px 12px; border-radius:6px; font-weight:600; font-size:0.85rem;'>🟢 Sensor Online</span>", unsafe_allow_html=True)

df_filtered = get_site_dataframe(st.session_state["selected_site"])
sample = st.session_state["active_sample"]
pred = st.session_state["active_prediction"]

# 2. Conditional Alert Banner (Appears during High Risk)
if pred["status_tier"] in ["High Risk", "Critical Threat"]:
    st.markdown(f"""
    <div class="alert-banner">
        <div>
            <b style="font-size:1.05rem;">⚠️ High Risk Alert at {st.session_state['selected_site']}</b> — pH level ({sample['ph']:.1f}) and Dissolved Oxygen ({sample['do']:.1f} mg/L) are outside safe limits. Immediate recourse intervention required.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Immediate Recourse Plan →", type="primary"):
        st.switch_page("pages/3_Recourse.py")

# 3. Top Metric Cards Row
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("**Current Status**")
    if pred["status_tier"] == "Safe / Low Risk":
        st.markdown("<span class='badge-safe'>🟢 SAFE (Low Risk)</span>", unsafe_allow_html=True)
    elif pred["status_tier"] == "Medium Risk":
        st.markdown("<span class='badge-medium'>🟡 MEDIUM RISK</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-high'>🔴 HIGH RISK</span>", unsafe_allow_html=True)
    st.caption(f"AI Confidence: {pred['confidence']*100:.1f}%")

import json
with m2:
    try:
        with open("metrics.json", "r") as mf:
            m_data = json.load(mf)
            f1 = m_data.get("f1_macro", 0.0)
            dvr_m = m_data.get("domain_violation_rate_pct", 0.0)
            st.metric("Model F1-Score", f"{f1*100:.1f}%", "Macro Avg (Test Set)")
    except:
        st.metric("Model F1-Score", "N/A")


with m3:
    try:
        with open("metrics.json", "r") as mf:
            m_data = json.load(mf)
            dvr_m = m_data.get("domain_violation_rate_pct", 0.0)
            st.metric("Domain Violation Rate (DVR)", f"{dvr_m:.1f}%", "Test Set (Critical/Severe)")
    except:
        st.metric("Domain Violation Rate (DVR)", "N/A")

with m4:
    st.metric("Site Records Monitored", f"{len(df_filtered)} Samples", f"Total Tank Data Points")

st.markdown("---")

# 4. Live Parameter Gauges Row
st.subheader("📊 Live Telemetry Gauges (Active Site)")
st.caption(f"Real-time sensor indicators for **{st.session_state['selected_site']}** comparing live values against safe regulatory bands.")

g1, g2 = st.columns(2)

with g1:
    fig_ph = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sample["ph"],
        title={'text': "pH Level (Safe: 6.5 - 8.5)", 'font': {'size': 15, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [2, 12]},
            'bar': {'color': "#0ea5e9"},
            'steps': [
                {'range': [2, 6.5], 'color': "#fee2e2"},
                {'range': [6.5, 8.5], 'color': "#dcfce7"},
                {'range': [8.5, 12], 'color': "#fee2e2"}
            ],
            'threshold': {'line': {'color': pred["badge_color"], 'width': 4}, 'value': sample["ph"]}
        }
    ))
    fig_ph.update_layout(height=230, margin=dict(l=20, r=20, t=35, b=10))
    st.plotly_chart(fig_ph, width='stretch')

with g2:
    fig_do = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sample["do"],
        number={'suffix': " mg/L"},
        title={'text': "Dissolved Oxygen (Safe: > 4.0 mg/L)", 'font': {'size': 15, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [0, 12]},
            'bar': {'color': "#22c55e"},
            'steps': [
                {'range': [0, 4.0], 'color': "#fee2e2"},
                {'range': [4.0, 12.0], 'color': "#dcfce7"}
            ],
            'threshold': {'line': {'color': "green", 'width': 4}, 'value': sample["do"]}
        }
    ))
    fig_do.update_layout(height=230, margin=dict(l=20, r=20, t=35, b=10))
    st.plotly_chart(fig_do, width='stretch')


st.markdown("---")

# 5. Site-Filtered Dynamic Telemetry Line Chart & Time-Range Filter
st.subheader("📈 Real Dataset Telemetry Trend Chart")
st.caption(f"Dynamic rolling trajectories extracted directly from **{len(df_filtered)} real data points** in `{st.session_state['selected_site']}`.")

tr_col1, tr_col2 = st.columns([3, 1])
with tr_col2:
    time_range = st.radio("Time Window:", ["1H", "24H", "7D", "30D"], index=1, horizontal=True)

time_window_sizes = {"1H": 10, "24H": 35, "7D": 70, "30D": 150}
win_len = time_window_sizes.get(time_range, 35)

cur_idx = min(st.session_state.get("selected_row_idx", 0), len(df_filtered)-1)
win_start = max(0, cur_idx - win_len)
df_sub = df_filtered.iloc[win_start:cur_idx+1].copy()

hist_rows = []
for idx, row in df_sub.iterrows():
    phys = unscale_row(row)
    hist_rows.append({
        "Sample Point": f"Pt-{idx:04d}",
        "pH Level": phys["ph"],
        "Dissolved Oxygen (mg/L)": phys["do"],
        
    })

df_chart = pd.DataFrame(hist_rows)

fig_trend = px.line(
    df_chart,
    x="Sample Point",
    y=["pH Level", "Dissolved Oxygen (mg/L)"],
    markers=True,
    color_discrete_sequence=["#0ea5e9", "#22c55e", "#f59e0b"],
    template="plotly_white"
)

fig_trend.add_hrect(y0=6.5, y1=8.5, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0, annotation_text="pH Safe Envelope (6.5-8.5)", annotation_position="top left")
fig_trend.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20), legend_title="Parameter")
st.plotly_chart(fig_trend, width='stretch')

st.markdown("---")

# 6. Real-Time Parameter Diagnostics Table
st.subheader("📋 Real-Time Parameter Diagnostics vs Regulatory Envelopes")
st.caption("Detailed breakdown of current parameter readings against WHO & TNPCB standard regulatory thresholds.")

diag_rows = [
    {"Parameter": "pH Level", "Current Reading": f"{sample['ph']:.2f}", "Regulatory Envelope": "6.5 - 8.5", "Status Check": "🟢 COMPLIANT" if 6.5 <= sample['ph'] <= 8.5 else "🔴 VIOLATION"},
    {"Parameter": "Dissolved Oxygen (DO)", "Current Reading": f"{sample['do']:.2f} mg/L", "Regulatory Envelope": "≥ 4.0 mg/L", "Status Check": "🟢 COMPLIANT" if sample['do'] >= 4.0 else "🔴 VIOLATION"},
    {"Parameter": "Turbidity", "Current Reading": f"{sample['turbidity']:.2f} NTU", "Regulatory Envelope": "≤ 5.0 NTU", "Status Check": "🟢 COMPLIANT" if sample['turbidity'] <= 5.0 else "🔴 WARNING"},
]
st.dataframe(pd.DataFrame(diag_rows), width='stretch', hide_index=True)

st.markdown("---")

# 7. Multi-Site Overview Matrix
st.subheader("🌐 Multi-Site Overview Matrix")
st.caption("Real-time summary of all monitored facility nodes. Filtered directly from site DataFrame subsets.")

overview_rows = []
for s_name in available_sites:
    sub_site = get_site_dataframe(s_name)
    if len(sub_site) > 0:
        p_row = unscale_row(sub_site.iloc[0])
        eval_p = predict_with_pytorch([float(sub_site.iloc[0][col]) for col in ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL']])
        
        if eval_p["status_tier"] == "Safe / Low Risk":
            b_html = "🟢 SAFE"
        elif eval_p["status_tier"] == "Medium Risk":
            b_html = "🟡 MEDIUM RISK"
        else:
            b_html = "🔴 HIGH RISK"
            
        overview_rows.append({
            "Site Name": s_name,
            "Total Samples": len(sub_site),
            "Status": b_html,
            "pH Level": f"{p_row['ph']:.1f}",
            "Dissolved Oxygen": f"{p_row['do']:.1f} mg/L",
                        "Turbidity": f"{p_row['turbidity']:.1f} NTU",
            "Last Updated": "Live Stream"
        })

st.dataframe(pd.DataFrame(overview_rows), width='stretch', hide_index=True)


