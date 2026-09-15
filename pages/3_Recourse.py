import sys
import os
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from model_backend import (
    init_global_state,
    render_persistent_header,
    calculate_recourse_actions
)

st.set_page_config(page_title="Recourse & Operator Action | AquaGuard", layout="wide")

# 1. Initialize Global State & Persistent Header
init_global_state()
render_persistent_header()

# Header
st.title("🛠️ Recommended Corrective Actions (Recourse Engine)")
st.caption("Minimal convex hull counterfactual corrections, operator action checklist, and live actuator telecontrol.")

# Retrieve active sample & prediction from session state
active_sample = st.session_state["active_sample"]
active_pred = st.session_state["active_prediction"]
selected_site = st.session_state["selected_site"]

rec_res = calculate_recourse_actions(
    active_sample["ph"],
    active_sample["do"],
    active_sample["bod"],
    active_sample["turbidity"]
)

# 1. Site / Sample Context Card & Urgency Badge
info_col1, info_col2 = st.columns([3, 1.2])

with info_col1:
    st.markdown(f"""
    <div style="background:white; padding:16px; border-radius:10px; border:1px solid #e2e8f0;">
        <h4 style="margin:0; color:#1e293b;">📍 Active Context: {selected_site}</h4>
        <p style="margin:4px 0 0 0; color:#64748b; font-size:0.9rem;"><b>Sample ID:</b> {active_sample['sample_id']} | <b>Timestamp:</b> {active_sample['timestamp']} | <b>Status:</b> {active_pred['status_tier']}</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    if "Immediate" in rec_res["urgency"]:
        st.markdown("<span style='background:#fee2e2; color:#991b1b; padding:10px 18px; border-radius:8px; font-weight:700; font-size:1.05rem; display:inline-block;'>🔴 Immediate Action</span>", unsafe_allow_html=True)
    elif "Maintenance" in rec_res["urgency"]:
        st.markdown("<span style='background:#fef3c7; color:#92400e; padding:10px 18px; border-radius:8px; font-weight:700; font-size:1.05rem; display:inline-block;'>🟡 Schedule Maintenance</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='background:#dcfce7; color:#166534; padding:10px 18px; border-radius:8px; font-weight:700; font-size:1.05rem; display:inline-block;'>🟢 Monitor Only</span>", unsafe_allow_html=True)

st.markdown("---")

# 2. Before -> After Counterfactual Table
st.subheader("🔄 Before → After Counterfactual Recourse Table")
st.caption("Calculates the minimal, actionable parameter adjustments required to restore water safety within convex hulls.")

if rec_res["before_after"]:
    df_ba = pd.DataFrame(rec_res["before_after"])[["parameter", "current", "recommended", "delta", "method"]]
    df_ba.columns = ["Parameter", "Current Value", "Recommended Value", "Δ Change", "Action Method"]
    st.dataframe(df_ba, width='stretch', hide_index=True)
else:
    st.success("✨ All parameters currently optimal! No parameter adjustments required.")

st.markdown("---")

# 3. Action Checklist Panel & Status Tracker
st.subheader("📋 Operator Action Checklist & Assignment Panel")
st.caption("Assignable task list for plant technicians and supervisors to execute and track remediation progress.")

if "checklist_state" not in st.session_state:
    st.session_state["checklist_state"] = {}

for idx, item in enumerate(rec_res["checklist"], start=1):
    chk_key = f"{active_sample['sample_id']}_chk_{idx}"
    ass_key = f"{active_sample['sample_id']}_ass_{idx}"
    stat_key = f"{active_sample['sample_id']}_stat_{idx}"

    c_chk, c_assign, c_status = st.columns([3, 1, 1])
    with c_chk:
        checked = st.checkbox(f"**Step {idx}:** {item['task']}", key=chk_key)
    with c_assign:
        assignee = st.selectbox("Assignee:", ["Operator_1", "Operator_2", "Plant_Manager"], index=0, key=ass_key)
    with c_status:
        default_idx = 2 if checked else 1
        status_val = st.selectbox("Status:", ["Pending", "In Progress", "Resolved"], index=default_idx, key=stat_key)

st.markdown("---")

# 4. Simulated Outcome Preview
st.subheader("🔮 Simulated Outcome Preview")
st.caption("Predictive simulation of water quality status after applying the recommended counterfactual fixes.")

prev_col1, prev_col2 = st.columns([2, 1])

with prev_col1:
    st.info(f"✨ **Predicted Fix Success Rate:** {rec_res['predicted_success_rate']}% confidence that the proposed recourse actions will restore water to certified **Safe** status.")

    # Use the dissolved-oxygen counterfactual directly, so the preview stays
    # aligned with the primary aeration recourse rather than showing pH.
    do_action = next(
        (action for action in rec_res["before_after"] if action["parameter"] == "Dissolved Oxygen (DO)"),
        None
    )
    do_before = active_sample["do"]
    do_after = float(do_action["recommended"].split()[0]) if do_action else do_before

    g_before, g_after = st.columns(2)
    with g_before:
        fig_b = go.Figure(go.Indicator(
            mode="gauge+number",
            value=do_before,
            number={'suffix': " mg/L"},
            title={'text': "BEFORE Fix (Dissolved Oxygen)"},
            gauge={'axis': {'range': [0, 12]}, 'bar': {'color': active_pred["badge_color"]}}
        ))
        fig_b.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_b, width='stretch')

    with g_after:
        fig_a = go.Figure(go.Indicator(
            mode="gauge+number",
            value=do_after,
            number={'suffix': " mg/L"},
            title={'text': "AFTER Fix (Simulated Dissolved Oxygen)"},
            gauge={'axis': {'range': [0, 12]}, 'bar': {'color': "#22C55E"}}
        ))
        fig_a.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_a, width='stretch')

with prev_col2:
    st.subheader("🔔 Notification Dispatch Log")
    notif_data = pd.DataFrame([
        {"Notified Party": "Operator_1", "Channel": "SMS / Push", "Time Sent": "10:32 AM", "Acknowledged": "✓ YES"},
        {"Notified Party": "Plant_Manager", "Channel": "Email", "Time Sent": "10:33 AM", "Acknowledged": "✓ YES"},
        {"Notified Party": "Compliance_Officer", "Channel": "Email", "Time Sent": "10:34 AM", "Acknowledged": "Pending"}
    ])
    st.dataframe(notif_data, width='stretch', hide_index=True)

st.markdown("---")

# 5. Live Actuator Telecontrol Panel
st.subheader("⚡ Live Actuator Telecontrol Panel")
st.caption("Override or manually engage biological dosing, mechanical aerators, and mineral buffer feeders.")

acts = st.session_state["actuators"]

ac_t1, ac_t2 = st.columns([3, 1])
with ac_t2:
    auto_toggle = st.toggle("🤖 Fully Autonomous Dosing", value=acts["autonomous_mode"])
    acts["autonomous_mode"] = auto_toggle

ac1, ac2, ac3, ac4 = st.columns(4)

with ac1:
    st.markdown("#### 🌬️ Aeration Blowers")
    st.caption("Mechanical Blower Bank A")
    new_speed = st.slider("Speed Control (%)", 0, 100, acts["aeration_speed"], key="rec_aer_slider")
    if st.button("Set Aeration Speed"):
        acts["aeration_speed"] = new_speed
        st.toast(f"Aerator Blower Bank A speed set to {new_speed}%!")

with ac2:
    st.markdown("#### 🦠 Bio-Strain Injectors")
    st.caption("Bacillus subtilis Unit 2")
    new_bio = st.selectbox("Dosing Rate", ["0.5 L/hr", "1.2 L/hr (Standard)", "2.5 L/hr (High BOD)", "5.0 L/hr (Shock Dosing)"], index=1, key="rec_bio_select")
    if st.button("Dispense Bio-Culture"):
        acts["bio_culture_rate"] = new_bio
        st.toast(f"Bio-culture dosing rate updated to {new_bio}!")

with ac3:
    st.markdown("#### 🪨 Mineral Buffers")
    st.caption("Limestone Slurry Feeder")
    new_kg = st.number_input("Dispense Amount (kg)", 0, 500, acts["mineral_buffer_kg"], step=10, key="rec_min_input")
    if st.button("Inject Mineral Buffer"):
        acts["mineral_buffer_kg"] = new_kg
        st.toast(f"Mineral buffer feed of {new_kg} kg queued!")

with ac4:
    st.markdown("#### 🛑 Emergency Shutoff")
    st.caption("Sluice Valve Lockout")
    if st.button("⚠️ EMERGENCY VALVE LOCK", type="primary", key="rec_emerg_btn"):
        acts["emergency_lock"] = True
        st.error("🚨 Emergency isolation command dispatched! Main intake sluice gate locked.")
