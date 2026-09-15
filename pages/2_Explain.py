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
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from model_backend import (
    init_global_state,
    render_persistent_header,
    explain_sample_shap,
    predict_with_pytorch
)

st.set_page_config(page_title="Explainability & Compliance | AquaGuard", layout="wide")

# 1. Initialize Global State & Persistent Header
init_global_state()
render_persistent_header()

# Header
st.title("🔍 Why This Result? (Explainability & Compliance)")
st.caption("Feature attribution analysis, domain constraint violation explanations, and certified regulatory audit reporting.")

# Sample Selector Dropdown from Session State Logs
logs = st.session_state["audit_logs"]
sample_options = [f"{log['Sample ID']} ({log['Site Name']} - {log['Risk Tier']})" for log in logs] if len(logs) > 0 else [f"{st.session_state['active_sample']['sample_id']} (Active Sample)"]

sel_col1, sel_col2 = st.columns([2, 1])
with sel_col1:
    selected_sample_str = st.selectbox(
        "Select Sample to Audit & Explain:",
        sample_options,
        index=0
    )

# Extract selected sample details
selected_id = selected_sample_str.split(" ")[0]
matched_log = next((l for l in logs if l["Sample ID"] == selected_id), None)

if matched_log:
    s_ph = float(matched_log["pH"])
    s_do = float(matched_log["DO (mg/L)"].replace(" mg/L", ""))
    s_bod = float(matched_log["BOD (mg/L)"].replace(" mg/L", ""))
    s_site = matched_log["Site Name"]
    s_raw_feats = matched_log.get("_raw_features")
    if s_raw_feats is None:
        st.warning("This older audit entry has no stored model input; showing the active sample instead.")
        s_raw_feats = st.session_state["active_sample"]["raw_features"]
else:
    active_s = st.session_state["active_sample"]
    s_ph, s_do, s_bod, s_site = active_s["ph"], active_s["do"], active_s["bod"], active_s["site_name"]
    s_raw_feats = active_s.get("raw_features", [0.0]*11)

# Ensure raw features is a list of clean floats to prevent format type errors
s_raw_feats = [float(val) for val in s_raw_feats]

pred_res = predict_with_pytorch(s_raw_feats)
explain_res = explain_sample_shap(s_raw_feats)

st.markdown("---")

# 2. Risk Tier Summary Card
st.subheader("📌 Risk Tier & Environmental Impact Summary")
st.caption(f"Physical environmental summary for **{selected_id}** at **{s_site}**.")

if pred_res["status_tier"] in ["High Risk", "Critical Threat"]:
    st.error(f"🔴 **{pred_res['status_tier']}** — Severe hypoxia & acidification detected. Dissolved oxygen ({s_do:.1f} mg/L) and pH levels ({s_ph:.1f}) are critically insufficient to support aquatic life or safe municipal treatment.")
elif pred_res["status_tier"] == "Medium Risk":
    st.warning(f"🟡 **Medium Risk / Operational Warning** — Moderate parameter deviation. Organic load ({s_bod:.1f} mg/L) requires active monitoring and automated bio-dosing intervention.")
else:
    st.success(f"🟢 **Safe / Low Risk Ecosystem** — Optimal hydro-chemical balance. All sensor readings fall within TNPCB regulatory safe envelopes.")

st.markdown("<br>", unsafe_allow_html=True)

# 3. Feature Contribution Chart (model-derived SHAP)
st.subheader("📊 SHAP Feature Attribution Analysis")
st.caption("Horizontal bar chart showing how much each parameter pushed the AI prediction towards **Unsafe (Red Bars)** or **Safe (Green Bars)**.")

if not explain_res.get("shap_available", True):
    st.error("Model-based SHAP attribution is unavailable for this model configuration.")
else:
    shap_data = []
    for feat, val in explain_res["feature_contributions"].items():
        attribution_impact = (
            "Pushes Unsafe (Hazard)"
            if val > 0
            else "Pushes Safe (Optimal)"
        )
        shap_data.append({
            "Feature": feat,
            "Contribution Score": val,
            "Attribution Impact": attribution_impact,
            "Color": "#EF4444" if attribution_impact == "Pushes Unsafe (Hazard)" else "#22C55E"
        })

    df_shap = pd.DataFrame(shap_data).sort_values(by="Contribution Score", ascending=True)
    fig_shap = px.bar(
        df_shap,
        x="Contribution Score",
        y="Feature",
        orientation="h",
        color="Attribution Impact",
        color_discrete_map={"Pushes Unsafe (Hazard)": "#EF4444", "Pushes Safe (Optimal)": "#22C55E"},
        template="plotly_white"
    )
    fig_shap.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Attribution Impact Score (SHAP)")
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("---")

# 4. Domain Violation Explanation Box
st.subheader("⚖️ Domain Physical Law & Regulation Audit")
st.caption("Verification of hard thermodynamic and hydro-chemical conservation boundaries.")

if pred_res["status_tier"] in {"High Risk", "Critical Threat"}:
    st.error(
        "**Physical Conservation Law or Regulatory Limit Breached (WHO/TNPCB)**\n\n"
        f"**Physical Audit Summary:** {pred_res['explanation']}\n\n"
        f"**Regulatory Standard Reference:** {explain_res['regulation_reference']}"
    )
elif pred_res["status_tier"] == "Safe / Low Risk":
    st.success(
        "**No Physical Conservation Laws Broken.** "
        f"Water quality conforms to regulatory standards: {explain_res['regulation_reference']}"
    )
else:
    st.warning(
        "**Operational monitoring required.** The sample is classified as Medium Risk; "
        f"review applicable limits: {explain_res['regulation_reference']}"
    )

st.markdown("---")

# 5. Weekly Compliance Summary Panel
st.subheader("📑 Weekly Compliance & Audit Summary Panel")
st.caption("Aggregate metrics computed dynamically from your session audit log.")

comp_col1, comp_col2 = st.columns([1, 1])

total_audited = len(logs)
violations_count = sum(1 for l in logs if "UNSAFE" in l["Prediction"])
compliance_pct = max(0.0, ((total_audited - violations_count) / total_audited) * 100) if total_audited > 0 else 100.0

with comp_col1:
    st.markdown("#### Dynamic Audit Session Summary")
    st.markdown(f"""
    - **Total Audited Samples:** {total_audited} Samples
    - **Violations Mitigated:** {violations_count} Violations
    - **Overall Compliance Score:** {compliance_pct:.1f}% (TNPCB / CPCB Standard)
    """)
    
    # PDF Generator
    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        audit_rows = [["Sample ID", "Site", "Risk tier", "Violation"]]
        audit_rows.extend([[log["Sample ID"], log["Site Name"], log["Risk Tier"], log["Violation Flag"]] for log in logs])
        action_rows = [["Time", "Sample", "Action", "Projected status"]]
        action_rows.extend([
            [entry["Timestamp"], entry["Sample ID"], entry["Action"], entry["Projected Post-Recourse Status"]]
            for entry in st.session_state.get("actuator_log", [])
        ])
        audit_table = Table(audit_rows, repeatRows=1)
        action_table = Table(action_rows, repeatRows=1)
        for table in (audit_table, action_table):
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
        story = [
            Paragraph("WATER QUALITY COMPLIANCE AUDIT REPORT", styles['Heading1']),
            Paragraph(f"<b>Issued By:</b> AquaGuard Enterprise AI Governance Platform", styles['Normal']),
            Paragraph(f"<b>Period:</b> Active Session Audit | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
            Spacer(1, 15),
            Paragraph(f"<b>Compliance Summary:</b> {total_audited} samples evaluated; {violations_count} violation flags recorded. Current compliance rate: {compliance_pct:.1f}%.", styles['Normal']),
            Spacer(1, 12),
            Paragraph("Per-sample audit trail", styles['Heading2']),
            audit_table,
            Spacer(1, 12),
            Paragraph("Operator action and counterfactual trail", styles['Heading2']),
            action_table,
        ]
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    pdf_bytes = generate_pdf()
    st.download_button("Download Compliance Audit PDF", data=pdf_bytes, file_name=f"AquaGuard_Compliance_Audit_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", type="primary")

with comp_col2:
    st.markdown("#### Risk Distribution (Active Session)")
    
    safe_cnt = sum(1 for l in logs if "Safe" in l["Risk Tier"])
    med_cnt = sum(1 for l in logs if "Medium" in l["Risk Tier"])
    high_cnt = sum(1 for l in logs if "High" in l["Risk Tier"] or "Critical" in l["Risk Tier"])
    
    df_pie = pd.DataFrame({
        "Status": ["Safe / Low Risk", "Medium Risk", "High Risk / Critical"],
        "Count": [safe_cnt, med_cnt, high_cnt]
    })
    
    fig_pie = px.pie(df_pie, values="Count", names="Status", color="Status",
                       color_discrete_map={"Safe / Low Risk": "#22C55E", "Medium Risk": "#F59E0B", "High Risk / Critical": "#EF4444"},
                       hole=0.4, template="plotly_white")
    fig_pie.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)
