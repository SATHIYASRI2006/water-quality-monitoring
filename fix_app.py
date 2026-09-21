import os
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update hardcoded string metrics to read from metrics.json
content = content.replace("with m2:\n    st.metric(\"Model F1-Score (Test)\", \"89.35%\", \"Cross-validated\")",
"""import json
with m2:
    try:
        with open("metrics.json", "r") as mf:
            m_data = json.load(mf)
            f1 = m_data.get("f1_macro", 0.0)
            dvr_m = m_data.get("domain_violation_rate_pct", 0.0)
            st.metric("Model F1-Score", f"{f1*100:.1f}%", "Macro Avg (Test Set)")
    except:
        st.metric("Model F1-Score", "N/A")
""")

# Note: The DVR metric logic is right after `with m3:`
content = re.sub(r'with m3:\n    logs = st\.session_state\.get\("audit_logs", \[\]\)\n    if len\(logs\) > 0:\n        dvr_count = sum\(1 for l in logs if "⚠️" in l\.get\("Violation Flag", ""\)\)\n        dvr = \(dvr_count / len\(logs\)\) \* 100\n    else:\n        dvr = 0\.0\n    st\.metric\("Domain Violation Rate \(DVR\)", f"\{dvr:\.1f\}%", "From active session logs"\)',
"""with m3:
    try:
        with open("metrics.json", "r") as mf:
            m_data = json.load(mf)
            dvr_m = m_data.get("domain_violation_rate_pct", 0.0)
            st.metric("Domain Violation Rate (DVR)", f"{dvr_m:.1f}%", "Test Set (Critical/Severe)")
    except:
        st.metric("Domain Violation Rate (DVR)", "N/A")""", content)

# 2. Fix the loop feature passing (7 features instead of 11/12)
content = re.sub(r"eval_p = predict_with_pytorch\(\[float\(sub_site\.iloc\[0\]\[col\]\) for col in \['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL', 'bod_mgL', 'hour', 'day', 'month', 'dayofweek'\]\]\)",
                 "eval_p = predict_with_pytorch([float(sub_site.iloc[0][col]) for col in ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL']])", content)
content = re.sub(r"eval_p = predict_with_pytorch\(\[float\(sub_site\.iloc\[0\]\[col\]\) for col in \['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL', 'hour', 'day', 'month', 'dayofweek'\]\]\)",
                 "eval_p = predict_with_pytorch([float(sub_site.iloc[0][col]) for col in ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL']])", content)


# 3. Remove BOD from Gauges
gauge_str = """
with g3:
    fig_bod = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sample["bod"],
        title={'text': "Est. BOD (Safe: < 5.0)", 'font': {'size': 15, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "#0ea5e9"},
            'steps': [
                {'range': [0, 5], 'color': "#dcfce7"},
                {'range': [5, 8], 'color': "#fef08a"},
                {'range': [8, 20], 'color': "#fee2e2"}
            ],
        }
    ))
    fig_bod.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_bod, use_container_width=True)
"""
content = content.replace(gauge_str, "")
# Re-adjust columns to be g1, g2 = st.columns(2) instead of g1, g2, g3 = st.columns(3)
content = content.replace("g1, g2, g3 = st.columns(3)", "g1, g2 = st.columns(2)")

# 4. Remove BOD from Diagnostics table
content = re.sub(r'        \{"Parameter": "BOD \(mg/L\)", "Current": f"\{sample\[\'bod\'\]:\.1f\}", "Safe Range": "< 5\.0", "Status": "⚠️ High" if sample\[\'bod\'\] > 5\.0 else "✅ Normal"\},\n', '', content)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("app.py successfully modified.")
