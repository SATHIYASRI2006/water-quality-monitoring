import os
import re

with open("model_backend.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Rename DomainConstrainedMLP -> WaterQualityMLP
content = content.replace("DomainConstrainedMLP", "WaterQualityMLP")

# 2. Update feat_cols
old_feat_cols = "feat_cols = ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL', 'bod_mgL', 'hour', 'day', 'month', 'dayofweek']"
new_feat_cols = "feat_cols = ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL']"
content = content.replace(old_feat_cols, new_feat_cols)
# Just in case it was the original one
old_feat_cols_2 = "feat_cols = ['orp_mV', 'ec_uScm', 'tds_mgL', 'turbidity_NTU', 'temp_C', 'pH', 'do_mgL', 'hour', 'day', 'month', 'dayofweek']"
content = content.replace(old_feat_cols_2, new_feat_cols)

# 3. Update unscale_row
old_unscale = """    ph = float(np.clip(physical["pH"], 2.0, 12.0))
    do = float(np.clip(physical["do_mgL"], 0.0, 15.0))
    turb = float(np.clip(physical["turbidity_NTU"], 0.1, 150.0))
    tds = float(np.clip(physical["tds_mgL"], 50.0, 3000.0))
    bod = float(np.clip(physical["bod_mgL"], 0.5, 30.0))

    return {
        "ph": ph,
        "do": do,
        "bod": bod,
        "bod_is_proxy": False,
        "turbidity": turb,
        "tds": tds
    }"""
new_unscale = """    ph = float(np.clip(physical["pH"], 2.0, 12.0))
    do = float(np.clip(physical["do_mgL"], 0.0, 15.0))
    turb = float(np.clip(physical["turbidity_NTU"], 0.1, 600.0))
    tds = float(np.clip(physical["tds_mgL"], 0.0, 5000.0))

    return {
        "ph": ph,
        "do": do,
        "turbidity": turb,
        "tds": tds
    }"""
content = content.replace(old_unscale, new_unscale)
# Just in case it was original
content = re.sub(r'    bod = float\(np\.clip\(tds \* 0\.01 \+ 2\.5, 0\.5, 20\.0\)\)\n\n    return \{\n        "ph": ph,\n        "do": do,\n        "bod": bod,\n        "bod_is_proxy": True,\n        "turbidity": turb,\n        "tds": tds\n    \}', new_unscale, content)

# 4. Remove BOD from predict_with_pytorch
content = re.sub(r'    bod_val = float\(np\.clip\(tds_val \* 0\.01 \+ 2\.5, 0\.5, 20\.0\)\)\n\n    allowed_bod = 2\.0 \+ \(0\.6 \* do_val\)\n    domain_violation = \(ph_val < 4\.0 or ph_val > 10\.5 or bod_val > allowed_bod\)', 
                 r'    domain_violation = (ph_val < 4.0 or ph_val > 10.5 or do_val < 3.0)', content)
content = re.sub(r'    allowed_bod = 2\.0 \+ \(0\.6 \* do_val\)\n    domain_violation = \(ph_val < 4\.0 or ph_val > 10\.5 or bod_val > allowed_bod\)', 
                 r'    domain_violation = (ph_val < 4.0 or ph_val > 10.5 or do_val < 3.0)', content)


# 5. Remove BOD from set_active_sample_from_row and audit_logs
content = content.replace('        "bod": phys["bod"],\n', '')
content = content.replace('            "BOD (mg/L)": f"{phys[\'bod\']:.1f}",\n', '')
content = content.replace('or bod > 8.0', '')

# 6. Rename explain_sample_shap
content = content.replace("explain_sample_shap", "explain_sample_heuristic")

# 7. Remove BOD logic from calculate_recourse_actions
content = re.sub(r'    if bod > 5\.0:\n        before_after\.append\(.*\n        before_after\.append\(.*\n        checklist\.append\(.*\n        checklist\.append\(.*\n', '', content)
content = re.sub(r'            elif action\["parameter"\] == "Biological Oxygen Demand \(BOD\)":\n                physical\["tds_mgL"\] = \(target - 2\.5\) / 0\.01\n', '', content)
content = content.replace(', bod', '')
content = content.replace('bod: float, ', '')


# 8. Fix global sidebar text
old_sidebar_text = """    st.sidebar.markdown(\"\"\"
    - **PINN Engine:** `Online (v2.4)`
    - **Telecontrol Link:** `Connected (0.4ms)`
    - **Standards:** `WHO / TNPCB 1974`
    - **Dataset:** `1,722 Real Rows`
    \"\"\")"""
new_sidebar_text = """    st.sidebar.markdown(\"\"\"
    - **Model:** `WaterQualityMLP`
    - **Standards:** `WHO / TNPCB 1974`
    - **Dataset:** `2,153 Real Rows`
    \"\"\")"""
content = content.replace(old_sidebar_text, new_sidebar_text)

# Also fix the exception for missing .pth to fail loudly
content = content.replace("if model_path.exists():\n        model.load_state_dict(torch.load(model_path))\n    model.eval()",
                          "if not model_path.exists():\n        raise FileNotFoundError(f'Missing {model_path}! Please run train.py first.')\n    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))\n    model.eval()")

with open("model_backend.py", "w", encoding="utf-8") as f:
    f.write(content)

print("model_backend.py successfully modified.")
