import os
import re

with open("pages/2_Explain.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Rename explain_sample_shap -> explain_sample_heuristic
content = content.replace("explain_sample_shap", "explain_sample_heuristic")

# 2. Remove BOD extraction
content = content.replace('    s_bod = float(matched_log["BOD (mg/L)"].replace(" mg/L", ""))\n', '')
content = content.replace('    s_ph, s_do, s_bod, s_site = active_s["ph"], active_s["do"], active_s["bod"], active_s["site_name"]\n', 
                          '    s_ph, s_do, s_site = active_s["ph"], active_s["do"], active_s["site_name"]\n')

# 3. Update Domain Rules to remove BOD
content = re.sub(r'        elif s_bod > 5\.0:\n            reason = f"BOD \(Biochemical Oxygen Demand\) is extremely high at \{s_bod:\.1f\} mg/L \(safe < 5\.0\)\."\n', '', content)
content = re.sub(r'        elif s_bod > 8\.0:\n            reason = f"Critical BOD Levels \(\{s_bod:\.1f\} mg/L\)\."\n', '', content)

with open("pages/2_Explain.py", "w", encoding="utf-8") as f:
    f.write(content)

print("pages/2_Explain.py successfully modified.")
