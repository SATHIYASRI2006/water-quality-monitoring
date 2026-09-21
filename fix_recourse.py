import os
import re

with open("pages/3_Recourse.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('    active_sample["bod"],\n', '')
content = content.replace('            elif item["parameter"] == "Biological Oxygen Demand (BOD)":\n                st.markdown(f"**{item[\'parameter\']}**  ➔  Reduce to **{item[\'recommended\']}**")\n', '')

with open("pages/3_Recourse.py", "w", encoding="utf-8") as f:
    f.write(content)

print("pages/3_Recourse.py successfully modified.")
