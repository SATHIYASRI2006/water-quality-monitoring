import os

with open("pages/2_Explain.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('Organic load ({s_bod:.1f} mg/L) requires active monitoring and automated bio-dosing intervention.', 'Organic load requires active monitoring and automated bio-dosing intervention.')

with open("pages/2_Explain.py", "w", encoding="utf-8") as f:
    f.write(content)
