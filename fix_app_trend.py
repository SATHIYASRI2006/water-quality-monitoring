import os

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('y=["pH Level", "Turbidity (NTU)", "BOD (mg/L)"]', 'y=["pH Level", "Turbidity (NTU)"]')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
