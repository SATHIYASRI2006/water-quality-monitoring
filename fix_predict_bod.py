import os

with open("pages/1_Predict.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('y=["pH Level", "Dissolved Oxygen (mg/L)", "BOD (mg/L)"]', 'y=["pH Level", "Dissolved Oxygen (mg/L)"]')

with open("pages/1_Predict.py", "w", encoding="utf-8") as f:
    f.write(content)
