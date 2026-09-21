import os

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('"BOD Level": f"{p_row[\'bod\']:.1f} mg/L",\n', '')
content = content.replace('"BOD Level": f"{p_row[\'bod\']:.1f} mg/L"', '')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
