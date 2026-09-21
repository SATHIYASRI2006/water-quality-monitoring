import os
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix NameError for g3 in app.py
# There is a block referring to g3, let's remove it if it exists.
content = re.sub(r'with g3:.*?st\.plotly_chart\(fig_bod, use_container_width=True\)\n', '', content, flags=re.DOTALL)
content = re.sub(r'with g3:\n.*?(?=\n\S|\Z)', '', content, flags=re.DOTALL)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
