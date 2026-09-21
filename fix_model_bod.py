import os
import re

with open("model_backend.py", "r", encoding="utf-8") as f:
    content = f.read()

# I will use a simple regex to replace the bod block
bod_block = r'    allowed_bod = 2\.0 \+ \(0\.6 \* do\)\n(.*?)(?=    checklist\.append)'

content = re.sub(bod_block, '', content, flags=re.DOTALL)

with open("model_backend.py", "w", encoding="utf-8") as f:
    f.write(content)
