import re

# Read the file
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all occurrences of 'from backend.api import SharePointManager' with 'from api import SharePointManager'
content = content.replace('from backend.api import SharePointManager', 'from api import SharePointManager')

# Write back to file
with open('server.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all import statements")