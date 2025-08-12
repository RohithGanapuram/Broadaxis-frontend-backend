import re

# Read the file
with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all occurrences to add type ignore comment
content = content.replace('from api import SharePointManager', 'from api import SharePointManager  # type: ignore')

# Write back to file
with open('server.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Added type ignore comments for Pylance")