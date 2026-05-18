import ast
import json
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent

# Extract Ripley quotes from Cornell
lines = {}
ripley_lines = set()
with open(RAW_DATA_DIR / "movie_lines.txt", encoding="iso-8859-1") as f:
    for line in f:
        parts = line.split(' +++$+++ ')
        if len(parts) == 5:
            lines[parts[0]] = parts[4].strip()
            if parts[3].strip() == 'RIPLEY':
                ripley_lines.add(parts[0])

convs = []
with open(RAW_DATA_DIR / "movie_conversations.txt", encoding="iso-8859-1") as f:
    for line in f:
        parts = line.split(' +++$+++ ')
        if len(parts) == 4:
            convs.append(ast.literal_eval(parts[3].strip()))

ripley_quotes = []
for conv in convs:
    if len(conv) >= 2:
        for i in range(len(conv) - 1):
            if conv[i+1] in ripley_lines:
                user_msg = lines.get(conv[i], '').strip()
                aura_msg = lines.get(conv[i+1], '').strip()
                if user_msg and aura_msg:
                    ripley_quotes.append({
                        "user": user_msg,
                        "assistant": aura_msg,
                        "source": "RIPLEY_CORNELL"
                    })

print(f"Extracted {len(ripley_quotes)} conversational pairs for Ripley from Cornell dataset.")

# Load the existing verbatim quotes
try:
    with open(RAW_DATA_DIR / "verbatim_quotes.json") as f:
        existing_quotes = json.load(f)
except Exception:
    existing_quotes = []

# Merge and save
all_quotes = existing_quotes + ripley_quotes
with open(RAW_DATA_DIR / "verbatim_quotes_expanded.json", "w") as f:
    json.dump(all_quotes, f, indent=2)

print(f"Total verbatim quotes now: {len(all_quotes)}")
