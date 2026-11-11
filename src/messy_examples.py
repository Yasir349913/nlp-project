import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(RAW_DIR, "imdb_train.csv"))
texts = train["text"].tolist()

ISSUES = {
    "HTML tags": re.compile(r"<[^>]+>"),
    "Elongated letters (sooo)": re.compile(r"\b\w*(\w)\1{3,}\w*\b"),
    "Repeated punctuation": re.compile(r"[!?.]{3,}"),
    "ALL-CAPS words": re.compile(r"\b[A-Z]{4,}\b"),
    "URLs": re.compile(r"http\S+|www\.\S+"),
    "HTML entities": re.compile(r"&\w+;|&#\d+;"),
    "Non-ASCII characters": re.compile(r"[^\x00-\x7F]"),
    "Missing space after full stop": re.compile(r"[a-z]\.[A-Z][a-z]"),
}

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


log("=== MESSY TEXT EXAMPLES (raw train data) ===")
for name, pat in ISSUES.items():
    count = sum(1 for t in texts if pat.search(t))
    log(f"\n--- {name}: {count} reviews ({count / len(texts):.1%}) ---")
    for t in texts:
        m = pat.search(t)
        if m:
            start = max(0, m.start() - 60)
            snippet = t[start:m.end() + 60].replace("\n", " ")
            log(f"  ...{snippet}...")
            break

with open(os.path.join(REPORT_DIR, "messy_examples.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nDone. reports/messy_examples.txt saved")