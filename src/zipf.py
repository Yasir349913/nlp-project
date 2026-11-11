import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FIG_DIR = os.path.join(BASE_DIR, "figures")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TOKEN_RE = re.compile(r"[a-z0-9']+")
lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


train = pd.read_csv(os.path.join(RAW_DIR, "imdb_train.csv"))
sample = train["text"].sample(5000, random_state=42)

# remove html tags only so that "br" does not dominate the counts
tokens = [t for text in sample for t in TOKEN_RE.findall(re.sub(r"<[^>]+>", " ", text).lower())]
counts = Counter(tokens)

log("=== VOCABULARY (5000-review sample, html removed) ===")
log(f"Tokens: {len(tokens)}")
log(f"Types (vocabulary size): {len(counts)}")
log(f"Type/token ratio: {len(counts) / len(tokens):.4f}")

hapax = sum(1 for c in counts.values() if c == 1)
log(f"Words appearing only once (hapax legomena): {hapax} ({hapax / len(counts):.1%} of vocabulary)")

top = counts.most_common(15)
log("\nTop 15 words:")
for w, c in top:
    log(f"  {w:10s} {c}")

ranked = [c for _, c in counts.most_common()]
top100 = sum(ranked[:100]) / len(tokens)
log(f"\nTop 100 words cover {top100:.1%} of all tokens")

plt.figure(figsize=(7, 5))
plt.loglog(range(1, len(ranked) + 1), ranked)
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title("Zipf plot: IMDB reviews (5000 sample)")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "zipf_plot.png"), dpi=150)

with open(os.path.join(REPORT_DIR, "zipf_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nDone. figures/zipf_plot.png and reports/zipf_summary.txt saved")