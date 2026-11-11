import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # window kholne ki jagah seedha file save karega
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FIG_DIR = os.path.join(BASE_DIR, "figures")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

summary_lines = []


def log(text=""):
    print(text)
    summary_lines.append(str(text))


# 1. Load
train = pd.read_csv(os.path.join(RAW_DIR, "imdb_train.csv"))
test = pd.read_csv(os.path.join(RAW_DIR, "imdb_test.csv"))

log("=== 1. SHAPE ===")
log(f"Train: {train.shape}")
log(f"Test: {test.shape}")

# 2. Quality checks
log("\n=== 2. QUALITY CHECKS ===")
log("Missing values (train):")
log(train.isnull().sum())
log(f"Duplicate reviews (train): {train['text'].duplicated().sum()}")
log(f"Duplicate reviews (test): {test['text'].duplicated().sum()}")
overlap = set(train["text"]).intersection(set(test["text"]))
log(f"Train/Test overlap: {len(overlap)}")

# 3. Label balance
log("\n=== 3. LABEL BALANCE (train) ===")
counts = train["label"].map({0: "negative", 1: "positive"}).value_counts()
log(counts)

plt.figure(figsize=(5, 4))
sns.barplot(x=counts.index, y=counts.values)
plt.title("Label balance (train)")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "label_balance.png"), dpi=150)
plt.close()

# 4. Review length
log("\n=== 4. REVIEW LENGTH (words) ===")
train["word_count"] = train["text"].str.split().str.len()
log(train["word_count"].describe())
log("\nBy label:")
log(train.groupby("label")["word_count"].describe())

plt.figure(figsize=(8, 4))
sns.histplot(data=train, x="word_count", hue="label", bins=60)
plt.xlim(0, 1500)
plt.title("Review length (words) by label")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "review_length.png"), dpi=150)
plt.close()

# 5. Noise check
log("\n=== 5. NOISE CHECK ===")
has_html = train["text"].str.contains(r"<[^>]+>", regex=True).mean()
has_url = train["text"].str.contains(r"http\S+", regex=True).mean()
log(f"Reviews with HTML tags: {has_html:.1%}")
log(f"Reviews with URLs: {has_url:.1%}")

html_rows = train[train["text"].str.contains("<br />", regex=False)]
if len(html_rows) > 0:
    log("\nHTML wali ek sample review:")
    log(html_rows.iloc[0]["text"][:500])

# 6. Sample reviews
log("\n=== 6. SAMPLE REVIEWS ===")
for label in [0, 1]:
    log("-" * 60)
    log("LABEL: " + ("positive" if label == 1 else "negative"))
    samples = train[train["label"] == label].sample(3, random_state=42)["text"]
    for text in samples:
        log("- " + text[:300].replace("\n", " ") + " ...\n")

# Save summary
with open(os.path.join(REPORT_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nDone. Figures saved in figures/, summary in reports/eda_summary.txt")