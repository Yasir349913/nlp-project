import argparse
import os
import re
import time

import pandas as pd

from cleaning import STOPWORDS, clean_pipeline, get_nlp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

TOKEN_RE = re.compile(r"[a-z0-9']+")
summary_lines = []


def log(text=""):
    print(text)
    summary_lines.append(str(text))


def vocab_size(texts):
    vocab = set()
    for t in texts:
        vocab.update(TOKEN_RE.findall(t.lower()))
    return len(vocab)


def stem_vs_lemma_examples(texts, n=12):
    """Words where Porter stem and spaCy lemma differ."""
    from nltk.stem import PorterStemmer  # imported here, not at top (keeps workers light)

    nlp = get_nlp()
    stemmer = PorterStemmer()
    found = {}
    for doc in nlp.pipe(texts):
        for tok in doc:
            if tok.is_alpha and tok.lower_ not in STOPWORDS and tok.lower_ not in found:
                lemma, stem = tok.lemma_.lower(), stemmer.stem(tok.lower_)
                if lemma != stem:
                    found[tok.lower_] = (lemma, stem)
            if len(found) >= n:
                return found
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="quick test on N rows, saves nothing")
    parser.add_argument("--procs", type=int, default=1, help="spaCy worker processes")
    args = parser.parse_args()

    start = time.time()
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ---------- 1. Load ----------
    train = pd.read_csv(os.path.join(RAW_DIR, "imdb_train.csv"))
    test = pd.read_csv(os.path.join(RAW_DIR, "imdb_test.csv"))
    log("=== 1. LOADED ===")
    log(f"Train: {train.shape}, Test: {test.shape}")

    # ---------- 2. Duplicates and leakage ----------
    log("\n=== 2. DUPLICATES / LEAKAGE ===")
    conflicts = train.groupby("text")["label"].nunique()
    log(f"Train duplicate texts with conflicting labels: {(conflicts > 1).sum()}")

    n_tr, n_te = len(train), len(test)
    train = train.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    test = test.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    log(f"Train duplicates removed: {n_tr - len(train)}")
    log(f"Test duplicates removed: {n_te - len(test)}")

    leak_mask = train["text"].isin(set(test["text"]))
    log(f"Train reviews also present in test (removed from train): {leak_mask.sum()}")
    train = train[~leak_mask].reset_index(drop=True)
    log(f"Train after dedupe: {train.shape}, Test after dedupe: {test.shape}")

    if args.sample:
        train = train.sample(args.sample, random_state=42).reset_index(drop=True)
        test = test.sample(args.sample, random_state=42).reset_index(drop=True)
        log(f"\n[SAMPLE MODE] using {args.sample} rows per split")

    vocab_before = vocab_size(train["text"])

    # ---------- 3. Cleaning pipeline ----------
    log("\n=== 3. CLEANING (clean_pipeline) ===")
    log(f"Using {args.procs} process(es)")
    train["text_clean"], train["text_lemma"] = clean_pipeline(train["text"].tolist(), n_process=args.procs)
    log(f"Train done ({time.time() - start:.0f}s elapsed)")
    test["text_clean"], test["text_lemma"] = clean_pipeline(test["text"].tolist(), n_process=args.procs)
    log(f"Test done ({time.time() - start:.0f}s elapsed)")

    # ---------- 4. Empty rows ----------
    for name, df in [("train", train), ("test", test)]:
        log(f"Empty text_lemma rows in {name}: {(df['text_lemma'].str.strip() == '').sum()}")
    train = train[train["text_lemma"].str.strip() != ""].reset_index(drop=True)
    test = test[test["text_lemma"].str.strip() != ""].reset_index(drop=True)

    # ---------- 5. Vocabulary before vs after ----------
    vocab_clean = vocab_size(train["text_clean"])
    vocab_after = vocab_size(train["text_lemma"])
    log("\n=== 4. VOCABULARY SIZE (train) ===")
    log(f"Raw text         : {vocab_before}")
    log(f"After text_clean : {vocab_clean}")
    log(f"After text_lemma : {vocab_after}")
    log(f"Reduction (raw -> lemma): {1 - vocab_after / vocab_before:.1%}")

    # ---------- 6. Stemming vs lemmatization ----------
    log("\n=== 5. LEMMA vs PORTER STEM (words where they differ) ===")
    log(f"{'word':14s}{'lemma (used)':16s}{'porter stem'}")
    for word, (lemma, stem) in stem_vs_lemma_examples(train["text_clean"].head(200).tolist()).items():
        log(f"{word:14s}{lemma:16s}{stem}")

    # ---------- 7. Before / after samples ----------
    log("\n=== 6. BEFORE / AFTER (2 samples) ===")
    for i in [0, 1]:
        log(f"[RAW]   {train.loc[i, 'text'][:250]}")
        log(f"[CLEAN] {train.loc[i, 'text_clean'][:250]}")
        log(f"[LEMMA] {train.loc[i, 'text_lemma'][:250]}")
        log("")

    log("=== 7. FINAL ===")
    log(f"Train: {train.shape}, label counts: {train['label'].value_counts().to_dict()}")
    log(f"Test: {test.shape}, label counts: {test['label'].value_counts().to_dict()}")
    log(f"Total time: {time.time() - start:.0f}s")

    if args.sample:
        print("\nSample mode: nothing saved.")
        return

    cols = ["label", "text_clean", "text_lemma"]
    train[cols].to_csv(os.path.join(PROC_DIR, "imdb_train_clean.csv"), index=False)
    test[cols].to_csv(os.path.join(PROC_DIR, "imdb_test_clean.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "preprocess_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("\nDone. Saved in data/processed/, summary in reports/preprocess_summary.txt")


if __name__ == "__main__":
    main()