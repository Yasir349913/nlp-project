import os
import re
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


train = pd.read_csv(os.path.join(RAW_DIR, "imdb_train.csv"))

# ---------- 1. Do tokenizers ----------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

# Regex tokenizer: words (with apostrophe), ratings like 8/10, punctuation runs
REGEX_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:/\d+)?|[!?.]+")


def regex_tokenize(text):
    return REGEX_TOKEN.findall(text)


def nltk_tokenize(text):
    return word_tokenize(text)


def spacy_tokenize(text):
    return [t.text for t in nlp.tokenizer(text)]


sentences = [
    "I didn't like it, honestly 3/10... Can't recommend!!!",
    "The plot is CURIOUS-YELLOW style<br /><br />but sooo boring.",
    "Directed by Mr. Smith in 1967, it costs $5.99 :) www.imdb.com",
    "It's a must-see, isn't it? Loved it!!! 10/10",
]

log("=== 1. TOKENIZER COMPARISON ===")
for s in sentences:
    log(f"\nSENTENCE: {s}")
    log(f"  regex : {regex_tokenize(s)}")
    log(f"  nltk  : {nltk_tokenize(s)}")
    log(f"  spacy : {spacy_tokenize(s)}")

# ---------- 2. Regex patterns ----------
PATTERNS = {
    "html_tag": (re.compile(r"<[^>]+>"),
                 "'<' then any chars except '>' then '>'; matches <br />"),
    "rating": (re.compile(r"\b\d{1,2}(?:\.\d)?\s*/\s*10\b"),
               "1-2 digits, optional decimal, '/10'; matches 8/10, 7.5/10"),
    "elongated": (re.compile(r"(\w)\1{2,}"),
                  "a char followed by itself 2+ more times; matches sooo, greaaat"),
    "url": (re.compile(r"http\S+|www\.\S+"),
            "http... or www... up to next whitespace"),
    "year": (re.compile(r"\b(?:19|20)\d{2}\b"),
             "4-digit year starting with 19 or 20; matches 1967, 2004"),
    "all_caps": (re.compile(r"\b[A-Z]{3,}\b"),
                 "3+ capital letters; matches GASLIGHT, CLASSICS (emphasis)"),
}

texts = train["text"].tolist()
log("\n=== 2. REGEX PATTERNS ON TRAIN ===")
for name, (pat, why) in PATTERNS.items():
    hits = sum(1 for t in texts if pat.search(t))
    example = next((pat.search(t).group(0) for t in texts if pat.search(t)), None)
    log(f"{name:10s} | pattern: {pat.pattern}")
    log(f"           | meaning: {why}")
    log(f"           | reviews matching: {hits} ({hits / len(texts):.1%}), example: {example!r}")

# ---------- 3. Token statistics on a sample ----------
sample = train["text"].sample(1000, random_state=42).tolist()
log("\n=== 3. TOKENS vs TYPES (1000 sample reviews) ===")
for name, fn in [("regex", regex_tokenize), ("nltk", nltk_tokenize), ("spacy", spacy_tokenize)]:
    toks = [tok.lower() for t in sample for tok in fn(t)]
    log(f"{name:6s} tokens: {len(toks):>8}  types: {len(set(toks)):>7}  "
        f"type/token ratio: {len(set(toks)) / len(toks):.3f}")

with open(os.path.join(REPORT_DIR, "tokenization_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nDone. Summary in reports/tokenization_summary.txt")