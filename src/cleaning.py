import html
import re
import unicodedata

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
KEEP_CHARS_RE = re.compile(r"[^a-z0-9'!?\s]")
SPACES_RE = re.compile(r"\s+")

# Negations and intensifiers carry sentiment, so they are NOT removed
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "neither",
    "cannot", "without", "against", "but", "very", "too", "really",
}
STOPWORDS = set(STOP_WORDS) - KEEP_WORDS

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def normalize_text(text):
    """Steps 1-6: unicode, html, url, case folding, noise, spaces."""
    text = unicodedata.normalize("NFC", text)          # 1. unicode (before tokenization)
    text = html.unescape(text)                         # 2. &amp; -> &
    text = HTML_RE.sub(" ", text)                      # 3. remove <br /> etc.
    text = URL_RE.sub(" ", text)                       # 4. remove URLs
    text = text.replace("\u2019", "'").replace("\u2018", "'")  # curly -> straight apostrophe
    text = text.lower()                                # 5. case folding
    text = KEEP_CHARS_RE.sub(" ", text)                # 6. keep letters, digits, ' ! ?
    return SPACES_RE.sub(" ", text).strip()


def lemmatize_texts(texts, n_process=1, batch_size=1000):
    """Steps 7-8: spaCy lemmatization + stop-word removal (negations kept)."""
    nlp = get_nlp()
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = []
        for tok in doc:
            if tok.lower_ == "n't":          # didn't -> did + n't -> not
                tokens.append("not")
                continue
            if not tok.is_alpha:
                continue
            lemma = tok.lemma_.lower()
            if lemma in STOPWORDS or tok.lower_ in STOPWORDS:
                continue
            tokens.append(lemma)
        results.append(" ".join(tokens))
    return results


def clean_pipeline(texts, n_process=1):
    """The single cleaning entry point.
    Accepts one string or a list of strings.
    Returns (text_clean, text_lemma), same type as input."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    cleaned = [normalize_text(t) for t in texts]
    lemmas = lemmatize_texts(cleaned, n_process=n_process)
    if single:
        return cleaned[0], lemmas[0]
    return cleaned, lemmas