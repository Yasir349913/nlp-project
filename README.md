# NLP Semester Project: IMDB Movie Review Sentiment Analysis


**Member:** <Yasir Maqsood (L1F23BSCS0209)>

## Topic

Binary sentiment classification of movie reviews (positive / negative).
Phase 1 covers data acquisition, exploration, tokenization and a reusable text cleaning pipeline.

## Dataset

| Item                | Details                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| Name                | IMDB Large Movie Review Dataset (Maas et al., 2011)                             |
| Source              | https://ai.stanford.edu/~amaas/data/sentiment/                                  |
| Size                | 50,000 labelled reviews (25,000 train + 25,000 test)                            |
| After deduplication | 24,781 train, 24,801 test (before empty-row removal)                            |
| Language            | English                                                                         |
| Licence             | Released by Stanford for research use; cite Maas et al. (2011)                  |
| Target              | `label`: positive (1) / negative (0), perfectly balanced (12,500 each in train) |

Raw data is NOT committed. It is downloaded by `src/download_data.py` into `data/raw/`.

## Project structure

```
nlp-project/
  data/            raw and processed data (git-ignored)
  src/
    download_data.py   download / load raw data
    eda.py             exploration and statistics
   tokenization.py    tokenizer comparison and regex patterns
  preprocess.py      cleaning pipeline, saves cleaned CSVs
  figures/         generated plots
  reports/         generated text summaries
  requirements.txt
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Python 3.11+ is required.

## How to reproduce (run in this order)

```
python src/download_data.py     # raw data -> data/raw/
python src/eda.py               # statistics + figures/ + reports/eda_summary.txt
python src/tokenization.py      # reports/tokenization_summary.txt
python src/preprocess.py        # cleaned data -> data/processed/
```

Output: `data/processed/imdb_train_clean.csv` and `imdb_test_clean.csv`
(columns: `label`, `text_clean`, `text_lemma`). Phase 2 loads these directly.

## Cleaning pipeline (summary)

1. Remove exact duplicates and train reviews that also appear in test (leakage).
2. Unescape HTML entities, remove HTML tags and URLs.
3. Lowercase, keep only letters, digits, apostrophes, `!` and `?`.
4. spaCy tokenization + lemmatization, stop-word removal.
5. Negations (not, no, never, ...) are kept on purpose, because they flip sentiment.

## AI assistance

Claude (Anthropic) was used for guidance and explanations. Cited here as required by the course.
