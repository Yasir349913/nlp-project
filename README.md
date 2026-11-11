@"
# NLP Semester Project

## Topic
Movie review sentiment analysis (IMDB Reviews). Details will be added after dataset approval.

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

## AI assistance
Claude (Anthropic) was used for guidance and explanations. Cited here as required by the course.
"@ | Out-File -Encoding utf8 README.md
