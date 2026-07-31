import re

import nltk
import pandas as pd
from nltk.corpus import stopwords


def load_stopwords() -> set[str]:
    """Lädt die englischen NLTK-Stopwörter."""

    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))


ENGLISH_STOPWORDS = load_stopwords()


def clean_text(text: str) -> str:
    """Bereinigt einen englischen Bewertungstext."""

    # Text in Kleinbuchstaben umwandeln
    text = text.lower()

    # HTML-Tags entfernen
    text = re.sub(r"<[^>]+>", " ", text)

    # Zahlen, Satzzeichen und Sonderzeichen entfernen
    text = re.sub(r"[^a-z\s]", " ", text)

    # Text in einzelne Wörter zerlegen
    words = text.split()

    # Stopwörter und sehr kurze Wörter entfernen
    words = [
        word
        for word in words
        if word not in ENGLISH_STOPWORDS and len(word) > 2
    ]

    # Wörter wieder zu einem Text zusammensetzen
    return " ".join(words)


def preprocess_dataframe(reviews: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt die Texte in einem DataFrame."""

    processed_reviews = reviews.copy()

    processed_reviews["CleanText"] = processed_reviews["Text"].apply(clean_text)

    # Leere Texte entfernen, die nach der Bereinigung entstanden sind
    processed_reviews = processed_reviews[
        processed_reviews["CleanText"].str.strip().ne("")
    ]

    processed_reviews.reset_index(drop=True, inplace=True)

    return processed_reviews