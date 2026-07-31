from pathlib import Path

import pandas as pd

from preprocessing import preprocess_dataframe
from vectorization import create_bow_matrix, create_tfidf_matrix, get_top_words
from topic_modeling import print_topics, train_lda, train_nmf


SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def load_reviews(data_path: Path) -> pd.DataFrame:
    """Lädt den Datensatz und erzeugt eine reproduzierbare Stichprobe."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Der Datensatz wurde nicht gefunden: {data_path}"
        )

    reviews = pd.read_csv(data_path, usecols=["Text", "Score"])

    if len(reviews) < SAMPLE_SIZE:
        raise ValueError(
            f"Der Datensatz enthält weniger als {SAMPLE_SIZE} Bewertungen."
        )

    reviews = reviews.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    reviews.reset_index(drop=True, inplace=True)

    return reviews


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "Reviews.csv"

    # Datensatz laden
    reviews = load_reviews(data_path)

    print(f"Geladene Bewertungen: {len(reviews)}")

    # Texte vorverarbeiten
    processed_reviews = preprocess_dataframe(reviews)

    print(f"Bewertungen nach der Bereinigung: {len(processed_reviews)}")

    print("\nVergleich eines ursprünglichen und bereinigten Textes:\n")

    print("Original:")
    print(processed_reviews.loc[0, "Text"])

    print("\nBereinigt:")
    print(processed_reviews.loc[0, "CleanText"])

    clean_texts = processed_reviews["CleanText"]

    bow_result = create_bow_matrix(clean_texts)
    tfidf_result = create_tfidf_matrix(clean_texts)

    print("\nBag-of-Words-Matrix:")
    print(f"Dokumente: {bow_result.matrix.shape[0]}")
    print(f"Merkmale: {bow_result.matrix.shape[1]}")
    print(f"Nicht-Null-Einträge: {bow_result.matrix.nnz}")

    print("\nTF-IDF-Matrix:")
    print(f"Dokumente: {tfidf_result.matrix.shape[0]}")
    print(f"Merkmale: {tfidf_result.matrix.shape[1]}")
    print(f"Nicht-Null-Einträge: {tfidf_result.matrix.nnz}")

    print("\nErste zehn Merkmale:")
    print(bow_result.feature_names[:10])

    print("\nHäufigste Wörter der BoW-Darstellung:")

    for word, count in get_top_words(bow_result):
        print(f"{word}: {count}")

    lda_result = train_lda(
        bow_matrix=bow_result.matrix,
        feature_names=bow_result.feature_names,
    )

    nmf_result = train_nmf(
        tfidf_matrix=tfidf_result.matrix,
        feature_names=tfidf_result.feature_names,
    )

    print_topics(lda_result)
    print_topics(nmf_result)

if __name__ == "__main__":
    main()