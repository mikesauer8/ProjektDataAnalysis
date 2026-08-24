from pathlib import Path

import pandas as pd

from preprocessing import preprocess_dataframe
from vectorization import (
    create_bow_matrix,
    create_tfidf_matrix,
    get_top_words,
)
from topic_modeling import (
    print_topics,
    train_lda,
    train_nmf,
)
from evaluation import (
    evaluate_topic_model,
    evaluate_topic_counts,
    get_optimal_topic_counts,
    print_evaluation,
    print_topic_count_evaluation,
    save_evaluation,
    save_topic_count_evaluation,
    save_topic_overview,
    save_topics,
)
from visualization import (
    plot_model_evaluation,
    plot_top_words,
)
from visualization import (
    plot_coherence_scores,
    plot_model_evaluation,
    plot_top_words,
)

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

MIN_TOPICS = 2
MAX_TOPICS = 10


def load_reviews(data_path: Path) -> pd.DataFrame:
    """Lädt den Datensatz und erzeugt eine reproduzierbare Stichprobe."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Der Datensatz wurde nicht gefunden: {data_path}"
        )

    reviews = pd.read_csv(
        data_path,
        usecols=["Text", "Score"],
    )

    if len(reviews) < SAMPLE_SIZE:
        raise ValueError(
            f"Der Datensatz enthält weniger als "
            f"{SAMPLE_SIZE} Bewertungen."
        )

    reviews = reviews.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    reviews.reset_index(
        drop=True,
        inplace=True,
    )

    return reviews


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "Reviews.csv"

    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # 1. Datensatz laden
    # ---------------------------------------------------------

    reviews = load_reviews(data_path)

    print(
        f"Geladene Bewertungen: "
        f"{len(reviews)}"
    )

    # ---------------------------------------------------------
    # 2. Textvorverarbeitung
    # ---------------------------------------------------------

    processed_reviews = preprocess_dataframe(
        reviews
    )

    print(
        f"Bewertungen nach der Bereinigung: "
        f"{len(processed_reviews)}"
    )

    print(
        "\nVergleich eines ursprünglichen "
        "und bereinigten Textes:\n"
    )

    print("Original:")
    print(
        processed_reviews.loc[0, "Text"]
    )

    print("\nBereinigt:")
    print(
        processed_reviews.loc[0, "CleanText"]
    )

    clean_texts = processed_reviews["CleanText"]

    # ---------------------------------------------------------
    # 3. Vektorisierung
    # ---------------------------------------------------------

    bow_result = create_bow_matrix(
        clean_texts
    )

    tfidf_result = create_tfidf_matrix(
        clean_texts
    )

    print("\nBag-of-Words-Matrix:")
    print(
        f"Dokumente: "
        f"{bow_result.matrix.shape[0]}"
    )
    print(
        f"Merkmale: "
        f"{bow_result.matrix.shape[1]}"
    )
    print(
        f"Nicht-Null-Einträge: "
        f"{bow_result.matrix.nnz}"
    )

    print("\nTF-IDF-Matrix:")
    print(
        f"Dokumente: "
        f"{tfidf_result.matrix.shape[0]}"
    )
    print(
        f"Merkmale: "
        f"{tfidf_result.matrix.shape[1]}"
    )
    print(
        f"Nicht-Null-Einträge: "
        f"{tfidf_result.matrix.nnz}"
    )

    print("\nErste zehn Merkmale:")
    print(
        bow_result.feature_names[:10]
    )

    print(
        "\nHäufigste Wörter "
        "der BoW-Darstellung:"
    )

    for word, count in get_top_words(
        bow_result
    ):
        print(
            f"{word}: {count}"
        )

    # ---------------------------------------------------------
    # 4. Optimale Themenanzahl anhand des Coherence Scores
    # ---------------------------------------------------------

    print(
        "\nBestimmung der optimalen "
        "Themenanzahl mittels Coherence Score..."
    )

    topic_count_results = evaluate_topic_counts(
        bow_result=bow_result,
        tfidf_result=tfidf_result,
        clean_texts=clean_texts,
        min_topics=MIN_TOPICS,
        max_topics=MAX_TOPICS,
    )

    print_topic_count_evaluation(
        topic_count_results
    )

    save_topic_count_evaluation(
        topic_count_results,
        results_dir / "coherence_scores.txt",
    )

    best_lda, best_nmf = get_optimal_topic_counts(
        topic_count_results
    )

    print("\nOptimale Themenanzahl:")

    print(
        f"LDA: {best_lda.n_topics} Topics "
        f"(Coherence: "
        f"{best_lda.lda_coherence:.3f})"
    )

    print(
        f"NMF: {best_nmf.n_topics} Topics "
        f"(Coherence: "
        f"{best_nmf.nmf_coherence:.3f})"
    )

    # ---------------------------------------------------------
    # 5. Finale Topic-Modelle
    # ---------------------------------------------------------

    lda_result = train_lda(
        bow_matrix=bow_result.matrix,
        feature_names=bow_result.feature_names,
        n_topics=best_lda.n_topics,
    )

    nmf_result = train_nmf(
        tfidf_matrix=tfidf_result.matrix,
        feature_names=tfidf_result.feature_names,
        n_topics=best_nmf.n_topics,
    )

    print_topics(
        lda_result
    )

    print_topics(
        nmf_result
    )

    # ---------------------------------------------------------
    # 6. Evaluation der finalen Modelle
    # ---------------------------------------------------------

    lda_evaluation = evaluate_topic_model(
        lda_result
    )

    nmf_evaluation = evaluate_topic_model(
        nmf_result
    )

    print_evaluation(
        lda_evaluation
    )

    print_evaluation(
        nmf_evaluation
    )

    # ---------------------------------------------------------
    # 7. Ergebnisse speichern
    # ---------------------------------------------------------

    save_topics(
        lda_result,
        results_dir / "lda_topics.txt",
    )

    save_topics(
        nmf_result,
        results_dir / "nmf_topics.txt",
    )

    save_evaluation(
        lda_evaluation,
        nmf_evaluation,
        results_dir / "evaluation.txt",
    )

    save_topic_overview(
        lda_result,
        nmf_result,
        results_dir / "topic_overview.txt",
        top_n=5,
    )

    # ---------------------------------------------------------
    # 8. Visualisierungen erzeugen
    # ---------------------------------------------------------

    plot_top_words(
        bow_result,
        figures_dir / "top_words_bow.png",
    )

    plot_model_evaluation(
        lda_evaluation,
        nmf_evaluation,
        figures_dir / "model_evaluation.png",
    )

    plot_coherence_scores(
        topic_count_results,
        figures_dir / "coherence_scores.png",
    )


if __name__ == "__main__":
    main()