from pathlib import Path

import matplotlib.pyplot as plt

from evaluation import EvaluationResult
from vectorization import VectorizationResult, get_top_words


def plot_top_words(
    bow_result: VectorizationResult,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Erstellt ein Balkendiagramm der häufigsten BoW-Wörter."""

    top_words = get_top_words(bow_result, top_n=top_n)

    words = [word for word, _ in top_words][::-1]
    counts = [count for _, count in top_words][::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(words, counts)
    plt.xlabel("Häufigkeit")
    plt.ylabel("Wort")
    plt.title("Häufigste Wörter der Bag-of-Words-Darstellung")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_evaluation(
    lda_result: EvaluationResult,
    nmf_result: EvaluationResult,
    output_path: Path,
) -> None:
    """Vergleicht die Evaluationskennzahlen beider Modelle."""

    model_names = ["LDA", "NMF"]
    diversity = [
        lda_result.topic_diversity,
        nmf_result.topic_diversity,
    ]
    overlap = [
        lda_result.average_topic_overlap,
        nmf_result.average_topic_overlap,
    ]

    x_positions = range(len(model_names))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        [position - width / 2 for position in x_positions],
        diversity,
        width,
        label="Topic Diversity",
    )

    plt.bar(
        [position + width / 2 for position in x_positions],
        overlap,
        width,
        label="Topic-Überschneidung",
    )

    plt.xticks(list(x_positions), model_names)
    plt.ylabel("Kennzahl")
    plt.title("Vergleich der Topic-Modelle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_coherence_scores(
    topic_count_results,
    output_path: Path,
) -> None:
    """Visualisiert die Coherence Scores für verschiedene Themenanzahlen."""

    topic_counts = [
        result.n_topics
        for result in topic_count_results
    ]

    lda_scores = [
        result.lda_coherence
        for result in topic_count_results
    ]

    nmf_scores = [
        result.nmf_coherence
        for result in topic_count_results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        topic_counts,
        lda_scores,
        marker="o",
        label="LDA",
    )

    plt.plot(
        topic_counts,
        nmf_scores,
        marker="o",
        label="NMF",
    )

    plt.xlabel("Anzahl der Topics")
    plt.ylabel(r"Coherence Score ($C_v$)")
    plt.title("Coherence Score in Abhängigkeit von der Themenanzahl")
    plt.xticks(topic_counts)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()