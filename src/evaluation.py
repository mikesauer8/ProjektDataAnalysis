from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from topic_modeling import (
    TopicModelResult,
    train_lda,
    train_nmf,
)


@dataclass
class EvaluationResult:
    model_name: str
    topic_diversity: float
    average_topic_overlap: float


@dataclass
class TopicCountResult:
    n_topics: int
    lda_coherence: float
    nmf_coherence: float


def get_topic_word_sets(
    result: TopicModelResult,
) -> list[set[str]]:
    """Gibt die wichtigsten Wörter jedes Topics als Mengen zurück."""

    return [
        {word for word, _ in topic}
        for topic in result.topics
    ]


def calculate_topic_diversity(
    result: TopicModelResult,
) -> float:
    """
    Berechnet den Anteil unterschiedlicher Wörter über alle Topics.

    Ein Wert nahe 1 bedeutet, dass sich die Topics nur wenig
    hinsichtlich ihrer wichtigsten Wörter überschneiden.
    """

    topic_word_sets = get_topic_word_sets(result)

    all_words = [
        word
        for topic_words in topic_word_sets
        for word in topic_words
    ]

    if not all_words:
        return 0.0

    return len(set(all_words)) / len(all_words)


def calculate_average_topic_overlap(
    result: TopicModelResult,
) -> float:
    """
    Berechnet die durchschnittliche Jaccard-Ähnlichkeit
    zwischen allen Topic-Paaren.

    Ein kleiner Wert bedeutet eine geringe Überschneidung.
    """

    topic_word_sets = get_topic_word_sets(result)

    if len(topic_word_sets) < 2:
        return 0.0

    overlaps = []

    for first_topic, second_topic in combinations(topic_word_sets, 2):
        union = first_topic | second_topic

        if not union:
            overlaps.append(0.0)
            continue

        intersection = first_topic & second_topic
        overlaps.append(len(intersection) / len(union))

    return sum(overlaps) / len(overlaps)


def evaluate_topic_model(
    result: TopicModelResult,
) -> EvaluationResult:
    """Berechnet die Evaluationskennzahlen eines Topic-Modells."""

    return EvaluationResult(
        model_name=result.model_name,
        topic_diversity=calculate_topic_diversity(result),
        average_topic_overlap=calculate_average_topic_overlap(result),
    )


def print_evaluation(result: EvaluationResult) -> None:
    """Gibt die Evaluation eines Topic-Modells aus."""

    print(f"\nEvaluation: {result.model_name}")
    print(f"Topic Diversity: {result.topic_diversity:.3f}")
    print(
        "Durchschnittliche Topic-Überschneidung: "
        f"{result.average_topic_overlap:.3f}"
    )


def prepare_coherence_data(
    clean_texts,
) -> tuple[list[list[str]], Dictionary]:
    """
    Bereitet die bereinigten Texte für die Berechnung
    des c_v-Coherence-Scores vor.
    """

    tokenized_texts = [
        text.split()
        for text in clean_texts
    ]

    dictionary = Dictionary(tokenized_texts)

    return tokenized_texts, dictionary


def calculate_coherence(
    result: TopicModelResult,
    tokenized_texts: list[list[str]],
    dictionary: Dictionary,
    top_n_words: int = 10,
) -> float:
    """
    Berechnet den c_v-Coherence-Score eines Topic-Modells.

    Ein höherer Wert weist auf semantisch konsistentere
    Themen hin.
    """

    topics = [
        [word for word, _ in topic[:top_n_words]]
        for topic in result.topics
    ]

    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v",
    )

    return coherence_model.get_coherence()


def evaluate_topic_counts(
    bow_result,
    tfidf_result,
    clean_texts,
    min_topics: int = 2,
    max_topics: int = 10,
) -> list[TopicCountResult]:
    """
    Trainiert LDA und NMF mit unterschiedlichen Themenanzahlen
    und berechnet jeweils den c_v-Coherence-Score.
    """

    tokenized_texts, dictionary = prepare_coherence_data(clean_texts)

    results = []

    for n_topics in range(min_topics, max_topics + 1):
        print(
            f"\nBerechne Coherence Scores für "
            f"{n_topics} Topics ..."
        )

        lda_result = train_lda(
            bow_matrix=bow_result.matrix,
            feature_names=bow_result.feature_names,
            n_topics=n_topics,
        )

        nmf_result = train_nmf(
            tfidf_matrix=tfidf_result.matrix,
            feature_names=tfidf_result.feature_names,
            n_topics=n_topics,
        )

        lda_coherence = calculate_coherence(
            lda_result,
            tokenized_texts,
            dictionary,
        )

        nmf_coherence = calculate_coherence(
            nmf_result,
            tokenized_texts,
            dictionary,
        )

        results.append(
            TopicCountResult(
                n_topics=n_topics,
                lda_coherence=lda_coherence,
                nmf_coherence=nmf_coherence,
            )
        )

    return results


def print_topic_count_evaluation(
    results: list[TopicCountResult],
) -> None:
    """Gibt die Coherence Scores für alle Themenanzahlen aus."""

    print("\nCoherence Score nach Themenanzahl:")

    for result in results:
        print(
            f"{result.n_topics} Topics | "
            f"LDA: {result.lda_coherence:.3f} | "
            f"NMF: {result.nmf_coherence:.3f}"
        )


def get_optimal_topic_counts(
    results: list[TopicCountResult],
) -> tuple[TopicCountResult, TopicCountResult]:
    """Ermittelt die beste Themenanzahl für LDA und NMF."""

    best_lda = max(
        results,
        key=lambda result: result.lda_coherence,
    )

    best_nmf = max(
        results,
        key=lambda result: result.nmf_coherence,
    )

    return best_lda, best_nmf


def save_topic_count_evaluation(
    results: list[TopicCountResult],
    output_path: Path,
) -> None:
    """Speichert die Coherence Scores für alle Themenanzahlen."""

    lines = [
        "Coherence Score nach Themenanzahl",
        "",
    ]

    for result in results:
        lines.append(
            f"{result.n_topics} Topics | "
            f"LDA: {result.lda_coherence:.3f} | "
            f"NMF: {result.nmf_coherence:.3f}"
        )

    best_lda, best_nmf = get_optimal_topic_counts(results)

    lines.extend(
        [
            "",
            "Optimale Themenanzahl:",
            (
                f"LDA: {best_lda.n_topics} Topics "
                f"(Coherence: {best_lda.lda_coherence:.3f})"
            ),
            (
                f"NMF: {best_nmf.n_topics} Topics "
                f"(Coherence: {best_nmf.nmf_coherence:.3f})"
            ),
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_topics(
    result: TopicModelResult,
    output_path: Path,
) -> None:
    """Speichert die Topics eines Modells als Textdatei."""

    lines = [result.model_name, ""]

    for topic_number, topic in enumerate(result.topics, start=1):
        words = ", ".join(word for word, _ in topic)
        lines.append(f"Thema {topic_number}: {words}")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_topic_overview(
    lda_result: TopicModelResult,
    nmf_result: TopicModelResult,
    output_path: Path,
    top_n: int = 5,
) -> None:
    """Speichert eine Übersicht der Topics mit den wichtigsten Keywords."""

    lines = [
        "Übersicht der identifizierten Themen",
        "",
    ]

    for result in (lda_result, nmf_result):
        lines.append(result.model_name)
        lines.append("")

        for topic_number, topic in enumerate(result.topics, start=1):
            keywords = [
                word
                for word, _ in topic[:top_n]
            ]

            lines.append(
                f"Thema {topic_number}: {', '.join(keywords)}"
            )

        lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_evaluation(
    lda_result: EvaluationResult,
    nmf_result: EvaluationResult,
    output_path: Path,
) -> None:
    """Speichert die Evaluationskennzahlen beider Modelle."""

    content = (
        f"{lda_result.model_name}\n"
        f"Topic Diversity: {lda_result.topic_diversity:.3f}\n"
        f"Durchschnittliche Topic-Überschneidung: "
        f"{lda_result.average_topic_overlap:.3f}\n\n"
        f"{nmf_result.model_name}\n"
        f"Topic Diversity: {nmf_result.topic_diversity:.3f}\n"
        f"Durchschnittliche Topic-Überschneidung: "
        f"{nmf_result.average_topic_overlap:.3f}\n"
    )

    output_path.write_text(content, encoding="utf-8")