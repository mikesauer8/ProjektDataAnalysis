from dataclasses import dataclass
from itertools import combinations

from topic_modeling import TopicModelResult

from pathlib import Path


@dataclass
class EvaluationResult:
    model_name: str
    topic_diversity: float
    average_topic_overlap: float


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