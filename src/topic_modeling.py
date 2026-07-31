from dataclasses import dataclass

from sklearn.decomposition import LatentDirichletAllocation, NMF


@dataclass
class TopicModelResult:
    model_name: str
    topics: list[list[tuple[str, float]]]


def extract_topics(
    model,
    feature_names: list[str],
    top_n_words: int = 10,
) -> list[list[tuple[str, float]]]:
    """Extrahiert die wichtigsten Wörter je Thema."""

    topics = []

    for topic_weights in model.components_:
        top_indices = topic_weights.argsort()[::-1][:top_n_words]

        topic = [
            (
                feature_names[index],
                float(topic_weights[index]),
            )
            for index in top_indices
        ]

        topics.append(topic)

    return topics


def train_lda(
    bow_matrix,
    feature_names: list[str],
    n_topics: int = 5,
    top_n_words: int = 10,
    random_state: int = 42,
) -> TopicModelResult:
    """Trainiert LDA auf der Bag-of-Words-Matrix."""

    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=20,
    )

    model.fit(bow_matrix)

    topics = extract_topics(
        model=model,
        feature_names=feature_names,
        top_n_words=top_n_words,
    )

    return TopicModelResult(
        model_name="LDA mit Bag of Words",
        topics=topics,
    )


def train_nmf(
    tfidf_matrix,
    feature_names: list[str],
    n_topics: int = 5,
    top_n_words: int = 10,
    random_state: int = 42,
) -> TopicModelResult:
    """Trainiert NMF auf der TF-IDF-Matrix."""

    model = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=400,
    )

    model.fit(tfidf_matrix)

    topics = extract_topics(
        model=model,
        feature_names=feature_names,
        top_n_words=top_n_words,
    )

    return TopicModelResult(
        model_name="NMF mit TF-IDF",
        topics=topics,
    )


def print_topics(result: TopicModelResult) -> None:
    """Gibt die extrahierten Themen übersichtlich aus."""

    print(f"\n{result.model_name}")

    for topic_number, topic in enumerate(result.topics, start=1):
        words = ", ".join(word for word, _ in topic)
        print(f"Thema {topic_number}: {words}")