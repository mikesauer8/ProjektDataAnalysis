import numpy as np

from dataclasses import dataclass

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass
class VectorizationResult:
    matrix: csr_matrix
    feature_names: list[str]


def create_bow_matrix(
    texts,
    max_features: int = 1000,
    min_df: int = 5,
    max_df: float = 0.95,
) -> VectorizationResult:
    """Erzeugt eine Bag-of-Words-Matrix."""

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return VectorizationResult(
        matrix=matrix,
        feature_names=feature_names,
    )


def create_tfidf_matrix(
    texts,
    max_features: int = 1000,
    min_df: int = 5,
    max_df: float = 0.95,
) -> VectorizationResult:
    """Erzeugt eine TF-IDF-Matrix."""

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return VectorizationResult(
        matrix=matrix,
        feature_names=feature_names,
    )


def get_top_words(
    result: VectorizationResult,
    top_n: int = 20,
) -> list[tuple[str, int]]:
    """Ermittelt die häufigsten Wörter einer Bag-of-Words-Matrix."""

    word_counts = np.asarray(
        result.matrix.sum(axis=0)
    ).flatten()

    top_indices = word_counts.argsort()[::-1][:top_n]

    return [
        (
            result.feature_names[index],
            int(word_counts[index]),
        )
        for index in top_indices
    ]