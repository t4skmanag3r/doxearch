import math
from collections import Counter


def compute_term_freq(document: list[str]) -> dict[str, float]:
    """Compute the term frequency for a document.
    TF = (number of times a term appears in a document) / (total number of terms in the document)
    """
    total_terms = len(document)
    term_count = Counter(document)
    term_freq = {term: count / total_terms for term, count in term_count.items()}
    return term_freq


def compute_idf(n_docs: int, doc_term_count: int) -> float:
    """Compute the IDF score for a term in a document.
    IDF = log((total number of documents) / (number of documents containing the term))
    """
    return math.log10(n_docs / doc_term_count)


def compute_tf_idf(tf: int, idf: int) -> float:
    """Compute the TF-IDF score for a term in a document.
    TF-IDF = TF * IDF
    """
    return tf * idf
