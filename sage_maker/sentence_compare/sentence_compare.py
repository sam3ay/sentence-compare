#!/usr/bin/env python3

from typing import List
from sentence_transformers import SentenceTransformer
import scipy.spatial


def sentence_compare(query: List[str], corpus: List[str]):
    """compares a sentence to a other sentences and returns similarities
    over 85%

    Args:
        query ([list]): list of a single sentence to match
        corpus ([list]): list of sentence(s) to be matched against
    """
    # load pre-trained model
    embedder = SentenceTransformer("roberta-large-nli-mean-tokens")
    # encode sentences
    corpus_embeddings = embedder.encode(corpus)
    query_embedding = embedder.encode(query)
    # determine cosine distance
    distances = scipy.spatial.distance.cdist(
        [query_embedding[0]], corpus_embeddings, "cosine"
    )[0]
    # map distances to index of corpus, sentences to distance
    results = zip(range(len(distances)), distances)
    # ascending sort
    results = sorted(results, key=lambda x: x[1])

    if 1 - results[0][1] > 0.85:
        return corpus[results[0][0]]
    else:
        return False
