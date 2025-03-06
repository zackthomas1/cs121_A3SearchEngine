import math
import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from inverted_index import InvertedIndex
from utils import tokenize_text, stem_tokens

# constants 
EPSILON = 0.0001

def expand_query():
    pass

def tokenize_query(query: str) -> list[str]: 
    return stem_tokens(tokenize_text(query.lower()))

def ranked_boolean_search(query_tokens: list[str], inverted_index: InvertedIndex) -> list[set[str, int]]:
    """

    Documents are ranked based on directional similarity rather than raw frequency.

    Parameters:
    query_tokens (list[str]): a query string 

    Returns:
    dict[str, list[tuple[int, int]]]: Inverted index containing only tokens formed from the query string
    """
   
    merged_index = inverted_index.construct_merged_index_from_disk(query_tokens)
    total_docs = len(inverted_index.get_doc_id_map_from_disk())
    scores = defaultdict(float)
    
    query_vector = defaultdict(float)
    for token in query_tokens:
        query_vector[token] += 1
    query_norm = norm(list(query_vector.values()))  # Compute Frobenius norm (Euclidean) of query vector
    
    # Computing document score: Each document that contains query terms gets a score based on tf-idf
    for token in query_tokens:
        relevent_documents = merged_index[token]
        doc_freq = len(relevent_documents)
        for doc_id, freq, tf in relevent_documents:
            scores[doc_id] += compute_tf_idf(tf, doc_freq, total_docs) * query_vector[token]

    # Compute Euclidean norm of document vector
    doc_vector = defaultdict(float)
    for doc_id in scores:
        for token in query_tokens: 
            relevent_documents = merged_index[token]
            doc_freq = len(relevent_documents)
            if token in merged_index:
                doc_vector[doc_id] += compute_tf_idf(tf, doc_freq, total_docs)
        doc_norm = norm(list(doc_vector.values()))  # Compute Frobenius norm (Euclidean) of doc vector

        # Final cosine similarity score obtained by dividing by the product of query and doc norms.
        # Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)
        if abs(doc_norm) > EPSILON and abs(query_norm) > EPSILON:
            scores[doc_id] /= (doc_norm * query_norm)

    # Sort the merged results by their "quality"
    def sorted_docs(item: tuple[int, int]) -> tuple[int, int]:
        doc_id, score = item 
        return (-score, doc_id)
    
    return sorted(scores.items(), key=sorted_docs)

def compute_tf_idf(tf: int, doc_freq: int, total_docs: int) -> int:
    """
    Computes the tf-idf score. 
    TF(Token Frequency): term_freq / doc_length 
    IDF (Inverse Document Frequency): math.log(total_docs / (1+doc_freq))

    Parameters:
    tf (int): The token frequency score calculated during document processing
    doc_freq (int): Number of documents that contain the token
    total_docs (int): Total number of documents in corpus

    Returns:
    int: The tf-idf score
    """

    idf = math.log(total_docs / (1+doc_freq))

    return tf * idf

