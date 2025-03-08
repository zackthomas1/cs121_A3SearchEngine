import math
import time
from numpy.linalg import norm
from collections import defaultdict
from inverted_index import InvertedIndex
from nltk.corpus import wordnet as wn
from utils import compute_tf_idf, get_logger

# constants 
EPSILON = 0.00001

search_logger = get_logger("SEARCH")

#TODO: Implement query token stemming

#TODO: Implement stop word remove from query

def expand_query(query: str, limit: int = 2) -> str: 
    """
    Expands input query by adding synonyms using nltk wordnet

    Parameters:
        query (str): The original query string
        limit (int): Maximum number of synonyms added to query. Prevents excessive expansion

    Returns: 
        str: The expanded query string containing original words and synonyms
    """
    words = query.lower().split()
    expanded_words = []

    for word in words: 
        # Add original word
        expanded_words.append(word)

        # get synonyms from WordNet
        synonyms = set()
        for syn in wn.synsets(word): 
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", "").lower()
                if synonym != word:
                    synonyms.add(synonym)

        # limit synonyms per word to avoid excessive expansion
        for synonym in list(synonyms)[:limit]: 
            expanded_words.append(synonym)
        
    # return expanded query as single string
    expanded_query = " ".join(expanded_words)
    search_logger.info(f"Query Expanded to: {expanded_query}")
    return expanded_query

def search_cosine_similarity(query_tokens: list[str], inverted_index: InvertedIndex, total_docs: int, precomputed_doc_norms: dict, token_to_file_map: dict) -> list:
    """

    Documents are ranked based on directional similarity rather than raw frequency.

    Parameters:
        query_tokens (list[str]): a query string 

    Returns:
        dict[str, list[tuple[int, int]]]: Inverted index containing only tokens formed from the query string
    """
    start_time = time.perf_counter() * 1000
    merged_index = inverted_index.construct_merged_index_from_disk(query_tokens, token_to_file_map)
    end_time = time.perf_counter() * 1000
    print(f"Merged Index Construction: {end_time - start_time:.0f} ms")
    scores = defaultdict(float)
    
    # Compute Query Vector: Uses raw term counts
    #TODO: Enhance by using idf
    query_vector = defaultdict(float)
    for token in query_tokens:
        query_vector[token] += 1
    # Compute Frobenius norm (Euclidean) of query vector
    query_norm = norm(list(query_vector.values()))
    
    # Computing dot product between query and document vectors
    for token in query_tokens:
        if token in merged_index:
            postings = merged_index[token]
            doc_freq = len(postings)
            for doc_id, freq, tf in postings:
                token_weight = compute_tf_idf(tf, doc_freq, total_docs)
                scores[doc_id] += token_weight * query_vector[token]

    # Normalize scores to obtain cosine similarity
    # Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)
    for doc_id in scores:
        doc_norm = precomputed_doc_norms[doc_id]
        if abs(doc_norm) > EPSILON and abs(query_norm) > EPSILON:
            scores[doc_id] /= (doc_norm * query_norm)

    # Sort the merged results by their cosine similarity (quality), tie-break with doc_id
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))