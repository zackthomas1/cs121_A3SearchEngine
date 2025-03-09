from numpy.linalg import norm
from collections import defaultdict
from inverted_index import InvertedIndex
from nltk.corpus import wordnet as wn
from utils import compute_tf_idf, get_logger, remove_stop_words, stem_tokens, tokenize_text

# constants 
EPSILON = 0.00001

query_logger = get_logger("QUERY")

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
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word:
                    synonyms.add(synonym)

        # limit synonyms per word to avoid excessive expansion
        for synonym in list(synonyms)[:limit]: 
            expanded_words.append(synonym)
        
    # return expanded query as single string
    expanded_query = " ".join(expanded_words)
    query_logger.info(f"Query Expanded to: {expanded_query}")
    return expanded_query

def process_query(query: str) -> list[str]:
    """
    Processes a query. Transforms query string into list of query tokens

    Parameters:
        query (str): a raw query string 

    Returns:
        list[str]: processed list of query tokens

    """
    expanded_query = expand_query(query)
    query_tokens = tokenize_text(expanded_query)
    query_tokens = remove_stop_words(query_tokens)
    query_tokens = query_tokens + stem_tokens(query_tokens)

    return query_tokens

def ranked_search_cosine_similarity(query_tokens: list[str], inverted_index: InvertedIndex, total_docs: int, precomputed_doc_norms: dict, token_to_file_map: dict) -> list:
    """

    Documents are ranked based on directional similarity rather than raw frequency.

    Parameters:
        query_tokens (list[str]): a query string 

    Returns:
        dict[str, list[tuple[int, int]]]: Inverted index containing only tokens formed from the query string
    """
    merged_index = inverted_index.construct_merged_index_from_disk(query_tokens, token_to_file_map)
    scores = defaultdict(float)
    
    # Compute query vector using raw frequency
    query_tf_counts = defaultdict(float)
    for token in query_tokens:
        query_tf_counts[token] += 1
    total_query_terms = sum(query_tf_counts.values())

    # Compute tf-idf weighted query vector
    query_weight_vector = {}
    for token, count in query_tf_counts.items():
        if token in merged_index: 
            # Normalize tf in query
            query_tf = count / total_query_terms
            df = len(merged_index[token])
            query_weight_vector[token] = compute_tf_idf(query_tf, df, total_docs)
        else: 
            # Toekn does not appear in any document, skip it
            query_weight_vector[token] = 0.0
    
    # Compute Euclidean norm of query vector
    query_norm = norm(list(query_weight_vector.values()))

    # Compute dot product between query and document vectors
    for token, query_weight in query_weight_vector.items():
        if token in merged_index:
            postings = merged_index[token]
            for doc_id, freq, tf in postings:
                df = len(postings)
                doc_token_weight = compute_tf_idf(tf, df, total_docs)
                scores[doc_id] += query_weight * doc_token_weight

    # Normalize scores to obtain cosine similarity
    # Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)
    for doc_id in scores:
        doc_norm = precomputed_doc_norms.get(doc_id, 0.0)
        if abs(doc_norm) > EPSILON and abs(query_norm) > EPSILON:
            scores[doc_id] /= (doc_norm * query_norm)
        else: 
            scores[doc_id] = 0.0

    # Sort the merged results by their cosine similarity (quality), tie-break with doc_id
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))