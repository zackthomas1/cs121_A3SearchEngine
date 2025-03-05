import math
from inverted_index import InvertedIndex
from utils import tokenize_text, stem_tokens

def expand_query():
    pass

def tokenize_query(query: str) -> list[str]: 
    return stem_tokens(tokenize_text(query.lower()))

def ranked_boolean_search(query_tokens: list[str], inverted_index: InvertedIndex) -> list[set[str, int]]:
    """
    Parameters:
    query_tokens (list[str]): a query string 

    Returns:
    dict[str, list[tuple[int, int]]]: Inverted index containing only tokens formed from the query string
    """
   
    merged_index = inverted_index.construct_merged_index_from_disk(query_tokens)
    total_docs = len(inverted_index.get_doc_id_map_from_disk())
    scores = {}
    # AND boolean implementation: merge docId results on token occurances
    for token in query_tokens:
        relevent_documents = merged_index[token]
        doc_freq = len(relevent_documents)
        for doc_id, freq, tf in relevent_documents:
            if doc_id in scores:
                # # measures quality of document by the frequency of query tokens
                # scores[doc_id] += freq
                # measures quality of document by tf-idf score
                scores[doc_id] += compute_tf_idf(tf, doc_freq, total_docs)
            else: 
                # # measures quality of document by the frequency of query tokens
                # scores[doc_id] = freq
                # measures quality of document by tf-idf score
                scores[doc_id] = compute_tf_idf(tf, doc_freq, total_docs)

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

