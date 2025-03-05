import math
from inverted_index import InvertedIndex
from utils import tokenize_text, stem_tokens

def tokenize_query(query: str) -> list[str]: 
    return stem_tokens(tokenize_text(query))

def ranked_boolean_search(query_tokens: list[str], inverted_index: InvertedIndex) -> list[set[str, int]]:
    """
    Parameters:
    query_tokens (list[str]): a query string 

    Returns:
    dict[str, list[tuple[int, int]]]: Inverted index containing only tokens formed from the query string
    """
   
    merged_index = inverted_index.construct_merged_index_from_disk(query_tokens)

    scores = {}
    # AND boolean implementation: merge docId results on token occurances
    for token in query_tokens:
        if not scores:
            scores = {doc_id: freq for doc_id, freq, tf in merged_index[token]}
        else:
            relevent_documents = merged_index[token]
            for doc_id, freq, tf in relevent_documents:
                if doc_id in scores:
                    # measures quality of document by the frequency of query tokens
                    scores[doc_id] += freq

    # Sort the merged results by their "quality"
    def sorted_docs(item: tuple[int, int]) -> tuple[int, int]:
        doc_id, score = item 
        return (-score, doc_id)
    
    return sorted(scores.items(), key=sorted_docs)

def compute_tf_idf(token: str, doc_id: int) -> int:
    """
    Call this function in loop for all documents like below.
        tfidf_table = 
        For text[contents] in all document files:
            for each token in each text[contents]:
                tf = __calculate_tfscore(token, text)
    """

    term_freq = 0
    doc_freq = 0 
    doc_length = 0
    tf = term_freq /  doc_length 
    total_docs = 0
    idf = math.log(total_docs / (1+doc_freq))

    return tf * idf

