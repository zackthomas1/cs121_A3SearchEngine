import os
import json
from inverted_index import InvertedIndex
from inverted_index import PARTIAL_INDEX_DIR, MASTER_INDEX_DIR, MASTER_INDEX_FILE, DOC_ID_DIR, DOC_ID_MAP_FILE
from typing import List
from utils import get_logger
# Code Report: 
# A table with assorted numbers pertaining to your index. 
# Contains:
#   Number of documents
#   Number of [unique] tokens
#   Total size (in KB) of index on disk.

def tokens_count():
    """
    Returns the total number of tokens
    """
    # Load master index if it exists
    if os.path.exists(MASTER_INDEX_FILE):
        with open(MASTER_INDEX_FILE, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {}

    return len(index_data.keys())

def unique_tokens_count():
    """Returns tokens that appear only once in a single document."""
    unique_tokens = set()

    # Load master index if it exists
    if os.path.exists(MASTER_INDEX_FILE):
        with open(MASTER_INDEX_FILE, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {}

    # Iterate through all tokens
    for token, postings in index_data.items():
        # Check if the token appears in exactly one document and has a frequency of 1
        if len(postings) == 1 and postings[0][1] == 1:
            unique_tokens.add(token)

    return len(list(unique_tokens))

def top_n_result_urls(query: str, n: int, index: InvertedIndex) -> List[str]:
    """
    Given a query, pulls and returns the top n Urls for each from an already
    built inverted index.

    Works by finding the head n values of a sorted intersection of the results for 
    individual tokens within a query:
    Ex: "Iftekhar Ahmed" -> (search(Iftekhar) ∧ search(Ahmed))

    
    {'ahm': [[111, 1], [238, 2], [499, 1], [4360, 1], [5006, 4], [592, 3], [686, 1], [744, 1], [745, 1], [5013, 2], [5030, 2], [5035, 2], [5038, 2], [5047, 6], [5051, 2], [5095, 2], [5118, 2], [5138, 2], [835, 1], [1110, 1], [1133, 1]]}
    """
    ordered_results = index.boolean_search(query)
    
    # Load the docId map to get URLS
    id_map = {}
    if os.path.exists(DOC_ID_MAP_FILE):
        with open(DOC_ID_MAP_FILE, "r", encoding="utf-8") as f: 
            id_map = json.load(f)

    # Get and return the top N urls
    ordered_url_list = set()

    num_index = 0
    while len(ordered_url_list) < n:
        if num_index > len(ordered_results):
            break
        doc_id = ordered_results[num_index][0]
        ordered_url_list.add(id_map[str(doc_id)])
        num_index += 1

    return list(ordered_url_list)

if __name__ == "__main__":
    index = InvertedIndex()
    #index.build_master_index()

    # m1 report
    # ---------------------------
    # print(f"Total number of tokens: {tokens_count()}")
    # print(f"Unique tokens: {unique_tokens_count()}")

    # m2 report
    # ---------------------------
    N_RESULTS = 5
    queries = ["cristina lopes",
               "machine learning",
               "ACM",
               "master of software engineering"]
    
    print(f"Top {N_RESULTS} Query Results:")
    for query in queries:
        print(f"\tQuery: {query}")
        top_query_results = top_n_result_urls(query, N_RESULTS, index)
        for ranked_pos, url in enumerate(top_query_results, start=1):
            print(f"\t {ranked_pos}. {url}")