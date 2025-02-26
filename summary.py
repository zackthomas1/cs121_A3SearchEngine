import os
import json
from inverted_index import InvertedIndex
from inverted_index import PARTIAL_INDEX_DIR, MASTER_INDEX_DIR, MASTER_INDEX_FILE, DOC_ID_DIR, DOC_ID_MAP_FILE
from typing import List
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

#TODO: WIP
def top_n_result_urls(query: str, n: int) -> List[str]:
    """
    Given a query, pulls and returns the top n Urls for each from an already
    built inverted index.

    Works by finding the head n values of a sorted intersection of the results for 
    individual tokens within a query:
    Ex: "Iftekhar Ahmed" -> (search(Iftekhar) ∧ search(Ahmed))

    
    {'ahm': [[111, 1], [238, 2], [499, 1], [4360, 1], [5006, 4], [592, 3], [686, 1], [744, 1], [745, 1], [5013, 2], [5030, 2], [5035, 2], [5038, 2], [5047, 6], [5051, 2], [5095, 2], [5118, 2], [5138, 2], [835, 1], [1110, 1], [1133, 1]]}
    """
    index = InvertedIndex()
    relevent_index = index.search(query)

    # merged_results = {docId: token_occurances}
    merged_results = {}

    # AND boolean implementation: merge docId results on token occurances
    for token in relevent_index:
        # Initialize 'merged_results' if empty
        if not merged_results:
            merged_results = {key: value for key, value in relevent_index[token]}

        # Find and merge relevent documents
        else:
            relevent_documents = relevent_index[token]

            for docId, token_occurances in relevent_documents:
                if docId in merged_results:
                    merged_results[docId] += token_occurances

    # TODO: tf-idf implementation would be somewhere here!
    # Sort the merged results by their "quality" [# of occurances]
    merged_results = sorted(merged_results.items(), key=lambda kv: (-kv[1], kv[0]))

    # Get the top n URLS
    id_map = {}

    if os.path.exists(DOC_ID_MAP_FILE):
        with open(DOC_ID_MAP_FILE, "r", encoding="utf-8") as f: 
            id_map = json.load(f)

    ordered_url_list = []
    for i in range(n):
        doc_id = merged_results[i][0]
        ordered_url_list.append(id_map[doc_id])

    return ordered_url_list


def print_token_data():
    """
    Gets and prints token data given an existing/built Invertedindex
    """
    print(f"Total number of tokens: {tokens_count()}")
    print(f"Unique tokens: {unique_tokens_count()}")

def print_m2_report():
    """
    Processes and prints the m2 requirements to the console
    """
    N_RESULTS = 5
    queries = ["cristina lopes",
               "machine learning",
               "ACM",
               "master of software engineering"]
    
    print(f"Top {N_RESULTS} Query Results")
    for query in queries:
        print(query)
        top_query_results = top_n_result_urls(N_RESULTS)
        count = 1
        for url in top_query_results:
            print(f"\t{count}. {url}")
            count += 1

if __name__ == "__main__":
    #index = InvertedIndex()
    # index.build_master_index()

    #print_token_data()

    print_m2_report()

