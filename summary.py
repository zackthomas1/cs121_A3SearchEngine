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

def top_n_result_urls(query: str, n: int, index: InvertedIndex, logger) -> List[str]:
    """
    Given a query, pulls and returns the top n Urls for each from an already
    built inverted index.

    Works by finding the head n values of a sorted intersection of the results for 
    individual tokens within a query:
    Ex: "Iftekhar Ahmed" -> (search(Iftekhar) ∧ search(Ahmed))

    
    {'ahm': [[111, 1], [238, 2], [499, 1], [4360, 1], [5006, 4], [592, 3], [686, 1], [744, 1], [745, 1], [5013, 2], [5030, 2], [5035, 2], [5038, 2], [5047, 6], [5051, 2], [5095, 2], [5118, 2], [5138, 2], [835, 1], [1110, 1], [1133, 1]]}
    """
    logger.info(f"Processing query: {query}")
    relevent_index = index.search(query)
    # merged_results = {docId: token_occurances}
    merged_results = {}

    # AND boolean implementation: merge docId results on token occurances
    for token in relevent_index:
        logger.info(f"Processing token: {token}")
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

    # Load the docId map to get URLS
    if os.path.exists(DOC_ID_MAP_FILE):
        with open(DOC_ID_MAP_FILE, "r", encoding="utf-8") as f: 
            id_map = json.load(f)

    logger.info(f'Info?: {merged_results}')
    # Get and return the top N urls
    ordered_url_list = set()

    num_index = 0
    while len(ordered_url_list) < n:
        if num_index > len(merged_results):
            break
        doc_id = merged_results[num_index][0]
        ordered_url_list.add(id_map[str(doc_id)])
        num_index += 1

    return list(ordered_url_list)


def print_token_data():
    """
    Gets and prints token data given an existing/built Invertedindex
    """
    print(f"Total number of tokens: {tokens_count()}")
    print(f"Unique tokens: {unique_tokens_count()}")

def print_m2_report():
    """
    Processes and writes the m2 requirements to report2.txt
    """
    N_RESULTS = 5
    queries = ["cristina lopes",
               "machine learning",
               "ACM",
               "master of software engineering"]
    index = InvertedIndex()
    logger = get_logger("TOP_N_RESULTS")

    with open("report2.txt", "w", encoding="utf-8") as file:
        file.write(f"Top {N_RESULTS} Query Results\n")
        for query in queries:
            file.write(f"{query}\n")
            top_query_results = top_n_result_urls(query, N_RESULTS, index, logger)
            for count, url in enumerate(top_query_results, start=1):
                file.write(f"\t{count}. {url}\n")


if __name__ == "__main__":
    #index = InvertedIndex()
    # index.build_master_index()

    #print_token_data()

    print_m2_report()

