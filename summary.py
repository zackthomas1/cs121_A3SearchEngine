import time
from inverted_index import InvertedIndex
from query import process_query, ranked_search_cosine_similarity

def count_tokens(index: InvertedIndex) -> int:
    """
    Counts total number of tokens in inverted index
    
    Parameters:
    index (InvertedIndex): 

    Returns:
    int: number of tokens in inverted index
    """

    index_data = index.load_master_index_from_disk()
    return len(index_data.keys())

def count_unique_tokens(index: InvertedIndex) -> int:
    """
    Counts tokens that appear only once in a single document.
    
    Parameters:
    index (InvertedIndex): 

    Returns:
    int: tokens that appear only once in a single document
    """

    unique_tokens = set()

    index_data = index.load_master_index_from_disk()

    # Iterate through all tokens
    for token, postings in index_data.items():
        # Check if the token appears in exactly one document and has a frequency of 1
        if len(postings) == 1 and postings[0][1] == 1:
            unique_tokens.add(token)

    return len(list(unique_tokens))

def retrive_relevant_urls(query: str, n: int, index: InvertedIndex) -> list[str]:
    """
    Given a query, pulls and returns the top n Urls for each from an already
    built inverted index.

    Works by finding the head n values of a sorted intersection of the results for 
    individual tokens within a query:
    Ex: "Iftekhar Ahmed" -> (search(Iftekhar) ∧ search(Ahmed))

    
    {'ahm': [[111, 1], [238, 2], [499, 1], [4360, 1], [5006, 4], [592, 3], [686, 1], [744, 1], [745, 1], [5013, 2], [5030, 2], [5035, 2], [5038, 2], [5047, 6], [5051, 2], [5095, 2], [5118, 2], [5138, 2], [835, 1], [1110, 1], [1133, 1]]}
    """
    doc_id_url_map      = index.load_doc_id_map_from_disk()
    # doc_lengths         = index.load_doc_lengths_from_disk()
    doc_norms           = index.load_doc_norms_from_disk()
    token_to_file_map   = index.load_token_to_file_map_from_disk()

    # avg_doc_length          = index.load_meta_data_from_disk()["aag_doc_length"]
    total_docs          = index.load_meta_data_from_disk()["total_doc_indexed"]    

    query_tokens = process_query(query)

    # Begin timing after recieving search query
    start_time = time.perf_counter() * 1000
    ranked_results = ranked_search_cosine_similarity(query_tokens, index, total_docs, doc_norms, token_to_file_map)
    end_time = time.perf_counter() * 1000
    print(f"Completed search: {end_time - start_time:.0f} ms")

    ranked_results = ranked_results[:n]

    # Get and return the top N urls
    ranked_urls = set()
    for result in ranked_results:
        doc_id, score = result
        url = doc_id_url_map[str(doc_id)]
        ranked_urls.add(url)

    return list(ranked_urls)

if __name__ == "__main__":
    index = InvertedIndex()
    # index.build_master_index()

    # m1 report
    # ---------------------------
    # print(f"Total number of tokens: {count_tokens(index)}")
    # print(f"Unique tokens: {count_unique_tokens(index)}")

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
        top_query_results = retrive_relevant_urls(query, N_RESULTS, index)
        for ranked_pos, url in enumerate(top_query_results, start=1):
            print(f"\t {ranked_pos}. {url}")

    # m3 report
    # ---------------------------

    N_RESULTS = 5
    queries = [
        "uci graphics",
        "graduate learning",
        "software engineering technology",
        "algorithm and data structure",
        "Women in Computer Science"
    ]
    
    print(f"Top {N_RESULTS} Query Results:")
    for query in queries:
        print(f"\tQuery: {query}")
        top_query_results = retrive_relevant_urls(query, N_RESULTS, index)
        for ranked_pos, url in enumerate(top_query_results, start=1):
            print(f"\t {ranked_pos}. {url}")