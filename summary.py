from inverted_index import InvertedIndex
from query import tokenize_query, ranked_boolean_search

def count_tokens(index: InvertedIndex) -> int:
    """
    Counts total number of tokens in inverted index
    
    Parameters:
    index (InvertedIndex): 

    Returns:
    int: number of tokens in inverted index
    """

    index_data = index.get_master_index_from_disk()
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

    index_data = index.get_master_index_from_disk()

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

    query_tokens = tokenize_query(query)
    ranked_results = ranked_boolean_search(query_tokens, index)
    ranked_results = ranked_results[:n]

    # Load the doc_id map to get urls
    doc_id_url_map = index.get_doc_id_map_from_disk()

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