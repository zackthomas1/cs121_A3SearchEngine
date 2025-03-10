from query import process_query, ranked_search_cosine_similarity
from inverted_index import InvertedIndex

def main():
    # Build or load your index
    index = InvertedIndex()
    index.build_index("data/")          # Build the index using your corpus
    index.build_master_index()            # Combine partial indexes into the master index
    index.precompute_doc_norms()          # Precompute document norms

    # Load necessary metadata for querying
    doc_id_url_map = index.load_doc_id_map_from_disk()
    doc_norms = index.load_doc_norms_from_disk()
    token_to_file_map = index.load_token_to_file_map_from_disk()
    total_docs = index.load_meta_data_from_disk()["total_doc_indexed"]

    # Process a multiword query
    query = "machine learning"
    processed_query = process_query(query, n=2)
    print("Processed Query Tokens:", processed_query)

    # Perform a ranked search (using cosine similarity, for example)
    results = ranked_search_cosine_similarity(processed_query, index, total_docs, doc_norms, token_to_file_map)
    print("\nSearch Results (doc_id, score):")
    for doc_id, score in results:
        url = doc_id_url_map.get(str(doc_id), "Unknown URL")
        print(f"Doc {doc_id} (URL: {url}) with score {score}")

if __name__ == "__main__":
    main()
