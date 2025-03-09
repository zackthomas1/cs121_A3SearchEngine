import time
from flask import Flask, render_template, request
from inverted_index import InvertedIndex
from query import process_query, ranked_search_cosine_similarity, ranked_search_bm25
from utils import get_logger

app = Flask(__name__)
app_logger = get_logger("APP")

# Create and load inverted index
inverted_index = InvertedIndex()

# Initialize globals when the module is loaded.
doc_id_url_map      = inverted_index.load_doc_id_map_from_disk()
doc_lengths         = inverted_index.load_doc_lengths_from_disk()
doc_norms           = inverted_index.load_doc_norms_from_disk()
token_to_file_map   = inverted_index.load_token_to_file_map_from_disk()

avg_doc_length      = inverted_index.load_meta_data_from_disk()["avg_doc_length"]
total_docs          = inverted_index.load_meta_data_from_disk()["total_doc_indexed"]

@app.route('/', methods=['GET', 'POST'])
def index():
    top_results     = []
    start_time      = 0.0
    end_time        = 0.0
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            start_time = time.perf_counter() * 1000
            # Tokenize and stem the query string
            query_tokens = process_query(query)
            # ranked_results = ranked_search_cosine_similarity(query_tokens, inverted_index, total_docs, doc_norms, token_to_file_map)
            ranked_results = ranked_search_bm25(query_tokens, inverted_index, total_docs, avg_doc_length, doc_lengths, token_to_file_map)
            end_time = time.perf_counter() * 1000

            # 
            for doc_id, score in ranked_results:
                url = doc_id_url_map.get(str(doc_id)) or doc_id_url_map.get(doc_id) or "Unknown URL"
                top_results.append(url)
    
    return render_template('index.html', top_results=top_results, results_count = len(top_results), delta_time = f"{end_time - start_time:.0f}")

if __name__ == '__main__':
    app.run(debug=True)