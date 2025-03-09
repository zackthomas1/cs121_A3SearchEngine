from flask import Flask, render_template, request
from inverted_index import InvertedIndex
from query import process_query, ranked_search_cosine_similarity
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

avg_doc_length      = inverted_index.load_meta_data_from_disk()["aavg_doc_length"]
total_docs          = inverted_index.load_meta_data_from_disk()["total_doc_indexed"]

@app.route('/', methods=['GET', 'POST'])
def index():
    top_results = []
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            # Tokenize and stem the query string
            query_tokens = process_query(query)
            ranked_results = ranked_search_cosine_similarity(query_tokens, inverted_index, total_docs, doc_norms, token_to_file_map)
            # 
            for doc_id, score in ranked_results:
                url = doc_id_url_map.get(str(doc_id)) or doc_id_url_map.get(doc_id) or "Unknown URL"
                top_results.append(url)
    
    return render_template('index.html', top_results=top_results[:10])

if __name__ == '__main__':
    app.run(debug=True)
