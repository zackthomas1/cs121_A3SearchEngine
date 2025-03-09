from flask import Flask, render_template, request
from inverted_index import InvertedIndex
from query import process_query, ranked_search_cosine_similarity

app = Flask(__name__)

# Create and load your inverted index instance.
# (Assume the index has been built and stored on disk previously.)
index = InvertedIndex()

# For example, if you have a method to load the master index or doc map, you could call it here.
doc_id_url_map      = index.load_doc_id_map_from_disk()
doc_norms           = index.load_doc_norms_from_disk()
token_to_file_map   = index.load_token_to_file_map_from_disk()
total_docs          = index.load_meta_data_from_disk()["total_doc_indexed"]

@app.route('/', methods=['GET', 'POST'])
def index():
    top_results = []
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            # Tokenize and stem the query string
            query_tokens = process_query(query)
            
            # Run the search function. It returns a list of (doc_id, score) tuples.
            ranked_results = ranked_search_cosine_similarity(query_tokens, index, total_docs, doc_norms, token_to_file_map)
            
            # Convert the doc_ids to URLs (show top 10 results)
            for doc_id, score in ranked_results[:10]:
                # Ensure doc_id key type matches that in the doc_id_map
                url = doc_id_url_map.get(str(doc_id)) or doc_id_url_map.get(doc_id) or "Unknown URL"
                top_results.append(url)
    
    return render_template('index.html', top_results=top_results)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

if __name__ == '__main__':
    app.run(debug=True)
