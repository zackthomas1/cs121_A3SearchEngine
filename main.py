import time
from utils import get_logger
from inverted_index import InvertedIndex
from summary import retrive_relevant_urls

from flask import Flask, render_template, request
from inverted_index import InvertedIndex
from search import expand_query, search_cosine_similarity  # assuming your search function is defined here
from utils import tokenize_text, stem_tokens

app = Flask(__name__)
search_engine_logger = get_logger("SEARCHENGINE")

@app.route('/', methods=['GET', 'POST'])
def index():
    top_results = []
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            # Tokenize and stem the query string
            query_tokens = tokenize_text(expand_query(query))
            # Run the search function. It returns a list of (doc_id, score) tuples.
            results = search_cosine_similarity(query_tokens, inverted_index)
            # Convert the doc_ids to URLs (show top 10 results)
            for doc_id, score in results[:10]:
                # Ensure doc_id key type matches that in the doc_id_map
                url = doc_id_map.get(str(doc_id)) or doc_id_map.get(doc_id) or "Unknown URL"
                top_results.append(url)
    return render_template('index.html', top_results=top_results)

if __name__ == "__main__":

    # Create and load your inverted index instance.
    # (Assume the index has been built and stored on disk previously.)
    inverted_index = InvertedIndex()
    # For example, if you have a method to load the master index or doc map, you could call it here.
    doc_id_map = inverted_index.get_doc_id_map_from_disk()

    app.run(debug=True)

    
    input_text = ""
    while (input_text != "quit"):
        print("Please enter the query you'd like to switch (or type 'quit' to exit)")
        input_text = input()

        if (input_text != "quit"):
            print(f'Searching for "{input_text}"')

            # Begin timing after recieving search query
            start_time = time.perf_counter() * 1000

            logger.info(f'Searching "{input_text}"')
            results = retrive_relevant_urls(input_text, RESULT_NUM, index, logger)
            for count, url in enumerate(results, start=1):
                print(f"{count}: {url}")
            end_time = time.perf_counter() * 1000
            logger.info(f"Completed search: {end_time - start_time:.0f} ms")

    print("'quit' detected, exiting...")
    logger.info("User ended searching.")