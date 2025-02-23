import os
import json
from inverted_index import InvertedIndex
from inverted_index import PARTIAL_INDEX_DIR, MASTER_INDEX_DIR, MASTER_INDEX_FILE

# Code Report: 
# A table with assorted numbers pertaining to your index. 
# Contains:
#   Number of documents
#   Number of [unique] tokens
#   Total size (in KB) of index on disk.

def tokens_count():
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

if __name__ == "__main__":
    index = InvertedIndex()
    # index.build_master_index()
    print(tokens_count())
    print(unique_tokens_count())
