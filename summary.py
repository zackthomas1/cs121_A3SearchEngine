import shelve
from inverted_index import InvertedIndex
from inverted_index import PARTIAL_INDEX_DIR, MASTER_INDEX_DIR

# Code Report: 
# A table with assorted numbers pertaining to your index. 
# Contains:
#   Number of documents
#   Number of [unique] tokens
#   Total size (in KB) of index on disk.


if __name__ == "__main__":
    index = InvertedIndex()
    # index.build_master_index()
    print(len(index.get_unique_tokens()))
    # list_urls()