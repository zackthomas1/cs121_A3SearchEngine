import shelve

# Code Report: 
# A table with assorted numbers pertaining to your index. 
# Contains:
#   Number of documents
#   Number of [unique] tokens
#   Total size (in KB) of index on disk.

def print_inverse_index(inverse_index_filepath: str) -> None: 
    with shelve.open(inverse_index_filepath) as db:
        for token, doc_freq in db.items(): 
            print(f"{token} - ", end=" ")
            for docid, freq in doc_freq:
                print(f"({docid}, {freq})", end= " ")
            print()

def list_urls(docid_index_filepath: str) -> None: 
    with shelve.open(docid_index_filepath) as db:
        return list(db.values())

if __name__ == "__main__":
    print_inverse_index('shelve/inverse_index.shelve')
    # list_urls()