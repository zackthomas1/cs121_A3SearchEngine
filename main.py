import nltk
from argparse import ArgumentParser
from inverted_index import InvertedIndex
from utils import get_logger

"""
Entry point
Call 'python main.py' from the command line to run program
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="developer\DEV")
    parser.add_argument("--restart", action="store_true", default=False)
    args = parser.parse_args()
    
    # Download NLTK resources
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    index = InvertedIndex()
    index.build_index(args.rootdir)
    index.build_master_index()
    #print(index.token_in_doc_count)

    print(f"[SYSMSG] Testing: Writing docid and tf-idf score for all tokens")
    # for token, posting in index.index_tfidf.items():
    #     print(f"{token}:")
    #     for post in set(posting):
    #         print(f"   {post[0]} - {post[1]}")
    print(f"[SYSMSG] Check directory: All tokens written to the file!")

    # print(index.search("Iftekhar Ahmed"))
    