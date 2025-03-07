from argparse import ArgumentParser
from inverted_index import InvertedIndex

"""
Call 'python build_index.py' from the command line to construct inverted index
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="dev/corpus")
    args = parser.parse_args()
    
    index = InvertedIndex()
    index.build_index(args.rootdir)
    index.build_master_index()
    index.precompute_document_norms()