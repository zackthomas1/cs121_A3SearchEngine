import os
import json
import shelve
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from collections import Counter
from utils import clean_url, get_logger
from inverted_index import InvertedIndex

# create logger
logger = get_logger("MAIN")

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

    index = InvertedIndex()
    index.build_index(args.rootdir)