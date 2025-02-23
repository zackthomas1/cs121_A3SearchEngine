import os
import json
from bs4 import BeautifulSoup
from collections import Counter
from typing import Dict, List, Counter
import nltk
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import re


# punkt is a tokenization model in the NLTK data package
#   which provides pre-trained tokenizers for splitting text into sentences and words
#   that effectively handles punctuation and contractions especially.
# nltk.word_tokenize() requires punkt to be downloaded before getting called.
# nltk.download("punkt")  # THIS IS MOVED TO main.py


class Preprocessor:
    def __init__(self, rootdir: str):
        self.root: str = rootdir                   # path to root directory
        self.data: Dict[str, Dict[str, str]] = {}  # dictionary data of all JSON file
        self.loaded: bool = False                  # True if data is loaded
        self.re_alnum = re.compile(r'[a-z0-9]+')
        self.stemmer = PorterStemmer()             # Porter Stemmer from NLTK package


    def load_data(self) -> None:
        """
        Parse path to directory that includes JSON files.
        Use os.walk to traverse through all subfolders to access files.

        Parameter:
            The relative path to directory to start crawling local files (e.g. ..\DEV)

        Returns (Saves to instance variable self.data):
            data dictionary that has following elements:
                key: relative path to each JSON files
                    ..\DEV\aiclub_ics_uci_edu\906c2...ecc3b.json
                value: dictionary that contains following elements:
                    key: data type
                        ['url', 'content', 'encoding']
                    value: corresponding values
                        "url": "https://aiclub.ics.uci.edu/"
                        "content": "<!DOCTYPE html>...</body>\r\n</html>"
                        "encoding": "utf-8"
                        NOTE: not yet exhasutively checked that content types are always same for all JSON files

        Reference Python Library Documents:
            os.walk:   https://docs.python.org/3.13/library/os.html#os.walk
            json.load: https://docs.python.org/3/library/json.html#json.load

        NOTE: Few questions with assumptions is annotated with (ask TA). Delete when resolved.
        """
        data: Dict[str, Dict[str, str]] = {}  # {filepath: {data type (e.g. url, html content): corresponding data value}}
        try:
            # Read all JSON files in all subfolders and load into dictionary
            for root, _, files in os.walk(self.root):  # os.walk returns [current folder, subfolders, files] and if subfolders exists, it 
                for file in files:
                    if file.endswith(".json"):  # NOTE: If all files are guaranteed to be JSON, we can get rid of this. (ask TA)
                        filepath = os.path.join(root, file)  # Generates full path (adaptive to operating system)
                        with open(filepath, "r", encoding="utf-8") as fp:
                            data[filepath] = json.load(fp)  # json.load converts JSON file to Python Dictionary
        except Exception as e:
            print(f"[ERROR] {type(e)} from load_data(): Data failed to load")
            # raise  # no need to close fp manually since "with open" handles it
        self.loaded = True
        self.data = data


    def parse_html(self, content: str) -> str:
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    

    def tokenize(self, text: str) -> List[str]:
        """
        Return:
            List of tokens stemmed by Porter Stemmer
        Tokenizer:
            https://www.nltk.org/howto/tokenize.html#regression-tests-nltkwordtokenizer
        Porter Stemmer:
            https://www.nltk.org/howto/stem.html#unit-tests-for-the-porter-stemmer
        """
        #words = word_tokenize(text.lower())  # 1. Use NLTK to tokenize all lowercased text
        text = text.lower()
        words = self.re_alnum.findall(text)  # Use compiled re expression to tokenize alphanumeric
        return list(map(self.stemmer.stem, words))  # Time and memory efficient than list comprehension
        # return [self.stemmer.stem(word) for word in words]  # 2. Stem them using PorterStemmer
    

    def get_tok_freq(self, text: str) -> Counter:
        """
        Counter for frequency {token: freq}
        Alternatively:
            https://www.nltk.org/api/nltk.probability.FreqDist.html#nltk.probability.FreqDist 
        """
        words = word_tokenize(text.lower())
        return Counter([self.stemmer.stem(word) for word in words])
        

    def get_root(self) -> str:
        return self.root
    

    def get_data(self) -> Dict[str, Dict[str, str]]:
        return self.data
    

    def is_data_loaded(self) -> bool:
        return self.loaded
    
    
